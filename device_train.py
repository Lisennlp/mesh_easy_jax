import argparse
import json
import time

import jax
import jax.numpy as np
import numpy as np
import optax

# import wandb
from tqdm import tqdm


from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from tfrecord_loader import TFRecordNewInputs, load_tfrecord_dataset
from smart_open import open
from google.cloud import storage

# from google.cloud.exceptions import NotFound

from mesh_transformer.util import clip_by_global_norm, additive_weight_decay, Timer  # XD
from mesh_transformer.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule
)

from mesh_transformer.jax_utils import (
    JaxRNG, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint
)

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
    To use, download the full checkpoint archive, extract and upload to a GCS bucket, and set that as --tune-model-path
    Modify the config file:
        - set `model_dir` to where the checkpoints should be written during training
        - set `train_set`, `val_set` to index files for your data
        - set `tpu_size` to 8 (if on a v3-8)
        - set `warmup_steps`, `anneal_steps`, `lr`, `end_lr` to the lr schedule for your finetuning run
        - the global step will reset to 0, keep that in mind when writing your lr schedule
        - set `name` to specify the name of the Weights & Biases run
        - set `wandb_project` to specify the Weights & Biases project to log to
    To prepare data in the expected data format:
        - use the script `create_finetune_tfrecords.py` in this repo to create data in the expected format
        - upload the .tfrecords files to GCS
        - save their GCS paths to a index file under `data/`, see existing files for examples
    """,
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--tune-model-path", type=str, default=None, help="Base model to finetune")
    parser.add_argument("--fresh-opt", default=False, action="store_true", help="Use a newly initialized optimizer, ignoring any optimizer state saved in the base checkpoint")
    parser.add_argument("--jax_method", type=str, default=None, help="Config file location")
    parser.add_argument("--opt_type", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


def save(network, step, bucket, path, mp, aux=None, keep_n=3, delete_old=True):
    assert path
    client = storage.Client()

    if aux is None:
        aux = {}

    try:
        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)
    except:
        # create metadata file
        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    for shard_id in range(mp):
        write_ckpt(network.state, f"gs://{bucket}/{path}/step_{step}/", shard_id)

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                # print(f"deleting {blob.name}")
                assert path in blob.name
                blob.delete()
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        json.dump(meta, f)


def train_step(network, data):
    data = data['input_ids']  # XD
    inputs = {
        "obs": data[:, :, :-1],
        "target": data[:, :, 1:],
    }
    loss, last_loss = network.train(inputs)  # XD: remove , grad_norm, grad_norm_micro

    return (
        np.array(loss).mean(),
        np.array(last_loss).mean(),
        # np.array(grad_norm).mean(),  # XDD
        # np.array(grad_norm_micro).mean(),  # XDD
    )


def eval_step(network, data):
    inputs = {
        "obs": data[:, :-1],
        "target": data[:, 1:],
    }

    out = network.eval(inputs)
    loss = out["loss"]

    return np.array(loss).mean()


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params.get("per_replica_batch")
    cores_per_replica = params.get("cores_per_replica")

    assert cores_per_replica <= 8

    bucket = params.get("bucket")
    model_dir = params.get("model_dir")
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]
    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    eval_tasks = params["eval_harness_tasks"]
    total_steps = params["total_steps"]
    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]
    intermediate_size = params['intermediate_size']
   
    
    # alpha parameter for the exponential moving averages used to compute B_simple
    noise_scale_alpha = params.get("noise_scale_alpha", 0.01)

    if args.opt_type == 'mesh':
        scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)
        opt = optax.chain(
            optax.scale(1 / 1),
            clip_by_global_norm(1, use_psum=params.get('transformation', 'xmap') == 'xmap'),  # XD
            optax.scale_by_adam(),
            additive_weight_decay(weight_decay),
            optax.scale(-1),
            optax.scale_by_schedule(scheduler)
        )
    else:
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.001,
                peak_value=0.01,
                warmup_steps=2000,
                decay_steps=500000,
                end_value=0.001,
            )
        opt = optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adafactor(
                        learning_rate=learning_rate_schedule,
                        multiply_by_parameter_scale=True,
                        momentum=0.9,
                        decay_rate=0.95,
                        factored=False,
                        clipping_threshold=None,
                        dtype_momentum=jnp.bfloat16,
                    ),
                    # optax_add_scheduled_weight_decay(
                    #     lambda step: -learning_rate_schedule(step) * config.weight_decay,
                    #     weight_decay_mask
                    # )
                    )
    params["optimizer"] = opt
    
    llama_config = LLaMAConfig(**params)
    llama_config.vocab_file = '/home/lishengping/models/trans_7b/tokenizer.model'
    llama_config.num_hidden_layers = layers
    llama_config.intermediate_size = intermediate_size
    llama_config.seed = 42
    for k, v in params.items():
        llama_config.k = v
    set_random_seed(llama_config.seed)

    start = time.time()
    tpu_size = jax.device_count()
    if tpu_size < cores_per_replica:
        msg = f"each shard needs a separate device, but device count ({tpu_size}) < shard count ({cores_per_replica})"
        raise ValueError(msg)
    print(f"jax devices: {tpu_size}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    step = 0
    initial_ckpt_state_path = None
    train_loader = None

    if args.tune_model_path:
        print('`--tune_model_path` passed: we are beginning a fine-tuning run')
        fine_tuning = True
        initial_ckpt_state_path = args.tune_model_path

    if initial_ckpt_state_path:
        print(f"path to load checkpoint from: {initial_ckpt_state_path}")
    else:
        print("not loading from a checkpoint")

    # set up datasets
    print("setting up datasets")
    # XD
    train_batch_size = (gradient_accumulation_steps, per_replica_batch * tpu_size // cores_per_replica)
    print('train_batch_size =', train_batch_size)
    train_dataset = load_tfrecord_dataset(f"data/{params['train_set']}", batch_size=train_batch_size, seq_len=params['seq'])

    global_val_batch = per_replica_batch * tpu_size // cores_per_replica

    val_sets = {}

    for k, v in params["val_set"].items():
        val_sets[k] = TFRecordNewInputs(
            f"data/{v}", batch_size=(global_val_batch,), sample_size=seq  # XDC: val bsz is different from train bsz!
        )

    # tok/sec metrics
    sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    tokens_per_step = params['seq'] * sequences_per_step

    # load + run
    with jax.experimental.maps.Mesh(devices, ('dp', 'mp')):  # XD: mesh -> Mesh
        with Timer("initializing network"):  # XD
            if args.jax_method == 'haiku':
                network = CausalTransformer(params)
            elif args.jax_method == 'flax':
                network = FlaxLLaMAForCausalLMModule(llama_config)
                network.init_state()

        if initial_ckpt_state_path:
            print("loading network")
            if fine_tuning:
                # get the scheduler step stored in the just-initialized optimizer
                # should be zero
                init_sched_state = network.state["opt_state"][-1]

            start = time.time()
            network.state = read_ckpt(network.state, initial_ckpt_state_path, devices.shape[1], load_opt=(not args.fresh_opt))

            if fine_tuning:
                network.state["opt_state"][-1] = init_sched_state

            print(f"network loaded in {time.time() - start:.06}s")

        print('compiling train fn')
        start = time.time()
        if args.jax_method == 'flax':
            loss, last_loss = train_step(network, next(train_dataset))
        else:
            loss, last_loss = train_step(network, next(train_dataset))
            
        step += 1
        print(f"Train fn compiled in {time.time() - start:.06}s")

        wandb = None

        G_noise_avg = None
        S_noise_avg = None

        while True:
            if (step > 1 and step % ckpt_every == 1) or step == total_steps:  # XD: add step > 1
                print(f"saving a checkpoint for step {step}")
                save(network, step, bucket, model_dir,
                     mp=cores_per_replica,
                    #  aux={"train_loader": train_dataset.get_state()},  # XD
                     aux={"train_loader": next(train_dataset)},
                     delete_old=True,
                     )

            if step % val_every == 1:  # 1 because we've already taken a step to compile train fn
                for name, val_set in val_sets.items():
                    val_loss = []
                    for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                     desc=f"validation for step {step}, set {name}",
                                     total=val_batches):
                        val_loss.append(eval_step(network, i))
                    val_set.reset()

                    val_loss = np.array(val_loss).mean()
                    print(f"validation loss for step {step}, set {name}: {val_loss}")

                    if wandb is not None: wandb.log({f'val/loss_{name}': float(val_loss)}, step)  # XD

            if step == total_steps:
                print("training completed!")
                exit()

            start = time.time()
            loss, last_loss = train_step(network, next(train_dataset))
            step += 1

            steps_per_sec = 1 / (time.time() - start)
            tokens_per_sec = tokens_per_step * steps_per_sec
            sequences_processed = sequences_per_step * step
            tokens_processed = tokens_per_step * step

            if False:  # XD
                ### compute summary stats about the gradient

                # converts from grads-summed-over-microbatch (what `CasualTransformer.train` computes)
                # to grads-averaged-over-microbatch (what we want)
                #
                # (when taking gradient steps, the same conversion happens inside the optimizer
                #  via optax.scale(1 / gradient_accumulation_steps))
                grad_norm = grad_norm / gradient_accumulation_steps

                # compute G_noise and S_noise
                # from "An Empirical Model of Large-Batch Training" Appendix A.1
                # here, B_big = gradient_accumulation_steps, and B_small = 1 for convenience
                gbsmall = grad_norm_micro ** 2
                gbbig = grad_norm ** 2
                G_noise = (gradient_accumulation_steps * gbbig - gbsmall) / (
                    gradient_accumulation_steps - 1
                )
                S_noise = (gbsmall - gbbig) / (1 - 1 / gradient_accumulation_steps)

                noise_scale_stats = {
                    "noise/G_noise": G_noise,
                    "noise/S_noise": S_noise,
                }

            # heuristic to avoid reporting G_noise in very early training when gradients are large
            # (these take a long time to wash out of the moving average that defines B_simple)
            use_step_in_noise_avgs = False # gbbig < 2  # XD

            if use_step_in_noise_avgs:
                # compute moving averages of G_noise and S_noise, for B_simple
                if G_noise_avg is None:
                    G_noise_avg = G_noise
                else:
                    G_noise_avg = (1 - noise_scale_alpha) * G_noise_avg + noise_scale_alpha * G_noise

                if S_noise_avg is None:
                    S_noise_avg = S_noise
                else:
                    S_noise_avg = (1 - noise_scale_alpha) * S_noise_avg + noise_scale_alpha * S_noise

                B_simple = S_noise_avg / G_noise_avg

                noise_scale_stats.update(
                    {
                        "noise/G_noise_avg": G_noise_avg,
                        "noise/S_noise_avg": S_noise_avg,
                        "noise/B_simple": B_simple,
                    }
                )

            wandb_stats = {
                "train/loss": loss,
                "train/last_loss": last_loss,
                "train/steps_per_sec": steps_per_sec,
                "train/tokens_per_sec": tokens_per_sec,
                # "train/grad_norm": grad_norm,  # XDD
                # "train/learning_rate": float(scheduler(network.state["opt_state"][-1].count[0].item())),  # XDD
                "sequences_processed": sequences_processed,
                "tokens_processed": tokens_processed,
            }
            print(wandb_stats); continue  # XD
            wandb_stats.update(noise_scale_stats)

            wandb.log(wandb_stats, step)
