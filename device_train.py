import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

# import wandb
from tqdm import tqdm
import mlxu

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from tfrecord_loader import TFRecordNewInputs, load_tfrecord_dataset
from smart_open import open
from google.cloud import storage

# from google.cloud.exceptions import NotFound

from mesh_transformer.util import clip_by_global_norm, additive_weight_decay, Timer  # XD
from easylm.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule, LLaMATokenizer
)

from easylm.jax_utils import (
    JaxRNG, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint
)
from easylm.checkpoint import StreamingCheckpointer
from easylm.data import DatasetFactory


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    # optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
)

import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
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
    parser.add_argument("--data_type", type=str, default=None, help="Config file location")

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
    if 'input_ids' not in data:
        inputs = {
            "obs": data['input_tokens'],
            "target": data['target_tokens'],
        }
    else:
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
    data = data['input_ids']  # XD
    inputs = {
        "obs": data[:, :-1],
        "target": data[:, 1:],
    }

    out = network.eval(inputs)
    loss = out#["loss"]  # XDD

    return np.array(loss).mean()


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
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
   
    # alpha parameter for the exponential moving averages used to compute B_simple
    noise_scale_alpha = params.get("noise_scale_alpha", 0.01)

#     scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)
    scheduler = util.gpt3_schedule(2000, 50000, 1.2e-4, 1.2e-5)
    
    eval_mode = False  # XD
    if not eval_mode:  # XD
#         opt = optax.chain(
#             optax.scale(1 / gradient_accumulation_steps),
#             clip_by_global_norm(1, use_psum=params.get('transformation', 'xmap') == 'xmap'),  # XD
#             optax.scale_by_adam(),
#             additive_weight_decay(weight_decay),
#             optax.scale(-1),
#             optax.scale_by_schedule(scheduler)
#         )
        
        opt = optax.chain(
            optax.scale(1 / 1),
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            util.additive_weight_decay(0.1),
            optax.scale(-1),
            optax.scale_by_schedule(scheduler)
        )
        params["optimizer"] = opt
        
    checkpoint_config = StreamingCheckpointer.get_default_config()
    checkpointer = StreamingCheckpointer(
        checkpoint_config, 'output/',
        enable=jax.process_index() == 0,
    )
    params['checkpointer'] = checkpointer
#     params['load_checkpoint'] = 'params::/home/lishengping/models/trans_7b/llama_trans_7b.stream'
    params['load_checkpoint'] = 'params::/home/lishengping/models/trans_belle_7b/belle_7b.stream'
    params['load_checkpoint'] = ''

    
        
    llama_config = LLaMAConfig(**params)
#     llama_config.vocab_file = '/home/lishengping/models/trans_7b/tokenizer.model'
    llama_config.vocab_file = '/home/lishengping/models/trans_belle_7b/tokenizer.model'
    
    llama_config.num_hidden_layers = layers
    llama_config.intermediate_size = 11008
    llama_config.seed = 42
#     llama_config.load_checkpoint = '/home/lishengping/models/llama_7b_streaming'
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

    # pick initial ckpt - based on tuning vs train from scratch

    step = 0
    initial_ckpt_state_path = None
    train_loader = None

    if args.tune_model_path:
        print('`--tune_model_path` passed: we are beginning a fine-tuning run')
        fine_tuning = True
        initial_ckpt_state_path = args.tune_model_path
    # else:  # XD
    #     print('`--tune_model_path` not passed: we are continuing a fine-tuning run from a checkpoint (or we are not fine-tuning)')
    #     fine_tuning = False
    #     initial_ckpt_model_dir = model_dir
    #     initial_ckpt_path = f"gs://{bucket}/{initial_ckpt_model_dir}"
    #     meta_path = f"{initial_ckpt_path}/meta.json"

    #     try:
    #         with open(meta_path, "r") as f:
    #             meta = json.load(f)
    #         ckpt_step = meta["checkpoints"][-1]
    #         initial_ckpt_state_path = f"{initial_ckpt_path}/step_{ckpt_step}/"
    #         print(f"state will be restored from checkpoint {ckpt_step}")

    #         step = ckpt_step
    #         train_loader = meta['aux'][str(ckpt_step)].get("train_loader", None)
    #     except NotFound:
    #         # no checkpoint, start at zero
    #         print(f"No checkpoint to load at {initial_ckpt_path}. Training from scratch.")

    if initial_ckpt_state_path:
        print(f"path to load checkpoint from: {initial_ckpt_state_path}")
    else:
        print("not loading from a checkpoint")

    # set up datasets
    print("setting up datasets")

    # train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
    #                                   batch_size=(
    #                                       gradient_accumulation_steps,
    #                                       per_replica_batch * tpu_size // cores_per_replica),
    #                                   sample_size=params['seq'],
    #                                   restore_state=train_loader)
    # XD
    train_batch_size = (gradient_accumulation_steps, per_replica_batch * tpu_size // cores_per_replica)
    print('train_batch_size =', train_batch_size)
    train_dataset = load_tfrecord_dataset(f"data/{params['train_set']}", batch_size=train_batch_size, seq_len=params['seq'])

    global_val_batch = per_replica_batch * tpu_size // cores_per_replica
    if eval_mode: eval_dataset = load_tfrecord_dataset(f"data/{params['train_set']}", batch_size=(global_val_batch,), seq_len=params['seq'])  # XD

    val_sets = {}

    for k, v in params["val_set"].items():
        val_sets[k] = TFRecordNewInputs(
            f"data/{v}", batch_size=(global_val_batch,), sample_size=seq  # XDC: val bsz is different from train bsz!
        )

    # tok/sec metrics
    sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica) \
        if not eval_mode else global_val_batch  # XD

    tokens_per_step = params['seq'] * sequences_per_step

    if args.data_type == 'json':
#         tokenizer = LLaMAConfig.get_tokenizer({'vocab_file': llama_config.vocab_file})
        tokenizer = LLaMATokenizer(
            vocab_file=llama_config.vocab_file,
            add_bos_token=False,
            add_eos_token=False,
            padding_side='left',
            truncation_side='right',
            add_tokens=6,
        )
        data_dict = {'type': 'json', 
                     'path': '/home/lishengping/data/trans_alpaca_data_cleaned.json',
                     'fields_from_example': 'fields',
                     'batch_size': 1,
                     'seq_length': 2048}
        train_dataset = DatasetFactory.load_dataset(data_dict, tokenizer)
    llama_config.vocab_size = tokenizer.vocab_size
    
    print(f'vocab_size: {llama_config.vocab_size}')
        
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
                # overwrite the loaded scheduler step with zeros
                # this makes fine-tuning use the lr schedule in
                network.state["opt_state"][-1] = init_sched_state

            print(f"network loaded in {time.time() - start:.06}s")

        if not eval_mode:  # XD
            print('compiling train fn')
            start = time.time()
            if args.data_type == 'json':
                for step_data in train_dataset:
                    step_data = step_data[0]
                    print(f'step_data: {step_data}')
                    break
            else:
                step_data = next(train_dataset)
            loss, last_loss = train_step(  # XD: remove , grad_norm, grad_norm_micro
                # network, train_dataset.get_samples()  # XD
                network, step_data
            )
            step += 1
            print(f"Train fn compiled in {time.time() - start:.06}s")
        else:
            print('compiling eval fn')
            start = time.time()
            eval_step(network, next(eval_dataset))  # XD
            # for val_set in val_sets.values():
            #     eval_step(network, val_set.get_samples())
            #     val_set.reset()
            print(f"Eval fn compiled in {time.time() - start:.06}s")

        # project = params.get("wandb_project", "mesh-transformer-jax")  # XD
        # wandb.init(project=project, name=params["name"], config=params)
        wandb = None

        G_noise_avg = None
        S_noise_avg = None

#         while True:
        for step_data in train_dataset:
            step_data = step_data[0]
            if (step > 1 and step % ckpt_every == 1) or step == total_steps:  # XD: add step > 1
                print(f"saving a checkpoint for step {step}")
                save(network, step, bucket, model_dir,
                     mp=cores_per_replica,
                    #  aux={"train_loader": train_dataset.get_state()},  # XD
                     aux={"train_loader": step_data},
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
            if not eval_mode:  # XD
                loss, last_loss = train_step(  # XD: remove , grad_norm, grad_norm_micro
                    # network, train_dataset.get_samples()  # XD
                    network, step_data
                )
            else:
                loss, last_loss = eval_step(network, next(eval_dataset)), 0.  # XD
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
