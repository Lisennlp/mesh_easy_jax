import argparse
import json
import time
import os
from collections import defaultdict

import numpy as np
# import wandb
from tqdm import tqdm

from mesh_transformer.build_model import build_model
from lm_eval import evaluator, tasks
from tasks.eval_harness import EvalHarnessAdaptor
from tfrecord_loader import TFRecordNewInputs, load_tfrecord_dataset
import multiprocessing
import tensorflow as tf

import jax

from easylm.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule, LLaMATokenizer
)

jax.config.update('jax_array', True)
tf.config.experimental.set_visible_devices([], "GPU")
os.environ['JAX_PLATFORMS'] = ''

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on.")
    parser.add_argument("--tpu_region", type=str, help="Region of TPU to train on.")
    parser.add_argument("--preemptible", action="store_true")

    parser.add_argument("--config", type=str, default='configs/6B_roto_256_test.json', help="Config file location")

    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")
    parser.add_argument("--version", type=int, default=3, help="Choose which model version to use, 1: pjit mesh-haiku-llama 2: xmap mesh-haiku-llama 3: pjit mesh-flax-llama")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # huggingface tokenizers gets very angry if you fork
    multiprocessing.set_start_method("spawn")

    args = parse_args()
    params = json.load(open(args.config, 'r'))

    if args.new:
        print(f"Starting experiment {params['name']} from scratch! "
              f"all data in gs://{params['bucket']}/{params['model_dir']}/ will be deleted")
        # input("Hit enter to continue")

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible
    clean_start = args.new

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    tpu_size = params["tpu_size"]
    cores_per_replica = params["cores_per_replica"]

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

    eval_batch_size = 16

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    t = build_model(params, tpu_name, region, preemptible, version=args.version)

    # try:
    print(f'bucket： {bucket} model_dir: {model_dir}')
    
    # t.save(0, bucket, model_dir, init=True, overwrite=clean_start)
    step = 0
    train_load_restore = None
    # except Exception as e:
    #     print(f"Save failed with error {e}, trying to load instead...", e)
    #     step, aux = t.load(bucket, model_dir)
    #     train_load_restore = aux.get("train_loader", None)

    #     if train_load_restore is None:
    #         print("Failed to restore train loader state")

    train_batch_size = (gradient_accumulation_steps, per_replica_batch * tpu_size // cores_per_replica)
    print('train_batch_size =', train_batch_size)
    train_dataset = load_tfrecord_dataset(f"{params['train_set']}", batch_size=train_batch_size, seq_len=params['seq'])

    global_val_batch = int(per_replica_batch * tpu_size // cores_per_replica * params.get("val_batch_multiplier", 1))

    sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    tokens_per_step = params['seq'] * sequences_per_step

    val_sets = {}

    for k, v in params['val_set'].items():
        val_sets[k] = load_tfrecord_dataset(f"{v}", batch_size=(1, global_val_batch), seq_len=params['seq'])

    # use dynamic seq length unless pe is fixed
    # adaptor = EvalHarnessAdaptor(t,
    #                              seq,
    #                              global_val_batch,
    #                              shrink=pe != "fixed",
    #                              min_seq=1024 if args.version == 2 else None)  # work around suboptimal pjit layout

    start = time.time()
    t.train(next(train_dataset))
    print(f"Train fn compiled in {time.time() - start:.06}s")

    start = time.time()
    for val_set in val_sets.values():
        t.eval(next(val_set))
    print(f"Eval fn compiled in {time.time() - start:.06}s")

    # project = params.get("wandb_project", "mesh-transformer-jax")
    # wandb.init(project=project, entity="eleutherai", name=params["name"], config=params)
    wandb = None
    # pbar = tqdm(initial=step, total=total_steps, desc="Training progress")

    while True:
        loss, last_loss = t.train(next(train_dataset))
        # wandb.log({'train/loss': loss, 'train/last_loss': last_loss}, step)
        if (step % ckpt_every == 0 and step) or step == total_steps:
            t.save(step, bucket, model_dir,
                   aux={"Train_loader": train_dataset.get_state()},
                   init=False,
                   delete_old=step % keep_every != 0)

            if step == total_steps:
                print("Training completed!")
                exit()

        if (step+1) % val_every == 0:
            eval_task_dict = defaultdict(dict)
            for val_name, val_set in val_sets.items():
                val_loss, val_acc = [], []
                while True:
                    if len(val_loss) % 100 == 0:
                        print(f'{val_name}-{len(val_loss)}-loss: {np.array(val_loss).mean()}')
                    try:
                        loss, acc = t.eval(next(val_set))
                    except:
                        break
                    val_loss.append(loss)
                    val_acc.append(acc)
                
                val_loss = np.array(val_loss).mean()
                val_acc = np.array(val_acc).mean()

                eval_task_dict[val_name]['loss'] = val_loss
                eval_task_dict[val_name]['acc'] = val_acc

                print(f"Validation loss for step {step}, dataset {val_name} loss: {val_loss} acc: {val_acc}")
                # wandb.log(f'{name} val loss: {float(val_loss)} acc: {float(val_acc)}', step)

            # results = evaluator.evaluate(adaptor, eval_task_dict, False, 0, None)

            flat_results = {}
            dumped = json.dumps(eval_task_dict, indent=2)
            print(f"Step {step} val results: {dumped}\n\n")
            # wandb.log(flat_results, step)
        step += 1

        steps_per_sec = step / (time.time() - start)
        tokens_per_sec = tokens_per_step * steps_per_sec
        sequences_processed = sequences_per_step * step
        tokens_processed = tokens_per_step * step

        wandb_stats = {
                "train/loss": loss,
                "train/last_loss": last_loss,
                "train/steps_per_sec": steps_per_sec,
                "train/tokens_per_sec": tokens_per_sec,
                "sequences_processed": sequences_processed,
                "tokens_processed": tokens_processed,
            }
        print(wandb_stats)
        # pbar.set_postfix({'loss': loss, 'last_loss': last_loss})
        # pbar.update()
