import argparse
import json
import time
import os
import re
from collections import defaultdict

import numpy as np
import wandb
from tqdm import tqdm

from mesh_transformer.build_model import build_model
from lm_eval import evaluator, tasks
from tasks.eval_harness import EvalHarnessAdaptor
from tfrecord_loader import TFRecordNewInputs, load_tfrecord_dataset
import multiprocessing
import tensorflow as tf
from google.cloud import storage

import jax

from easylm.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule, LLaMATokenizer
)

jax.config.update('jax_array', True)
tf.config.experimental.set_visible_devices([], "GPU")
os.environ['JAX_PLATFORMS'] = ''
os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

wandb.login(key='7988c805dfe3fed4d6e4017f616555a5160fd2c2')


def search_newest_train_state(params):
    """auto search bucket newest checkpoint path"""
    bucket_name = params['bucket']
    directory_path = params['model_dir']

    client = storage.Client()
    model_dirs = {}
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        if 'step_' in blob.name:
            step = re.findall('step_(\d+)',blob.name)[0]
            model_dirs[int(step)] = blob.name
    model_dirs = sorted(model_dirs.items(), key=lambda x: x[0])
    step, model_path = model_dirs[-1]
    if step > 0:
        model_path = f'trainstate::gs://{bucket_name}/{model_path}'
        assert 'train_state' in model_path
    else:
        model_path = f'params::gs://{bucket_name}/{model_path}'
    return step, model_path


def update_llama_params(params):
    params['skip_step'], params['load_checkpoint'] = search_newest_train_state(params)
    print(f'load_checkpoint: {params["load_checkpoint"]}')
    # params['vocab_file'] = '/home/lishengping/models/trans_belle_7b/tokenizer.model'
    params['num_hidden_layers'] = params.get('layers', 32)
    params['seed'] = params.get('seed', 42)
    params['rng_keys'] = ('params', 'dropout', 'fcm')
    params['gradient_checkpointing'] = params.get('gradient_checkpointing', 'nothing_saveable')
    params['embd_pdrop'] = params.get('embd_pdrop', 0.1)
    params['attn_pdrop'] = params.get('attn_pdrop', 0.0)
    params['resid_pdrop'] = params.get('resid_pdrop', 0.05)
    params['transformation'] = params.get('transformation', 'pjit')
    params['initializer_range'] = params.get('initializer_range', 0.02)
    params['fcm_min_ratio'] = params.get('fcm_min_ratio', 0.0)
    params['fcm_max_ratio'] = params.get('fcm_max_ratio', 0.0)
    params['use_cache'] = params.get('use_cache', True)
    params['rms_norm_eps'] = params.get('rms_norm_eps', 1e-6)
    params['max_sequence_length'] = params.get('seq', 2048)
    params['num_attention_heads'] = params.get('n_heads', 32)
    params['hidden_size'] = params.get('d_model', 4096)
    params['vocab_size'] = params.get('n_vocab', 32000)
    params['tie_word_embeddings'] = params.get('tie_word_embeddings', False)
    params['save_optimizer_state'] = params.get('save_optimizer_state', True)


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/6B_roto_256_test.json', help="Config file location")
    parser.add_argument("--version", type=int, default=3, help="Choose which model version to use, 1: pjit mesh-haiku-llama 2: xmap mesh-haiku-llama 3: pjit mesh-flax-llama")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # huggingface tokenizers gets very angry if you fork
    multiprocessing.set_start_method("spawn")

    args = parse_args()
    params = json.load(open(args.config, 'r'))

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
    eopch_num = params.get('epoch_num', 10)

    clip_norm = params.get('clip_norm', 1.0)

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    if int(args.version) == 3:
        update_llama_params(params)

    print(f'version: {args.version}\nparams: {params}')
    t = build_model(params, version=args.version)
    # try:
    print(f'bucket： {bucket} model_dir: {model_dir}')
    train_batch_size = (gradient_accumulation_steps, per_replica_batch * tpu_size // cores_per_replica)
    print(f'train_batch_size: {train_batch_size}')
    train_dataset = load_tfrecord_dataset(f"{params['train_set']}", batch_size=train_batch_size, seq_len=params['seq'], repeat=eopch_num)

    global_val_batch = int(per_replica_batch * tpu_size // cores_per_replica * params.get("val_batch_multiplier", 1))

    sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    tokens_per_step = params['seq'] * sequences_per_step

    val_sets = {}

    for k, v in params['val_set'].items():
        val_sets[k] = load_tfrecord_dataset(f"{v}", batch_size=(1, global_val_batch), seq_len=params['seq'], repeat=eopch_num)

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

    project = params.get("wandb_project", "mesh-transformer-jax")
    wandb.init(project=project, name=params["name"], config=params)
    skip_step = params['skip_step']
    print(f'skip_step: {skip_step}')
    step = 0
    while True:
        input_data = next(train_dataset)
        if step < skip_step:
            step += 1
            continue
        loss, acc = t.train(input_data)
        if (step % ckpt_every == 0 and step) or step == total_steps:
            t.save(step, bucket, model_dir,
                #    aux={"Train_loader": train_dataset.get_state()},
                   init=False,
                   delete_old=step % keep_every != 0)

            if step == total_steps:
                print("Training completed!")
                exit()

        if step % val_every == 0:
            eval_task_dict = defaultdict(dict)
            for val_name, val_set in val_sets.items():
                val_loss, val_acc = [], []
                val_start = time.time()
                for _ in range(val_batches):
                    loss, acc = t.eval(next(val_set))
                    val_loss.append(loss)
                    val_acc.append(acc)
                
                val_loss = np.array(val_loss).mean()
                val_acc = np.array(val_acc).mean()

                eval_task_dict[val_name]['loss'] = val_loss
                eval_task_dict[val_name]['acc'] = val_acc

                print(f"Validation loss for step {step}, dataset {val_name} loss: {val_loss} acc: {val_acc}")

            print(f"Step {step} val results: {dict(eval_task_dict)}\n\n")
            wandb.log(eval_task_dict, step)
        step += 1

        steps_per_sec = step / (time.time() - start)
        tokens_per_sec = tokens_per_step * steps_per_sec
        sequences_processed = sequences_per_step * step
        tokens_processed = tokens_per_step * step

        wandb_stats = {
                "train/loss": loss,
                "train/acc": acc,
                "train/steps_per_sec": steps_per_sec,
                "train/tokens_per_sec": tokens_per_sec,
                "sequences_processed": sequences_processed,
                "tokens_processed": tokens_processed,
            }
        print(f'step: {step}: {wandb_stats}')
        wandb.log(wandb_stats, step)
