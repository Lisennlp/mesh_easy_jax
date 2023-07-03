import argparse
import json
import time
import os
import re
from collections import defaultdict
import subprocess

import numpy as np
import wandb
from tqdm import tqdm
from jax.experimental.multihost_utils import host_local_array_to_global_array, global_array_to_host_local_array

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
from jax.experimental import PartitionSpec as P


jax.distributed.initialize(num_processes=jax.process_count(), process_id=jax.process_index())

# jax.config.update('jax_array', True)
# tf.config.experimental.set_visible_devices([], "GPU")
# os.environ['JAX_PLATFORMS'] = ''
# os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

# wandb.login(key='7988c805dfe3fed4d6e4017f616555a5160fd2c2')


def search_newest_train_state(params, debug=False):
    """auto search bucket newest checkpoint path"""
    bucket_name = params['bucket']
    directory_path = params['model_dir']

    client = storage.Client()
    model_dirs = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        if 'step_' in blob.name:
            step = re.findall('step_(\d+)',blob.name)[0]
            if int(step) > 100:
                continue
            model_dirs[int(step)].append(blob.name)
    print(f'model_dirs: {model_dirs}')
    model_dirs = sorted(model_dirs.items(), key=lambda x: x[0])
    if model_dirs:
        step, model_dir = model_dirs[-1]
        model_paths = [f'trainstate::gs://{bucket_name}/{model_path}' if step > 0 else f'params::gs://{bucket_name}/{model_path}' for model_path in model_dir]
    else:
        step, model_paths = 0, []
    # step, model_paths = 0, []
    return step, model_paths

def search_newest_step_orbax(params):
    model_dir = f'gs://{params["bucket"]}/{params["model_dir"]}'
    command = f'gsutil ls {model_dir}'
    response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    step_map_path = {}
    for path in response.stdout.decode('utf-8').split('\n'):
        step = re.findall('\d+', path)
        if step:
            step_map_path[int(step[0])] = os.path.split(path)[0]
    step_map_path = sorted(step_map_path.items())
    return step_map_path[-1][0], model_dir

def update_llama_params(params):
    if params['save_mode'] == 'orbax':
        params['skip_step'], params['load_checkpoint'] = search_newest_step_orbax(params)
    else:
        params['skip_step'], params['load_checkpoint'] = search_newest_train_state(params, debug=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/6B_roto_256_test.json', help="Config file location")
    parser.add_argument("--version", type=int, default=3, help="Choose which model version to use, 1: pjit mesh-haiku-llama 2: xmap mesh-haiku-llama 3: pjit mesh-flax-llama")

    args = parser.parse_args()
    return args


def build_sample(data, mesh):
    m = data['labels'] > 0
    d = data['input_ids']
    sample = {
         "obs": d[:, :, :-1],
        "target": d[:, :, 1:],
        "masks": m[:, :, 1:],
        }
    sample = host_local_array_to_global_array(sample, mesh, P(None, 'dp'))
    return sample


if __name__ == "__main__":
    # node必须删除
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
    total_steps = params["total_steps"]
    eopch_num = params.get('epoch_num', 10)

    clip_norm = params.get('clip_norm', 1.0)

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    if int(args.version) == 3:
        update_llama_params(params)

    print(f'version: {args.version}\nparams: {params}')
    print(f'bucket： {bucket} model_dir: {model_dir}')


    devices = np.array(jax.devices()).reshape(tpu_size // cores_per_replica, cores_per_replica)
    mesh = jax.sharding.Mesh(devices, ('dp', 'mp'))
    print(f'mesh: {mesh}')

    from praxis import py_utils
    py_utils.sync_global_devices('Train start.......')

    # project = params.get("wandb_project", "Linli-chinese-llama-finetune")
    # wandb.init(project=project, name=params["name"], config=params, resume=True)
    skip_step = params['skip_step']
    print(f'skip_step: {skip_step}')
    host_count = tpu_size // cores_per_replica
    with mesh:
        train_batch_size = (gradient_accumulation_steps, per_replica_batch)
        print(f'train_batch_size: {train_batch_size}')
        train_dataset = load_tfrecord_dataset(f"{params['train_set']}", batch_size=train_batch_size, seq_len=params['seq'], repeat=eopch_num)
        global_val_batch = int(per_replica_batch * params.get("val_batch_multiplier", 1))
        sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
        tokens_per_step = params['seq'] * sequences_per_step
        val_sets = {}
        for k, v in params['val_set'].items():
            val_sets[k] = load_tfrecord_dataset(f"{v}", batch_size=(1, global_val_batch), seq_len=params['seq'], repeat=int(2.5 * eopch_num))

        # ==== init =====
        start = time.time()
        t = build_model(params, version=args.version, ray=False)
        model = t()
        model.init_state()
        print(f'init state time: {time.time() - start}')

        start = time.time()
        # train complie
        model.train(build_sample(next(train_dataset), mesh=mesh))
        print(f"Train fn compiled in {time.time() - start:.06}s")
        # eval complie
        for val_set in val_sets.values():
            model.eval(build_sample(next(val_set), mesh=mesh))
        print(f"Eval fn compiled in {time.time() - start:.06}s")
        # start train
        step = 0
        print(f'host_count: {host_count} process id: {jax.process_index()}')
        data_count = 0
        start = time.time()
        while True:
            input_data = next(train_dataset)
            if step < skip_step:
                step += 1
                start = time.time()
                continue
            loss, acc = model.train(build_sample(input_data, mesh=mesh))
            loss = loss.item()
            acc = acc.item()
            if (step % ckpt_every == 0 and step) or step == total_steps:
                save_path = f"gs://{bucket}/{model_dir}/step_{step}/"
                model.write_ckpt(save_path)
                if step == total_steps:
                    print("Training completed!")
                    exit()

            if step % val_every == 0:
                eval_task_dict = defaultdict(dict)
                for val_name, val_set in val_sets.items():
                    val_loss, val_acc = [], []
                    val_start = time.time()
                    for _ in range(val_batches):
                        loss, acc = model.eval(build_sample(next(val_set), mesh=mesh))
                        loss = loss.item()
                        acc = acc.item()
                        val_loss.append(loss)
                        val_acc.append(acc)
                    
                    val_loss = np.array(val_loss).mean()
                    val_acc = np.array(val_acc).mean()

                    eval_task_dict[val_name]['loss'] = val_loss.item()
                    eval_task_dict[val_name]['acc'] = val_acc.item()

                    print(f"Validation loss for step {step}, dataset {val_name} loss: {val_loss} acc: {val_acc}")

                print(f"Step {step} val results: {dict(eval_task_dict)}\n\n")
                # wandb.log(eval_task_dict, step)
            step += 1

            steps_per_sec = (step - skip_step) / (time.time() - start)
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
            # wandb.log(wandb_stats, step)
        py_utils.sync_global_devices('Train finished.......')
        
