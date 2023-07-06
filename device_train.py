import argparse
import json
import time
import os
import re
from collections import defaultdict
import subprocess
import multiprocessing
import logging
import datetime

import numpy as np
import wandb
from tqdm import tqdm
from lm_eval import evaluator, tasks
from tasks.eval_harness import EvalHarnessAdaptor
import tensorflow as tf
from google.cloud import storage
import jax
from jax.experimental import PartitionSpec as P
from jax.experimental.multihost_utils import host_local_array_to_global_array, global_array_to_host_local_array
from praxis import py_utils

from mesh_transformer.build_model import build_model
from tfrecord_loader import TFRecordNewInputs, load_tfrecord_dataset
from easylm.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule, LLaMATokenizer
)

tf.config.experimental.set_visible_devices([], "GPU")
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".8"

jax.distributed.initialize()

# jax.config.update('jax_array', True)
# os.environ['JAX_PLATFORMS'] = ''
# os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

# wandb.login(key='7988c805dfe3fed4d6e4017f616555a5160fd2c2')


class CustomLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if extra is None:
            extra = {}
        extra['host_id'] = jax.process_index()
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

def setup_logger(host_id):
    logger = CustomLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"logs/log_{current_time}_{host_id}.txt"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(host_id)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # 注册一个清理函数，在关闭日志处理器时上传日志文件
    def cleanup():
        file_handler.close()
        command = f'gsutil cp {log_filename} gs://jax_llm_logs/'
        response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    # 在 Python 解释器关闭时自动执行清理函数
    if jax.process_index() == 0:
        import atexit
        atexit.register(cleanup)
    return logger

# 使用示例
logger = setup_logger(jax.process_index())

DEFAULT_PARAMS = {
    # model
    'num_hidden_layers': 32,
    'rng_keys': ('params', 'dropout', 'fcm'),
    'gradient_checkpointing': 'nothing_saveable',
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.0,
    'resid_pdrop': 0.05,
    'transformation': 'pjit',
    'initializer_range': 0.02,
    'fcm_min_ratio': 0.0,
    'fcm_max_ratio': 0.0,
    'use_cache': True,
    'rms_norm_eps': 1e-6,
    'max_sequence_length': 2048,
    'num_attention_heads': 32,
    'hidden_size': 4096,
    'vocab_size': 32000,
    'tie_word_embeddings': False,
    'pe': 'rotary',
    'pe_rotary_dims':64,
    # train
    'dp': 1,
    'fsdp': 1,
    'mp': 1,
    'seed': 42,
    'save_optimizer_state': True,
    'rotary_from': 'easylm',
    'gradient_accumulation_steps': 1,
    'epoch_num': 10,
    'load_mode': 'orbax',
    'save_mode': 'orbax',
    'skip_step': 0,
    'load_checkpoint': [],
}


def search_newest_train_state(params):
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
    logger.info(f'model_dirs: {model_dirs}')
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
    sample = host_local_array_to_global_array(sample, mesh, P(None, ('dp', 'fsdp')))
    return sample


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_args()
    params = DEFAULT_PARAMS
    user_params = json.load(open(args.config, 'r'))
    params.update(user_params)

    gradient_accumulation_steps = params["gradient_accumulation_steps"]
    per_replica_batch = params["per_replica_batch"]
    tpu_size = params["tpu_size"]
    cores_per_replica = params["cores_per_replica"]

    bucket = params["bucket"]
    model_dir = params["model_dir"]

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    total_steps = params["total_steps"]
    eopch_num = params['epoch_num']
    # mesh-transformer-jax
    assert params["pe"] in ["fixed", "rotary", "t5"]

    if int(args.version) == 3:
        if params['load_mode'] == 'orbax':
            params['skip_step'], params['load_checkpoint'] = search_newest_step_orbax(params)
        else:
            params['skip_step'], params['load_checkpoint'] = search_newest_train_state(params)

    logger.info(f'Version: {args.version}\nparams: {params}')

    dp = int(params['dp'])
    mp = int(params['mp'])
    fsdp = int(params['fsdp'])

    assert dp * fsdp * mp == tpu_size
    devices = np.array(jax.devices()).reshape(dp, fsdp, mp)
    mesh = jax.sharding.Mesh(devices, ('dp', 'fsdp', 'mp'))
    logger.info(f'Mesh: {mesh}')

    py_utils.sync_global_devices('Train start.......')
    # project = params.get("wandb_project", "Linli-chinese-llama-finetune")
    # wandb.init(project=project, name=params["name"], config=params, resume=True)
    host_count = tpu_size // cores_per_replica
    with mesh:
        logger.info(f'Host count: {host_count} Process id: {jax.process_index()}')
        train_batch_size = (gradient_accumulation_steps, per_replica_batch)
        logger.info(f'Train_batch_size: {train_batch_size}')
        train_dataset = load_tfrecord_dataset(f"{params['train_set']}", batch_size=train_batch_size, seq_len=params['seq'], repeat=eopch_num)
        val_batch_size = per_replica_batch
        sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
        tokens_per_step = params['seq'] * sequences_per_step

        val_sets = {
                    k: load_tfrecord_dataset(index_fname=v, 
                                            batch_size=(1, val_batch_size), 
                                            seq_len=params['seq'], 
                                            repeat=int(2.5 * eopch_num)) 
                    for k, v in params['val_set'].items()
                                    }
        # ==== init =====
        start = time.time()
        t = build_model(params, version=args.version, ray=False)
        model = t()
        model.init_state()
        logger.info(f'Init state time: {time.time() - start}')

        start = time.time()
        # train complie
        model.train(build_sample(next(train_dataset), mesh=mesh))
        logger.info(f"Train fn compiled in {time.time() - start:.06}s")
        # eval complie
        start = time.time()
        for val_set in val_sets.values():
            model.eval(build_sample(next(val_set), mesh=mesh))
        logger.info(f"Eval fn compiled in {time.time() - start:.06}s")
        # start train
        step = 0
        skip_step = params['skip_step']
        logger.info(f'Skip_step: {skip_step}, train start step is set to {skip_step}')
        start = time.time()
        while True:
            input_data = next(train_dataset)
            if step < skip_step:
                step += 1
                start = time.time()
                continue
            loss, acc = model.train(build_sample(input_data, mesh=mesh))
            loss, acc = loss.item(), acc.item()

            if (step % ckpt_every == 0 and step) or step == total_steps:
                save_path = f"gs://{bucket}/{model_dir}/step_{step}/"
                model.write_ckpt(save_path)
                if step == total_steps:
                    logger.info("Training completed!")
                    exit()

            if step % val_every == 0 and step:
                logger.info(f'Start to evaluate....')
                eval_task_dict = defaultdict(dict)
                for val_name, val_set in val_sets.items():
                    val_loss, val_acc = [], []
                    val_start = time.time()
                    for _ in range(val_batches):
                        loss, acc = model.eval(build_sample(next(val_set), mesh=mesh))
                        loss, acc = loss.item(), acc.item()
                        val_loss.append(loss)
                        val_acc.append(acc)
                    
                    val_loss = np.array(val_loss).mean()
                    val_acc = np.array(val_acc).mean()

                    eval_task_dict[val_name]['loss'] = val_loss.item()
                    eval_task_dict[val_name]['acc'] = val_acc.item()

                    logger.info(f"Validation loss for step {step}, dataset {val_name} loss: {val_loss} acc: {val_acc} take time: {time.time() - val_start}")

                logger.info(f"Step {step} val results: {dict(eval_task_dict)}\n\n")
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
            logger.info(f'Step: {step}: {wandb_stats}')
            # wandb.log(wandb_stats, step)
        py_utils.sync_global_devices('Train finished.......')
        
