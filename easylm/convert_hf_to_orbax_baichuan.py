import os
import time
import argparse
import socket

import torch
import numpy as np
import flax
import orbax
import orbax.checkpoint
import jax

# usage:
# python convert_hf_to_orbax_baichuan.py --read_dir gs://llm_base_models/baichuan-7b/hf --save_dir gs://llm_base_models/baichuan-7b --model_size 7b



name = socket.getfqdn(socket.gethostname())
if 'tpu' in name:
    jax.distributed.initialize()
else:
    jax.distributed.initialize('0.0.0.0', num_processes=1, process_id=0)


parser = argparse.ArgumentParser(description='hf to orbax format script')

parser.add_argument('--read_dir', type=str, help='Need to be converted model weight dir. it is a dir')
parser.add_argument('--save_dir', type=str,  help='Save model weight file path, it is a dir.')
parser.add_argument('--model_size', type=str, default='7b', choices=['7b', '13b', '30b', '65b'], help='model size')
parser.add_argument('--step', type=int, default=0, help='checkpoint step')

args = parser.parse_args()

model_size = args.model_size
read_dir = args.read_dir
save_dir = args.save_dir
step = args.step


print(f'read_dir: {args.read_dir}')
print(f'save_dir: {args.save_dir}')
print(f'model_size: {args.model_size}')

# model_size = '7b'
# # read_dir = '/home/lishengping/baichuan-7b/hf'
# read_dir = 'gs://llm_base_models/baichuan-7b/hf'
# save_dir = 'gs://llm_base_models/baichuan-7b/'


LLAMA_STANDARD_CONFIGS = {
    '7b': {
        'dim': 4096,
        'intermediate_size': 11008,
        'n_layers': 32,
        'n_heads': 32,
        'norm_eps': 1e-6,
    },
    '13b': {
        'dim': 5120,
        'intermediate_size': 13824,
        'n_layers': 40,
        'n_heads': 40,
        'norm_eps': 1e-6,
    },
    '30b': {
        'dim': 6656,
        'intermediate_size': 17920,
        'n_layers': 60,
        'n_heads': 52,
        'norm_eps': 1e-6,
    },
    '65b': {
        'dim': 8192,
        'intermediate_size': 22016,
        'n_layers': 80,
        'n_heads': 64,
        'norm_eps': 1e-5,
    },
}


if 'gs:' in read_dir:
    import mlxu
    from google.cloud import storage
    client = storage.Client()
    bucket_name, *model_dir = read_dir.split('//')[1].split('/')
    model_dir = '/'.join(model_dir)
    ckpt_paths = [os.path.join('gs://', bucket_name, blob.name) for blob in client.list_blobs(bucket_name, prefix=model_dir)
         if blob.name.endswith('.bin')]
else:
    from pathlib import Path
    ckpt_paths = Path(read_dir).glob("*.bin")
ckpt_paths = sorted(ckpt_paths)

start = time.time()


ckpt_paths = sorted(ckpt_paths)
ckpt = {}
for i, ckpt_path in enumerate(ckpt_paths):
    if 'gs:' in ckpt_path:
        with mlxu.open_file(ckpt_path) as fin:
            checkpoint = torch.load(fin, map_location="cpu")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    for k, v in checkpoint.items():
        if k.startswith('model.'):
            k = k[6:]
        ckpt[k] = v.to(torch.float16)
print(f'Load model weight take time: {time.time() - start}')

params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]

def inverse_permute(w):
    reshaped_w = w.reshape(n_heads, 2, dim // n_heads // 2,  dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(dim, dim)
    return inverted_w

jax_weights = {
        'transformer': {
            'wte': {'embedding': ckpt['embed_tokens.weight'].numpy()},
            'ln_f': {'kernel': ckpt['norm.weight'].numpy()},
            'h': {
                '%d' % (layer): {
                    'attention': {
                        'wq': {'kernel': inverse_permute(ckpt[f'layers.{layer}.self_attn.W_pack.weight'][:dim].numpy()).transpose()},
                        'wk': {'kernel': inverse_permute(ckpt[f'layers.{layer}.self_attn.W_pack.weight'][dim:2*dim].numpy()).transpose()},
                        'wv': {'kernel': ckpt[f'layers.{layer}.self_attn.W_pack.weight'][-dim:].numpy().transpose()},
                        'wo': {'kernel': ckpt[f'layers.{layer}.self_attn.o_proj.weight'].numpy().transpose()},
                    },
                    'feed_forward': {
                        'w1': {'kernel': ckpt[f'layers.{layer}.mlp.gate_proj.weight'].numpy().transpose()},
                        'w2': {'kernel': ckpt[f'layers.{layer}.mlp.down_proj.weight'].numpy().transpose()},
                        'w3': {'kernel': ckpt[f'layers.{layer}.mlp.up_proj.weight'].numpy().transpose()},
                    },
                    'attention_norm': {'kernel': ckpt[f'layers.{layer}.input_layernorm.weight'].numpy()},
                    'ffn_norm': {'kernel': ckpt[f'layers.{layer}.post_attention_layernorm.weight'].numpy()},
                }
            for layer in range(params['n_layers'])},
        },
        'lm_head': {'kernel': ckpt['lm_head.weight'].numpy().transpose()},
    }
print(f'Convert weight to easylm format finished...')

item = {
        'opt_state': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        'params': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        'step': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.ArrayCheckpointHandler()),
                }
mngr = orbax.checkpoint.CheckpointManager(save_dir, item)

start = time.time()
mngr.save(step, {'params':  flax.core.frozen_dict.freeze({'params': jax_weights}), 'step': jax.numpy.array([step])})
print(f'Save orbax format take time: {time.time() - start}')