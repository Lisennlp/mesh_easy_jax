import functools
import multiprocessing

import optax
import ray

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerV2
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay
from ray_tpu import create_tpu, wait_til, get_connection, start_ray

from easylm.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule, LLaMATokenizer
)
from easylm.checkpoint import StreamingCheckpointer

import jax.numpy as jnp
import jax

def build_model(params, tpu_name, region, preemptible, version=1):
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    cores_per_replica = params["cores_per_replica"]
    tpu_size = params["tpu_size"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]

    assert tpu_size in [8, 32, 128, 256, 512]

    tpu_name, region = 'llm-jax-v3-32', 'us-east1-d'
    conns = get_connection(tpu_name, region)

    assert len(conns) * 8 == tpu_size, "wrong size TPU for config"

    head_info = ray.init()
    address = head_info['redis_address'] if head_info.get('redis_address') else head_info['address']
    print(f'address: {address}')

 #   with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
  #      p.map(functools.partial(start_ray, address=address, version=version), conns)

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
       clip_by_global_norm(1, use_psum=(version != 2)),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr))
    )

    params["optimizer"] = opt
    # lsp 这里只是把参数传进去，还没有执行任何init操作
    if version == 2:
        model_fn = functools.partial(CausalTransformerV2, params)
    elif version == 1:
        model_fn = functools.partial(CausalTransformer, params)
    elif version == 3:
        llama_params = get_llama_params(params)
        print(f'llama_params init finished...')
        print(f'llama_params: \n{llama_params}')
        model_fn = functools.partial(FlaxLLaMAForCausalLMModule, llama_params, dtype=jnp.bfloat16)
    else:
        raise Exception(f"Version {version} does not exist")

    t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), len(conns), model_fn, version=version)
    return t, params


# python  EasyLM/models/llama/convert_torch_to_easylm.py \
#     --checkpoint_dir='/home/lishengping/models/llama/7B' \
#     --output_file='/home/lishengping/models/llama/easylm_trans_7B.stream' \
#     --streaming=True
def get_llama_params(params):
#     params['load_checkpoint'] = 'params::/home/lishengping/models/trans_7b/llama_trans_7b.stream'
    # params['load_checkpoint'] = 'params::/home/lishengping/models/trans_belle_7b/belle_7b.stream'
    # params['load_checkpoint'] = params.get('load_checkpoint', 'params::gs://llm_base_models/easylm/lama_trans_7b.stream')
    params['load_checkpoint'] = ''
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


#     llama_config.load_checkpoint = '/home/lishengping/models/llama_7b_streaming'
    return params

