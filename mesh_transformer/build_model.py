import functools

import optax
# import ray
import jax.numpy as jnp

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerV2
from mesh_transformer.util import additive_weight_decay
from easylm.llama_model import FlaxLLaMAForCausalLMModule

import orbax
from orbax import checkpoint


def build_model(params, version=1, ray=True):
    cores_per_replica = params["cores_per_replica"]
    assert cores_per_replica == 8
    tpu_size = params["tpu_size"]
    host_count = tpu_size // cores_per_replica
    assert tpu_size in [8, 16, 32, 64, 128, 256, 512]

    if ray:
        head_info = ray.init()
        address = head_info['redis_address'] if head_info.get('redis_address') else head_info['address']
        print(f'address: {address}')

    # warmup_cosine_decay_schedule
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=params['lr'],
        warmup_steps=params['warmup_ratio'] * params['total_steps'],
        decay_steps=params['anneal_steps'],
        end_value=params["end_lr"],
        exponent=1.0,
    )
    # constant_with_warmup
    # scheduler = util.constant_with_warmup(params["warmup_steps"], params["anneal_steps"], params["lr"], params["end_lr"])
    # Linear_schedule
    # scheduler = util.gpt3_schedule(params["warmup_steps"], params["anneal_steps"], params["lr"], params["end_lr"])
    # chain就是依次经过这些函数，最后调用def scale_by_schedulestep_size_fn: base.Schedule)
    # 接受一个Schedule函数作为输入，然后在内部将opt_state.count（step）传入值该函数中
    # https://github.com/deepmind/optax/blob/507ce1369241a9c05490bf0e0e020cbf51d249c7/optax/_src/transform.py#L786
    opt = optax.chain(
        optax.scale(1 / params['gradient_accumulation_steps']),
        # clip_by_global_norm(1, use_psum=(version != 2)), # tpu pod 不能使用
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        additive_weight_decay(params["weight_decay"]),
        optax.scale(-1),  # updates = jax.tree_util.tree_map(lambda g: step_size * g, updates)
        optax.scale_by_schedule(scheduler),
    )
    params["optimizer"] = opt

    # lsp 这里只是把参数传进去，还没有执行任何init操作
    if version == 2:
        model_fn = functools.partial(CausalTransformerV2, params)
    elif version == 1:
        model_fn = functools.partial(CausalTransformer, params)
    elif version == 3:
        model_fn = functools.partial(FlaxLLaMAForCausalLMModule, params, dtype=jnp.bfloat16)
    else:
        raise Exception(f"Version {version} does not exist")
    if ray:
        t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), host_count, model_fn, version=version)
    else:
        t = model_fn
    return t

# ray run
# gcloud compute tpus tpu-vm ssh llm-jax-v3-32 --zone=us-east1-d --worker=0 --command="/home/lishengping/miniconda3/bin/ray start --head --port=3333 --resources='{\"tpu\": 1}'"  --project=llm-tpu
