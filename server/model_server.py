import re
import sys
import os
import time
from functools import partial
import json
import types
import datetime
import logging
import subprocess

import numpy as np
import mlxu
from google.cloud import storage
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from transformers import GenerationConfig, FlaxLogitsProcessorList, AutoTokenizer
import orbax

sys.path.append('/home/lishengping/projects/mesh_easy_jax')

from easylm.checkpoint import StreamingCheckpointer
from easylm.llama_model import LLaMAConfig, LLaMAConfig2, FlaxLLaMAForCausalLM, LLaMATokenizer
from easylm.jax_utils import (
    JaxRNG, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from transformers import FlaxLogitsWarper

from log_utils import setup_logger


try:
    __import__('flask')
except ImportError:
    subprocess.run('pip install flask', stdout=subprocess.PIPE, shell=True)

from flask import Flask, request, jsonify


jax.distributed.initialize()
logger = setup_logger(jax.process_index(), prefix='server')
# 命令行传入config path
config_path = sys.argv[1]
configs = json.load(open(config_path, 'r'))
model_objs = {n: types.SimpleNamespace(**config)  for n, config in configs.items()}
logger.info(f'model_objs:\n{model_objs}\n')

# config
FLAGS_DEF = {
    'temperature': 0.2,
    'top_k': 30,
    'top_p': 0.9,
    'do_sample': True,
    'num_beams': 1,
    'max_new_tokens': 1280,
    'seed': 42,
    'mesh_dim': '1,1,8',
    'dtype': 'bf16',
    'input_length': 768,
    'add_bos_token': False,
    'penalty': 1.5,
}
logger.info(f'FLAGS_DEF: {FLAGS_DEF}')



class FlaxPenaltyLogitsWarper(FlaxLogitsWarper):
    """ JIT traceable version of FlaxLogitsWarper that performs penalty."""
    def __init__(self, penalty, eos_token_id):
        self.penalty = penalty
        self.eos_token_id = eos_token_id

    def _penalty(self, x):
        return jnp.where(x < 0, x / self.penalty, x * self.penalty)

    def __call__(self, input_ids, scores, cur_len):
        score = jnp.take_along_axis(scores, input_ids, axis=1)
        # 获取惩罚历史id之后的分数，但是结束符也一起惩罚了
        score = jnp.where(score > 0 , score / self.penalty, score * self.penalty)
        # 赋值分数
        scores = scores.at[jnp.arange(scores.shape[0])[:, None], input_ids].set(score)
        # 还原结束符的概率
        # scores = jnp.where(jnp.broadcast_to(jnp.arange(scores.shape[1]), scores.shape) == self.eos_token_id, 
        #                             self._penalty(scores), 
        #                             scores)
        print(f'scores: {scores.shape}')
        return scores


def update_params(params, config):
    for k, v in params.items():
        setattr(config, k, v)
        

def update_extra_params(config):
    config.add_cross_attention = False
    config.is_encoder_decoder = False
    config.cache = True
    config.gradient_checkpointing = ''
    config.output_attentions = False
    config.output_hidden_states = False
    config.return_dict = True
    
        
item = {'params': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler())}
for name, model_obj in model_objs.items():
    start = time.time()
    model_obj.tokenizer = LLaMATokenizer(
                    vocab_file=model_obj.vocab_file,
                    padding_side='right',
                    truncation_side='right',
        )
    model_obj.prefix_tokenizer = LLaMATokenizer(
                vocab_file=model_obj.vocab_file,
                padding_side='left',
                truncation_side='left',
            )
    model_obj.user_params = json.load(open(model_obj.config, 'r'))
    if model_obj.bucket_name:
        model_obj.mngr = orbax.checkpoint.CheckpointManager(f'gs://{model_obj.bucket_name}/{model_obj.model_dir}', item)
    else:
        model_obj.mngr = orbax.checkpoint.CheckpointManager(f'{model_obj.model_dir}', item)
    with jax.default_device(jax.devices("cpu")[0]):
        llama_config = LLaMAConfig2.get_default_config()
        update_params(model_obj.user_params, llama_config)
        update_extra_params(llama_config)
        model_obj.hf_model = FlaxLLaMAForCausalLM(llama_config, input_shape=(1, 2048), seed=42, _do_init=False)
        if model_obj.load_step:
            load_step = model_obj.load_step
        else:
            load_step = model_obj.mngr.latest_step()
        logger.info(f'load step: {load_step}')
        train_state = model_obj.mngr.restore(int(load_step))
        if train_state['params'].get('params', None) is not None:
            params = train_state['params']['params']
        else:
            params = train_state
        model_ps = match_partition_rules(LLaMAConfig.get_partition_rules(), params)
        shard_fns, _ = make_shard_and_gather_fns(model_ps, get_float_dtype_by_name(FLAGS_DEF['dtype']))

        model_obj.params = params
        model_obj.model_ps = model_ps
        model_obj.shard_fns = shard_fns
        
    logger.info(f'load {model_obj.model_name} weight take: {time.time() - start}')


def get_jax_mesh(axis_dims, names):
    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert(set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)
    return Mesh(np.array(jax.devices()).reshape(dims), dim_names)


mesh_dim = FLAGS_DEF['mesh_dim']
mesh = get_jax_mesh(mesh_dim, names=('dp', 'fsdp', 'mp'))
set_random_seed(FLAGS_DEF['seed'])

for name, model_obj in model_objs.items():
    start = time.time()
    with mesh:
        model_obj.params = tree_apply(model_obj.shard_fns, model_obj.params)
    logger.info(f'put to {model_obj.model_name} device time: {time.time() - start}')


def generate(text, temperature, penalty, model_name, rng):
    model_obj = model_objs[model_name]
    inputs = model_obj.prefix_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=FLAGS_DEF['input_length'],
        return_tensors='np',
    )
    input_tokens = inputs.input_ids
    input_mask = inputs.attention_mask
    if FLAGS_DEF['add_bos_token']:
        input_tokens[:, 0] = model_obj.tokenizer.bos_token_id
        input_mask[:, 0] = 1
    batch = dict(
        input_tokens=input_tokens,
        attention_mask=input_mask,
    )
    with mesh:
        output = model_obj.compile_generate(
                model_obj.params, rng, batch, temperature, penalty
            )
        output = jax.device_get(output)
    output_text = []
    for text in list(model_obj.tokenizer.batch_decode(output)):
        if model_obj.tokenizer.eos_token in text:
            text = text.split(model_obj.tokenizer.eos_token, maxsplit=1)[0]
        output_text.append(text)

    return output_text, output


# 如果改变tokp和topk，需要重新编译，因为这两个参数会改变函数的输出size
def forward_generate(params, rng, batch, temperature, penalty, model_name='ziya-13b'):
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    # rng_generator = JaxRNG(rng)
    logger.info(f'batch shape: {batch["input_tokens"].shape}')
    output = model_objs[model_name].hf_model.generate(
        batch['input_tokens'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=rng,
        logits_processor=FlaxLogitsProcessorList(
            [FlaxTemperatureLogitsWarper(temperature), 
            FlaxPenaltyLogitsWarper(penalty, 
                                    eos_token_id=model_objs[model_name].tokenizer.eos_token_id)]
        ),
        generation_config=GenerationConfig(
            max_new_tokens=FLAGS_DEF['max_new_tokens'], # lsp
            pad_token_id=model_objs[model_name].tokenizer.eos_token_id,
            bos_token_id=model_objs[model_name].tokenizer.bos_token_id,
            eos_token_id=model_objs[model_name].tokenizer.eos_token_id,
            do_sample=FLAGS_DEF['do_sample'],
            num_beams=FLAGS_DEF['num_beams'],
            top_k=FLAGS_DEF['top_k'],
            top_p=FLAGS_DEF['top_p'],
        )
    ).sequences[:, batch['input_tokens'].shape[1]:]
    return output

# 编译
for name, model_obj in model_objs.items():
    model_obj.compile_generate = pjit(partial(forward_generate, model_name=name),
                     in_shardings=(model_obj.model_ps, PS(), PS(('dp', 'fsdp')), PS(), PS()),
                    out_shardings=(PS())
                    )

app = Flask(__name__)

response = {
            'text': '',
            'history': '',
            'temperature': 0.1,
            'top_p': FLAGS_DEF['top_p'],
            'top_k': FLAGS_DEF['top_k'],
            'penalty': 1.0,
            'max_new_tokes': FLAGS_DEF['max_new_tokens'],
            'code': 0
        }

@app.route('/generate', methods=['POST'])
def server():
    request_json = request.json
    input_texts = request_json['text']
    temperature = request_json.get('temperature', FLAGS_DEF['temperature'])
    model_name = request_json.get('model_name', list(model_objs.keys())[0])
    seed = request_json.get('seed', 42)
    penalty = request_json.get('penalty', 1.0)

    response['temperature'] = temperature
    assert model_name in model_objs

    logger.info(f'request_json: \n{request_json}')

    if isinstance(temperature, list):
        temperature = temperature[0]

    if not isinstance(input_texts, list):
        input_texts = [input_texts]

    if 'history' not in request_json:
        histories = [''] * len(input_texts)
    else:
        histories = request_json['history']

    if not isinstance(histories, list):
        histories = [histories]

    if not histories:
        histories = [''] * len(input_texts)

    assert len(histories) == len(input_texts)

    format_texts = []
    for i in range(len(input_texts)):
        inp = histories[i] + input_texts[i]
        format_texts.append(inp)
    
    decoded_outputs, _ = generate(format_texts, temperature, penalty, model_name, rng=jax.random.PRNGKey(seed))

    if decoded_outputs is None:
        response['text'] = input_texts
        response['error_message'] = f'model name must in ‘{model_objs.keys()}’...'
        response['code'] = 400
        return jsonify(response)

    format_outputs = []
    format_histories = []
    for i, decoded_output in enumerate(decoded_outputs):
        if decoded_output:
            decoded_output = decoded_output[1:] if decoded_output[0] in [':', '：'] else decoded_output
        format_outputs.append(decoded_output)
        history = input_texts[i].rstrip() + '\n' + decoded_output.lstrip()
        format_histories.append(history)
    
    response['text'] = format_outputs
    response['history'] = format_histories
    response['code'] = 200

    logger.info(f'response:\n{response}\n\n\n')

    return jsonify(response)

app.run(debug=False, host='0.0.0.0', port=5000)



