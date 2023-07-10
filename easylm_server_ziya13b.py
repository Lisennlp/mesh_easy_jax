import re
import sys
import os
import jax
import time
from functools import partial
import json

import numpy as np
import mlxu
from google.cloud import storage
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from transformers import GenerationConfig, FlaxLogitsProcessorList, AutoTokenizer
import orbax
from flask import Flask, request, jsonify

sys.path.append('/home/lishengping/projects/mesh_easy_jax')

from easylm.checkpoint import StreamingCheckpointer
from easylm.llama_model import LLaMAConfig, LLaMAConfig2, FlaxLLaMAForCausalLM, LLaMATokenizer
from easylm.jax_utils import (
    JaxRNG, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)

print(f'000: {jax.devices()}')
jax.distributed.initialize()
print(f'111: {jax.devices()}')


bucket_name = 'llm_base_models'
model_path = 'Ziya-LLaMA-13B-Pretrain-v1-easylm'


FLAGS_DEF = {
    'temperatrue': 0.2,
    'top_k': 30,
    'top_p': 0.9,
    'do_sample': True,
    'num_beams': 1,
    'max_new_tokens': 1024,
    'seed': 42,
    'initialize_jax_distributed': False,
    'mesh_dim': '1,1,8',
    'dtype': 'bf16',
    'input_length': 768,
    'seq_length': 2048,
    'add_bos_token': False,
    'load_llama_config': '',
    'load_checkpoint': '',
}


vocab_file = '/home/lishengping/models/ziya-13b/tokenizer.model'

tokenizer = LLaMATokenizer(
            vocab_file=vocab_file,
            padding_side='right',
            truncation_side='right',
        )
prefix_tokenizer = LLaMATokenizer(
            vocab_file=vocab_file,
            padding_side='left',
            truncation_side='left',
        )
# tokenizer = AutoTokenizer.from_pretrained('/home/lishengping/models/ziya-13b', use_fast=False)
user_params = json.load(open('/home/lishengping/projects/mesh_easy_jax/configs/v3_64_13b.json', 'r'))

def update_params(params, config):
    for k, v in params.items():
        setattr(config, k, v)

with jax.default_device(jax.devices("cpu")[0]):
    llama_config = LLaMAConfig2.get_default_config()
    update_params(user_params, llama_config)

    llama_config.add_cross_attention = False
    llama_config.is_encoder_decoder = False
    llama_config.cache = True
    llama_config.gradient_checkpointing = ''
    llama_config.output_attentions = False
    llama_config.output_hidden_states = False
    llama_config.return_dict = True
    hf_model = FlaxLLaMAForCausalLM(llama_config, input_shape=(1, 2048), seed=42, _do_init=False)


# 初始化mngr
item = {
    'params': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler())
       }
mngr = orbax.checkpoint.CheckpointManager(f'gs://{bucket_name}/{model_path}', item)


start = time.time()
with jax.default_device(jax.devices("cpu")[0]):
    train_state = mngr.restore(mngr.latest_step())
print(f'load weight time: {time.time() - start}')


if train_state['params'].get('params', None) is not None:
    params = train_state['params']['params']
else:
    params = train_state

start = time.time()
model_ps = match_partition_rules(LLaMAConfig.get_partition_rules(), params)
shard_fns, _ = make_shard_and_gather_fns(model_ps, get_float_dtype_by_name('bf16'))

print(f'shard weight time: {time.time() - start}')

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


# @staticmethod
def generate(text, temperature):
    global sharded_rng
    inputs = prefix_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=FLAGS_DEF['input_length'],
        return_tensors='np',
    )
    input_tokens = inputs.input_ids
    input_mask = inputs.attention_mask
    if FLAGS_DEF['add_bos_token']:
        input_tokens[:, 0] = tokenizer.bos_token_id
        input_mask[:, 0] = 1
    batch = dict(
        input_tokens=input_tokens,
        attention_mask=input_mask,
    )
    with mesh:
        output, sharded_rng = forward_generate(
            params, sharded_rng, batch, temperature
        )
        output = jax.device_get(output)
    output_text = []
    for text in list(tokenizer.batch_decode(output)):
        if tokenizer.eos_token in text:
            text = text.split(tokenizer.eos_token, maxsplit=1)[0]
        output_text.append(text)

    return output_text, output


start = time.time()
mesh_dim = FLAGS_DEF['mesh_dim']
mesh = get_jax_mesh(mesh_dim, names=('dp', 'fsdp', 'mp'))
set_random_seed(42)
with mesh:
    params = tree_apply(shard_fns, params)
    sharded_rng = next_rng()
print(f'put to device time: {time.time() - start}')

# 如果改变tokp和topk，需要重新编译，因为这两个参数会改变函数的输出size
@partial(
    pjit,
    in_shardings=(model_ps, PS(), PS(('dp', 'fsdp')), PS()),
    out_shardings=(PS(), PS())
)
def forward_generate(params, rng, batch, temperature):
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    rng_generator = JaxRNG(rng)
    print(f'batch shape: {batch["input_tokens"].shape}')
    output = hf_model.generate(
        batch['input_tokens'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=rng_generator(),
        logits_processor=FlaxLogitsProcessorList(
            [FlaxTemperatureLogitsWarper(temperature)]
        ),
        generation_config=GenerationConfig(
            max_new_tokens=FLAGS_DEF['max_new_tokens'], # lsp
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=FLAGS_DEF['do_sample'],
            num_beams=FLAGS_DEF['num_beams'],
            top_k=FLAGS_DEF['top_k'],
            top_p=FLAGS_DEF['top_p'],
        )
    ).sequences[:, batch['input_tokens'].shape[1]:]
    return output, rng_generator()


app = Flask(__name__)

# def restart_server():
#     # 关闭当前服务器进程
#     os.kill(os.getpid(), signal.SIGINT)
#     # 这里可以添加额外的重启逻辑，例如清理缓存或执行其他操作
#     # 启动新的服务器进程
#     time.sleep(10)
#     os.execvp(sys.executable, [sys.executable] + sys.argv)


@app.route('/generate', methods=['POST'])
def server():
    print(f'input: {request.json}')
    histories = request.json['history']
    input_texts = request.json['text']
    temperature = request.json['temperature']

    if isinstance(temperature, list):
        temperature = temperature[0]
    if not isinstance(histories, list):
        histories = [histories]
    if not isinstance(input_texts, list):
        input_texts = [input_texts]

    if not histories:
        histories = [''] * len(input_texts)

    if  len(histories) != len(input_texts):
        return jsonify({'text': [], 'error': 'history lenght is not equal to text', 'code': 400})

    format_texts = []
    for i in range(len(input_texts)):
        inp = histories[i] + input_texts[i]
        format_texts.append(inp)

    decoded_outputs = generate(format_texts, temperature)[0]
    format_outputs = []
    format_histories = []
    for i, decoded_output in enumerate(decoded_outputs):
        if decoded_output:
            decoded_output = decoded_output[1:] if decoded_output[0] in [':', '：'] else decoded_output
        format_outputs.append(decoded_output)
        history = input_texts[i].rstrip() + '\n\n' + decoded_output.lstrip()
        format_histories.append(history)
    print(f'output: {format_outputs}')
    return jsonify({
                'text': format_outputs,
                'history': format_histories,
                'temperature': temperature,
                'top_p': FLAGS_DEF['top_p'],
                'top_k': FLAGS_DEF['top_k'],
                'code': 200})

app.run(debug=False, host='0.0.0.0', port=5000)