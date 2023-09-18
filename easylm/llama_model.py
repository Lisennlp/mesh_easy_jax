import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import tempfile
import time
import math
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask

# from flax.linen.attention import dot_product_attention_weights
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PrecisionLike

from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning
from flax.core.frozen_dict import FrozenDict

import sentencepiece as spm
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from mlxu import function_args_to_config, load_pickle, open_file

from jax.experimental.maps import thread_resources
from mesh_transformer.util import to_f32, to_bf16, maybe_shard, global_norm
from jax.experimental.pjit import pjit

from easylm.jax_utils import (
    JaxRNG,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    named_tree_map,
    global_norm,
    set_random_seed,
    average_metrics,
    get_weight_decay_mask,
    make_shard_and_gather_fns,
    with_sharding_constraint,
)
import optax
from easylm.jax_utils import with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy, tree_apply
from easylm.checkpoint import StreamingCheckpointer
import orbax
from orbax import checkpoint

from flax.serialization import from_bytes, to_bytes, to_state_dict, from_state_dict
from log_utils import logger

remat = nn_partitioning.remat

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

OPT_EMPTY_OBJ = optax._src.base.EmptyState()


LLAMA_STANDARD_CONFIGS = {
    '3b': {
        'vocab_size': 32000,
        'hidden_size': 3200,
        'intermediate_size': 8640,
        'num_hidden_layers': 26,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '7b': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '30b': {
        'vocab_size': 32000,
        'hidden_size': 6656,
        'intermediate_size': 17920,
        'num_hidden_layers': 60,
        'num_attention_heads': 52,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '65b': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 22016,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    'debug': {  # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
}


class LLaMAConfig(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_partition_rules():
        return (
            # embeddings
            ("transformer/wte/embedding", PS("mp", "fsdp")),
            # atention
            ("attention/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("attention/wo/kernel", PS("mp", "fsdp")),
            # mlp
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            ('.*', PS(None)),
        )


class LLaMAConfig2(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        # pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        gradient_checkpointing='nothing_saveable',
        fcm_min_ratio=0.0,
        fcm_max_ratio=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        super().__init__(
            # pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'mp'))

    @staticmethod
    def get_partition_rules():
        """Parition rules for GPTJ. Note that these rules are orderd, so that
        the beginning rules match first. It is important to use
        PartitionSpec() instead of None here because JAX does not treat
        None as a pytree leaf.
        """
        return (
            # embeddings
            ("transformer/wte/embedding", PS("mp", None)),
            # atention
            ("attention/(wq|wk|wv)/kernel", PS(None, "mp")),
            ("attention/wo/kernel", PS("mp", None)),
            # mlp
            ("feed_forward/w1/kernel", PS(None, "mp")),
            ("feed_forward/w2/kernel", PS("mp", None)),
            ("feed_forward/w3/kernel", PS(None, "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS(None, "mp")),
            ('.*', PS(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')

    @staticmethod
    def get_tokenizer_config(updates=None):
        config = ConfigDict()
        config.vocab_file = ''
        config.add_bos_token = False
        config.add_eos_token = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_tokenizer(cls, config, padding_side='left', truncation_side='right'):
        config = cls.get_tokenizer_config(config)
        assert config.vocab_file != '', 'vocab_file must be specified'
        tokenizer = LLaMATokenizer(
            vocab_file=config.vocab_file,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        return tokenizer

    @classmethod
    def load_config(cls, path):
        if path in LLAMA_STANDARD_CONFIGS:
            return cls.from_dict(LLAMA_STANDARD_CONFIGS[path])
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['llama_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


class BaseLayer(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None
    multable = 'intermediates'

    def add_summary(self, name, input, level=0):
        logger.info(f'[{self.multable} -- {name}] level is {level}')
        if level:
            self.sow(self.multable, name, input, init_fn=lambda: 0, reduce_fn=lambda a, b: b)


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    # def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
    #     x_square = jnp.square(x)
    #     self.add_summary('x_square', x_square[1], level=self.config.summary_level)

    #     x_mean = x_square.mean(-1, keepdims=True)
    #     self.add_summary('x_mean', x_mean[1], level=self.config.summary_level)

    #     x_eps = x_mean + self.eps
    #     self.add_summary('x_eps', x_eps[1], level=self.config.summary_level)

    #     x_rsqrt = jax.lax.rsqrt(x_eps)
    #     self.add_summary('x_rsqrt', x_rsqrt[1], level=self.config.summary_level)

    #     x_cheng = x * x_rsqrt
    #     self.add_summary('x_cheng', x_cheng[1], level=self.config.summary_level)
    #     return x_cheng

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self._norm(x.astype(self.dtype)).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


# def precompute_freqs_cis_praxis(dim: int, end: int, theta: float=10000.0, dtype: jnp.dtype=jnp.float32, min_timescale=1.0):
#     min_timescale = 1.0
#     fraction = 2 * jnp.arange(0, dim // 2) / dim
#     timescale = (
#         min_timescale
#         * (theta / min_timescale) ** fraction
#     )
#     position = jnp.arange(end, dtype=dtype)[jnp.newaxis, :]
#     position = position[:, :, jnp.newaxis, jnp.newaxis]
#     timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
#     sinusoid_inp = (position / timescale).astype(dtype)
#     sin = jnp.sin(sinusoid_inp)
#     cos = jnp.cos(sinusoid_inp)
#     return sin, cos


def apply_rotary_emb_praxis(
    inputs: jnp.ndarray,
    position: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    min_timescale = 1.0
    dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, dim // 2) / dim
    timescale = min_timescale * (10000.0 / min_timescale) ** fraction
    # position = jnp.arange(end, dtype=dtype)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = (position / timescale).astype(dtype)
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    # sin, cos = freqs_cis
    qk = inputs.astype(dtype).reshape(*inputs.shape[:-1], -1, 2)
    # split和reshape分割不同
    # first_half, second_half = jnp.split(inputs, 2, axis=-1)
    # lsp
    first_half, second_half = qk[..., 0], qk[..., 1]
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    # lsp
    final = jnp.stack((first_part, second_part), axis=-1).reshape(*first_part.shape[:-1], -1).astype(dtype)
    # stack和concatenate不同
    return final


class FlaxLLaMAAttention(BaseLayer):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")

        if self.config.rotary_from == 'easylm':
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim,
                config.max_sequence_length * 2,
                dtype=self.dtype,
            )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def dot_product_attention_weights(
        self,
        query: Array,
        key: Array,
        bias: Optional[Array] = None,
        mask: Optional[Array] = None,
        broadcast_dropout: bool = True,
        dropout_rng: Optional[PRNGKey] = None,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        dtype: Optional[Dtype] = None,
        precision: PrecisionLike = None,
    ):
        query, key = promote_dtype(query, key, dtype=dtype)
        dtype = query.dtype
        logger.info(f'[Dot] -- precision: {precision} || query and key dtype: {dtype} || dropout_rate: {dropout_rate}')
        logger.info(f'[Dot] -- bias: {bias} || mask: {mask} || deterministic: {deterministic}')

        assert query.ndim == key.ndim, 'q, k must have same rank.'
        assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
        assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # calculate attention matrix
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(dtype)
        self.add_summary('scale_query', query[1], level=self.config.summary_level)
        # attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key, precision=precision)
        self.add_summary('q_dot_k', attn_weights[1], level=self.config.summary_level)
        # apply attention bias: masking, dropout, proximity bias, etc. # not None
        if bias is not None:
            attn_weights = attn_weights + bias
        # apply attention mask: lsp: None
        if mask is not None:
            big_neg = jnp.finfo(dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        self.add_summary('q_dot_k_mask', attn_weights[1], level=self.config.summary_level)
        # normalize the attention weights
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
        self.add_summary('q_dot_k_mask_softmax', attn_weights[1], level=self.config.summary_level)
        # apply attention dropout
        if not deterministic and dropout_rate > 0.0:
            keep_prob = 1.0 - dropout_rate
            if broadcast_dropout:
                # dropout is broadcast across the batch + head dimensions
                dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
                keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
            else:
                keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
            multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
            attn_weights = attn_weights * multiplier

        return attn_weights

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
        idx=None,
    ):
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        self.add_summary('query_proj', xq[1], level=self.config.summary_level)
        self.add_summary('key_proj', xk[1], level=self.config.summary_level)
        self.add_summary('value_proj', xv[1], level=self.config.summary_level)

        # 加上会稍微耗时一点
        # xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        # xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        # xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        if not self.config.alibi:
            if self.config.rotary_from == 'easylm':
                # lsp easylm
                freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis, dtype=self.dtype)
            else:
                # lsp paxml
                xq = apply_rotary_emb_praxis(xq, position=position_ids, dtype=self.dtype)
                xk = apply_rotary_emb_praxis(xk, position=position_ids, dtype=self.dtype)

        query_length, key_length = xq.shape[1], xk.shape[1]

        self.add_summary('query_rotary', xq[1], level=self.config.summary_level)
        self.add_summary('key_rotary', xk[1], level=self.config.summary_level)

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # lsp: 从self.causal_mask中取shape：(1, 1, query_length, max_decoder_length) 。 且从索引(0, 0, mask_shift, 0)开始取
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            mask_shift = None
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        # batch * len
        # attention_mask: batch
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        if self.config.alibi:
            attention_mask = combine_masks(attention_mask, causal_mask)
        else:
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

        # transform boolean mask into float mask
        # >0用0.0赋值，否则用jnp.finfo(self.dtype).min赋值 attention_mask: 前面为0，后面为负无穷
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )
        # attention_bias = with_sharding_constraint(attention_bias, PS(("dp", "fsdp"), "mp", None, None))

        if self.config.alibi:
            # fcm_mask : n_head * qlen * klen  ||  attention_bias: bsz * num_heads * qlen * klen
            if mask_shift is not None:
                fcm_mask = lax.dynamic_slice(
                    fcm_mask, (0, mask_shift, 0), (fcm_mask.shape[0], query_length, max_decoder_length)
                )
                fcm_mask = fcm_mask[jnp.newaxis, ...]
            else:
                fcm_mask = fcm_mask[jnp.newaxis, :, : attention_bias.shape[-2], : attention_bias.shape[-1]]
            # print(f'fcm_mask: {fcm_mask.dtype} value:\n {fcm_mask[10, 10, 10].item()}') # error
            if len(attention_bias.shape) == 3:
                attention_bias = jnp.expand_dims(attention_bias, axis=(1,))
            attention_bias += fcm_mask
        self.add_summary('attn_mask', attention_bias[1], level=self.config.summary_level)
        # lsp: batch:8 -> 256M
        # q, k作为 q.math
        # query = query / jnp.sqrt(depth).astype(dtype) -> depth: head_dim
        # attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key, precision=precision)
        # attn_weights: b * n_head * qlen * klen
        # xq: b * qlen * n_head * head_size || xk: b * klen * n_head * head_size
        attn_weights = self.dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )
        self.add_summary('attn_score_softmax', attn_weights[1], level=self.config.summary_level)

        # lsp shape: b * qlen * n_head * klen
        # attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
        # lsp: paxml
        # attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), None, "mp", None))

        # lsp: batch:8 -> 256M
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)
        self.add_summary('attn_score_dot_value', attn_output[1], level=self.config.summary_level)
        attn_output = self._merge_heads(attn_output)
        # lsp: paxml ?
        # attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"),  None, "mp"))
        attn_output = self.wo(attn_output)
        self.add_summary('attn_post', attn_output[1], level=self.config.summary_level)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxLLaMAMLP(BaseLayer):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w3 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        ffn1_gate = self.w1(x)
        self.add_summary('ffn1_gate', ffn1_gate[1], level=self.config.summary_level)
        ffn1_gate_activ = nn.silu(ffn1_gate)
        self.add_summary('ffn1_gate_activ', ffn1_gate_activ[1], level=self.config.summary_level)
        ffn1_output = self.w3(x)
        self.add_summary('ffn1_output', ffn1_output[1], level=self.config.summary_level)

        ffn1_output_activ = ffn1_gate_activ * ffn1_output
        self.add_summary('ffn1_gate_dot_ffn1_output', ffn1_output_activ[1], level=self.config.summary_level)

        ffn2_output = self.w2(ffn1_output_activ)
        self.add_summary('ffn_residual', ffn2_output[1], level=self.config.summary_level)
        x = self.dropout(ffn2_output, deterministic=deterministic)
        self.add_summary('ffn_residual_drop', x[1], level=self.config.summary_level)
        #         x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        return x


class FlaxLLaMABlock(BaseLayer):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.attention = FlaxLLaMAAttention(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.feed_forward = FlaxLLaMAMLP(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask: Optional[jnp.ndarray] = None,
        idx=None,
    ):
        self.add_summary('block_input', hidden_states[1], level=self.config.summary_level)
        attn_norm = self.attention_norm(hidden_states)
        self.add_summary('attn_pre_norm', attn_norm[1], level=self.config.summary_level)

        attn_outputs = self.attention(
            attn_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            fcm_mask=fcm_mask,
            idx=idx,
        )
        attn_output = attn_outputs[0]
        self.add_summary('attn_residual_drop', attn_output[1], level=self.config.summary_level)
        hidden_states = hidden_states + attn_output
        self.add_summary('attn_output', hidden_states[1], level=self.config.summary_level)
        ffn_norm = self.ffn_norm(hidden_states)
        self.add_summary('ffn_pre_norm', ffn_norm[1], level=self.config.summary_level)
        feed_forward_hidden_states = self.feed_forward(
            ffn_norm,
            deterministic=deterministic,
        )
        self.add_summary('ffn_residual', feed_forward_hidden_states[1], level=self.config.summary_level)
        hidden_states = hidden_states + feed_forward_hidden_states
        self.add_summary('ffn_output', feed_forward_hidden_states[1], level=self.config.summary_level)
        return (hidden_states,) + attn_outputs[1:]


class FlaxLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: LLaMAConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to
        # ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxLLaMABlockCollection(BaseLayer):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        block = FlaxLLaMABlock
        if self.config.gradient_checkpointing != '':
            FlaxLLaMACheckpointBlock = remat(
                block,
                static_argnums=(
                    3,
                    4,
                    5,
                    6,
                ),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
            )
            block = FlaxLLaMACheckpointBlock
        self.blocks = [
            block(self.config, name=str(i), dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
            for i in range(self.config.num_hidden_layers)
        ]

    def _get_interleave(self, n):
        def _get_interleave_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                _get_interleave_power_of_2(closest_power_of_2)
                + self._get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def _fill_with_neg_inf(self, t):
        return jax.lax.select(
            t.astype(self.dtype) > 0,
            jnp.full(t.shape, 0.0, dtype=self.dtype),
            jnp.full(t.shape, jnp.finfo(self.dtype).min),
        )

    def _gen_alibi_mask(self, n_head, max_pos):
        slopes = jnp.array(self._get_interleave(n_head)).astype(self.dtype)
        # n_head: head数量,  n_head * 1 * 1
        slopes = slopes[..., jnp.newaxis, jnp.newaxis]
        # 1 * 1 * position_len
        baichuan1
        position = jnp.arange(max_pos, dtype=self.dtype)[jnp.newaxis, jnp.newaxis, ...]
        alibi = jnp.broadcast_to(slopes * position, (n_head, 1, max_pos))

        # # baichuan2
        # position_point = jnp.arange(max_pos) - max_pos + 1
        # position_point = jnp.broadcast_to(position_point[jnp.newaxis, jnp.newaxis, ...], (n_head, 1, max_pos))
        # diag = jnp.diag(position_point[0])
        # position_point = position_point - diag[jnp.newaxis, jnp.newaxis, ...].transpose(0, -1, -2)
        # alibi = jnp.broadcast_to(slopes * position_point, (n_head, 1, max_pos))

        alibi_mask = jnp.triu(self._fill_with_neg_inf(jnp.zeros([max_pos, max_pos])), 1)
        alibi_mask = jnp.expand_dims(alibi_mask, axis=(0,)) + alibi
        return alibi_mask

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if not deterministic and self.config.fcm_max_ratio > 0:
            # Apply forgetful causal mask
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng('fcm'),
                shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio,
            )
            fcm_mask = (
                jax.random.uniform(self.make_rng('fcm'), shape=(batch_size, 1, seq_length, seq_length)) > fcm_ratio
            )
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype('bool')
        else:
            fcm_mask = None

        if self.config.alibi:
            # num_head * seq * seq
            fcm_mask = self._gen_alibi_mask(self.config.num_attention_heads, self.config.seq)

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                fcm_mask,
                idx=i,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLLaMAModule(BaseLayer):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.embed_dim = self.config.hidden_size
        if self.config.gradient_checkpointing != '':
            wte = remat(
                nn.Embed,
                static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
            )
            rmsnorm = remat(
                RMSNorm,
                static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
            )
        else:
            wte = nn.Embed
            rmsnorm = RMSNorm

        self.wte = wte(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        # 不能rematFlaxLLaMABlockCollection
        self.h = FlaxLLaMABlockCollection(
            self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.ln_f = rmsnorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=False,  # lsp
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        #         deterministic = Flase
        print(f'deterministic: {deterministic} self.config.embd_pdrop: {self.config.embd_pdrop}')

        input_embeds = self.wte(input_ids.astype("i4"))
        self.add_summary(f'input_embeds', input_embeds[1], level=self.config.summary_level)
        hidden_states = self.dropout(input_embeds, deterministic=deterministic)
        self.add_summary(f'input_embeds_drop', hidden_states[1], level=self.config.summary_level)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        self.add_summary(f'last_output', hidden_states[1], level=self.config.summary_level)
        hidden_states = self.ln_f(hidden_states)
        self.add_summary(f'last_norm', hidden_states[1], level=self.config.summary_level)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


@add_start_docstrings("", "")
class FlaxLLaMAModel(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAModule


class FlaxLLaMAForCausalLMModule(BaseLayer):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        if self.config.gradient_checkpointing != '':
            t = remat(
                FlaxLLaMAModule,
                static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
            )
            d = remat(
                nn.Dense,
                static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
            )
        else:
            t = FlaxLLaMAModule
            d = nn.Dense

        self.transformer = t(self.config, dtype=self.dtype)
        self.lm_head = d(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def train_step(self, train_state, input_tokens, target_tokens, masks, rngs):
        # 减少编译函数内部的print，可以加速一丢丢
        # if masks is not None:
        #     logger.info(f'Train input_tokens: {input_tokens.shape} Target_tokens: {target_tokens.shape} Masks: {masks.shape}')
        # else:
        #     logger.info(f'Train input_tokens: {input_tokens.shape} Target_tokens: {target_tokens.shape} Masks: None')
        def loss_and_accuracy(params, input_token, target_token, mask=None):
            # deterministic=False的时候有Dropout，否则无
            output, inter_values = self.apply(
                params,
                input_token,
                deterministic=False,
                rngs=rngs,
                mutable=['intermediates'],
            )
            logits = output.logits
            loss, forward_output = cross_entropy_loss_and_accuracy(logits, target_token, valid=mask)
            forward_output.update(inter_values)
            return loss, forward_output

        def microbatch(old_grad, batch):
            input_token, target_token, mask = batch
            val_grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            # loss函数返回两个值，第一个值有梯度，第二个值可以随意
            (loss, forward_output), grads = val_grad_fn(to_bf16(train_state['params']), input_token, target_token, mask)
            new_grad = getattr(jax, 'tree_multimap', jax.tree_map)(lambda a, b: a + b, old_grad, grads)
            return new_grad, (loss.mean(), forward_output)

        def one_step_grad(x):
            val_grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (loss, forward_output), grads = val_grad_fn(
                to_bf16(train_state['params']), input_tokens[0], target_tokens[0], masks[0]
            )
            return (loss.mean(), forward_output), grads

        def multi_step_grad(x):
            # jax.lax.scan传入的第一个参数执行的是加和，且第一个参数和返回的第一个参数要相同
            # 返回的第二个参数执行的是cat
            grads, (loss, forward_output) = jax.lax.scan(
                microbatch,
                jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16), train_state['params']),
                (input_tokens, target_tokens, masks),
            )
            return (loss, forward_output), grads

        # 采用jax的控制流会稍微快些, if input_tokens.shape[0] == 1 true, 执行one_step_grad。None表示传入两个函数的参数
        #         (loss, inter_values), grads = jax.lax.cond(input_tokens.shape[0] == 1, one_step_grad, multi_step_grad, None)
        (loss, forward_output), grads = one_step_grad(input_tokens.shape[0])
        updates, new_opt_state = self.optimizer.update(grads, train_state["opt_state"], train_state["params"])
        forward_output['loss'] = loss
        return (
            to_f32(forward_output),
            None,
            {
                "params": optax.apply_updates(train_state["params"], to_f32(updates)),
                "step": train_state["step"] + 1,
                "opt_state": new_opt_state,
            },
        )

    def eval_step(self, train_state, input_tokens, target_tokens, masks):
        # rng = next_rng()
        # rng_generator = JaxRNG(rng) # 会造成泄露
        # 泄露报错：jax._src.traceback_util.UnfilteredStackTrace: jax._src.errors.UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect, allowing for a reference to an intermediate value with type uint32[2] wrapped in a DynamicJaxprTracer to escape the scope of the transformation.
        # JAX transformations require that functions explicitly return their outputs, and disallow saving intermediate values to global state.
        # The function being traced when the value leaked was train_step at /home/lishengping/projects/mesh_easy_jax/easylm/llama_model.py:934 traced for pjit.
        # logger.info(f'Eval input_tokens: {input_tokens.shape} target_tokens: {target_tokens.shape}')
        def loss_and_accuracy(params, input_token, target_token, mask=None):
            # deterministic=False的时候有Dropout，否则无
            logits = self.apply(
                params,
                input_token,
                deterministic=True
                # rngs=rng_generator(self.config.rng_keys),
            ).logits
            forward_output = cross_entropy_loss_and_accuracy(logits, target_token, valid=mask)
            return forward_output

        if len(input_tokens.shape) == 3:
            input_token = input_tokens.reshape(-1, input_tokens.shape[-1])
            target_token = target_tokens.reshape(-1, target_tokens.shape[-1])
            mask = masks.reshape(-1, masks.shape[-1])
        forward_output = loss_and_accuracy(train_state['params'], input_token, target_token, mask=mask)
        token_log_prob = forward_output['token_log_prob']
        return to_f32(forward_output['loss']), to_f32(forward_output['accuracy'])

    def init_from_params(self, params, step=0):
        opt_state = self.optimizer.init(params)
        return {'params': params, 'opt_state': opt_state, 'step': step + 1}

    def init_fn(self, rng):
        rng_generator = JaxRNG(rng)
        params = self.init(
            input_ids=jnp.zeros((4, self.config.seq - 1), dtype=jnp.int32),
            position_ids=jnp.zeros((4, self.config.seq - 1), dtype=jnp.int32),
            attention_mask=jnp.ones((4, self.config.seq - 1), dtype=jnp.int32),
            rngs=rng_generator(self.config.rng_keys),
        )
        opt_state = self.optimizer.init(params)
        return {'params': params, 'opt_state': opt_state, 'step': 0}

    # 根据保存的params实现的
    def recovery_train_state(self):
        for k, v in self.state.items():
            if k == 'opt_state':
                if isinstance(v, list):
                    temp_v = {}
                    for v_ in v:
                        if isinstance(v_, dict) and 'mu' in v_:
                            temp_v['2'] = v_
                            break
                    v = temp_v
                assert '2' in v
                # 不是很明白保存的优化器为什么是这样
                # 正好对应optax.chain()的6个位置的对象。
                self.state[k] = (
                    OPT_EMPTY_OBJ,
                    OPT_EMPTY_OBJ,
                    optax._src.transform.ScaleByAdamState(
                        count=v['2']['count'],
                        mu=v['2']['mu'],  # 如果报错的话，要包上FrozenDict
                        nu=v['2']['nu'],
                    ),
                    OPT_EMPTY_OBJ,
                    OPT_EMPTY_OBJ,
                    optax._src.transform.ScaleByScheduleState(v['2']['count']),
                )
            elif k == 'params' and isinstance(v, dict):
                self.state[k] = v
            # FrozenDict 如果报错的话，要去掉FrozenDict

    def init_mngr(self, model_dir):
        self.item = {
            'opt_state': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
            'params': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
            'step': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.ArrayCheckpointHandler()),
        }
        self.mngr = orbax.checkpoint.CheckpointManager(f'gs://{model_dir}', self.item)

    def load_orbax_async_checkpoint(self):
        if 'step' not in self.shard_fns:
            target.pop('step')
        lastest_step = int(self.mngr.latest_step())
        logger.info(f'Latest step: {lastest_step}')
        try:
            self.state = self.mngr.restore(lastest_step)
            self.recovery_train_state()
            logger.info(f'State: {self.state.keys()}')
            logger.info(f'Shard keys: {self.shard_fns.keys()}')  # step, params, opt_states
            self.state = tree_apply(self.shard_fns, self.state)

        except Exception as error:
            logger.info(f'Load error: {error}!!! now pop ‘opt_state’ key and load it again...')
            # del opt_state key, because load 0 step model has not this key.
            self.item.pop('opt_state')
            self.state = self.mngr.restore(lastest_step)
            self.recovery_train_state()
            logger.info(f'State params: {self.state["params"].keys()}')  # params
            logger.info(f'Shard params keys: {self.shard_fns["params"].keys()}')  # params, step
            restored_params = tree_apply(self.shard_fns['params'], self.state['params'])
            self.state = self.init_from_params(restored_params, step=lastest_step)
            del restored_params
            # but save model need key:opt_state
            self.item['opt_state'] = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler())

    def init_state(self):
        self.config = LLaMAConfig(**self.config)
        set_random_seed(self.config.seed)
        self.optimizer = self.config.optimizer
        train_state_shapes = jax.eval_shape(self.init_fn, next_rng())
        train_state_partition = match_partition_rules(self.config.get_partition_rules(), train_state_shapes)
        self.init_ = pjit(self.init_fn, in_shardings=PS(), out_shardings=train_state_partition)
        self.init_from_params_ = pjit(
            self.init_from_params,
            in_shardings=(train_state_partition['params'],),
            out_shardings=train_state_partition,
            donate_argnums=(0,),
        )
        self.train_ = pjit(
            self.train_step,
            in_shardings=(
                train_state_partition,
                PS(None, ('dp', 'fsdp')),
                PS(None, ('dp', 'fsdp')),
                PS(None, ('dp', 'fsdp')),
                PS(),
            ),
            out_shardings=(PS(), PS(), train_state_partition),
            donate_argnums=(0,),
        )
        self.eval_ = pjit(
            self.eval_step,
            in_shardings=(
                train_state_partition,
                PS(None, ('dp', 'fsdp')),
                PS(None, ('dp', 'fsdp')),
                PS(None, ('dp', 'fsdp')),
            ),
            out_shardings=(PS(), PS()),
            #   donate_argnums=(0, ),
        )
        # 保存的是每个参数怎么进行shard和gather的函数
        self.shard_fns, self.gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)
        import haiku as hk

        key = hk.PRNGSequence(42)
        assert thread_resources.env.shape['mp'] == self.config.mp
        assert thread_resources.env.shape['dp'] == self.config.dp
        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']
        fsdp = thread_resources.env.shape['fsdp']
        vocab = self.config.vocab_size
        logger.info('============Init state============')
        example_shape = (max(dp, 1), self.config.seq - 1)
        logger.info(f'Example_shape: {example_shape}')
        x = jax.random.uniform(next(key), example_shape, minval=0, maxval=vocab).astype(jnp.uint32)  # batch, len
        logger.info(f'dp: {dp}')
        logger.info(f"fsdp : {fsdp}")
        logger.info(f"mp: {mp}")
        self.gen_length = 1
        self.rng = next_rng()

        checkpoint_config = StreamingCheckpointer.get_default_config(
            {'save_optimizer_state': self.config.save_optimizer_state}
        )
        model_save_dir = os.path.join(self.config.bucket, self.config.model_dir)

        self.checkpointer = StreamingCheckpointer(checkpoint_config, model_save_dir, enable=jax.process_index() == 0)
        if 'orbax' in [self.config.load_mode, self.config.save_mode]:
            logger.info(f'Init mngr....')
            self.init_mngr(model_save_dir)

        if self.config.load_checkpoint:
            start = time.time()
            logger.info(f'Start load pretrained weight -> {self.config.load_checkpoint}')
            if self.config.load_mode == 'orbax':
                logger.info(f'Load_mode1: {self.config.load_mode}')
                # init orbax async checkpointer and load latest checkpoint
                self.load_orbax_async_checkpoint()
            else:
                logger.info(f'load_mode: {self.config.load_mode}')
                if 'train_state' in self.config.load_checkpoint[0]:
                    logger.info(f'Loading train_state')
                    self.state, _ = self.checkpointer.load_trainstate_checkpoint(
                        load_from=self.config.load_checkpoint,
                        trainstate_target=None,
                        trainstate_shard_fns=self.shard_fns,
                    )
                    # 为什么要进行恢复参数，因为self.train_的pjit编译的时候，对self.state进行的donate（加入缓冲区）节省内存，加快速度。
                    self.recovery_train_state()
                else:
                    logger.info(f'Loading params')
                    _, restored_params = self.checkpointer.load_trainstate_checkpoint(
                        load_from=self.config.load_checkpoint,
                        trainstate_target=train_state_shapes['params'],
                        trainstate_shard_fns=self.shard_fns['params'],
                    )
                    self.state = self.init_from_params(restored_params)
                    del restored_params
            logger.info(f'Loaded pretrained weight finished!!! take time: {time.time() - start}s')
        else:
            logger.info(f'Train model from scratch!!!')
            self.state = self.init_(self.rng)
        # error: jaxlib.xla_extension.XlaRuntimeError: UNIMPLEMENTED: PJRT C API does not support Defragment
        # jax.lib.xla_bridge.get_backend().defragment()
        param_count = hk.data_structures.tree_size(self.state['params'])
        logger.info(f"Total parameters: {param_count}")

    def train(self, sample):
        input_tokens, target_tokens, masks = sample['obs'], sample['target'], sample['masks']
        rng_generator = JaxRNG(self.rng)
        rngs = rng_generator(self.config.rng_keys)
        self.rng = rng_generator()
        forward_output, _, self.state = self.train_(self.state, input_tokens, target_tokens, masks, rngs)
        return forward_output

    def eval(self, sample):
        input_tokens, target_tokens, masks = sample['obs'], sample['target'], sample['masks']
        loss, acc = self.eval_(self.state, input_tokens, target_tokens, masks)
        return loss, acc

    def write_ckpt(self, path=None, shard=None):
        start = time.time()
        logger.info(f'Start to save model: ‘{path}’')
        if self.config.save_mode == 'orbax':
            self.mngr.save(self.state['step'].item(), self.state)
            # self.mngr.wait_until_finished()
        else:
            self.checkpointer.save_all(self.state, self.gather_fns, model_dir=path)
        logger.info(f'Model save finished. take time: {time.time() - start}')

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        self.add_summary('input_ids', input_ids, level=self.config.summary_level)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0), (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        self.add_summary('last_output', hidden_states[1], level=self.config.summary_level)
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        self.add_summary('lm_logits', lm_logits[1], level=self.config.summary_level)

        if not return_dict:
            return (lm_logits,) + outputs[1:]
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings("", "")
class FlaxLLaMAForCausalLM(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTJ uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # attention_mask上的值，赋值到extended_attention_mask上，其实索引为0，如果不为0，attention_mask的shape和extended_attention_mask一样的话，也是按照0开始的
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # broadcast_to多个卡引用，只会保存一份副本
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}
PRETRAINED_VOCAB_FILES_MAP = {}


class LLaMATokenizer(PreTrainedTokenizer):
    """
    Construct a LLaMA tokenizer. Based on byte-level Byte-Pair-Encoding.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=False,
        add_eos_token=False,
        add_tokens=0,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)

        with tempfile.NamedTemporaryFile() as tfile:
            with open_file(self.vocab_file, 'rb') as fin:
                tfile.write(fin.read())
                tfile.flush()
                tfile.seek(0)
            self.sp_model.Load(tfile.name)
        """ Initialisation"""
        self.add_special_tokens(
            dict(
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
            )
        )
        self.pad_token_id = self.unk_token_id
        if add_tokens:
            self.added_tokens_encoder = {f'makeword{i}': 79458 + i for i in range(add_tokens)}

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size() + len(self.added_tokens_encoder)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = output + [self.eos_token_id]

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
