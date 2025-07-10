# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma-3n model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Gemma3nTextConfig(PretrainedConfig):
    def __init__(
        self,
        activation_sparsity_pattern=None,
        altup_active_idx=0,
        altup_coef_clip=120.0,
        altup_correct_scale=True,
        altup_lr_multiplier=1.0,
        altup_num_inputs=4,
        attention_bias=False,
        attention_dropout=0.0,
        final_logit_softcapping=30.0,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=2048,
        hidden_size_per_layer_input=256,
        initializer_range=0.02,
        intermediate_size=8192,
        laurel_rank=64,
        layer_types=None,
        max_position_embeddings=32768,
        model_type="gemma3n_text",
        num_attention_heads=8,
        num_hidden_layers=30,
        num_key_value_heads=2,
        num_kv_shared_layers=10,
        query_pre_attn_scalar=256,
        rms_norm_eps=1e-06,
        rope_local_base_freq=10000.0,
        rope_scaling=None,
        rope_theta=1000000.0,
        sliding_window=512,
        use_cache=True,
        vocab_size=262400,
        vocab_size_per_layer_input=262144,
        **kwargs,
    ):
        self.activation_sparsity_pattern = activation_sparsity_pattern
        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.altup_lr_multiplier = altup_lr_multiplier
        self.altup_num_inputs = altup_num_inputs
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.final_logit_softcapping = final_logit_softcapping
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.laurel_rank = laurel_rank
        self.layer_types = layer_types
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_shared_layers = num_kv_shared_layers
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.rms_norm_eps = rms_norm_eps
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        super().__init__(**kwargs)


class Gemma3nConfig(PretrainedConfig):
    model_type = "gemma3n"

    def __init__(
        self,
        text_config=None,
        boa_token_id=256000,
        boi_token_id=255999,
        eoa_token_id=262272,
        eoi_token_id=262144,
        image_token_id=262145,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Gemma3nTextConfig with default values.")

        # If loading from dict, text_config may already be a Gemma3nTextConfig
        if isinstance(text_config, dict):
            self.text_config = Gemma3nTextConfig(**text_config)
        else:
            self.text_config = text_config

        # Always set these at the top level for compatibility
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.boa_token_id = boa_token_id
        self.boi_token_id = boi_token_id
        self.eoa_token_id = eoa_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.initializer_range = initializer_range

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Keep hidden_size and vocab_size in sync with text_config
        if name == "text_config" and value is not None:
            super().__setattr__("hidden_size", value.hidden_size)
            super().__setattr__("vocab_size", value.vocab_size)
