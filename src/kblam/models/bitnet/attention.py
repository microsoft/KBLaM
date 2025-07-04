# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# Adapted for KBLaM from the Llama and Phi-3 implementations.
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

import torch
from torch import nn
import os
import numpy as np
from typing import Optional, Tuple

from transformers.models.bitnet import configuration_bitnet, modeling_bitnet
from transformers.utils import logging

from ..kblam_config import KBLaMConfig

logger = logging.get_logger(__name__)

class KBLaMBitNetAttention(nn.Module):
    """
    Multi-head self-attention module for BitNet, with KBLaM extensions.

    Supports standard attention as well as knowledge base (KB) integration for retrieval-augmented generation.
    Implements rotary position embeddings and optional rectangular attention for KB.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig, layer_idx: Optional[int] = None):
        """
        Initialize the attention module.
        Args:
            config: BitNetConfig with attention parameters.
            layer_idx: Index of the decoder layer (for KB integration).
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # New query head for KB interaction
        self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # Projection for KB embeddings
        self.kb_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.kb_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kb_config: Optional[KBLaMConfig] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head self-attention.
        Handles both standard and KB-augmented attention, with rotary embeddings.
        Returns attention output, (optionally) attention weights, and cache for fast decoding.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = position_embeddings
        query_states, key_states = modeling_bitnet.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # KBLaM Rectangular Attention Logic
        if kb_config is not None and self.layer_idx is not None and self.layer_idx % kb_config.kb_layer_frequency == 0:
            kb_key_states_full, kb_value_states_full = kb_kvs

            # Calculate the index for the current layer's slice of the KB embedding
            kb_layer_index = self.layer_idx // kb_config.kb_layer_frequency
            start_index = kb_layer_index * self.hidden_size
            end_index = (kb_layer_index + 1) * self.hidden_size

            # Slice the embeddings for the current layer
            kb_key_states_layer = kb_key_states_full[:, :, start_index:end_index]
            kb_value_states_layer = kb_value_states_full[:, :, start_index:end_index]

            # Project the sliced KB embeddings to the correct dimension for K/V heads
            kb_key_states_proj = self.kb_k_proj(kb_key_states_layer)
            kb_value_states_proj = self.kb_v_proj(kb_value_states_layer)

            # Reshape for attention
            num_kb_entries = kb_key_states_proj.shape[1]
            kb_key_states = kb_key_states_proj.view(
                bsz, num_kb_entries, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            kb_value_states = kb_value_states_proj.view(
                bsz, num_kb_entries, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            kb_query_states = self.q_proj_new(hidden_states)
            kb_query_states = kb_query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Repeat KV heads for GQA
            key_states = modeling_bitnet.repeat_kv(key_states, self.num_key_value_groups)
            value_states = modeling_bitnet.repeat_kv(value_states, self.num_key_value_groups)
            kb_key_states = modeling_bitnet.repeat_kv(kb_key_states, self.num_key_value_groups)
            kb_value_states = modeling_bitnet.repeat_kv(kb_value_states, self.num_key_value_groups)
            
            # No RoPE for KB queries
            attn_weights_kb = torch.matmul(kb_query_states, kb_key_states.transpose(2, 3)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32)
            )
            
            # Pruning logic for top-k KB selection
            if kb_config.top_k_kb > 0:
                kb_len = kb_key_states.shape[2]
                topk = min(kb_len, kb_config.top_k_kb)
                if topk < kb_len:
                    # Sum attention weights across heads and query sequence length to get a score per KB entry
                    top_idx = attn_weights_kb.sum(dim=(1, 2)).topk(topk, dim=-1)[1]

                    # Gather the top-k keys, values, and corresponding attention weights
                    idx_expanded_kv = top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, topk, self.head_dim)
                    kb_key_states = torch.gather(kb_key_states, 2, idx_expanded_kv)
                    kb_value_states = torch.gather(kb_value_states, 2, idx_expanded_kv)

                    idx_expanded_attn = top_idx.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, q_len, topk)
                    attn_weights_kb = torch.gather(attn_weights_kb, 3, idx_expanded_attn)

            # Attention score scaling for KB length generalization
            if kb_config.kb_length_scaling:
                attn_weights_kb += (torch.log(torch.tensor(kb_config.kb_max_train_triples)) - torch.log(torch.tensor(kb_key_states.shape[2]))).to(attn_weights_kb.device)

            attn_weights_prompt = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32)
            )

            if attention_mask is not None:
                attn_weights_prompt = attn_weights_prompt + attention_mask

            # Combine weights and apply softmax over all context (prompt + KB)
            attn_weights = torch.cat([attn_weights_kb, attn_weights_prompt], dim=-1)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            if save_attention_weights:
                detached_weights = attn_weights_kb.detach().cpu().numpy()
                save_path = os.path.join(attention_save_loc, f"{attention_file_base_name}_{self.layer_idx}.npy")
                np.save(save_path, detached_weights)

            # The split in the original code is tricky. A single matmul after concatenating
            # the value states is equivalent and less error-prone.
            combined_value_states = torch.cat((kb_value_states, value_states), dim=2)
            attn_output = torch.matmul(attn_weights, combined_value_states)
        else:
            # Standard attention if not a KB layer
            key_states = modeling_bitnet.repeat_kv(key_states, self.num_key_value_groups)
            value_states = modeling_bitnet.repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32)
            )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
