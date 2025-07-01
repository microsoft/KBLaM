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
import torch.nn.functional as F
import torch.utils.checkpoint
import os
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, List, Union

from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.bitnet import configuration_bitnet, modeling_bitnet
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.generation.utils import GenerationMixin

from kblam.models.kblam_config import KBLaMConfig

logger = logging.get_logger(__name__)

# --- Sequence Classification Head ---
class KBLaMBitNetForSequenceClassification(modeling_bitnet.BitNetPreTrainedModel):
    """
    Sequence classification head for BitNet, adapted for KBLaM.

    This class enables BitNet to be fine-tuned and evaluated on tasks where a single label
    is assigned to an entire input sequence (e.g., sentiment analysis, topic classification).
    It pools the last non-padding token's hidden state and applies a linear classifier.
    """
    def __init__(self, config):
        """
        Initialize the sequence classification head.
        Args:
            config: Model configuration with num_labels, classifier_dropout, classifier_bias, etc.
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.model = KBLaMBitNetModel(config)
        # Expose classifier_dropout and classifier_bias in config, with robust defaults
        classifier_dropout = float(getattr(config, "classifier_dropout", 0.0))
        if hasattr(config, "enable_dropout") and not config.enable_dropout:
            logger.info("Classifier dropout is DISABLED via config.enable_dropout (ablation mode)")
            classifier_dropout = 0.0
        classifier_bias = bool(getattr(config, "classifier_bias", False))
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=classifier_bias)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        """
        Forward pass for sequence classification.
        Returns:
            SequenceClassifierOutput or tuple: HuggingFace output type (or tuple if return_dict=False).
        Pools the last non-padding token and applies a linear classifier.
        Computes loss if labels are provided.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            logger.error("Both input_ids and inputs_embeds were provided to SequenceClassification forward. Only one should be set.")
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        # Ensure classifier head is on same device/dtype as model output
        self.score = self.score.to(hidden_states.device, dtype=hidden_states.dtype)
        self.dropout = self.dropout.to(hidden_states.device, dtype=hidden_states.dtype)
        # Apply classifier dropout before the score layer
        hidden_states = self.dropout(hidden_states)
        logits = self.score(hidden_states)

        # Pool the last non-padding token for each sequence (ONNX-friendly)
        logger.debug("Pooling last non-padding token for sequence classification (ONNX-friendly logic)")
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            if self.config.pad_token_id is not None:
                # ONNX-friendly: find first pad token, subtract 1, modulo to stay in bounds
                pad_mask = (input_ids == self.config.pad_token_id).int()
                # argmax returns first occurrence of pad_token_id or 0 if none
                sequence_lengths = pad_mask.argmax(dim=-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = (input_ids.shape[-1] - 1) * torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        else:
            batch_size = inputs_embeds.shape[0]
            sequence_lengths = (inputs_embeds.shape[1] - 1) * torch.ones(batch_size, dtype=torch.long, device=logits.device)

        # Ensure gather is on correct device/dtype
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths.to(logits.device)]
        pooled_logits = pooled_logits.to(logits.device, dtype=logits.dtype)

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

# --- Token Classification Head ---
class KBLaMBitNetForTokenClassification(modeling_bitnet.BitNetPreTrainedModel):
    """
    Token classification head for BitNet, adapted for KBLaM.

    This class enables BitNet to be fine-tuned and evaluated on tasks where each token
    in the input sequence receives a label (e.g., NER, POS tagging, slot filling).
    Applies a linear classifier to each token's hidden state.
    """
    def __init__(self, config):
        """
        Initialize the token classification head.
        Args:
            config: Model configuration with num_labels, classifier_dropout, classifier_bias, etc.
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.model = KBLaMBitNetModel(config)
        classifier_dropout = float(getattr(config, "classifier_dropout", 0.1))
        if hasattr(config, "enable_dropout") and not config.enable_dropout:
            logger.info("Classifier dropout is DISABLED via config.enable_dropout (ablation mode)")
            classifier_dropout = 0.0
        classifier_bias = bool(getattr(config, "classifier_bias", False))
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels, bias=classifier_bias)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        """
        Forward pass for token classification.
        Returns:
            TokenClassifierOutput or tuple: HuggingFace output type (or tuple if return_dict=False).
        Applies a linear classifier to each token.
        Computes loss for active (non-masked) tokens if labels are provided.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # Ensure classifier head is on same device/dtype as model output
        self.classifier = self.classifier.to(sequence_output.device, dtype=sequence_output.dtype)
        self.dropout = self.dropout.to(sequence_output.device, dtype=sequence_output.dtype)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only compute loss for active tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

# This is the squared ReLU activation function specified in the BitNet config.json
def relu2(x):
    """
    Squared ReLU activation function as specified in BitNet config.
    Used in the MLP layers for non-linearity.
    """
    return torch.pow(F.relu(x), 2)

def swiglu(x):
    """
    SwiGLU activation function (optionally supported).
    x is expected to be split in half along the last dimension.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2

# Copied from transformers.models.llama.modeling_llama._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Create a causal mask for self-attention, preventing tokens from attending to future tokens.
    Used in decoder-only transformer architectures.
    ONNX-friendly: avoids dynamic shapes, always returns a mask of the correct size.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.llama.modeling_llama._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands a 2D attention mask to 4D for use in multi-head attention.
    This allows masking of padding tokens in the attention mechanism.
    ONNX-friendly: avoids dynamic shapes, always returns a mask of the correct size.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class KBLaMBitNetMLP(nn.Module):
    """
    Feed-forward (MLP) block for BitNet, using squared ReLU activation.
    This is used in each decoder layer after self-attention.
    Supports a standard MLP and an experimental gated (MoE-like) version.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_gated_mlp = getattr(config, "use_gated_mlp", False)
        self.num_experts = getattr(config, "gated_mlp_num_experts", 4)

        act_fn_name = getattr(config, "activation_function", "swiglu")
        if act_fn_name == "swiglu":
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        if self.use_gated_mlp:
            logger.info(f"Using Gated MLP with {self.num_experts} experts.")
            # Create a list of expert down_projs
            self.down_proj_experts = nn.ModuleList(
                [nn.Linear(self.intermediate_size, self.hidden_size, bias=False) for _ in range(self.num_experts)]
            )
            # Routing layer
            self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
            self.down_proj = None # Not used in this mode
        else:
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.down_proj_experts = None
            self.router = None

        # Select activation function
        if act_fn_name == "squared_relu":
            self.act_fn = relu2
        elif act_fn_name == "gelu":
            self.act_fn = F.gelu
        elif act_fn_name == "swiglu":
            self.act_fn = swiglu
        else:
            raise ValueError(f"Unknown activation_function '{act_fn_name}'. Supported: squared_relu, gelu, swiglu.")

        # Dropout after MLP (residual)
        mlp_pdrop = getattr(config, "mlp_pdrop", 0.0)
        if hasattr(config, "enable_dropout") and not config.enable_dropout:
            logger.info("MLP dropout is DISABLED via config.enable_dropout (ablation mode)")
            mlp_pdrop = 0.0
        self.mlp_dropout = nn.Dropout(mlp_pdrop) if mlp_pdrop > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Custom initialization for all linear layers.
        Uses Kaiming Uniform for layers before ReLU-based activations.
        Uses a scaled normal initializer for the final output projection.
        """
        act_fn_name = getattr(self.config, "activation_function", "swiglu")
        if act_fn_name in ("squared_relu", "gelu"):
            # Kaiming init is designed for ReLU-family activations
            nn.init.kaiming_uniform_(self.gate_proj.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.up_proj.weight, a=0, mode='fan_in', nonlinearity='relu')
        else:
            # Xavier is a safe default for other activations like SwiGLU
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)

        if self.use_gated_mlp:
            for expert_proj in self.down_proj_experts:
                nn.init.normal_(expert_proj.weight, mean=0.0, std=0.02 / torch.sqrt(torch.tensor(2 * self.config.num_hidden_layers, dtype=torch.float32)))
            nn.init.xavier_uniform_(self.router.weight)
        else:
            # For the final projection layer in the MLP, a scaled normal init is often better
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02 / torch.sqrt(torch.tensor(2 * self.config.num_hidden_layers, dtype=torch.float32)))

        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            if module and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Common up-projection part
        if getattr(self.config, "activation_function", "squared_relu") == "swiglu":
            intermediate_states = self.act_fn(self.gate_proj(x) + self.up_proj(x))
        else:
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        if self.use_gated_mlp:
            # Route to experts
            routing_weights = F.softmax(self.router(x), dim=-1) # (bsz, seq_len, num_experts)
            expert_outputs = torch.stack([expert(intermediate_states) for expert in self.down_proj_experts], dim=-1) # (bsz, seq_len, hidden_size, num_experts)
            
            # Weighted sum of expert outputs
            # (bsz, seq_len, 1, num_experts) * (bsz, seq_len, hidden_size, num_experts) -> sum over last dim
            out = torch.sum(routing_weights.unsqueeze(-2) * expert_outputs, dim=-1)
        else:
            out = self.down_proj(intermediate_states)

        return self.mlp_dropout(out)



class KBLaMBitNetAttention(nn.Module):
    """
    Multi-head self-attention module for BitNet, with KBLaM extensions.
    Supports standard attention as well as knowledge base (KB) integration for retrieval-augmented generation.
    Implements rotary position embeddings and optional rectangular attention for KB.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig, layer_idx: Optional[int] = None):
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
        
        # --- Task 1: Gated Attention ---
        self.use_gated_attention = getattr(config, "use_gated_attention", False)
        if self.use_gated_attention:
            self.kb_fusion_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            logger.info(f"Layer {layer_idx}: Using experimental Gated Attention for KB fusion.")

        # --- Task 2: Sinking Token ---
        self.use_sinking_token = getattr(config, "use_sinking_token", True) # Enabled by default
        if self.use_sinking_token:
            self.sinking_key = nn.Parameter(torch.randn(1, self.num_key_value_heads, 1, self.head_dim))
            self.sinking_value = nn.Parameter(torch.randn(1, self.num_key_value_heads, 1, self.head_dim))
            logger.info(f"Layer {layer_idx}: Using Sinking Token for KB sparsification.")

        self.reset_parameters()

    def reset_parameters(self):
        """
        Custom initialization for all linear layers.
        Uses Xavier uniform for query/key/value projections.
        Uses a scaled normal initializer for the final output projection.
        """
        for module in [self.q_proj, self.k_proj, self.v_proj, self.q_proj_new, self.kb_k_proj, self.kb_v_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        if self.use_gated_attention:
            nn.init.xavier_uniform_(self.kb_fusion_gate.weight)

        if self.use_sinking_token:
            nn.init.normal_(self.sinking_key, mean=0.0, std=0.02)
            nn.init.normal_(self.sinking_value, mean=0.0, std=0.02)

        # Use a scaled normal distribution for the output projection, similar to GPT-2/3
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02 / torch.sqrt(torch.tensor(2 * self.config.num_hidden_layers, dtype=torch.float32)))
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

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
        - Causal masking and padding are ONNX-friendly.
        - All softmax and attention scores are computed in float32 for numerical stability, then cast back to model dtype.
        - repeat_kv logic matches Llama/Phi-3 reference for GQA/MQA.
        - Rotary embeddings are applied to both Q/K, theta is configurable.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Ensure all new tensors are on the correct device
        device = hidden_states.device

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

        # KBLaM Rectangular Attention Logic with Dynamic KB Sparsify
        if kb_config is not None and self.layer_idx is not None and self.layer_idx % kb_config.kb_layer_frequency == 0:
            use_efficient_kb_proj = getattr(kb_config, "use_efficient_kb_proj", False)

            if use_efficient_kb_proj:
                # In this path, kb_kvs is already pre-projected by the shared layer in KBLaMBitNetModel.
                # The per-layer kb_k_proj and kb_v_proj act as adapters on this shared representation.
                kb_key_states_layer, kb_value_states_layer = kb_kvs
            else:
                # --- Original KBLaM Implementation ---
                # Slice the full KB embedding tensor for the current layer.
                kb_key_states_full, kb_value_states_full = kb_kvs
                kb_layer_index = self.layer_idx // kb_config.kb_layer_frequency
                start_index = kb_layer_index * self.hidden_size
                end_index = (kb_layer_index + 1) * self.hidden_size
                kb_key_states_layer = kb_key_states_full[:, :, start_index:end_index]
                kb_value_states_layer = kb_value_states_full[:, :, start_index:end_index]

            # Project the (potentially sliced) KB embeddings to the correct dimension for K/V heads
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

            # --- Task 2: Sinking Token ---
            if self.use_sinking_token:
                # Prepend the sinking token to the KB entries for this batch
                expanded_sinking_key = self.sinking_key.expand(bsz, -1, -1, -1)
                expanded_sinking_value = self.sinking_value.expand(bsz, -1, -1, -1)
                kb_key_states = torch.cat([expanded_sinking_key, kb_key_states], dim=2)
                kb_value_states = torch.cat([expanded_sinking_value, kb_value_states], dim=2)

            kb_query_states = self.q_proj_new(hidden_states)
            kb_query_states = kb_query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Repeat KV heads for GQA
            key_states = modeling_bitnet.repeat_kv(key_states, self.num_key_value_groups)
            value_states = modeling_bitnet.repeat_kv(value_states, self.num_key_value_groups)
            kb_key_states = modeling_bitnet.repeat_kv(kb_key_states, self.num_key_value_groups)
            kb_value_states = modeling_bitnet.repeat_kv(kb_value_states, self.num_key_value_groups)

            # No RoPE for KB queries
            attn_weights_kb = torch.matmul(
                kb_query_states, kb_key_states.transpose(2, 3)
            ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))

            # --- Dynamic KB Sparsify ---
            dynamic_sparsify = getattr(kb_config, "dynamic_sparsify", False)
            if dynamic_sparsify and kb_config.top_k_kb > 0:
                kb_len = kb_key_states.shape[2]
                # Add 1 to topk if using sinking token to account for it
                topk_val = kb_config.top_k_kb + 1 if self.use_sinking_token else kb_config.top_k_kb
                topk = min(kb_len, topk_val)
                if topk < kb_len:
                    # Sum attention weights across heads and query sequence length to get a score per KB entry
                    # (batch, heads, q_len, kb_len) -> (batch, kb_len)
                    attn_scores = attn_weights_kb.sum(dim=(1, 2))
                    top_idx = attn_scores.topk(topk, dim=-1)[1]
                    # Logging for debug
                    logger.debug(f"Dynamic KB Sparsify: Pruning KB from {kb_len} to {topk} entries for layer {self.layer_idx}.")
                    # Gather the top-k keys, values, and corresponding attention weights
                    idx_expanded_kv = top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, topk, self.head_dim)
                    kb_key_states = torch.gather(kb_key_states, 2, idx_expanded_kv)
                    kb_value_states = torch.gather(kb_value_states, 2, idx_expanded_kv)
                    idx_expanded_attn = top_idx.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, q_len, topk)
                    attn_weights_kb = torch.gather(attn_weights_kb, 3, idx_expanded_attn)
            elif kb_config.top_k_kb > 0:
                # Legacy: prune only if top_k_kb is set, for backward compatibility
                kb_len = kb_key_states.shape[2]
                topk_val = kb_config.top_k_kb + 1 if self.use_sinking_token else kb_config.top_k_kb
                topk = min(kb_len, topk_val)
                if topk < kb_len:
                    attn_scores = attn_weights_kb.sum(dim=(1, 2))
                    top_idx = attn_scores.topk(topk, dim=-1)[1]
                    logger.debug(f"KB Pruning (legacy): Pruning KB from {kb_len} to {topk} entries for layer {self.layer_idx}.")
                    idx_expanded_kv = top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, topk, self.head_dim)
                    kb_key_states = torch.gather(kb_key_states, 2, idx_expanded_kv)
                    kb_value_states = torch.gather(kb_value_states, 2, idx_expanded_kv)
                    idx_expanded_attn = top_idx.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, q_len, topk)
                    attn_weights_kb = torch.gather(attn_weights_kb, 3, idx_expanded_attn)

            # Attention score scaling for KB length generalization
            if kb_config.kb_length_scaling:
                attn_weights_kb += (torch.log(torch.tensor(kb_config.kb_max_train_triples)) - torch.log(torch.tensor(kb_key_states.shape[2]))).to(attn_weights_kb.device)

            attn_weights_prompt = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))

            if attention_mask is not None:
                attn_weights_prompt = attn_weights_prompt + attention_mask

            # --- Task 1: Gated Attention ---
            if self.use_gated_attention:
                # Compute prompt and KB attention outputs separately
                attn_weights_prompt_softmax = nn.functional.softmax(attn_weights_prompt.to(torch.float32), dim=-1).to(query_states.dtype)
                attn_weights_kb_softmax = nn.functional.softmax(attn_weights_kb.to(torch.float32), dim=-1).to(query_states.dtype)

                attn_output_prompt = torch.matmul(attn_weights_prompt_softmax, value_states)
                attn_output_kb = torch.matmul(attn_weights_kb_softmax, kb_value_states)

                # Combine using a learned gate
                gate = torch.sigmoid(self.kb_fusion_gate(hidden_states))
                # Reshape gate for broadcasting: (bsz, q_len, hidden_size) -> (bsz, q_len, num_heads, head_dim) -> (bsz, num_heads, q_len, head_dim)
                gate = gate.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                attn_output = gate * attn_output_kb + (1 - gate) * attn_output_prompt
                
                # For output, combine the softmaxed weights for potential analysis (optional)
                attn_weights = torch.cat([attn_weights_kb_softmax, attn_weights_prompt_softmax], dim=-1)

            else:
                # Original approach: Combine weights and apply softmax over all context (prompt + KB)
                attn_weights = torch.cat([attn_weights_kb, attn_weights_prompt], dim=-1)
                # Numerical stability: always compute softmax in float32, then cast back
                attn_weights = nn.functional.softmax(attn_weights.to(torch.float32), dim=-1).to(query_states.dtype)

                if save_attention_weights:
                    # Note: this now saves the combined weights, not just KB
                    detached_weights = attn_weights.detach().cpu().numpy()
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
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Numerical stability: always compute softmax in float32, then cast back
            attn_weights = nn.functional.softmax(attn_weights.to(torch.float32), dim=-1).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class KBLaMBitNetDecoderLayer(nn.Module):
    """
    Single decoder layer for BitNet, with self-attention and MLP blocks.
    Integrates KBLaM attention for knowledge base retrieval if configured.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        logger.info(f"Instantiating KBLaMBitNetDecoderLayer (layer_idx={layer_idx}) with hidden_size={config.hidden_size}")
        self.self_attn = KBLaMBitNetAttention(config=config, layer_idx=layer_idx)
        self.mlp = KBLaMBitNetMLP(config)
        # RMSNorm: match Llama/Phi-3 epsilon, dtype
        self.input_layernorm = modeling_bitnet.BitNetRMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6)).to(torch.float32)
        self.post_attention_layernorm = modeling_bitnet.BitNetRMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6)).to(torch.float32)
        
        # --- Task 2: LayerScale for Training Stability ---
        self.use_layerscale = getattr(config, "use_layerscale", True)
        self.layerscale_init_value = getattr(config, "layerscale_init_value", 1e-5) # Task 3
        if self.use_layerscale:
            self.attn_layerscale = nn.Parameter(torch.ones(config.hidden_size))
            self.mlp_layerscale = nn.Parameter(torch.ones(config.hidden_size))

        # Dropout for residual connections (after attn, after MLP)
        resid_pdrop = getattr(config, "resid_pdrop", 0.0)
        if hasattr(config, "enable_dropout") and not config.enable_dropout:
            logger.info(f"Residual dropout is DISABLED via config.enable_dropout (ablation mode) for decoder layer {layer_idx}")
            resid_pdrop = 0.0
        if resid_pdrop > 0:
            logger.info(f"Residual dropout enabled for decoder layer {layer_idx}: resid_pdrop={resid_pdrop}")
        self.resid_dropout = nn.Dropout(resid_pdrop) if resid_pdrop > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all submodules for reproducibility.
        """
        if hasattr(self.self_attn, 'reset_parameters'):
            self.self_attn.reset_parameters()
        if hasattr(self.mlp, 'reset_parameters'):
            self.mlp.reset_parameters()
        # RMSNorm layers: no learnable weights except scale, which is initialized in the RMSNorm class
        if self.use_layerscale:
            # --- Task 3: LayerScale Initialization ---
            logger.info(f"Initializing LayerScale for layer {self.self_attn.layer_idx} with value: {self.layerscale_init_value}")
            nn.init.constant_(self.attn_layerscale, self.layerscale_init_value)
            nn.init.constant_(self.mlp_layerscale, self.layerscale_init_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kb_config: Optional[KBLaMConfig] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for a single decoder layer.
        Applies self-attention, MLP, and layer normalization, with optional KB integration.
        Returns hidden states and (optionally) attention weights and cache.
        """
        if kb_config is not None and self.self_attn.layer_idx is not None and self.self_attn.layer_idx % kb_config.kb_layer_frequency == 0:
            logger.debug(f"KB-ATTN: DecoderLayer {self.self_attn.layer_idx} received kb_config.")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
            position_embeddings=position_embeddings,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )
        
        if self.use_layerscale:
            hidden_states = self.attn_layerscale * hidden_states
        hidden_states = residual + self.resid_dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.use_layerscale:
            hidden_states = self.mlp_layerscale * hidden_states
        hidden_states = residual + self.resid_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class KBLaMBitNetModel(modeling_bitnet.BitNetPreTrainedModel):
    """
    Main BitNet model body for KBLaM.
    This class wraps the BitNet transformer stack, providing input/output embedding layers,
    a stack of decoder layers, and rotary position embeddings. It supports both standard
    language modeling and knowledge base-augmented tasks.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        logger.info(f"Instantiating KBLaMBitNetModel with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # --- Task 4: Efficient KB Projection (Shared Layer) ---
        self.use_efficient_kb_proj = getattr(config, "use_efficient_kb_proj", False)
        if self.use_efficient_kb_proj:
            # This assumes the raw KB embedding dimension matches hidden_size.
            # A more robust implementation might get this from the dataset config.
            self.kb_input_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            logger.info("Using efficient KB projection mode.")

        # Dropout for embeddings
        embd_pdrop = getattr(config, "embd_pdrop", 0.0)
        if hasattr(config, "enable_dropout") and not config.enable_dropout:
            logger.info("Embedding dropout is DISABLED via config.enable_dropout (ablation mode)")
            embd_pdrop = 0.0
        if embd_pdrop > 0:
            logger.info(f"Embedding dropout enabled: embd_pdrop={embd_pdrop}")
        self.embd_dropout = nn.Dropout(embd_pdrop) if embd_pdrop > 0 else nn.Identity()
        self.layers = nn.ModuleList(
            [KBLaMBitNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # RMSNorm: match Llama/Phi-3 epsilon, dtype
        self.norm = modeling_bitnet.BitNetRMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6)).to(torch.float32)
        self.rotary_emb = modeling_bitnet.BitNetRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all submodules for reproducibility.
        """
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # RMSNorm and rotary_emb: rely on their own init
        if self.use_efficient_kb_proj:
            nn.init.xavier_uniform_(self.kb_input_proj.weight)

    def get_input_embeddings(self):
        """
        Return the input embedding layer.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embedding layer.
        """
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        Prepare the combined attention mask for the decoder, including causal and padding masks.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kb_config: Optional[KBLaMConfig] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass for the BitNet model body.
        Returns:
            BaseModelOutputWithPast or tuple: HuggingFace output type (or tuple if return_dict=False).
        Handles input embedding, rotary position embedding, attention mask prep, and decoder stack.
        Supports both standard and KB-augmented tasks.
        Returns hidden states, (optionally) attention weights, and cache.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            logger.error("Both input_ids and inputs_embeds were provided to KBLaMBitNetModel forward. Only one should be set.")
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache and past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Apply embedding dropout
        inputs_embeds = self.embd_dropout(inputs_embeds)

        # --- Task 4: Efficient KB Projection (Forward Pass) ---
        projected_kb_kvs = None
        if kb_kvs is not None and self.use_efficient_kb_proj:
            # Project raw KB embeddings once before the decoder stack
            raw_kb_keys, raw_kb_values = kb_kvs
            projected_keys = self.kb_input_proj(raw_kb_keys)
            projected_values = self.kb_input_proj(raw_kb_values)
            projected_kb_kvs = (projected_keys, projected_values)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # 4d mask is passed through the layers
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    projected_kb_kvs if self.use_efficient_kb_proj else kb_kvs, # Pass the correct KB tensor
                    kb_config,
                    position_embeddings,
                    save_attention_weights,
                    attention_save_loc,
                    attention_file_base_name,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    kb_kvs=projected_kb_kvs if self.use_efficient_kb_proj else kb_kvs, # Pass the correct KB tensor
                    kb_config=kb_config,
                    position_embeddings=position_embeddings,
                    save_attention_weights=save_attention_weights,
                    attention_save_loc=attention_save_loc,
                    attention_file_base_name=attention_file_base_name,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KBLaMBitNetForCausalLM(GenerationMixin, modeling_bitnet.BitNetPreTrainedModel):
    """
    Causal Language Modeling head for BitNet, adapted for KBLaM.

    This class enables BitNet to be used for standard left-to-right language modeling tasks,
    including text generation, continuation, and knowledge-augmented generation. It supports
    HuggingFace's generation utilities and integrates with KBLaM's knowledge base features.
    """
    def __init__(self, config):
        """
        Initialize the CausalLM head.
        Args:
            config: Model configuration with vocab size and hidden size.
        """
        super().__init__(config)
        self.model = KBLaMBitNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        """
        Return the input embedding layer from the underlying model.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embedding layer for the underlying model.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Return the output (language modeling) head.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output (language modeling) head.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Set the underlying transformer model (decoder stack).
        """
        self.model = decoder

    def get_decoder(self):
        """
        Return the underlying transformer model (decoder stack).
        """
        return self.model

    def load_query_head(self, query_head_path):
        """
        Load a query head (adapter weights) from a file for knowledge base integration.
        Args:
            query_head_path: Path to the saved query head weights.
        """
        self.model.load_state_dict(torch.load(query_head_path), strict=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kb_config: Optional[KBLaMConfig] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
        tokenizer: Optional[object] = None, # Included for compatibility with the eval script
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for Causal Language Modeling.
        Returns:
            CausalLMOutputWithPast or tuple: HuggingFace output type (or tuple if return_dict=False).
        Computes logits and (optionally) loss for next-token prediction.
        Supports knowledge base integration and HuggingFace generation utilities.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if kb_config is not None:
            logger.debug("KB-ATTN: CausalLM received kb_config.")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )

        hidden_states = outputs[0]
        # Ensure output head is on correct device/dtype
        self.lm_head = self.lm_head.to(hidden_states.device, dtype=hidden_states.dtype)
        logits = self.lm_head(hidden_states)
        logits = logits.to(hidden_states.device, dtype=hidden_states.dtype)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare model inputs for HuggingFace's generation utilities.
        Handles batch expansion, position id creation, and KBLaM-specific arguments.
        """
        # Omit `kwargs` that are not used by `prepare_inputs_for_generation`
        # to avoid redundant warnings.
        kwarg_keys = ["kb_kvs", "kb_config"]
        for key in kwarg_keys:
            if key not in kwargs:
                kwargs[key] = None
        
        # Handle topk_size for KB pruning during evaluation
        kb_config = kwargs.get("kb_config")
        topk_size = kwargs.pop("topk_size", -1)
        if kb_config is not None and topk_size > -1:
            kb_config.top_k_kb = topk_size

        # Expand kb_kvs to match batch size
        kb_kvs = kwargs.get("kb_kvs")
        if kb_kvs is not None:
            if kb_kvs[0].ndim == 2:
                batch_size = input_ids.shape[0]
                keys = kb_kvs[0].unsqueeze(0).expand(batch_size, -1, -1)
                values = kb_kvs[1].unsqueeze(0).expand(batch_size, -1, -1)
                kwargs["kb_kvs"] = (keys, values)

        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "kb_kvs": kwargs.get("kb_kvs"),
                "kb_config": kb_config,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder the cache for beam search during generation.
        Args:
            past_key_values: Tuple of cached key/value states for each layer.
            beam_idx: Indices for reordering the batch.
        Returns:
            Reordered cache tuple.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
