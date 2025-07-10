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
"""
A KBLaM model based on the Gemma-3N architecture from Google.

This module adapts the Gemma-3N model to integrate with a Knowledge Base (KB)
by modifying the attention mechanism to process and incorporate KB-derived information
during generation. It follows the established KBLaM project pattern of composition and
surgical modification, enabling knowledge injection at configurable layers.

Key Features:
- Replaces standard attention with a custom KBLaM-aware attention module.
- Injects KB key/value vectors at configurable layer intervals.
- Robust config patching to ensure all required attributes are present.
- Designed for maintainability and extensibility, with detailed documentation for new contributors.

References:
- See KBLaM whitepaper, Section 4, for the knowledge injection mechanism.
- See HuggingFace Gemma-3N documentation for base model details.
"""

import torch
from torch import nn
import math
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nConfig,
    Gemma3nPreTrainedModel,
    Gemma3nTextModel,
    Gemma3nTextAttention,
    Gemma3nTextConfig,
)
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging
from kblam.models.kblam_config import KBLaMConfig
from typing import Optional, Tuple

logger = logging.get_logger(__name__)

class KblamGemma3nAttention(Gemma3nTextAttention):
    """
    Custom attention mechanism for Gemma-3N that integrates Knowledge Base (KB) information.

    This class overrides the standard Gemma-3N attention to inject knowledge tokens at configurable
    intervals, using pre-computed key/value vectors from a sentence encoder and learned adapters.
    The injection frequency and other parameters are controlled by the KBLaMConfig.

    Args:
        original_attention: The original Gemma3nTextAttention instance to wrap and extend.
        kblam_config (KBLaMConfig): Configuration object specifying KB injection parameters.

    See Also:
        - KBLaM whitepaper, Section 4 (Knowledge Injection)
        - Llama and Phi KBLaM attention implementations for architectural parallels
    """
    def __init__(self, original_attention, kblam_config: KBLaMConfig):
        # Note: We avoid calling _init_rope in the base class, as we want to re-initialize
        # with the correct config from the original attention module.
        config = original_attention.config
        if isinstance(config, dict):
            config = Gemma3nTextConfig.from_dict(config)
        super().__init__(config, original_attention.layer_idx)

        # Copy all weights and attributes from the original attention module
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj

        # Ensure all necessary attributes are set for correct initialization and forward pass
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # KBLaM-specific configuration
        self.kblam_config = kblam_config

        # The input dimension for the KB projection is the hidden size of the model
        kb_embed_dim = self.hidden_size

        # This projection is for the separate query head used to score KB entries (future extensibility)
        self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor,
        past_key_value,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_altup: bool = False,
        kb_kvs: Optional[tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for KBLaM Gemma3n attention.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch, seq_len, hidden_dim).
            position_embeddings: Rotary position embeddings (tuple or tensor).
            attention_mask (torch.Tensor): Attention mask for input tokens.
            past_key_value: Cached key/value states for autoregressive decoding.
            output_attentions (bool): Whether to return attention weights.
            use_cache (bool): Whether to use cache for fast generation.
            use_altup (bool): Whether to use alternative upsampling (Gemma3n-specific).
            kb_kvs (Optional[tuple]): Tuple of (kb_keys, kb_values) for knowledge injection.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (attn_output, attn_weights, [optional past_key_value]).
        """
        bsz, q_len, _ = hidden_states.size()

        # Project input to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings (if provided)
        if position_embeddings is not None:
            # Gemma3n passes a tuple for position_embeddings
            global_pos_emb, local_pos_emb = position_embeddings
            # Handle both (cos, sin) tuple and single tensor cases for global_pos_emb
            if isinstance(global_pos_emb, (tuple, list)) and len(global_pos_emb) == 2:
                cos, sin = global_pos_emb
                query_states, key_states = (
                    query_states * cos + self._rotate_half(query_states) * sin,
                    key_states * cos + self._rotate_half(key_states) * sin,
                )
            else:
                # Fallback: just pass through (or use base class logic if needed)
                pass  # No rotary applied, or already applied upstream

        # Update with past key/value states if present (for generation)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # Repeat key/value heads as needed for grouped attention
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # === KBLaM Knowledge Injection ===
        # Inject KB key/value vectors at configured layers
        if kb_kvs is not None and self.layer_idx % self.kblam_config.kb_layer_frequency == 0:
            kb_keys, kb_values = kb_kvs
            kb_idx = self.layer_idx // self.kblam_config.kb_layer_frequency

            if len(kb_keys.shape) == 2:  # Shared KB embeddings across batch
                kb_len = kb_keys.shape[0]
                num_kb_layers = 1 + self.config.num_hidden_layers // self.kblam_config.kb_layer_frequency

                # Select KB vectors for this layer
                kb_keys_layer = kb_keys.reshape(kb_len, num_kb_layers, -1)[:, kb_idx]
                kb_values_layer = kb_values.reshape(kb_len, num_kb_layers, -1)[:, kb_idx]

                # Reshape to (num_heads, kb_len, head_dim) and expand to batch
                kb_keys_layer = kb_keys_layer.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1)
                kb_values_layer = kb_values_layer.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1)
                kb_keys_layer = kb_keys_layer.unsqueeze(0).expand(bsz, -1, -1, -1)
                kb_values_layer = kb_values_layer.unsqueeze(0).expand(bsz, -1, -1, -1)
            else:  # Batch-specific KB embeddings
                kb_len = kb_keys.shape[1]
                num_kb_layers = 1 + self.config.num_hidden_layers // self.kblam_config.kb_layer_frequency

                # Select KB vectors for this layer and batch
                kb_keys_layer = kb_keys.view(bsz, kb_len, num_kb_layers, -1)[:, :, kb_idx]
                kb_values_layer = kb_values.view(bsz, kb_len, num_kb_layers, -1)[:, :, kb_idx]

                # Reshape to (batch, num_heads, kb_len, head_dim)
                kb_keys_layer = kb_keys_layer.view(bsz, kb_len, self.num_heads, self.head_dim).transpose(1, 2)
                kb_values_layer = kb_values_layer.view(bsz, kb_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Concatenate KB vectors to the left of the normal key/value states
            key_states = torch.cat([kb_keys_layer, key_states], dim=2)
            value_states = torch.cat([kb_values_layer, value_states], dim=2)

            # Modify attention mask to allow attending to KB tokens
            if attention_mask is not None:
                kb_len = kb_keys_layer.shape[2]
                kb_attention_mask = torch.zeros(bsz, 1, q_len, kb_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([kb_attention_mask, attention_mask], dim=-1)

        # Standard scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output to (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
    
    def _rotate_half(self, x):
        """
        Helper for rotary position embeddings: rotates half the hidden dims of the input.
        Args:
            x (torch.Tensor): Input tensor of shape (..., head_dim)
        Returns:
            torch.Tensor: Rotated tensor of same shape as input.
        """
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def repeat_kv(hidden_states, n_rep):
        """
        Repeat key/value hidden states n_rep times for grouped attention.
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch, heads, seq_len, head_dim)
            n_rep (int): Number of repetitions
        Returns:
            torch.Tensor: Repeated tensor
        """
        if n_rep == 1:
            return hidden_states
        return hidden_states.repeat_interleave(n_rep, dim=1)



class KblamGemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    """
    KBLaM model for conditional generation based on Gemma-3N.

    This class wraps the base Gemma-3N model and replaces its attention layers with KBLaM-aware
    attention modules, enabling knowledge injection at configurable intervals. It also robustly
    patches the config to ensure all required attributes are present, and exposes standard
    HuggingFace model interfaces for compatibility.

    Args:
        config (KBLaMConfig): Configuration object specifying both base model and KBLaM parameters.

    See Also:
        - KblamGemma3nAttention for the custom attention logic
        - KBLaM whitepaper, Section 4 (Knowledge Injection)
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: KBLaMConfig):
        """
        Initialize a KBLaM Gemma-3N model for conditional generation.

        This constructor robustly patches the config to ensure all required attributes are present,
        loads the base Gemma-3N model, and replaces its attention layers with KBLaM-aware modules.

        Args:
            config (KBLaMConfig): Configuration object specifying both base model and KBLaM parameters.
        """
        # Load the base Gemma3nConfig and update it with KBLaM parameters
        base_model_name_or_path = config.base_model_name_or_path if hasattr(config, "base_model_name_or_path") else config._name_or_path
        orig_config = Gemma3nConfig.from_pretrained(base_model_name_or_path)
        gemma_config = Gemma3nConfig.from_pretrained(base_model_name_or_path)

        # Ensure 'attention_bias' is present (default to False if missing)
        if not hasattr(gemma_config, "attention_bias"):
            gemma_config.attention_bias = False

        # Transfer KBLaM specific attributes to the Gemma3nConfig, but do not overwrite config objects with dicts
        for key, value in config.to_dict().items():
            if key == "text_config" and isinstance(value, dict) and hasattr(gemma_config, "text_config"):
                for subkey, subval in value.items():
                    setattr(gemma_config.text_config, subkey, subval)
            elif not isinstance(value, dict):
                setattr(gemma_config, key, value)

        # Always ensure 'layer_types' is present and valid
        num_layers = getattr(gemma_config, "num_hidden_layers", None)
        if num_layers is not None:
            if hasattr(orig_config, "layer_types"):
                gemma_config.layer_types = orig_config.layer_types
            else:
                gemma_config.layer_types = [
                    "full_attention" if i % 5 == 0 else "sliding_attention" for i in range(num_layers)
                ]

        # Ensure 'attention_dropout' is present (default to 0.0 if missing)
        if not hasattr(gemma_config, "attention_dropout"):
            gemma_config.attention_dropout = 0.0

        # Ensure 'attention_bias' is present (default to False if missing)
        if not hasattr(gemma_config, "attention_bias"):
            gemma_config.attention_bias = False

        # --- PATCH START ---
        # Promote all essential attributes from text_config to the main config, or set safe defaults if missing
        if hasattr(gemma_config, "text_config") and isinstance(gemma_config.text_config, dict):
            gemma_config.text_config = Gemma3nTextConfig(**gemma_config.text_config)

        # List of all attributes to ensure at the top level
        # NOTE: If you encounter new AttributeErrors for missing config fields, add them here with a safe default.
        required_attrs = {
            "vocab_size_per_layer_input": 262144,  # Used for per-layer vocab partitioning
            # Rope/rotary embedding related attributes
            "rope_theta": 1000000.0,
            "rope_scaling": None,
            "rope_local_base_freq": 10000.0,
            # AltUp-related attributes (Gemma3n-specific upsampling)
            "altup_num_inputs": 4,
            "altup_active_idx": 0,
            "altup_coef_clip": 120.0,
            "altup_correct_scale": True,
            "altup_lr_multiplier": 1.0,
            "activation_sparsity_pattern": None,
            "hidden_size_per_layer_input": 256,
            # Standard transformer attributes
            "vocab_size": 262400,
            "hidden_size": 2048,
            "num_hidden_layers": 30,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": lambda cfg: ["self_attention"] * getattr(cfg, "num_hidden_layers", 30),
            "attention_dropout": 0.0,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "initializer_range": 0.02,
            "max_position_embeddings": 32768,
            "intermediate_size": 8192,
            "sliding_window": 512,
            "use_cache": True,
            "model_type": "gemma3n_text",
            "laurel_rank": 64,
            "num_kv_shared_layers": 10,
            "query_pre_attn_scalar": 256,
            "final_logit_softcapping": 30.0,
            "hidden_activation": "gelu_pytorch_tanh",
            # Token IDs (from config or fallback)
            "boa_token_id": 256000,
            "boi_token_id": 255999,
            "eoa_token_id": 262272,
            "eoi_token_id": 262144,
            "image_token_id": 262145,
        }

        for attr, default in required_attrs.items():
            if not hasattr(gemma_config, attr):
                # Try to get from text_config if available
                val = None
                if hasattr(gemma_config, "text_config") and hasattr(gemma_config.text_config, attr):
                    val = getattr(gemma_config.text_config, attr)
                elif callable(default):
                    val = default(gemma_config)
                else:
                    val = default
                setattr(gemma_config, attr, val)
        # --- PATCH END ---

        # Now, initialize the parent class with the fully-featured config
        super().__init__(gemma_config)

        # Load the base Gemma3nTextModel
        self.model = Gemma3nTextModel.from_pretrained(base_model_name_or_path, config=gemma_config, torch_dtype=config.torch_dtype)

        # Replace attention layers with our custom KBLaM version
        for layer in self.model.layers:
            original_attention = layer.self_attn
            # Pass the augmented gemma_config to the attention layer
            layer.self_attn = KblamGemma3nAttention(original_attention, gemma_config)

        # Standard LM head for language modeling
        self.vocab_size = gemma_config.vocab_size
        self.lm_head = nn.Linear(gemma_config.hidden_size, self.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        """
        Returns the input embedding layer (token embeddings) of the underlying model.
        """
        return self.model.embed_tokens


    def set_input_embeddings(self, value):
        """
        Sets the input embedding layer (token embeddings) of the underlying model.
        Args:
            value (nn.Module): New embedding layer.
        """
        self.model.embed_tokens = value


    def get_output_embeddings(self):
        """
        Returns the output embedding (LM head) of the model.
        """
        return self.lm_head


    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embedding (LM head) of the model.
        Args:
            new_embeddings (nn.Module): New output embedding layer.
        """
        self.lm_head = new_embeddings


    def set_decoder(self, decoder):
        """
        Sets the decoder (transformer body) of the model.
        Args:
            decoder (nn.Module): New decoder module.
        """
        self.model = decoder


    def get_decoder(self):
        """
        Returns the decoder (transformer body) of the model.
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kb_kvs: Optional[list[torch.FloatTensor]] = None,
        **kwargs,
    ):
        """
        Forward pass for KBLaM Gemma-3N conditional generation.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            position_ids (Optional[torch.LongTensor]): Position IDs.
            past_key_values (Optional[list[torch.FloatTensor]]): Cached key/value states for generation.
            inputs_embeds (Optional[torch.FloatTensor]): Optional input embeddings.
            labels (Optional[torch.LongTensor]): Target labels for language modeling loss.
            use_cache (Optional[bool]): Whether to use cache for fast generation.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dict or tuple.
            kb_kvs (Optional[list[torch.FloatTensor]]): KB key/value vectors for knowledge injection.
            **kwargs: Additional arguments.

        Returns:
            CausalLMOutputWithPast or tuple: Model outputs, including loss (if labels provided), logits, and optional states.
        """
        if return_dict is None:
            return_dict = True
        # The `use_altup` flag is handled internally by the original Gemma3nTextModel.forward
        use_altup_flag = kwargs.get("use_altup", None)

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
            use_altup=use_altup_flag,
            kb_kvs=kb_kvs,
            # Pass the entire kblam_config to the attention layers
            kblam_config=self.config
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepare model inputs for generation (used by HuggingFace GenerationMixin).
        Ensures KB-related arguments are passed through if present.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            past_key_values (Optional[list[torch.FloatTensor]]): Cached key/value states.
            **kwargs: Additional arguments, including KB vectors.

        Returns:
            dict: Model input dictionary for generation.
        """
        if hasattr(self.model, "prepare_inputs_for_generation"):
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        else:
            # Fallback: basic input dict for generation
            model_inputs = {"input_ids": input_ids}
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
            # Pass through any other kwargs
            model_inputs.update(kwargs)
        # Add KB-related arguments if they are present in the generation call
        if "kb_kvs" in kwargs:
            model_inputs["kb_kvs"] = kwargs["kb_kvs"]
        return model_inputs
