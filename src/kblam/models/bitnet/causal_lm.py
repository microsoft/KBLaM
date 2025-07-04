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
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List, Union

from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bitnet import modeling_bitnet
from transformers.generation.utils import GenerationMixin

from .modeling import KBLaMBitNetModel
from ..kblam_config import KBLaMConfig

logger = logging.get_logger(__name__)

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
        input_ids: torch.LongTensor = None,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for Causal Language Modeling.
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
