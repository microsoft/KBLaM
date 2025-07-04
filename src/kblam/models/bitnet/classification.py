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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.bitnet import modeling_bitnet
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from .modeling import KBLaMBitNetModel

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
            config: Model configuration with num_labels and other settings.
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.model = KBLaMBitNetModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for sequence classification.
        Pools the last non-padding token and applies a linear classifier.
        Computes loss if labels are provided.
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
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        # Pool the last non-padding token for each sequence
        # This is standard for transformer-based sequence classification
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            if self.config.pad_token_id is not None:
                sequence_lengths = (input_ids != self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
        else:
            batch_size = inputs_embeds.shape[0]
            sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

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
            config: Model configuration with num_labels and classifier_dropout.
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.model = KBLaMBitNetModel(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for token classification.
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
