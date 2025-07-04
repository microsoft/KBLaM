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

from torch import nn
from transformers.models.bitnet import configuration_bitnet
from .utils import relu2

class KBLaMBitNetMLP(nn.Module):
    """
    Feed-forward (MLP) block for BitNet, using squared ReLU activation.
    This is used in each decoder layer after self-attention.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig):
        """
        Initialize the MLP block.
        Args:
            config: BitNetConfig with hidden and intermediate sizes.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = relu2

    def forward(self, x):
        """
        Forward pass for the MLP block.
        Applies two linear projections with squared ReLU activation and elementwise multiplication.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
