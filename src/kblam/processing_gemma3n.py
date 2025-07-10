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
"""Processor for Gemma-3n"""

from transformers import PreTrainedTokenizer, AutoTokenizer, SiglipImageProcessor, Gemma3nAudioFeatureExtractor
from .models.gemma3n_config import Gemma3nConfig

class Gemma3nProcessor:
    def __init__(self, config, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path, **kwargs)
        self.image_processor = SiglipImageProcessor.from_pretrained(config.vision_config._name_or_path, **kwargs)
        self.audio_feature_extractor = Gemma3nAudioFeatureExtractor.from_pretrained(config.audio_config._name_or_path, **kwargs)

    def __call__(self, text=None, images=None, audio=None, **kwargs):
        if text is not None:
            inputs = self.tokenizer(text, **kwargs)
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors="pt")
            # Logic to combine image and text inputs will be implemented here
        if audio is not None:
            audio_inputs = self.audio_feature_extractor(audio, return_tensors="pt")
            # Logic to combine audio and text inputs will be implemented here
        return inputs

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = Gemma3nConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config, **kwargs)
