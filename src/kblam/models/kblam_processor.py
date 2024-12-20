from typing import Union
from transformers.processing_utils import ProcessorMixin
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers import BatchFeature

from kblam.kb_encoder import KBEncoder
import torch

from dataclasses import dataclass


@dataclass
class EncoderArgs:
    encoder_name: str
    hidden_size: int
    num_hidden_layers: int
    kb_layer_frequency: int
    encoder_dir: str
    projector_type: str
    endpoint_url: str


class KBLaMProcessor(ProcessorMixin):
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer: AutoTokenizer, args: EncoderArgs, **kwargs):
        self.kb_encoder = self.load_encoder(args)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(self.kb_encoder, self.tokenizer)

    def load_encoder(self, args: EncoderArgs):
        encoder = KBEncoder(
            encoder_name=args.encoder_name,
            projector_type=args.projector_type,
            endpoint_url=args.endpoint_url,
            out_dim=args.hidden_size
            * (args.num_hidden_layers // args.kb_layer_frequency + 1),
            frozen_base_model=True,
            projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
            get_oai_embd_online=False,
        )

        encoder.load_state_dict(torch.load(args.encoder_dir))
        return encoder

    def __call__(
        self,
        knowledge_base: list[tuple[torch.Tensor]] | list[tuple[str]] = None,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
    ) -> BatchFeature:
        # Process the knowledge base if needed
        if (
            knowledge_base
            and isinstance(knowledge_base, list)
            and isinstance(knowledge_base[0][0], torch.Tensor)
        ):
            knowledge_base = self.kb_encoder.encode_base_embeddings(knowledge_base)
        elif (
            knowledge_base
            and isinstance(knowledge_base, list)
            and isinstance(knowledge_base[0][0], str)
        ):
            knowledge_base = self.kb_encoder.encode(knowledge_base)

        # Process the text
        input_str = (
            "<|start_header_id|>user<|end_header_id|> "
            + text
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>"
        )

        text_inputs = self.tokenizer(input_str, return_tensors="pt", padding=True).to(
            self.device
        )
        return BatchFeature(data={**text_inputs, "kb_kvs": knowledge_base})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
