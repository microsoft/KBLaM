import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_processor import EncoderArgs, KBLaMProcessor
from kblam.models.llama_model import KblamLlamaForCausalLM


def load_model_and_processor(
    model_path: str, encoder_name: str, kb_layer_frequency: int, encoder_dir: str
) -> tuple[AutoModelForCausalLM, KBLaMProcessor]:
    model = KblamLlamaForCausalLM.from_pretrained(model_path).bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    args = EncoderArgs(
        encoder_name=encoder_name,
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        kb_layer_frequency=kb_layer_frequency,
        encoder_dir=encoder_dir,
    )

    processor = KBLaMProcessor(tokenizer, args)
    return model, processor
