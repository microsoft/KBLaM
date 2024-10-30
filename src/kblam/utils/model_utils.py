import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_processor import EncoderArgs, KBLaMProcessor
from kblam.models.llama_model import KblamLlamaForCausalLM
from kblam.models.llama_model_old import KblamForCausalLM


def load_model_and_tokenizer_old(model_name, hf_token, encoder_name, kb_layer_frequency, query_head_path, encoder_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=hf_token)

    tokenizer.pad_token = tokenizer.eos_token

    model = KblamForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.load_query_head(query_head_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = 128009
    model.model.config._attn_implementation_internal = 'eager'
    model.model.config._attn_implementation = 'eager'
    model.eval()

    encoder = KBEncoder(
        encoder_name=encoder_name,
        projector_type='linear',
        endpoint_url="",
        out_dim=model.config.hidden_size * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={'mlp_depth': 1, 'mlp_hidden_dim': 512},
        get_oai_embd_online=True,
    )

    encoder.load_state_dict(torch.load(encoder_dir))

    return model, tokenizer, encoder


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
