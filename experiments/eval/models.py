import os
import torch
import transformers
from transformers import AutoTokenizer

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM

def _prepare_models(
    encoder_spec,
    encoder_path,
    llm_type,
    llm_base_dir,
    model_path,
    query_head_path,
    kb_layer_frequency,
    kb_scale_factor,
):
    """Prepares and loads the tokenizer, encoder, and language model for evaluation.

    This function initializes the tokenizer, loads the specified language model
    (Llama3, Phi3, or BitNet), and sets up the knowledge base encoder. It also
    configures the model for generation and loads pre-trained weights and query heads
    if provided.

    Args:
        encoder_spec (str): The specification for the encoder model.
        encoder_path (str): The path to the pre-trained encoder model.
        llm_type (str): The type of the language model ('llama3', 'phi3', 'bitnet').
        llm_base_dir (str): The base directory for the language model.
        model_path (str): The path to the pre-trained language model.
        query_head_path (str): The path to the pre-trained query head.
        kb_layer_frequency (int): The frequency of knowledge base layers.
        kb_scale_factor (int): The scaling factor for the knowledge base.

    Returns:
        tuple: A tuple containing the tokenizer, encoder, model, and knowledge base configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )

    if llm_type == "llama3":
        if query_head_path:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            model.load_query_head(query_head_path)
        else:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
    elif llm_type == "bitnet":
        model = KBLaMBitNetForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if query_head_path:
            model.load_query_head(query_head_path)
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if model.generation_config is None:
        model.generation_config = transformers.GenerationConfig.from_model_config(
            model.config
        )

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    # config = model.config.to_dict()
    kb_config = KBLaMConfig(
        sep_query_head=True,
        kb_layer_frequency=kb_layer_frequency,
        kb_scale_factor=kb_scale_factor,
    )
    # config.update(kb_config.to_dict())
    # new_config = KBLaMConfig(**config)
    # model.config = new_config

    encoder = KBEncoder(
        encoder_name=encoder_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size
        * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )

    encoder.load_state_dict(torch.load(os.path.join(encoder_path, "encoder.pt")))
    return tokenizer, encoder, model, kb_config
