import re
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM
from kblam.models.gemma3n_model import KblamGemma3nForConditionalGeneration

def _get_parameter_count(encoder):
    """Calculates the number of trainable parameters in the encoder.

    Args:
        encoder: The encoder model.

    Returns:
        float: The total number of trainable parameters.
    """
    param_count = 0.0
    for p in encoder.parameters():
        if p.requires_grad:
            param_count += p.numel()
    return param_count


def _get_phi3_query_head_parameters(
    model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM | KBLaMBitNetForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    """Retrieves the query head parameters for the Phi-3 model.

    This function identifies and returns the parameters of the query projection
    layers in the Phi-3 model that are designated for the knowledge base.
    It supports both separate and shared query heads.

    Args:
        model (KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM | KBLaMBitNetForCausalLM): The model.
        sep_query_head (bool): Whether to use a separate query head.
        kb_token_layer_frequency (int): The frequency of KB token layers.

    Returns:
        list: A list of the query head parameters.
    """
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            # For phi3
            if "qkv_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight[: model.config.hidden_size, :])  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


def _get_llama3_query_head_parameters(
    model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM | KBLaMBitNetForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    """Retrieves the query head parameters for the Llama-3 model.

    This function identifies and returns the parameters of the query projection
    layers in the Llama-3 model that are designated for the knowledge base.
    It supports both separate and shared query heads.

    Args:
        model (KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM | KBLaMBitNetForCausalLM): The model.
        sep_query_head (bool): Whether to use a separate query head.
        kb_token_layer_frequency (int): The frequency of KB token layers.

    Returns:
        list: A list of the query head parameters.
    """
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:  # TODO: this is different for each model type
            # For llama3
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


def _get_bitnet_query_head_parameters(
    model: KBLaMBitNetForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    """Retrieves the query head parameters for the BitNet model.

    This function identifies and returns the parameters of the query projection
    layers in the BitNet model that are designated for the knowledge base.
    It supports both separate and shared query heads.

    Args:
        model (KBLaMBitNetForCausalLM): The BitNet model.
        sep_query_head (bool): Whether to use a separate query head.
        kb_token_layer_frequency (int): The frequency of KB token layers.

    Returns:
        list: A list of the query head parameters.
    """
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


def _get_gemma3n_query_head_parameters(
    model: KblamGemma3nForConditionalGeneration,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    """Retrieves the query head parameters for the Gemma3n model.

    This function identifies and returns the parameters of the query projection
    layers in the Gemma3n model that are designated for the knowledge base.
    It supports both separate and shared query heads.

    Args:
        model (Gemma3nForConditionalGeneration): The Gemma3n model.
        sep_query_head (bool): Whether to use a separate query head.
        kb_token_layer_frequency (int): The frequency of KB token layers.

    Returns:
        list: A list of the query head parameters.
    """
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params
