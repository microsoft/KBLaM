from typing import Optional

import numpy as np
import torch
import transformers
from transformers import Gemma3nConfig

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM
from kblam.models.gemma3n_model import KblamGemma3nForConditionalGeneration

instruction_prompts = """
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
"""

instruction_prompts_multi_entities = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

zero_shot_prompt = """
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
"""

zero_shot_prompt_multi_entities = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


def _prune_for_llama(S: str) -> str:
    S = S.replace("<|eot_id|>", "")
    S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "\n\n")
    S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
    S = S.replace("<|end_of_text|>", "")
    return S


def _prune_for_phi3(S: str) -> str:
    S = S.replace("<|end|>", "")
    S = S.replace("<|assistant|>", "\n\n")
    S = S.replace("<|user|>", "")
    return S


def _prune_for_bitnet(S: str) -> str:
    S = S.replace("<s>", "").replace("</s>", "")
    # The model output contains the prompt, so we remove it.
    assistant_marker = "ASSISTANT:"
    marker_pos = S.find(assistant_marker)
    if marker_pos != -1:
        return S[marker_pos + len(assistant_marker) :].strip()
    return S


def _prune_for_gemma3n(S: str) -> str:
    S = S.replace("<s>", "").replace("</s>", "")
    # The model output contains the prompt, so we remove it.
    assistant_marker = "ASSISTANT:"
    marker_pos = S.find(assistant_marker)
    if marker_pos != -1:
        return S[marker_pos + len(assistant_marker) :].strip()
    return S


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )


def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


def _format_Q_bitnet(Q: str):
    return f"USER: {Q} ASSISTANT:"


def _format_Q_gemma3n(Q: str):
    return f"USER: {Q} ASSISTANT:"


model_question_format_mapping = {
    KblamLlamaForCausalLM: _format_Q_llama,
    KBLaMPhi3ForCausalLM: _format_Q_phi3,
    KBLaMBitNetForCausalLM: _format_Q_bitnet,
    KblamGemma3nForConditionalGeneration: _format_Q_gemma3n,
}
model_prune_format_mapping = {
    KblamLlamaForCausalLM: _prune_for_llama,
    KBLaMPhi3ForCausalLM: _prune_for_phi3,
    KBLaMBitNetForCausalLM: _prune_for_bitnet,
    KblamGemma3nForConditionalGeneration: _prune_for_gemma3n,
}


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KBLaMBitNetForCausalLM | KblamGemma3nForConditionalGeneration,
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig | Gemma3nConfig] = None,
    attention_save_loc: Optional[str] = None,
    save_attention_weights: bool = False,
    attention_file_base_name: Optional[str] = None,
    topk_size: int = -1,
):
    print("[DEBUG][answer_question] Entered function")
    print(f"[DEBUG][answer_question] Model type: {type(model)}")
    print(f"[DEBUG][answer_question] Q: {Q}")
    print(f"[DEBUG][answer_question] kb is None: {kb is None}")
    print(f"[DEBUG][answer_question] kb_config: {kb_config}")
    print(f"[DEBUG][answer_question] topk_size: {topk_size}")
    for m in model_question_format_mapping:
        if isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
    print(f"[DEBUG][answer_question] input_str: {input_str}")
    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )
    print(f"[DEBUG][answer_question] input_ids shape: {input_ids.shape}, device: {input_ids.device}")
    print(f"[DEBUG][answer_question] attention_masks shape: {attention_masks.shape}, device: {attention_masks.device}")
    if kb is not None:
        if isinstance(kb, tuple):
            print(f"[DEBUG][answer_question] kb tuple devices: {[k.device for k in kb]}")
        else:
            print(f"[DEBUG][answer_question] kb device: {kb.device}")

    if kb_config is not None and topk_size > -1:
        kb_config.top_k_kb = topk_size

    with torch.autograd.no_grad():
        # Set pad_token_id logic for each model type
        if isinstance(model, (KblamLlamaForCausalLM, KBLaMPhi3ForCausalLM, KBLaMBitNetForCausalLM)):
            pad_token_id = tokenizer.eos_token_id
        elif isinstance(model, KblamGemma3nForConditionalGeneration):
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "kb_kvs": kb,
            "max_new_tokens": 150,
            "tokenizer": tokenizer,
            "output_attentions": True,
            "kb_config": kb_config,
            "pad_token_id": pad_token_id,
            "save_attention_weights": save_attention_weights,
            "attention_file_base_name": attention_file_base_name,
            "attention_save_loc": attention_save_loc,
        }
        # The generate method in some models might not accept all these arguments.
        # We can filter them based on the model's generate method signature.
        import inspect
        sig = inspect.signature(model.generate)
        filtered_kwargs = {k: v for k, v in generate_kwargs.items() if k in sig.parameters}
        print(f"[DEBUG][answer_question] About to call model.generate with keys: {list(filtered_kwargs.keys())}")
        print(f"[DEBUG][answer_question] model.generate input_ids shape: {filtered_kwargs.get('input_ids').shape if 'input_ids' in filtered_kwargs else 'N/A'}")
        print(f"[DEBUG][answer_question] model.generate attention_mask shape: {filtered_kwargs.get('attention_mask').shape if 'attention_mask' in filtered_kwargs else 'N/A'}")
        try:
            outputs = model.generate(**filtered_kwargs).squeeze()
            print("[DEBUG][answer_question] model.generate completed successfully")
        except Exception as e:
            print(f"[DEBUG][answer_question] model.generate raised exception: {e}")
            raise
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)
    print(f"[DEBUG][answer_question] Decoded outputs: {outputs[:200]}...")

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
    print(f"[DEBUG][answer_question] Returning pruned_output: {pruned_output[:200]}...")
    return pruned_output
