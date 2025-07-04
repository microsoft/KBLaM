import torch
from typing import List

def _format_QA_llama(Q: str, A: str):
    """Formats a question and answer for the Llama model.

    Args:
        Q (str): The question.
        A (str): The answer.

    Returns:
        str: The formatted string for the Llama model.
    """
    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )


def _format_QA_phi3(Q: str, A: str):
    """Formats a question and answer for the Phi-3 model.

    Args:
        Q (str): The question.
        A (str): The answer.

    Returns:
        str: The formatted string for the Phi-3 model.
    """
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"


def _format_QA_bitnet(Q: str, A: str):
    """Formats a question and answer for the BitNet model.

    Args:
        Q (str): The question.
        A (str): The answer.

    Returns:
        str: The formatted string for the BitNet model.
    """
    return f"USER: {Q} ASSISTANT: {A}"


def _create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    """Creates labels for the Llama model by masking the question part.

    This function generates labels for training the Llama model by masking out the
    user's question, so that the model only learns to predict the assistant's answer.

    Args:
        input_ids (torch.Tensor): The input tensor of token IDs.
        input_strs (List[str]): The list of input strings.
        tokenizer: The tokenizer used to encode the strings.

    Returns:
        torch.Tensor: The tensor of labels with the question part masked.
    """
    # Not sure this is correct. This method simply masks the <|start_header_id|>user<|end_header_id|> then leaves the rest in the labels
    # Possibly what they want is to mask out the query. To do that swap the index from the tokenizer below from 1 to 2
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][1]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 2)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels


def _create_labels_for_phi3(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    """Creates labels for the Phi-3 model by masking the user part.

    This function generates labels for training the Phi-3 model by masking out the
    user's question, so the model only learns to predict the assistant's answer.

    Args:
        input_ids (torch.Tensor): The input tensor of token IDs.
        input_strs (List[str]): The list of input strings.
        tokenizer: The tokenizer used to encode the strings.

    Returns:
        torch.Tensor: The tensor of labels with the user part masked.
    """
    # We just want to mask out the starting token.
    # The tokenized values are left padded so we want to know where our Q/A pairs start
    # Not 100% this is correct
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|user|>")["input_ids"][0]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 1)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels


def _create_labels_for_bitnet(
    input_ids: torch.Tensor, input_strs: List[str], tokenizer, offset_mapping: torch.Tensor
):
    """Creates labels for the BitNet model by masking the question part.

    This function generates labels for training the BitNet model by finding the
    'ASSISTANT: ' marker and masking all tokens before it.

    Args:
        input_ids (torch.Tensor): The input tensor of token IDs.
        input_strs (List[str]): The list of input strings.
        tokenizer: The tokenizer used to encode the strings.
        offset_mapping (torch.Tensor): The offset mapping from tokens to characters.

    Returns:
        torch.Tensor: The tensor of labels with the question part masked.
    """
    labels = input_ids.clone()
    for i, text in enumerate(input_strs):
        answer_marker = "ASSISTANT: "
        marker_pos = text.find(answer_marker)

        if marker_pos == -1:
            labels[i, :] = -100
            continue

        # The answer starts right after the marker
        answer_start_char_pos = marker_pos + len(answer_marker)

        current_offsets = offset_mapping[i]
        mask_end_token_idx = -1

        # Find the first token that is part of the answer
        for j, (start_char, end_char) in enumerate(current_offsets):
            # Find the token where the answer begins.
            # We look for the first token whose character span includes the start of the answer.
            if start_char <= answer_start_char_pos < end_char:
                mask_end_token_idx = j
                break

        if mask_end_token_idx != -1:
            # Mask all tokens up to (but not including) the first token of the answer
            labels[i, :mask_end_token_idx] = -100
        else:
            # If we couldn't find the start of the answer, mask the whole sequence
            labels[i, :] = -100

    return labels
