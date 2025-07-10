import json
from pathlib import Path
from typing import Any

def format_question(question: str, fancy: bool = False) -> str:
    """Format a question string for model input, optionally with fancy/augmented formatting."""
    if not fancy:
        return question
    # Example fancy formatting: add special tokens, bold, or other augmentations
    # This can be customized as needed for your models
    return f"[QUESTION]\n**{question.strip()}**\n[/QUESTION]"

def postprocess_output(output: str, pad_token: str = None, strip_special: bool = True) -> str:
    """Clean and normalize model output, optionally stripping pad tokens and special tokens."""
    if pad_token:
        output = output.replace(pad_token, "")
    if strip_special:
        # Remove common special tokens (customize as needed)
        for special in ["<pad>", "<eos>", "<s>", "</s>"]:
            output = output.replace(special, "")
    return output.strip()

def get_topk_confidence(logits, tokenizer, k=5):
    """Return top-k tokens and their softmax confidence scores from logits."""
    import torch
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k)
    topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices.tolist()]
    return list(zip(topk_tokens, topk_probs.tolist()))

def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    """Writes a dictionary to a JSON file with specified formatting.

    This function serializes a Python dictionary to a JSON file with error handling.
    It allows for custom indentation and encoding.

    Args:
        data (Any): The dictionary or other serializable object to write to the file.
        filepath (str): The path to the output JSON file.
        indent (int, optional): The number of spaces for JSON indentation. Defaults to 4.
        encoding (str, optional): The file encoding. Defaults to 'utf-8'.

    Returns:
        bool: True if the file was written successfully, although the function does not explicitly return a value.
    """

    try:
        # Convert string path to Path object
        file_path = Path(filepath)

        # Write the JSON file
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=True,  # For consistent output
                default=str,  # Handle non-serializable objects by converting to string
            )

    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
