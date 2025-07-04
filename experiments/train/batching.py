import torch
import numpy as np
from typing import Callable, Dict, List
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa, get_i_dont_know_ans

def get_batch(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable, torch.Tensor], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    B: int = 20,
    random_sample=True,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
):
    """Generates a batch of data for training.

    This function creates a batch of input IDs, attention masks, and labels for
    training the model. It can handle various data augmentation strategies,
    including outliers, multi-entity questions, and extended Q&A pairs.

    Args:
        qa_format_func (Callable[[str, str], str]): A function to format the question and answer into a single string.
        label_func (Callable[[torch.Tensor, List, Callable, torch.Tensor], torch.Tensor]): A function to create labels for the model.
        dataset (List[Dict]): The dataset to sample from.
        tokenizer: The tokenizer for the model.
        device (torch.device): The device to place the tensors on.
        B (int, optional): The batch size. Defaults to 20.
        random_sample (bool, optional): Whether to sample randomly from the dataset. Defaults to True.
        use_data_aug (bool, optional): Whether to use data augmentation. Defaults to False.
        include_outlier (bool, optional): Whether to include outlier questions. Defaults to False.
        multi_entities (int, optional): The number of entities for multi-entity questions. Defaults to None.
        use_extended_qa (bool, optional): Whether to use extended Q&A pairs. Defaults to False.

    Returns:
        tuple: A tuple containing input IDs, attention masks, labels, and batch indices.
    """
    labels = []
    if multi_entities is not None:
        assert not include_outlier

    if random_sample:
        if multi_entities is not None:
            batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)
        else:
            batch_indices = np.random.choice(len(dataset), B, replace=False)
    else:
        batch_indices = np.arange(B)

    def get_question_and_answer(idx: int) -> tuple[str, str]:
        if use_extended_qa:
            Q, A = dataset[idx]["extended_Q"], dataset[idx]["extended_A"]

        elif multi_entities is not None:
            Q, A = generate_multi_entity_qa(
                [dataset[i]["name"] for i in idx],
                [dataset[i]["description_type"] for i in idx],
                [dataset[i]["description"] for i in idx],
            )
        else:
            Q = augment_row(dataset[idx]) if use_data_aug else dataset[idx]["Q"]
            A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]
        return Q, A

    with torch.autograd.no_grad():
        input_strs = []
        real_batch_indices = []
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            if Q is not None and A is not None:
                input_strs.append(qa_format_func(Q, A))
                real_batch_indices.append(idx)
            else:
                print("Q or Answer is none")
        batch_indices = real_batch_indices
        tokenizer_output = tokenizer(
            input_strs, return_tensors="pt", padding=True, return_offsets_mapping=True
        ).to(device)
        input_ids, attention_masks, offset_mapping = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
            tokenizer_output["offset_mapping"],
        )

        labels = label_func(input_ids, input_strs, tokenizer, offset_mapping)
    if include_outlier:
        # Generate a new set of indices, such that the KB does not contain the entity where the question comes from
        batch_indices = np.random.choice(len(dataset), B, replace=False)
    return input_ids, attention_masks, labels, batch_indices
