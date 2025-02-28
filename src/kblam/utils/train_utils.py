import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import argparse

from torch.optim.optimizer import ParamsT
from torch.nn.parallel import DistributedDataParallel


def get_tensor_config(x: torch.tensor) -> dict[str, any]:
    return {"dtype": x.dtype, "layout": x.layout, "device": x.device}


def preprocess_embds(emb1: list, emb2: list, kb_mask_val: int = 1):
    """
    emb1: List of embeddings of the KB.
    emb2: List of embeddings of the query.
    kb_mask_val: Attention mask value for the KB, i.e. emb1, part.

    The function would first pad emb1 on the left and then concat emb1 and emb2.

    Return:
    A single 2-D embedding tensor, the attention mask, and the position_ids,
    where the position_ids for the KB embeddings parts are set 0.
    """
    assert isinstance(emb1, list)
    assert isinstance(emb2, list)
    assert len(emb1) == len(emb2)
    max_length = max([e1.shape[0] + e2.shape[0] for e1, e2 in zip(emb1, emb2)])
    joint_embs = []
    attention_masks = []
    position_ids = []
    kb_masks = []
    for e1, e2 in zip(emb1, emb2):
        tensor_config = get_tensor_config(e1)
        pad_size = max_length - e1.shape[0] - e2.shape[0]
        padding = torch.zeros((pad_size, e1.shape[1]), **tensor_config)
        joint_embs.append(torch.concat([padding, e1, e2]))
        attention_mask = torch.cat(
            [
                torch.zeros(pad_size, **tensor_config),
                torch.zeros(e1.shape[0], **tensor_config) + kb_mask_val,  # Attention mask for KB
                torch.ones(e2.shape[0], **tensor_config),  # Attention mask for the question
            ]
        )

        attention_masks.append(attention_mask)
        position_id = torch.cat(
            [
                torch.zeros(max_length - e2.shape[0], **tensor_config) - 1,
                torch.arange(1, e2.shape[0] + 1, **tensor_config) - 1,
            ]
        )
        position_ids.append(position_id)

        kb_mask = torch.cat(
            [
                torch.zeros(pad_size, **tensor_config),
                torch.ones(e1.shape[0], **tensor_config),
                torch.zeros(e2.shape[0], **tensor_config),
            ]
        )
        kb_masks.append(kb_mask)

    return (
        torch.stack(joint_embs),
        torch.stack(attention_masks),
        torch.stack(position_ids),
        torch.stack(kb_masks),
    )


def kb_to_embd(kb_encoder, kb_dict=None, precomputed_base_embd=None):
    if isinstance(kb_encoder, DistributedDataParallel):
        kb_encoder = kb_encoder.module
    key_embds, value_embds = [], []
    if precomputed_base_embd is not None:
        for key_base_embd, value_base_embd in zip(*precomputed_base_embd):
            key_embds.append(kb_encoder.encode_key(base_emb=key_base_embd))
            value_embds.append(kb_encoder.encode_val(base_emb=value_base_embd))
    else:
        for entity in kb_dict:
            key_embds.append(kb_encoder.encode_key(S=entity["key_string"]))
            value_embds.append(kb_encoder.encode_val(S=entity["description"]))
    return (torch.stack(key_embds), torch.stack(value_embds))


def get_kb_embd(
    kb_encoder: torch.nn.Module,
    indices: list[int],
    kb_dict: dict = None,
    precomputed_embd: tuple[torch.tensor] = None,
) -> tuple[torch.tensor]:
    if precomputed_embd:
        key_embds, value_embds = precomputed_embd
        train_set_key, train_set_val = kb_to_embd(
            kb_encoder,
            precomputed_base_embd=np.stack([key_embds[indices], value_embds[indices]]),
        )
    elif kb_dict:
        if len(indices.shape) == 2:
            # Sampling batch of multi entities
            train_set_key, train_set_val = [], []
            for indices_row in indices.T:
                _train_set_key, _train_set_val = kb_to_embd(kb_encoder, kb_dict=[kb_dict[i] for i in indices_row])
                (train_set_key.append(_train_set_key),)
                train_set_val.append(_train_set_val)
            train_set_key = torch.stack(train_set_key, 1)
            train_set_val = torch.stack(train_set_val, 1)
        elif len(indices.shape) == 1:
            train_set_key, train_set_val = kb_to_embd(kb_encoder, kb_dict=[kb_dict[i] for i in indices])
    return train_set_key, train_set_val


def weighted_nll(model, input_ids, attention_mask, labels, kb=None):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        kb_kv=kb,
        output_attentions=True,
    )
    logits = out["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    weights = weights.view(-1)
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_labels = shift_labels.to(shift_logits.device)
    loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights).mean()
    return loss


def compute_perplexity_gain(model, kb, input_ids, attention_mask, labels):
    with torch.autograd.no_grad():
        unconditioned_nll = weighted_nll(model, input_ids, attention_mask, labels, kb=None)
        conditioned_nll = weighted_nll(model, input_ids, attention_mask, labels, kb)
    return unconditioned_nll, conditioned_nll  # Loss should decrease


def context_set_size_scheduler(curr_step: int, kb_size: list[int] | int | str) -> int:
    """Determines the KB size for the current training step.
    The KB size can be a fixed number, a list of numbers or a "dynamic" value.
    If no KB size is provided, the KB size is dynamicly increased every 100 steps."""

    dynamic_range = (10, 200)
    if kb_size == "dynamic":
        return np.random.randint(dynamic_range[0], dynamic_range[1])

    if isinstance(kb_size, list):
        return np.random.randint(kb_size[0], kb_size[1])

    increase_kb_size_every = 100
    if not kb_size:
        round = (curr_step) // increase_kb_size_every
        return 4 * (round + 1)

    return kb_size


def get_prefix_str(args: argparse.Namespace) -> str:
    kb_size = args.kb_size
    if kb_size == -1:
        kb_size = None  # Progressively increase size
    elif kb_size == 0:
        kb_size = "dynamic"  # Random size

    prefix_string = f"stage1_lr_{args.lr}"
    if args.kb_token_layer_frequency is not None:
        prefix_string += f"KBTokenLayerFreq{args.kb_token_layer_frequency}"
    if args.use_extended_qa:
        prefix_string += "UseExtendedQA"
    if args.multi_entities is not None:
        prefix_string += f"MultiEntities{args.multi_entities}"
    if args.outlier_num > 0:
        prefix_string += f"UseOutlier{args.outlier_num}"
    if args.length_invariance:
        prefix_string += "LengthInvariant"
    if kb_size is not None:
        prefix_string += f"KBSize{kb_size}"
    if args.sep_query_head:
        prefix_string += "SepQueryHead"
    if args.use_data_aug:
        prefix_string += "UseDataAug"
    return prefix_string


def setup_scheduler_and_optimizer(model_parapmeters: ParamsT, lr: float, max_iter: int) -> tuple:
    optim = torch.optim.AdamW(model_parapmeters, lr=lr, weight_decay=0.0)  # type: ignore

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_iter, eta_min=lr * 0.01)  # type: ignore
    return scheduler, optim
