import argparse
import json
import logging
import os
import pathlib
import re
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.theme import Theme
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa, get_i_dont_know_ans
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd, setup_scheduler_and_optimizer

LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

# setup logging
# Create a custom theme for Rich
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)

# Create a Rich console with the custom theme
console = Console(theme=custom_theme)

# Configure the root logger to WARNING
logging.basicConfig(
    level=logging.WARNING,  # Set the root logger to WARNING
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--train_dataset",type=str,default="synthetic")
parser.add_argument("--N", type=int, default=120000, help="Size of training set, select the first N samples for training")
parser.add_argument("--B", type=int, default=10, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, help="Train a separate query head")
parser.add_argument("--use_oai_embd", action="store_true", help="Use OpenAI embedding")
parser.add_argument("--use_cached_embd", action="store_true", help="Choose to use pre-computed KV embeddings")
parser.add_argument("--total_steps", type=int, default=20000, help="Total steps")
parser.add_argument("--encoder_spec", type=str, default="OAI")
parser.add_argument("--key_embd_src", type=str, default="key", choices=["key", "answer", "questions", None], help="Source of key embedding")
parser.add_argument("--use_data_aug", action="store_true", help="Randomly pick templates for the question")
parser.add_argument("--use_lr_decay", action="store_true")
parser.add_argument("--dataset_dir", type=str, default="synthetic_data")
parser.add_argument("--model_dir_to_resume", type=str, default=None, help="Checkpoint directory to resume training")
parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct", choices=["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-3.2-1B-Instruct", "microsoft/bitnet-b1.58-2B-4T-bf16"])
parser.add_argument("--hf_token", type=str,default=None,help="Huggingface token")
parser.add_argument("--model_save_dir", type=str, default="output", help="Place to save the checkpoints")
parser.add_argument("--kb_size", type=int, default=None, help="The size of the KB set size")
parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None, help="The size of the KB set size. Set a dynamic range for the kbsize specify min and max")
parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True, help="Duplicate true entity's KB token")
parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False, help="Scale the raw attention score")
parser.add_argument("--outlier_num", type=int, default=1, help="Introduce questions without correct KB entites")
parser.add_argument("--multi_entities", type=int, default=None, help="Introduce questions involving multiple entities")
parser.add_argument("--use_extended_qa", action="store_true", help="Introduce QA with extended open-ended parts")
parser.add_argument("--kb_token_layer_frequency", type=int, default=3, help="Introduce QA with extended open-ended parts")
parser.add_argument("--gradient_accm_step", type=int, default=20, help="Introduce QA with extended open-ended parts")
parser.add_argument("--verbose", action="store_true", help="Set logging to debug")
parser.add_argument("--log_to_file", action="store_true", help="Log to file as well as stdout")
parser.add_argument("--llm_type",type=str,default="llama3",choices=["llama3", "phi3", "bitnet"])
parser.add_argument("--max_seq_len",type=int,default=None)
# fmt: on


def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
    disable=False,
) -> Progress:
    """
    Create a custom progress bar using Rich, optionally including loss reporting.

    :param description: Description of the task
    :param total: Total number of steps
    :param console: Rich Console object (if None, a new one will be created)
    :param color: Color of the progress bar
    :param show_time: Whether to show the time remaining
    :param show_spinner: Whether to show a spinner
    :param spinner_style: Style of the spinner (e.g., "dots", "dots12", "line", "arrow")
    :param show_loss: Whether to show loss information
    :return: A Rich Progress object and task ID
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend(
        [
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
        ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=disable)
    return progress


def _format_QA_llama(Q: str, A: str):
    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )


def _format_QA_phi3(Q: str, A: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"


def _format_QA_bitnet(Q: str, A: str):
    return f"USER: {Q} ASSISTANT: {A}"


def _create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
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
    labels = input_ids.clone()
    for i, text in enumerate(input_strs):
        answer_marker = "ASSISTANT: "
        marker_pos = text.find(answer_marker)

        if marker_pos == -1:
            labels[i, :] = -100
            continue

        prompt_end_char_pos = marker_pos + len(answer_marker)
        
        current_offsets = offset_mapping[i]
        token_id = -1

        # Find the token that contains the character at the end of the prompt
        for j, (start_char, end_char) in enumerate(current_offsets):
            if start_char < prompt_end_char_pos <= end_char:
                token_id = j
                break

        if token_id != -1:
            mask_end_index = token_id + 1
            labels[i, :mask_end_index] = -100
        else:
            labels[i, :] = -100

    return labels


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
    """
    dataset: List of dictionary, denoting the KB, used to extract QA pairs
    model: The LLM, used to provide the embedding
    kb_embedding: KB embedding (differentiable)
    B: Batchsize
    include_outlier : Create a batch of question without answer in the KB.
    multi_entities : Create a batch of question that involves more than one entities.
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


def get_prefix_str(args):
    use_data_aug = args.use_data_aug
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size

    if dynamic_kb_size is not None:
        kb_size = "dynamic"  # Random size

    duplicate_true_kb = args.duplicate_true_kb
    length_invariance = args.length_invariance
    outlier_ratio = args.outlier_num
    use_outlier = outlier_ratio != -1
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    lr = args.lr

    prefix_string = f"stage1_lr_{str(lr).replace('.', 'p')}"
    if kb_token_layer_frequency is not None:
        prefix_string += f"KBTokenLayerFreq{kb_token_layer_frequency}"
    if use_extended_qa:
        prefix_string += "UseExtendedQA"
    if multi_entities is not None:
        prefix_string += f"MultiEntities{multi_entities}"
    if use_outlier:
        prefix_string += f"UseOutlier{outlier_ratio}"
    if length_invariance:
        prefix_string += "LengthInvariant"
    if not duplicate_true_kb:
        prefix_string += "NoDuplicate"
    if kb_size is not None:
        prefix_string += f"KBSize{kb_size}"
    if sep_query_head:
        prefix_string += "SepQueryHead"
    if use_data_aug:
        prefix_string += "UseDataAug"
    return prefix_string


def _load_cached_embeddings(encoder_model_spec: str, dataset_dir: str, dataset_name: str, key_embd_src: str):
    if encoder_model_spec == "OAI":
        encoder_model_spec_str = "oai"
    else:
        encoder_model_spec_str = encoder_model_spec
    key_embds = np.load(
        os.path.join(
            dataset_dir,
            f"{dataset_name}_{encoder_model_spec_str}_embd_{key_embd_src}.npy",
        )
    ).astype("float32")
    if key_embd_src == "answer":
        # If we are using the answer string as the key, we also use it as the value string
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"{dataset_name}_{encoder_model_spec_str}_embd_answer.npy",
            )
        ).astype("float32")
    else:
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"{dataset_name}_{encoder_model_spec_str}_embd_value.npy",
            )
        ).astype("float32")
    return key_embds, value_embds


def get_step_config(
    current_accum_step: int,
    total_accum_step: int,
    use_data_aug: bool,
    outlier_num: int,
    multi_entities: int | None,
    use_extended_qa: bool,
):
    """
    Our instruction tuning dataset is composed of different types of instructions.
    Strategies:
    Outlier QA takes the last `outlier_num` accum steps;
    Multiple entites QA (if included) takes 1/3 of the rest accum_steps;
    Extended QA (if included) takes 1/3 of the rest accum_steps;
    Standard QA takes the rest.
    """
    config = {}
    config["use_data_aug"] = use_data_aug
    config["include_outlier"] = False
    config["multi_entities"] = None
    config["use_extended_qa"] = False
    include_outlier = current_accum_step >= total_accum_step - 1 - outlier_num
    # Decide to include outlier and has reached the time
    if include_outlier:
        config["include_outlier"] = True
        return config
    if current_accum_step % 3 == 0:
        # multi_entities could be None,
        # in which case we just use standard QA
        config["multi_entities"] = multi_entities
        return config
    if current_accum_step % 3 == 1:
        config["use_extended_qa"] = use_extended_qa
        return config
    return config


def _get_parameter_count(encoder):
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


class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray],
        value_embds: Optional[np.ndarray],
    ):
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices, batch_size, step, kb_size):
        if self._use_cached_embd():
            train_set_key, train_set_val = get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            train_set_key, train_set_val = get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)

        if len(train_set_key.shape) == 2:
            # Add comment on why we need this line
            train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
            train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_index = np.random.choice(len(self.dataset), context_set_size, replace=False)  # type: ignore
        if self._use_cached_embd():
            context_set_key, context_set_val = get_kb_embd(
                self.encoder,
                context_set_index,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            context_set_key, context_set_val = get_kb_embd(self.encoder, context_set_index, kb_dict=self.dataset)
        context_set_key = context_set_key.unsqueeze(0).expand(batch_size, *context_set_key.shape)
        context_set_val = context_set_val.unsqueeze(0).expand(batch_size, *context_set_val.shape)
        # context_set_val = torch.randn_like(context_set_val)
        # Idea: Try torch.randn here context_set_tokens??
        true_kb_copy = 1
        kb_embedding = (
            torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
            torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        )
        return kb_embedding


class Trainer:
    def __init__(
        self,
        llm_model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KBLaMBitNetForCausalLM,
        kbretriever: KBRetriever,
        tokenizer: transformers.PreTrainedTokenizer,
        kb_token_layer_frequency: int,
        num_steps: int,
        lr: float,
        device: torch.device | None,
        use_lr_decay: bool,
        kb_size: int | List[int],
        llm_savename: str,
        output_dir: str,
        sep_query_head: bool = False,
        max_seq_len: int | None = None,
    ):
        self.accelerator = Accelerator()
        self.logger = logging.getLogger("training")
        self.tokenizer = tokenizer
        self.sep_query_head = sep_query_head
        self.kb_token_layer_frequency = kb_token_layer_frequency
        self.num_steps = num_steps
        self.lr = lr
        self.max_seq_len = max_seq_len

        self.model = llm_model
        self.model.gradient_checkpointing_enable()

        self.device = device if device is not None else self.accelerator.device
        self.kbretriever = kbretriever
        self.kb_size = kb_size
        self.use_lr_decay = use_lr_decay
        self.llm_savename = llm_savename
        self.output_path = pathlib.Path(output_dir)

        if isinstance(llm_model, KBLaMPhi3ForCausalLM):  # Phi3
            self._get_batch = partial(get_batch, _format_QA_phi3, _create_labels_for_phi3)
            self._get_params = _get_phi3_query_head_parameters
        elif isinstance(llm_model, KblamLlamaForCausalLM):  # llama
            self._get_batch = partial(get_batch, _format_QA_llama, _create_labels_for_llama)
            self._get_params = _get_llama3_query_head_parameters
        elif isinstance(llm_model, KBLaMBitNetForCausalLM):
            self._get_batch = partial(get_batch, _format_QA_bitnet, _create_labels_for_bitnet)
            self._get_params = _get_bitnet_query_head_parameters
        else:
            raise ValueError(f"{llm_model} not recognised")

        self.scheduler, self.optim = self.setup_scheduler_and_optim()

        self.model, self.optim, self._get_batch, self.kbretriever.encoder = self.accelerator.prepare(
            self.model, self.optim, self._get_batch, self.kbretriever.encoder
        )

    def setup_scheduler_and_optim(self):
        if self.sep_query_head:
            self.logger.info("Query head being fine tuned!")
            llm_q_params = self._get_params(self.model, self.sep_query_head, self.kb_token_layer_frequency)
            scheduler, optim = setup_scheduler_and_optimizer(
                chain(self.kbretriever.encoder.parameters(), llm_q_params),
                self.lr,
                self.num_steps,
            )
            self.logger.info("Optimizer recreated")
        else:
            scheduler, optim = setup_scheduler_and_optimizer(
                self.kbretriever.encoder.parameters(), self.lr, self.num_steps
            )
            self.logger.info("Optimizer recreated")
        return scheduler, optim

    def train(
        self,
        training_set: List[Dict],
        batch_size,
        grad_accum_steps: int,
        outlier_num: int,
        use_data_aug: bool = False,
        multi_entities: bool = False,
        use_extended_qa: bool = False,
        save_period: int = 2000,
        resumed_step: int = 0,
        kb_config: KBLaMConfig = None,
    ):
        train_losses = []
        start_step = resumed_step

        loss_fct = CrossEntropyLoss(reduction="none")

        # Calculate accumulation steps per GPU
        num_processes = self.accelerator.num_processes
        accum_steps_per_gpu = max(1, grad_accum_steps // num_processes)
        effective_batch_size = batch_size * grad_accum_steps

        if self.accelerator.is_main_process:
            self.logger.info(f"Training with {num_processes} GPUs")
            self.logger.info(f"Total accumulation steps: {grad_accum_steps}, Steps per GPU: {accum_steps_per_gpu}")
            self.logger.info(f"Batch size: {batch_size}")
            self.logger.info(f"Effective batch size: {effective_batch_size}")

        with create_custom_progress_bar(console=console, disable=not self.accelerator.is_main_process) as pbar:
            task = pbar.add_task("Training", total=self.num_steps, loss=100)
            for step in range(start_step, self.num_steps, 1):
                self.optim.zero_grad()
                losses = []

                # Calculate which accumulation steps this GPU should process
                process_rank = self.accelerator.process_index
                start_accum_step = process_rank * accum_steps_per_gpu
                end_accum_step = min(start_accum_step + accum_steps_per_gpu, grad_accum_steps)

                # Accumulate gradients
                for a_step in range(start_accum_step, end_accum_step):
                    step_config = get_step_config(
                        a_step,
                        grad_accum_steps,
                        use_data_aug,
                        outlier_num,
                        multi_entities,
                        use_extended_qa,
                    )
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(
                        training_set,
                        self.tokenizer,
                        self.device,
                        B=batch_size,
                        random_sample=True,
                        **step_config,
                    )

                    if a_step == 0 and step % 10 == 0:
                        self.logger.info(f"INPUT IDs SHAPE: {input_ids.shape}")

                    if self.max_seq_len is not None:
                        input_ids = input_ids[:, : self.max_seq_len]
                        attention_masks = attention_masks[:, : self.max_seq_len]
                        labels = labels[:, : self.max_seq_len]
                        if a_step == 0 and step % 10 == 0:
                            self.logger.info(f"TRUNCATED INPUT IDs SHAPE: {input_ids.shape}")

                    kb_embedding = self.kbretriever.get_key_embeddings(
                        batch_indices, len(input_ids), step, self.kb_size
                    )
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        kb_kvs=kb_embedding,
                        output_attentions=True,
                        kb_config=kb_config,
                    )
                    logits = out["logits"]

                    # display ground truth and model prediction to quickly check model
                    if a_step == 0 and step % 10 == 0:
                        batch_index = 0  # Which example in the batch to select
                        max_logits = logits.argmax(axis=2)
                        decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])
                        sel_labels = labels[batch_index, :]
                        sel_labels = sel_labels[sel_labels >= 0]  # Remove padding token -100
                        decoded_gt = self.tokenizer.decode(sel_labels)
                        self.logger.info(f"KB SHAPE: {kb_embedding[0].shape}")
                        self.logger.info(f"GT: {decoded_gt}")
                        self.logger.info(f"PRED: {decoded_pred}")
                        wandb.log({"kbsize": kb_embedding[0].shape[1]})

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
                    # Flatten the tokens
                    model_config = (
                        self.model.config
                        if not isinstance(self.model, DistributedDataParallel)
                        else self.model.module.config
                    )
                    shift_logits = shift_logits.view(-1, model_config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    weights = weights.view(-1)

                    shift_labels = shift_labels.to(shift_logits.device)

                    loss = (
                        loss_fct(shift_logits, shift_labels) * weights.max() / weights
                    ).mean()  # Make sure each sample is equally weighted

                    self.accelerator.backward(loss)
                    losses.append(loss.item())

                self.optim.step()
                if self.use_lr_decay:
                    self.scheduler.step()

                # Gather and average losses from all GPUs for reporting
                if losses:  # Only if this GPU processed any batches
                    local_loss = torch.tensor(np.mean(losses), device=self.device)
                else:
                    local_loss = torch.tensor(0.0, device=self.device)

                # Gather losses from all processes
                all_losses = self.accelerator.gather(local_loss)
                valid_losses = all_losses[all_losses > 0]  # Filter out zeros from GPUs that didn't process batches
                avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0

                # Only log from main process
                if self.accelerator.is_main_process:
                    self.logger.info(f"step: {step}, loss: {avg_loss}")
                    wandb.log({'train_loss': np.mean(losses)})
                    train_losses.append(avg_loss)
                    pbar.update(task, advance=1, loss=avg_loss)

                if (step % save_period) == 0 and (step != start_step):
                    try:
                        # Log memory usage before synchronization
                        self.logger.info(
                            f"Is main process: {self.accelerator.is_main_process}, GPU memory before save: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
                        )

                        # Try to free up memory
                        torch.cuda.empty_cache()

                        # Synchronize before saving
                        self.accelerator.wait_for_everyone()

                        if self.accelerator.is_main_process:

                            self.logger.info("Saving checkpoint...")
                            self.logger.info("Making dirs...")
                            # Save model - using proper directory creation
                            model_ckpt_name = self.output_path / f"{self.llm_savename}_step_{step}"
                            model_ckpt_name.mkdir(parents=True, exist_ok=True)

                            # Also create encoder directory
                            encoder_dir = self.output_path / f"{self.llm_savename}_step_{step}_encoder"
                            encoder_dir.mkdir(parents=True, exist_ok=True)

                            self.logger.info("Saving model...")
                            # Unwrap and save model
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(
                                model_ckpt_name,
                                is_main_process=self.accelerator.is_main_process,
                                save_function=self.accelerator.save,
                                safe_serialization=False,
                            )

                            self.logger.info("Saving tokenizer...")
                            self.tokenizer.save_pretrained(model_ckpt_name)

                            self.logger.info("Saving encoder...")
                            # Save encoder and config from main process
                            encoder_ckpt_name = encoder_dir / "encoder.pt"
                            torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)

                            self.logger.info("Saving config...")
                            # Explicitly save config as JSON
                            config_path = model_ckpt_name / "kb_config_explicit.json"
                            with open(config_path, 'w') as f:
                                f.write(kb_config.to_json_string())

                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {e}")
                        self.logger.error(f"Error details: {str(e)}")
                        raise e


def main():
    os.environ["NCCL_TIMEOUT"] = "1200000"
    logger = logging.getLogger("training")

    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    print(vars(args))
    dataset_name = args.train_dataset
    seed = args.seed
    N = args.N
    B = args.B

    total_steps = args.total_steps
    encoder_spec = args.encoder_spec
    key_embd_src = args.key_embd_src
    use_data_aug = args.use_data_aug
    use_lr_decay = args.use_lr_decay
    use_cached_embd = args.use_cached_embd
    dataset_dir = args.dataset_dir
    model_dir_to_resume = args.model_dir_to_resume
    model_save_dir = args.model_save_dir
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size
    max_seq_len = args.max_seq_len

    if kb_size is not None and dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")

    kb_size = kb_size if kb_size is not None else dynamic_kb_size

    gradient_accm_step = args.gradient_accm_step

    length_invariance = args.length_invariance
    outlier_num = args.outlier_num
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    llm_type = args.llm_type
    hf_model_spec = args.hf_model_spec
    hf_token = args.hf_token
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")

    torch.manual_seed(seed)
    np.random.seed(seed)

    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if Accelerator().is_main_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project="kb-llm",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                'sep_query_head': sep_query_head,
                'kb_size': kb_size,
                'length_invariance': length_invariance,
                'dataset': dataset_name,
                'outlier_num': outlier_num,
                'multi_entities': multi_entities,
                'use_extended_qa': use_extended_qa,
                'kb_token_layer_frequency': kb_token_layer_frequency,
                'gradient_accm_step': gradient_accm_step,
                "encoder_spec": encoder_spec,
                "max_seq_len": max_seq_len,
            },
        )

    # Try to free up memory
    torch.cuda.empty_cache()

    if args.log_to_file:
        formatter = logging.Formatter(LOGFORMAT)
        f_handler = logging.FileHandler(model_save_dir / "log.txt")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.info(f"Running on {device}")

    logger.info("Started training")
    logger.info(f"Saving to  {model_save_dir}")
    if sep_query_head:
        os.environ["SEP_QUERY_HEAD"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    if length_invariance:
        os.environ["LENGTH_INVARIANCE"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    os.environ["SCALE_FACTOR"] = ""

    if use_cached_embd:
        # We load the pre-computed version stored on the disk rather
        # than computing them on the fly to make things faster
        logger.info(f"Using pre-computed {encoder_spec} embedding")
        key_embds, value_embds = _load_cached_embeddings(encoder_spec, dataset_dir, dataset_name, key_embd_src)

    prefix_string = get_prefix_str(args)
    logger.info(f"Experiment prefix {get_prefix_str(args)}")

    if use_extended_qa:
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json")))
    else:
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}.json")))

    training_set = dataset[:N]

    # Set up the LLM
    llm_model_spec = model_dir_to_resume if model_dir_to_resume else hf_model_spec

    resumed_step = 0 if not model_dir_to_resume else int(model_dir_to_resume.split("_")[-1])

    if llm_model_spec is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    if hf_token is None and args.llm_type == "llama3":
        raise ValueError("Please supply HuggingFace token(hf_token) when loading model Llama weights from HuggingFace")

    # Tokenizer comes from the base model
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_spec,
        trust_remote_code=True,
        token=hf_token if hf_token is args.llm_type == "llama3" else None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            #token=hf_token,
        )
    elif args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif args.llm_type == "bitnet":
        model = KBLaMBitNetForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16, # BitNet uses bfloat16
            trust_remote_code=True,
        )
    else:
        ValueError(f"LLM type {args.llm_type} not recognised")

    logger.info(model.config)  # type: ignore

    model.eval()  # type: ignore
    # freeze model
    for _, param in model.named_parameters():  # type: ignore
        param.requires_grad = False

    # Set up the encoder
    encoder = KBEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=device,
    )

    if model_dir_to_resume:
        encoder.load_state_dict(torch.load(os.path.join(model_dir_to_resume, "encoder.pt")))
        kb_config = KBLaMConfig.from_pretrained(os.path.join(model_dir_to_resume, "kb_config.json"))
    else:
        kb_config = KBLaMConfig(
            sep_query_head=sep_query_head,
            kb_layer_frequency=kb_token_layer_frequency,
            kb_length_scaling=length_invariance,
            kb_max_train_triples=N,
        )

    encoder.train()

    kbretriever = KBRetriever(
        encoder,
        training_set,
        key_embds=key_embds,  # type: ignore
        value_embds=value_embds,  # type: ignore
    )

    logger.info("Model ready")

    # Get the training started
    llm_ckpt_name = f"{prefix_string}KeyFrom{key_embd_src}_{encoder_spec}_{dataset_name}_{llm_type}"

    trainer = Trainer(
        model,  # type: ignore
        kbretriever,
        tokenizer,
        kb_token_layer_frequency,
        total_steps,
        args.lr,
        device,
        use_lr_decay,
        kb_size,  # type: ignore
        llm_ckpt_name,
        model_save_dir,
        sep_query_head=sep_query_head,
        max_seq_len=max_seq_len,
    )

    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    trainer.train(
        training_set,
        B,
        gradient_accm_step,
        outlier_num,
        use_data_aug=use_data_aug,
        multi_entities=multi_entities,
        use_extended_qa=use_extended_qa,
        save_period=100,
        resumed_step=resumed_step,
        kb_config=kb_config,
    )


if __name__ == "__main__":
    main()
