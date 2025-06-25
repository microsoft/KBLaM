"""Script for evaluating KB models"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import nltk
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa
from kblam.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    _format_Q_llama,
    _format_Q_phi3,
    _format_Q_bitnet,
    model_prune_format_mapping,
    answer_question,
    softmax,
)
from kblam.utils.train_utils import get_kb_embd

nltk.download("wordnet")
logging.set_verbosity_warning()

rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")

class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[np.ndarray] = None,
    ):
        self.encoder = encoder
        self.dataset = dataset
        if precomputed_embed_keys_path is not None:
            self.key_embds = np.load(precomputed_embed_keys_path).astype("float32")
        else:
            self.key_embds = None
        if precomputed_embed_values_path is not None:
            self.value_embds = np.load(precomputed_embed_values_path).astype("float32")
        else:
            self.value_embds = None

        if precomputed_embed_keys_path is not None:
            assert len(dataset) == len(self.key_embds)

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices):
        if self._use_cached_embd():
            return get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            return get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)


def perform_eval(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    encoder_model_spec: str,
    kb_config: KBLaMConfig,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    topk_size: int = -1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
):
    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    key_str = [row["key_string"] for row in test_kb]
    value_str = [row["description"] for row in test_kb]
    prompt_strs = ""
    for k, v in zip(key_str, value_str):
        prompt_strs += f"{k} is {v}; "

    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    full_outputs = []
    # answer_question
    subset_size = min(
        400, len(test_kb)
    )  # Regardless of KB size, always test 250 questions, otherwise it will be too slow
    subset_size = min(
        400, len(test_kb)
    )  # Regardless of KB size, always test 250 questions, otherwise it will be too slow
    # subset_size = 50
    for row in tqdm(test_kb[:subset_size]):
        if multi_entites == -1:
            Q = row["Q"]
            answer = row["A"]
        else:
            kb_subset_idx = np.random.randint(0, len(test_kb), multi_entites)
            Q, A = generate_multi_entity_qa(
                [test_kb[i]["name"] for i in kb_subset_idx],
                [test_kb[i]["description_type"] for i in kb_subset_idx],
                [test_kb[i]["description"] for i in kb_subset_idx],
            )
            answer = A

        if eval_mode == "kb":
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                topk_size=topk_size,
                kb_config=kb_config,
            ).split(Q)[1]
        elif eval_mode == "icl":
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            model_output = answer_question(
                tokenizer,
                model,
                ins_prompt + prompt_strs + Q,
                kb=None,
                kb_config=kb_config,
            ).split(Q)[1]
        elif eval_mode == "zeroshot":
            if multi_entites != -1:
                ins_prompt = zero_shot_prompt_multi_entities
            else:
                ins_prompt = zero_shot_prompt
            model_output = answer_question(
                tokenizer, model, ins_prompt + Q, kb=None, kb_config=kb_config
            ).split(Q)[1]
        # print(model_output)
        if remove_sorry:
            if "sorry" in model_output:
                continue
        full_outputs.append((model_output, answer))
        if multi_entites == -1:
            pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
            match = re.search(pattern, model_output)
            answers.append(row["description"])
            if match:
                model_output = match.group(1)
        else:
            pattern = r"(?:is|are) (.*?)(?:\.|;)"
            matches = re.findall(pattern, model_output)
            model_output = "; ".join(matches)
            answers.append(";".join(re.findall(r"(?:is|are) (.*?);", answer)))
        model_outputs.append(model_output)

    print(f"KB size: {kb_size}, mode: {eval_mode}")
    rouge = evaluate.load("rouge")

    for pred, gt in zip(model_outputs, answers):
        print(f"PREDICTION: {pred}")
        print(f"GT: {gt}")
    rouge_scores = rouge.compute(predictions=model_outputs, references=answers)
    print(rouge_scores)

    results_dict = {k: float(v) for k, v in rouge_scores.items()}

    bertscore = bert_score.compute(
        predictions=model_outputs,
        references=answers,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
    )
    # bert_scores = []
    # bert_scores = {}
    for k, v in bertscore.items():
        if isinstance(v, list):
            # bert_scores.append(np.mean(v))
            results_dict[f"bert_score_{k}"] = float(np.mean(v))
            print(k, np.mean(v))
    results = ""
    for a, A in full_outputs:
        results += f"Model output: {a}\nTrue answer: {A}\n-------\n"
    if eval_mode == "kb":
        eval_mode = encoder_model_spec + eval_mode

    return results, results_dict


def perform_eval_refusal(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: Optional[KBLaMConfig] = None,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    outlier_ratio: float = 0.2,
    topk_size: int = -1,
    question_size: int = 100,
):
    instruction_prompts = (
        'Please answer questions based on the given text with format: "The {property} of {name} is {description}",'
        ' if relevant information cannot be found in the text, please respond "I am sorry I cannot find relevant information in the KB".'
    )
    zero_shot_prompt = """
    Please answer the question in a very compact manner with format: The {property} of {name} is {description}
    """

    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    key_str = [row["key_string"] for row in test_kb]
    value_str = [row["description"] for row in test_kb]
    prompt_strs = ""
    for k, v in zip(key_str, value_str):
        prompt_strs += f"{k} is {v}; "

    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    # answer_question
    outlier_idx = np.arange(len(kb_retriever.dataset))
    outlier_idx = outlier_idx[~np.isin(outlier_idx, kb_idx)]
    np.random.shuffle(outlier_idx)
    question_size = min(kb_size, question_size)
    outlier_idx = outlier_idx[: int(question_size * outlier_ratio)]
    test_kb = test_kb[: int(question_size * (1 - outlier_ratio))] + [
        kb_retriever.dataset[idx] for idx in outlier_idx
    ]
    change_point = int(question_size * (1 - outlier_ratio))
    for i, row in tqdm(enumerate(test_kb)):
        Q = row["Q"]
        if eval_mode == "kb":
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                topk_size=topk_size,
                kb_config=kb_config,
            ).split(Q)[1]

        elif eval_mode == "icl":
            model_output = answer_question(
                tokenizer,
                model,
                instruction_prompts + prompt_strs + Q,
                kb=None,
                kb_config=kb_config,
            ).split(Q)[1]
        elif eval_mode == "zeroshot":
            model_output = answer_question(
                tokenizer,
                model,
                zero_shot_prompt + Q,
                kb=None,
                kb_config=kb_config,
            ).split(Q)[1]
        model_outputs.append(model_output)
        if i < change_point:
            answers.append(row["description"])
        else:
            answers.append("Cannot find relevant information in the KB")
    true_label = [0] * change_point + [1] * int(question_size * outlier_ratio)
    prediction = [int("sorry" in model_output) for model_output in model_outputs]
    print(f"KB size: {kb_size}, mode: {eval_mode}, outlier ratio: {outlier_ratio}")
    results = ""
    for a, A in zip(model_outputs, answers):
        results += f"Model output: {a}\nTrue answer: {A}\n-------\n"
    return results, np.array([prediction, true_label])


parser = argparse.ArgumentParser(description="Evaluation script")

# Add arguments that will be shared across all subcommands
parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument(
    "--dataset_dir", type=str, help="Directory containing the dataset"
)
parent_parser.add_argument(
    "--encoder_dir", type=str, help="Directory containing the encoder model"
)
parent_parser.add_argument(
    "--encoder_spec",
    type=str,
    default="OAI",
    help="Specification for the encoder model",
)
parent_parser.add_argument(
    "--fancy_instruction",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use fancy instructions",
)
parent_parser.add_argument(
    "--kb_layer_frequency",
    type=int,
    default=3,
    help="Frequency of knowledge base layers",
)
parent_parser.add_argument(
    "--kb_scale_factor",
    type=int,
    default=None,
    help="Scaling factor for knowledge base",
)
parent_parser.add_argument(
    "--kb_size", type=int, default=200, help="Size of the knowledge base"
)
parent_parser.add_argument(
    "--llm_base_dir",
    type=str,
    help="llm to load, can be HF location or local directory",
)
parent_parser.add_argument(
    "--llm_type",
    type=str,
    default="phi3",
    choices=["llama3", "phi3", "bitnet"],
    help="Type of language model to use",
)
parent_parser.add_argument(
    "--model_dir", type=str, help="Directory containing the model"
)
parent_parser.add_argument("--save_dir", type=str, help="Directory to save outputs")
parent_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
parent_parser.add_argument(
    "--test_dataset", type=str, help="Source of test KB (assumes KV pair format)"
)
parent_parser.add_argument(
    "--precomputed_embed_keys_path", type=str, help="Path to precomputed key embeddings"
)
parent_parser.add_argument(
    "--precomputed_embed_values_path",
    type=str,
    help="Path to precomputed value embeddings",
)
parent_parser.add_argument(
    "--query_head_path", type=str, default="", help="Path to load KB head from"
)
parent_parser.add_argument(
    "--topk_size",
    type=int,
    default=-1,
    help="The number of top-k entities to retrieve from the knowledge base.",
)

# Create subparsers
subparsers = parser.add_subparsers(dest="command", required=True)

# Create the parser for the generation command
gen_parser = subparsers.add_parser(
    "generation", parents=[parent_parser], help="Evaluate generation"
)
gen_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
gen_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="generation_results",
    help="Name of the experiment configuration",
)
gen_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="Frequency of knowledge base token layers",
)
gen_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="Number of entities to process (-1 for unlimited)",
)
gen_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use checkpoints trained without outliers",
)
gen_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)
gen_parser.add_argument(
    "--topk_size", type=int, default=-1, help="Size of top-k selection (-1 for all)"
)


# Create the parser for the accuracy command
acc_parser = subparsers.add_parser(
    "accuracy", parents=[parent_parser], help="Evaluate accuracy"
)

acc_parser.add_argument(
    "--attn_save_dir", type=str, default="", help="Directory to save attention masks"
)
acc_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="accuracy_results",
    help="Name of the experiment configuration",
)
acc_parser.add_argument(
    "--fancy_question",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable fancy question format",
)
acc_parser.add_argument(
    "--log_save_dir", type=str, help="Directory to save accuracy results"
)
acc_parser.add_argument(
    "--test_batch_size", type=int, default=50, help="Batch size for testing"
)
acc_parser.add_argument(
    "--use_shift_match",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable shift matching",
)

# Create the parser for the accuracy eval
acc_results_parser = subparsers.add_parser(
    "acc_results", parents=[acc_parser], help="run accuracy eval", add_help=False
)


# Create the parser for the refusal command
ref_parser = subparsers.add_parser(
    "refusal", parents=[parent_parser], help="Evaluate refusal"
)
ref_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
ref_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="refusal_results",
    help="Name of the experiment configuration",
)
ref_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="Frequency of knowledge base token layers",
)
ref_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="Number of entities to process (-1 for unlimited)",
)
ref_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use checkpoints trained without outliers",
)
ref_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)
ref_parser.add_argument(
    "--topk_size", type=int, default=-1, help="Size of top-k selection (-1 for all)"
)

# Create the parser for the standard command
basic_parser = subparsers.add_parser(
    "standard", parents=[parent_parser], help="Evaluate basic performance"
)
basic_parser.add_argument(
    "--attn_summary_save_dir",
    type=str,
    default="",
    help="Directory to save attention masks",
)
basic_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
basic_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="basic_results",
    help="Name of the experiment configuration",
)
basic_parser.add_argument(
    "--exp_config_str", type=str, help="Experiment configuration string"
)
basic_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="Frequency of knowledge base token layers",
)
basic_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use checkpoints trained without outliers",
)
basic_parser.add_argument(
    "--sample_size", default=5, type=int, help="Number of samples to process"
)
basic_parser.add_argument(
    "--subset_size", default=100, type=int, help="Size of the data subset to use"
)
basic_parser.add_argument(
    "--topk_size", type=int, default=-1, help="Size of top-k selection (-1 for all)"
)


def eval_generate():
    """Evaluate generation using KB"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    query_head_path = args.query_head_path
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    gen_results, score_results = perform_eval(
        model,
        tokenizer,
        kb_retriever,
        encoder_model_spec,
        kb_config,
        eval_mode,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        multi_entites=args.multi_entites,
    )
    mem_cost = torch.cuda.max_memory_reserved("cuda")
    score_results["mem_cost"] = mem_cost

    (Path(args.save_dir) / exp_config).mkdir(exist_ok=True, parents=True)
    write_to_json(score_results, Path(args.save_dir) / f"{exp_config}.json")
    print(score_results)
    text_file = open(os.path.join(args.save_dir, exp_config + ".txt"), "w")
    text_file.write(gen_results)


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
    tokenizer = AutoTokenizer.from_pretrained(
        llm_base_dir, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = "^"

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


def eval_accuracy(
    tokenizer,
    kb_retriever,
    model,
    dataset,
    exp_config,
    fancy_question,
    kb_config,
    kb_size,
    llm_type,
    test_batch_size,
    save_dir,
    attn_save_dir,
):
    """Evaluate accuracy using KB"""

    if kb_size == len(dataset):
        dataset_subset_idx = range(len(dataset))
    elif kb_size > len(dataset):
        raise IndexError(
            f"The KB size {kb_size} is greater than the dataset size {len(dataset)}"
        )
    else:
        dataset_subset_idx = np.random.choice(len(dataset), kb_size, replace=False)

    dataset_subset = [dataset[i] for i in dataset_subset_idx]

    kb_embedding_real = kb_retriever.get_key_embeddings(dataset_subset_idx)

    format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3, "bitnet": _format_Q_bitnet}

    if not fancy_question:
        input_strs_gen = (dataset_subset[i]["Q"] for i in range(test_batch_size))
    else:
        input_strs_gen = (augment_row(dataset_subset[i]) for i in range(test_batch_size))
    input_strs = [format_func_map[llm_type](ex) for ex in input_strs_gen]

    tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
        "cuda"
    )
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb_embedding_real,
            max_new_tokens=60,
            tokenizer=tokenizer,
            output_attentions=True,
            save_attention_weights=True,
            kb_config=kb_config,
            attention_save_loc=attn_save_dir,
            attention_file_base_name=exp_config,
        )
        outputs = tokenizer.batch_decode(outputs.squeeze(), skip_special_tokens=False)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    with open(save_path / f"{exp_config}_acc.txt", "w+") as text_file:
        for output in outputs:
            output_string = output.strip("^")
            text_file.write(f"{str(output_string)}\n")

    accs = []
    with torch.autograd.no_grad():
        for idx in range(0, 32, kb_config.kb_layer_frequency):
            weight = np.load(os.path.join(attn_save_dir, f"{exp_config}_{idx}.npy"))
            weight = weight[..., :kb_size]
            label = np.arange(test_batch_size)
            weight = weight.reshape(test_batch_size, -1, kb_size)
            acc = (weight.sum(1).argmax(1) == label).mean()
            top_5_predictions = torch.topk(torch.from_numpy(weight.sum(1)), 5, dim=1)[1]
            top_5_acc = (top_5_predictions.numpy() == label[:, None]).any(1).mean()
            if idx == 15:
                print(f"ACC & TOP 5 ACC: {idx} {(acc, top_5_acc)}")
                print(f"min: {np.min(weight)}  max: {np.max(weight)}")
            accs.append(
                {
                    "idx": idx,
                    "acc": float(acc),
                    "top5acc": float(top_5_acc),
                }
            )

    np.save(
        save_path / f"{exp_config}_acc.npy",
        np.array([(a["acc"], a["top5acc"]) for a in accs]),
    )

    return accs


def eval_accuracy_cli():
    """Evaluate accuracy using KB"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_path = args.encoder_dir
    encoder_spec = args.encoder_spec
    exp_config = args.exp_config_name
    fancy_question = args.fancy_question
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = llm_type = args.llm_type
    model_path = args.model_dir
    test_batch_size = args.test_batch_size
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    query_head_path = args.query_head_path
    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )
    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    eval_accuracy(
        tokenizer,
        kb_retriever,
        model,
        dataset,
        exp_config,
        fancy_question,
        kb_config,
        kb_size,
        llm_type,
        test_batch_size,
        args.log_save_dir,
        args.attn_save_dir,
    )


def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    """
    Write a dictionary to a JSON file with error handling and formatting options.

    Args:
        data: Dictionary to write to JSON file
        filepath: Path where the JSON file should be saved
        indent: Number of spaces for indentation (default: 4)
        encoding: File encoding (default: 'utf-8')

    Raises:
        TypeError: If data is not a dictionary
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


def run_accuracy_evalution():
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_path = args.encoder_dir
    encoder_spec = args.encoder_spec
    exp_config = args.exp_config_name
    fancy_question = args.fancy_question
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    llm_base_dir = args.llm_base_dir
    llm_type = llm_type = args.llm_type
    model_path = args.model_dir
    test_dataset = args.test_dataset

    query_head_path = args.query_head_path
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))
    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    xs = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    accuracy_results = []
    for x in xs:
        print(f"kb_size {x}")

        accs = eval_accuracy(
            tokenizer,
            kb_retriever,
            model,
            dataset,
            exp_config,
            fancy_question,
            kb_config,
            x,
            llm_type,
            min(x, 200),
            args.log_save_dir,
            args.attn_save_dir,
        )
        shutil.rmtree(args.attn_save_dir)
        os.mkdir(args.attn_save_dir)
        accuracy_results.append({"kb_size": x, "accuracy_results": accs})
    write_to_json(
        accuracy_results, os.path.join(args.log_save_dir, "accuracy_results.json")
    )


def eval_refusal():
    """Evaluate refusal to answer questions for which the answer does not exist in the KB"""
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path
    query_head_path = args.query_head_path

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    gen_results, refusal_results = perform_eval_refusal(
        model,
        tokenizer,
        kb_retriever,
        eval_mode=eval_mode,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        kb_config=kb_config,
    )

    np.save(os.path.join(args.save_dir, "OutLierTest" + exp_config), refusal_results)
    text_file = open(
        os.path.join(args.save_dir, "OutLierTest" + exp_config + ".txt"), "w"
    )
    text_file.write(gen_results)


def eval():
    """Evaluate the KB model"""
    args = parser.parse_args()
    attn_summary_save_dir = args.attn_summary_save_dir
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    exp_config_str = args.exp_config_str
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    output_dir = args.save_dir
    sample_size = args.sample_size
    seed = args.seed
    subset_size = args.subset_size
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path
    query_head_path = args.query_head_path
    sep_query_head = True
    actual_kb_token_layer_frequency = 3

    if kb_size == -1:
        kb_size = None

    # validation_part_start_idx = 120000 if 'gpt' in test_dataset else 0
    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    if sep_query_head:
        print("Having seperate query head for KB!")

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["ATTN_SAVE_DIR"] = output_dir
    os.environ["EVAL_MODE"] = "1"

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    for param in model.parameters():
        param.requires_grad = False

    # Set up the encoder
    encoder = KBEncoder(
        encoder_name=encoder_model_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // actual_kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=torch.device("cuda"),
    )
    encoder.load_state_dict(torch.load(encoder_path))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )
    no_kb_predictions = []
    predictions = []
    answer = []

    for _ in range(sample_size):
        print("******")
        dataset_subset_idx = np.random.choice(len(dataset), subset_size, replace=False)
        dataset_subset = [dataset[i] for i in dataset_subset_idx]
        encoder.eval()
        with torch.autograd.no_grad():
            kb_embedding_real = kb_retriever.get_key_embeddings(dataset_subset_idx)
            kb_embedding_key, kb_embedding_val = kb_embedding_real
            kb_embedding_real = (kb_embedding_key, kb_embedding_val)

        format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

        input_strs = [
            format_func_map[llm_type](dataset_subset[i]["Q"])
            for i in range(subset_size)
        ]

        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
            "cuda"
        )
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        kb_embedding_real = (kb_embedding_real[0], kb_embedding_real[1])

        config_str = f"{exp_config_str}__kb_{subset_size}__seed_{seed}"
        with torch.autograd.no_grad():
            outputs_no_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=None,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=False,
                kb_config=kb_config,
            )

            outputs_true_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=kb_embedding_real,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=True,
                save_attention_weights=True,
                attention_save_loc=output_dir,
                attention_file_base_name=config_str,
                kb_config=kb_config,
            )
        print("decoding")
        outputs_no_kb = tokenizer.batch_decode(outputs_no_kb, skip_special_tokens=False)

        outputs_true_kb = tokenizer.batch_decode(
            outputs_true_kb, skip_special_tokens=False
        )
        print("KB:")
        for i in range(subset_size):
            print(
                "{} : {}".format(
                    dataset_subset[i]["name"], dataset_subset[i]["description"]
                )
            )

        for m in model_prune_format_mapping:
            if isinstance(model, m):
                prune_str = model_prune_format_mapping[m]

        print("------------------")
        for i in range(subset_size):
            print("True KB", prune_str(outputs_true_kb[i]))
            print("True answer: ", dataset_subset[i]["A"])
            no_kb_predictions.append(
                prune_str(outputs_no_kb[i]).split(dataset_subset[i]["Q"])[1]
            )
            predictions.append(
                prune_str(outputs_true_kb[i]).split(dataset_subset[i]["Q"])[1]
            )
            answer.append(dataset_subset[i]["A"])
            print("--------------------")
        print("******")

    rogue_score = rouge.compute(predictions=predictions, references=answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_rouge.npy"), **rogue_score
    )

    rogue_score_no_kb = rouge.compute(predictions=no_kb_predictions, references=answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_rouge_no_kb.npy"),
        **rogue_score_no_kb,
    )

    # Start inspecting attention masks
    ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 32)]

    save_dir = output_dir
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    accs, confidences = [], []
    for left, right in ranges:
        weights = []
        kb_size = subset_size
        for idx in range(32)[left:right]:
            if idx % 3 == 0:
                weight = np.load(os.path.join(save_dir, f"{config_str}_{idx}.npy"))
                weights.append(weight[..., :kb_size].reshape(kb_size, -1, kb_size))
        print(len(weights))
        weights = np.stack(weights)
        weights = weights.transpose(1, 0, 2, 3).reshape(kb_size, -1, kb_size)
        acc = (weights.sum(1).argmax(1) == np.arange(kb_size)).mean()
        top_5_predictions = torch.topk(torch.from_numpy(weights.sum(1)), 5, dim=1)[1]
        top_5_acc = (
            (top_5_predictions == torch.arange(kb_size)[:, None]).any(1).float().mean()
        )
        accs.append((acc, top_5_acc))
        confidence = softmax(weights.mean(1), -1).max()
        confidences.append(confidence)
    np.save(
        os.path.join(attn_summary_save_dir, f"{config_str}_acc.npy"), np.array(accs)
    )
    np.save(
        os.path.join(attn_summary_save_dir, f"{config_str}_conf.npy"),
        np.array(confidences),
    )


def main():
    args = parser.parse_args()
    print(args)
    if args.command == "generation":
        eval_generate()
    elif args.command == "accuracy":
        eval_accuracy_cli()
    elif args.command == "acc_results":
        run_accuracy_evalution()
    elif args.command == "refusal":
        eval_refusal()
    elif args.command == "standard":
        eval()
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()
