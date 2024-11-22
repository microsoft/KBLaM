""" Script for evaluating KB models
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import nltk
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.utils.data_utils import aug_row, generate_multi_entity_qa
from kblam.utils.train_utils import get_kb_embd

nltk.download('wordnet')
logging.set_verbosity_warning()

rouge = evaluate.load('rouge')
bert_score = evaluate.load('bertscore')

instruction_prompts = '''
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
'''

instruction_prompts_multi_entities = '''
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
'''

zero_shot_prompt = '''
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
'''

zero_shot_prompt_multi_entities = '''
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
'''


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )


def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


model_question_format_mapping = {KblamLlamaForCausalLM: _format_Q_llama, KBLaMPhi3ForCausalLM: _format_Q_phi3}


def _prune_for_llama(S):
    S = S.replace('<|eot_id|>', '')
    S = S.replace('<|start_header_id|>assistant<|end_header_id|>', '')
    S = S.replace('<|start_header_id|>user<|end_header_id|>', '')
    S = S.replace('<|end_of_text|>', '')
    return S


def _prune_for_phi3(S):
    S = S.replace('<|end|>', '')
    S = S.replace('<|assistant|>', '')
    S = S.replace('<|user|>', '')
    return S


model_prune_format_mapping = {KblamLlamaForCausalLM: _prune_for_llama, KBLaMPhi3ForCausalLM: _prune_for_phi3}


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    Q: str,
    kb=None,
    kb_layer_frequency: int = 3,
    topk_size: int | None = None,
    kb_scale_factor: int = -1,
):
    for m in model_question_format_mapping:
        if isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
    tokenizer_output = tokenizer(input_str, return_tensors='pt', padding=True).to('cuda')
    input_ids, attention_masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']

    with torch.autograd.no_grad():
        if topk_size != -1:
            dynamic_sparsify = True
        else:
            dynamic_sparsify = False
        if kb_scale_factor == -1:
            kb_scale_factor = None

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            tokenizer=tokenizer,
            output_attentions=True,
        ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
    return pruned_output


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
    eval_mode: str = 'kb',
    kb_layer_frequency: int = 3,
    kb_size: int = 250,
    seed: int = 1,
    topk_size: int = -1,
    kb_scale_factor: int = -1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
    use_mlflow: bool = False,
    config_name: str = 'generation_results',
):
    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    key_str = [row['key_string'] for row in test_kb]
    value_str = [row['description'] for row in test_kb]
    prompt_strs = ''
    for k, v in zip(key_str, value_str):
        prompt_strs += f'{k} is {v}; '

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
            Q = row['Q']
            answer = row['A']
        else:
            kb_subset_idx = np.random.randint(0, len(test_kb), multi_entites)
            Q, A = generate_multi_entity_qa(
                [test_kb[i]['name'] for i in kb_subset_idx],
                [test_kb[i]['description_type'] for i in kb_subset_idx],
                [test_kb[i]['description'] for i in kb_subset_idx],
            )
            answer = A

        if eval_mode == 'kb':
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                kb_layer_frequency=kb_layer_frequency,
                topk_size=topk_size,
                kb_scale_factor=kb_scale_factor,
            ).split(Q)[1]
        elif eval_mode == 'icl':
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            model_output = answer_question(
                tokenizer, model, ins_prompt + prompt_strs + Q, kb=None, kb_layer_frequency=kb_layer_frequency
            ).split(Q)[1]
        elif eval_mode == 'zeroshot':
            if multi_entites != -1:
                ins_prompt = zero_shot_prompt_multi_entities
            else:
                ins_prompt = zero_shot_prompt
            model_output = answer_question(
                tokenizer, model, ins_prompt + Q, kb=None, kb_layer_frequency=kb_layer_frequency
            ).split(Q)[1]
        # print(model_output)
        if remove_sorry:
            if 'sorry' in model_output:
                continue
        full_outputs.append((model_output, answer))
        if multi_entites == -1:
            pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
            match = re.search(pattern, model_output)
            answers.append(row['description'])
            if match:
                model_output = match.group(1)
        else:
            pattern = r'(?:is|are) (.*?)(?:\.|;)'
            matches = re.findall(pattern, model_output)
            model_output = '; '.join(matches)
            answers.append(';'.join(re.findall(r'(?:is|are) (.*?);', answer)))
        model_outputs.append(model_output)

    print(f'KB size: {kb_size}, mode: {eval_mode}')
    rouge = evaluate.load('rouge')

    for pred, gt in zip(model_outputs, answers):
        print(f"PREDICTION: {pred}")
        print(f"GT: {gt}")

    rouge_score = rouge.compute(predictions=model_outputs, references=answers)
    print("ROUGE:", rouge_score)

    if use_mlflow:
        import mlflow

        for rouge_type, scores in rouge_score.items():
            mlflow.log_metric(config_name + rouge_type, np.mean(scores))
    bertscore = bert_score.compute(
        predictions=model_outputs, references=answers, lang="en", model_type='microsoft/deberta-xlarge-mnli'
    )
    bert_scores = []
    for k, v in bertscore.items():
        if isinstance(v, list):
            bert_scores.append(np.mean(v))
            print(k, np.mean(v))
            if use_mlflow:
                mlflow.log_metric(config_name + k, np.mean(v))

    results = ''
    for a, A in full_outputs:
        results += f'Model output: {a}\nTrue answer: {A}\n-------\n'
    if eval_mode == 'kb':
        eval_mode = encoder_model_spec + eval_mode
    return results, bert_scores + list(rouge_score.values())


def perform_eval_refusal(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    eval_mode: str = 'kb',
    kb_layer_frequency: int = 3,
    kb_size: int = 250,
    seed: int = 1,
    outlier_ratio: float = 0.2,
    topk_size: int = -1,
    question_size: int = 100,
    kb_scale_factor: int = 100,
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
    test_kb = test_kb[: int(question_size * (1 - outlier_ratio))] + [kb_retriever.dataset[idx] for idx in outlier_idx]
    change_point = int(question_size * (1 - outlier_ratio))
    for i, row in tqdm(enumerate(test_kb)):
        Q = row["Q"]
        if eval_mode == "kb":
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                kb_layer_frequency=kb_layer_frequency,
                topk_size=topk_size,
                kb_scale_factor=kb_scale_factor,
            ).split(Q)[1]

        elif eval_mode == "icl":

            model_output = answer_question(
                tokenizer,
                model,
                instruction_prompts + prompt_strs + Q,
                kb=None,
                kb_layer_frequency=kb_layer_frequency,
            ).split(Q)[1]
        elif eval_mode == "zeroshot":
            model_output = answer_question(
                tokenizer,
                model,
                zero_shot_prompt + Q,
                kb=None,
                kb_layer_frequency=kb_layer_frequency,
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

parent_parser.add_argument('--ckpt_idx', type=int, default=10000, help='Checkpoint to use')
parent_parser.add_argument('--dataset_dir', type=str, help='Directory containing the dataset')
parent_parser.add_argument('--encoder_dir', type=str, help='Directory containing the encoder model')
parent_parser.add_argument('--encoder_spec', type=str, default='OAI', help='Specification for the encoder model')
parent_parser.add_argument(
    '--fancy_instruction',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Whether to use fancy instructions',
)
parent_parser.add_argument('--kb_layer_frequency', type=int, default=3, help='Frequency of knowledge base layers')
parent_parser.add_argument('--kb_scale_factor', type=int, default=None, help='Scaling factor for knowledge base')
parent_parser.add_argument('--kb_size', type=int, default=200, help='Size of the knowledge base')
parent_parser.add_argument('--llm_base_dir', type=str, help='llm to load, can be HF location or local directory')
parent_parser.add_argument(
    '--llm_type', type=str, default="phi3", choices=["llama3", "phi3"], help='Type of language model to use'
)
parent_parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parent_parser.add_argument('--model_dir', type=str, help='Directory containing the model')
parent_parser.add_argument('--tokenizer_path', type=str, help='Directory containing the tokenizer', default="")
parent_parser.add_argument('--save_dir', type=str, help='Directory to save outputs')
parent_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
parent_parser.add_argument('--test_dataset', type=str, help='Source of test KB (assumes KV pair format)')
parent_parser.add_argument('--query_head_path', type=str, default="")
parent_parser.add_argument('--sep_query_head', dest='sep_query_head', action='store_const', const=True, default=None)
parent_parser.add_argument('--no-sep_query_head', dest='sep_query_head', action='store_const', const=False)
parent_parser.add_argument(
    '--mlflow',
    default=False,
    action='store_true',
)

# Create subparsers
subparsers = parser.add_subparsers(dest='command', required=True)

# Create the parser for the generation command
gen_parser = subparsers.add_parser('generation', parents=[parent_parser], help='Evaluate generation')
gen_parser.add_argument(
    '--eval_mode',
    type=str,
    choices=['kb', 'icl', 'zeroshot'],
    default='kb',
    help='Evaluation mode: knowledge base, in-context learning, or zero-shot',
)
gen_parser.add_argument(
    '--exp_config_name', type=str, default="generation_results", help='Name of the experiment configuration'
)
gen_parser.add_argument(
    '--kb_token_layer_frequency', type=int, default=None, help='Frequency of knowledge base token layers'
)
gen_parser.add_argument(
    '--multi_entites', type=int, default=-1, help='Number of entities to process (-1 for unlimited)'
)
gen_parser.add_argument(
    '--no_outlier',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Use checkpoints trained without outliers',
)
gen_parser.add_argument(
    '--remove_sorry',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)
gen_parser.add_argument('--topk_size', type=int, default=-1, help='Size of top-k selection (-1 for all)')
gen_parser.add_argument(
    '--use_precomputed_embd', action='store_true', default=False, help='Use pre-computed embeddings'
)
gen_parser.add_argument('--test_batch_size', type=int, default=50, help='Batch size for testing')
gen_parser.add_argument('--log_save_dir', type=str, help='Directory to save accuracy results')

# Create the parser for the generation command
acc_parser = subparsers.add_parser('accuracy', parents=[parent_parser], help='Evaluate accuracy')
acc_parser.add_argument('--attn_save_dir', type=str, default="", help='Directory to save attention masks')
acc_parser.add_argument(
    '--exp_config_name', type=str, default="accuracy_results", help='Name of the experiment configuration'
)
acc_parser.add_argument(
    '--fancy_question', action=argparse.BooleanOptionalAction, default=False, help='Enable fancy question format'
)
acc_parser.add_argument('--log_save_dir', type=str, help='Directory to save accuracy results')
acc_parser.add_argument('--test_batch_size', type=int, default=50, help='Batch size for testing')
acc_parser.add_argument(
    '--use_shift_match', action=argparse.BooleanOptionalAction, default=False, help='Enable shift matching'
)

# Create the parser for the refusal command
ref_parser = subparsers.add_parser('refusal', parents=[parent_parser], help='Evaluate refusal')
ref_parser.add_argument(
    '--eval_mode',
    type=str,
    choices=['kb', 'icl', 'zeroshot'],
    default='kb',
    help='Evaluation mode: knowledge base, in-context learning, or zero-shot',
)
ref_parser.add_argument(
    '--exp_config_name', type=str, default="refusal_results", help='Name of the experiment configuration'
)
ref_parser.add_argument(
    '--kb_token_layer_frequency', type=int, default=None, help='Frequency of knowledge base token layers'
)
ref_parser.add_argument(
    '--multi_entites', type=int, default=-1, help='Number of entities to process (-1 for unlimited)'
)
ref_parser.add_argument(
    '--no_outlier',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Use checkpoints trained without outliers',
)
ref_parser.add_argument(
    '--remove_sorry',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)
ref_parser.add_argument('--topk_size', type=int, default=-1, help='Size of top-k selection (-1 for all)')
ref_parser.add_argument(
    '--use_precomputed_embd', action='store_true', default=False, help='Use pre-computed embeddings'
)

# Create the parser for the standard command
basic_parser = subparsers.add_parser('standard', parents=[parent_parser], help='Evaluate basic performance')
basic_parser.add_argument(
    '--eval_mode',
    type=str,
    choices=['kb', 'icl', 'zeroshot'],
    default='kb',
    help='Evaluation mode: knowledge base, in-context learning, or zero-shot',
)
basic_parser.add_argument(
    '--exp_config_name', type=str, default="basic_results", help='Name of the experiment configuration'
)
basic_parser.add_argument(
    '--kb_token_layer_frequency', type=int, default=None, help='Frequency of knowledge base token layers'
)
basic_parser.add_argument(
    '--no_outlier',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Use checkpoints trained without outliers',
)
basic_parser.add_argument('--output_dir', type=str, default="", help='Directory to save output files')
basic_parser.add_argument('--sample_size', default=5, type=int, help='Number of samples to process')
basic_parser.add_argument('--subset_size', default=100, type=int, help='Size of the data subset to use')
basic_parser.add_argument('--topk_size', type=int, default=-1, help='Size of top-k selection (-1 for all)')
basic_parser.add_argument(
    '--use_precomputed_embd', action='store_true', default=False, help='Use pre-computed embeddings'
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
    use_precomputed_embd = args.use_precomputed_embd
    query_head_path = args.query_head_path
    use_mlflow = args.mlflow
    save_dir = args.log_save_dir
    remove_sorry = args.remove_sorry
    tokenizer_path = args.tokenizer_path
    sep_query_head = args.sep_query_head

    print("ARGS: ", args)

    validation_part_start_idx = 120000 if 'gpt' in test_dataset else 0

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset + '.json')))[validation_part_start_idx:]

    key_embds = None
    value_embds = None
    if use_precomputed_embd:
        key_embds = np.load(os.path.join(dataset_dir, f'{test_dataset}_{encoder_model_spec}_embd_key.npy')).astype(
            'float32'
        )[validation_part_start_idx:]
        value_embds = np.load(os.path.join(dataset_dir, f'{test_dataset}_{encoder_model_spec}_embd_value.npy')).astype(
            'float32'
        )[validation_part_start_idx:]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = '^'

    if llm_type == "llama3":
        if query_head_path:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            print("PATHS:", os.listdir(os.path.dirname(query_head_path)))
            model.load_query_head(query_head_path)
        else:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
        kb_config = KBLaMConfig(
            kb_layer_frequency=kb_layer_frequency,
            kb_scale_factor=kb_scale_factor,
            **model.config.to_dict(),
        )
        if sep_query_head is not None:
            kb_config.sep_query_head = sep_query_head
        model.config = kb_config
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        kb_config = KBLaMConfig(
            kb_layer_frequency=kb_layer_frequency,
            kb_scale_factor=kb_scale_factor,
            **model.config.to_dict(),
        )
        if sep_query_head is not None:
            kb_config.sep_query_head = sep_query_head
        model.config = kb_config

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    encoder = KBEncoder(
        encoder_name=encoder_model_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // kb_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=torch.device("cuda"),
    )
    encoder.load_state_dict(torch.load(encoder_path))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        key_embds=key_embds,
        value_embds=value_embds,
    )

    gen_results, score_output = perform_eval(
        model,
        tokenizer,
        kb_retriever,
        encoder_model_spec,
        eval_mode,
        kb_layer_frequency,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        multi_entites=args.multi_entites,
        kb_scale_factor=kb_scale_factor,
        config_name=f"kb_size_{kb_size}",
        use_mlflow=use_mlflow,
        remove_sorry=remove_sorry,
    )
    mem_cost = torch.cuda.max_memory_reserved('cuda')
    score_output.append(mem_cost)

    (Path(save_dir) / exp_config).mkdir(exist_ok=True, parents=True)
    np.save(os.path.join(args.save_dir, exp_config), np.array(score_output))
    text_file = open(os.path.join(args.save_dir, exp_config + '.txt'), "w")
    text_file.write(gen_results)


def eval_accuracy_cli():
    args = parser.parse_args()

    print("ARGS: ", args)

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
    use_shift_match = args.use_shift_match
    query_head_path = args.query_head_path
    save_dir = args.log_save_dir
    attn_save_dir = args.attn_save_dir
    seed = args.seed
    use_mlflow = args.mlflow
    sep_query_head = args.sep_query_head
    tokenizer_path = args.tokenizer_path

    test_batch_size = min(test_batch_size, kb_size)

    torch.manual_seed(seed)
    np.random.seed(seed)

    return eval_accuracy(
        dataset_dir,
        encoder_path,
        encoder_spec,
        exp_config,
        fancy_question,
        kb_layer_frequency,
        kb_scale_factor,
        kb_size,
        llm_base_dir,
        llm_type,
        model_path,
        test_batch_size,
        test_dataset,
        use_shift_match,
        query_head_path,
        save_dir=save_dir,
        attn_save_dir=attn_save_dir,
        use_mlflow=use_mlflow,
        sep_query_head=sep_query_head,
        tokenizer_path=tokenizer_path,
    )


def eval_accuracy(
    dataset_dir: str,
    encoder_path,
    encoder_spec,
    exp_config,
    fancy_question,
    kb_layer_frequency,
    kb_scale_factor,
    kb_size,
    llm_base_dir,
    llm_type,
    model_path,
    test_batch_size,
    test_dataset,
    use_shift_match,
    query_head_path,
    save_dir,
    attn_save_dir,
    sep_query_head=None,
    model=None,
    dataset=None,
    key_embds=None,
    value_embds=None,
    tokenizer=None,
    encoder=None,
    use_mlflow=False,
    tokenizer_path=None,
):
    """Evaluate accuracy using KB"""

    if kb_scale_factor == -1:
        kb_scale_factor = None

    validation_part_start_idx = 120000 if "gpt" in test_dataset else 0
    encoder_model_spec = encoder_spec

    if dataset is None:
        dataset = json.load(open(os.path.join(dataset_dir, test_dataset) + ".json"))[validation_part_start_idx:]

    if key_embds is None:
        sm_string = "" if not use_shift_match else "_sm"

        key_embds = np.load(
            os.path.join(dataset_dir, f"{test_dataset}_{encoder_model_spec}_embd_key{sm_string}.npy")
        ).astype("float32")[validation_part_start_idx:]
        value_embds = np.load(
            os.path.join(dataset_dir, f"{test_dataset}_{encoder_model_spec}_embd_value{sm_string}.npy")
        ).astype("float32")[validation_part_start_idx:]

    if kb_layer_frequency == -1:
        kb_layer_frequency = 3

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_base_dir if tokenizer_path is None else tokenizer_path, trust_remote_code=True, padding_side="left"
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
                print("PATHS:", os.listdir(os.path.dirname(query_head_path)))
                model.load_query_head(query_head_path)
            else:
                model = KblamLlamaForCausalLM.from_pretrained(
                    model_path,
                    device_map="cuda",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
        else:
            model = KBLaMPhi3ForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )

        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.eval()

        kb_config = KBLaMConfig(
            kb_layer_frequency=kb_layer_frequency,
            kb_scale_factor=kb_scale_factor,
            **model.config.to_dict(),
        )
        if sep_query_head is not None:
            kb_config.sep_query_head = sep_query_head
        model.config = kb_config

        encoder = KBEncoder(
            encoder_name=encoder_spec.upper(),
            projector_type="linear",
            endpoint_url="",
            out_dim=model.config.hidden_size * (model.config.num_hidden_layers // kb_layer_frequency + 1),
            frozen_base_model=True,
            projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
            device=torch.device("cuda"),
        )

        encoder.load_state_dict(torch.load(encoder_path))

    if kb_size == len(dataset):
        dataset_subset_idx = range(len(dataset))
    elif kb_size > len(dataset):
        raise IndexError(f"The KB size {kb_size} is greater than the dataset size {len(dataset)}")
    else:
        dataset_subset_idx = np.random.choice(len(dataset), kb_size, replace=False)

    dataset_subset = [dataset[i] for i in dataset_subset_idx]
    encoder.eval()
    with torch.autograd.no_grad():
        # Could also pass this in but I can't be bothered for now
        kb_embedding_real = get_kb_embd(encoder, dataset_subset_idx, precomputed_embd=(key_embds, value_embds))
        kb_embedding_key, kb_embedding_val = kb_embedding_real
        kb_embedding_real = (kb_embedding_key, kb_embedding_val)

    format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}

    if not fancy_question:
        input_strs_gen = (dataset_subset[i]["Q"] for i in range(test_batch_size))
    else:
        input_strs_gen = (aug_row(dataset_subset[i]) for i in range(test_batch_size))
    input_strs = [format_func_map[llm_type](ex) for ex in input_strs_gen]

    tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )
    kb_embedding_real = (kb_embedding_real[0], kb_embedding_real[1])

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb_embedding_real,
            max_new_tokens=60,
            tokenizer=tokenizer,
            output_attentions=True,
            save_attention_weights=True,
            attention_save_loc=attn_save_dir,
            attention_file_base_name=exp_config,
        )
        outputs = tokenizer.batch_decode(outputs.squeeze(), skip_special_tokens=False)
        print(outputs[:10])

    accs = []
    with torch.autograd.no_grad():
        for idx in range(0, 32, kb_layer_frequency):
            weight = np.load(os.path.join(attn_save_dir, f"{exp_config}_{idx}.npy"))[..., :kb_size]
            label = np.arange(test_batch_size)
            weight = weight.reshape(test_batch_size, -1, kb_size)
            acc = (weight.sum(1).argmax(1) == label).mean()
            top_5_predictions = torch.topk(torch.from_numpy(weight.sum(1)), 5, dim=1)[1]
            top_5_acc = (top_5_predictions.numpy() == label[:, None]).any(1).mean()
            accs.append((acc, top_5_acc))
            if use_mlflow and idx < 17 and idx > 13:
                import mlflow

                mlflow.log_metric(f"kb_size_{kb_size}_acc_{idx}", acc)
                mlflow.log_metric(f"kb_size_{kb_size}_top_5_acc_{idx}", top_5_acc)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    print("ACC & TOP 5 ACC:", accs)

    np.save(save_path / f"{exp_config}_acc.npy", np.array(accs))
    with open(save_path / f"{exp_config}_acc.txt", "w+") as text_file:
        for output in outputs:
            output_string = output.strip("^")
            text_file.write(f"{str(output_string)}\n")


def eval_refusal():
    "Evaluate refusal to answer questions the KB is not relevant for"
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
    use_precomputed_embd = args.use_precomputed_embd

    validation_part_start_idx = 120000 if "gpt" in test_dataset else 0

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset + ".json")))[validation_part_start_idx:]

    if use_precomputed_embd:
        key_embds = np.load(os.path.join(dataset_dir, f"{test_dataset}_{encoder_model_spec}_embd_key.npy")).astype(
            "float32"
        )[validation_part_start_idx:]
        value_embds = np.load(os.path.join(dataset_dir, f"{test_dataset}_{encoder_model_spec}_embd_value.npy")).astype(
            "float32"
        )[validation_part_start_idx:]

    encoder_spec = encoder_model_spec
    tokenizer = AutoTokenizer.from_pretrained(llm_base_dir, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = "^"

    if llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    encoder = KBEncoder(
        encoder_name=encoder_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},  # Some arbitary numbers,
        get_oai_embd_online=False if args.use_precomputed_embd else True,
    )
    encoder.load_state_dict(torch.load(encoder_path))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        key_embds=key_embds,
        value_embds=value_embds,
    )

    gen_results, refusal_results = perform_eval_refusal(
        model,
        tokenizer,
        kb_retriever,
        eval_mode,
        kb_layer_frequency,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        kb_scale_factor=kb_scale_factor,
    )

    np.save(os.path.join(args.save_dir, "OutLierTest" + exp_config), refusal_results)

    text_file = open(os.path.join(args.save_dir, "OutLierTest" + exp_config + ".txt"), "w")
    text_file.write(gen_results)


def eval():
    """Evaluate the KB model"""
    args = parser.parse_args()
    attn_save_dir = args.attn_save_dir
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    exp_config_name = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    sample_size = args.sample_size
    seed = args.seed
    subset_size = args.subset_size
    test_dataset = args.test_dataset
    save_dir = args.log_save_dir
    use_mlflow = args.mlflow

    sep_query_head = True
    actual_kb_token_layer_frequency = 3

    if kb_size == -1:
        kb_size = None

    validation_part_start_idx = 120000 if 'gpt' in test_dataset else 0
    dataset = json.load(open(os.path.join(dataset_dir, test_dataset + '.json')))[validation_part_start_idx:]

    key_embds = np.load(os.path.join(dataset_dir, f'{test_dataset}_{encoder_model_spec}_embd_key.npy')).astype(
        'float32'
    )[validation_part_start_idx:]
    value_embds = np.load(os.path.join(dataset_dir, f'{test_dataset}_{encoder_model_spec}_embd_value.npy')).astype(
        'float32'
    )[validation_part_start_idx:]

    if sep_query_head:
        print("Having seperate query head for KB!")

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["EVAL_MODE"] = "1"

    llm_model_spec = llm_base_dir
    tokenizer = AutoTokenizer.from_pretrained(llm_model_spec, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token_id = 128001
    tokenizer.pad_token = '^'
    if model_path:  # TODO: make it load the default llm checkpoint
        if llm_type == "llama3":
            model = KblamLlamaForCausalLM.from_pretrained(
                llm_model_spec,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
        else:
            model = KBLaMPhi3ForCausalLM.from_pretrained(
                llm_model_spec,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )

    else:
        if llm_type == "llama3":
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
        else:
            model = KBLaMPhi3ForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )

    model.eval()

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
        key_embds=key_embds,
        value_embds=value_embds,
    )

    assert len(dataset) == len(key_embds)

    no_kb_predictions = []
    predictions = []
    answer = []

    kb_config = KBLaMConfig(
        sep_query_head=True,
        kb_layer_frequency=kb_layer_frequency,
        kb_scale_factor=kb_scale_factor,
    )

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

        input_strs = [format_func_map[llm_type](dataset_subset[i]["Q"]) for i in range(subset_size)]

        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to("cuda")
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        kb_embedding_real = (kb_embedding_real[0], kb_embedding_real[1])

        with torch.autograd.no_grad():

            outputs_no_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=input_ids,
                kb_kvs=None,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=True,
                kb_config=kb_config,
            )

            outputs_true_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=kb_embedding_real,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=True,
                kb_config=kb_config,
            )

        outputs_no_kb = tokenizer.batch_decode(outputs_no_kb, skip_special_tokens=False)

        outputs_true_kb = tokenizer.batch_decode(outputs_true_kb, skip_special_tokens=False)
        print("KB:")
        for i in range(subset_size):
            print("{} : {}".format(dataset_subset[i]["name"], dataset_subset[i]["description"]))

        for m in model_prune_format_mapping:
            if isinstance(model, m):
                prune_str = model_prune_format_mapping[m]

        print("------------------")
        for i in range(subset_size):

            print("True KB", prune_str(outputs_true_kb[i]))
            print("True answer: ", dataset_subset[i]["A"])
            no_kb_predictions.append(prune_str(outputs_no_kb[i]).split(dataset_subset[i]["Q"])[1])
            predictions.append(prune_str(outputs_true_kb[i]).split(dataset_subset[i]["Q"])[1])
            answer.append(dataset_subset[i]["A"])
            print("--------------------")
        print("******")

    config_str = exp_config_name  # Assume that the user has set this up outside of this script.

    # TODO: Add mlflow
    rouge_score = rouge.compute(predictions=predictions, references=answer)
    np.savez(os.path.join(save_dir, f"{config_str}_rouge.npy"), **rouge_score)

    rogue_score_no_kb = rouge.compute(predictions=no_kb_predictions, references=answer)
    np.savez(
        os.path.join(save_dir, f"{config_str}_rouge_no_kb.npy"),
        **rogue_score_no_kb,
    )

    # Start inspecting attention masks
    ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 32)]

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    accs, confidences = [], []
    for left, right in ranges:
        weights = []
        kb_size = subset_size
        for idx in range(32)[left:right]:
            if idx % 3 == 0:
                weight = np.load(os.path.join(save_dir, f"{config_str}_{idx}.npy"))
                weights.append(weight[..., :kb_size].reshape(kb_size, -1, kb_size))
        weights = np.stack(weights)
        weights = weights.transpose(1, 0, 2, 3).reshape(kb_size, -1, kb_size)
        acc = (weights.sum(1).argmax(1) == np.arange(kb_size)).mean()
        top_5_predictions = torch.topk(torch.from_numpy(weights.sum(1)), 5, dim=1)[1]
        top_5_acc = (top_5_predictions == torch.arange(kb_size)[:, None]).any(1).float().mean()
        accs.append((acc, top_5_acc))
        confidence = softmax(weights.mean(1), -1).max()
        confidences.append(confidence)
    np.save(os.path.join(save_dir, f"{config_str}_acc.npy"), np.array(accs))
    np.save(os.path.join(dsave_dir, f"{config_str}_conf.npy"), np.array(confidences))


def main():
    args = parser.parse_args()
    print(args)
    if args.command == 'generation':
        eval_generate()
    elif args.command == 'accuracy':
        eval_accuracy_cli()
    elif args.command == 'refusal':
        eval_refusal()
    elif args.command == 'standard':
        eval()
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()
