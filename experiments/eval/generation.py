import json
import os
import re
import numpy as np
import torch
import transformers
import evaluate
from tqdm import tqdm
from pathlib import Path

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.utils.data_utils import generate_multi_entity_qa
from kblam.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    answer_question,
)

from .retriever import KBRetriever
from .utils import write_to_json
from .models import _prepare_models

bert_score = evaluate.load("bertscore")

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
    """Performs evaluation of the model's generation capabilities.

    This function evaluates the model on a generation task using different modes:
    knowledge base (kb), in-context learning (icl), or zero-shot. It computes
    ROUGE and BERT scores for the generated outputs and saves the results.

    Args:
        model (KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM): The language model to evaluate.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        kb_retriever (KBRetriever): The knowledge base retriever.
        encoder_model_spec (str): The specification of the encoder model.
        kb_config (KBLaMConfig): The configuration for the knowledge base.
        eval_mode (str, optional): The evaluation mode ('kb', 'icl', 'zeroshot'). Defaults to "kb".
        kb_size (int, optional): The size of the knowledge base. Defaults to 250.
        seed (int, optional): The random seed. Defaults to 1.
        topk_size (int, optional): The number of top-k entities to retrieve. Defaults to -1.
        multi_entites (int, optional): The number of entities for multi-entity questions. Defaults to -1.
        remove_sorry (bool, optional): Whether to remove answers containing "sorry". Defaults to False.

    Returns:
        tuple: A tuple containing the raw generation results as a string and a dictionary of scores.
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

        prompt_to_split = Q
        if eval_mode == "kb":
            raw_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                topk_size=topk_size,
                kb_config=kb_config,
            )
        elif eval_mode == "icl":
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            prompt_to_split = ins_prompt + prompt_strs + Q
            raw_output = answer_question(
                tokenizer,
                model,
                prompt_to_split,
                kb=None,
                kb_config=kb_config,
            )
        elif eval_mode == "zeroshot":
            if multi_entites != -1:
                ins_prompt = zero_shot_prompt_multi_entities
            else:
                ins_prompt = zero_shot_prompt
            prompt_to_split = ins_prompt + Q
            raw_output = answer_question(
                tokenizer, model, prompt_to_split, kb=None, kb_config=kb_config
            )
        
        output_parts = raw_output.split(prompt_to_split)
        model_output = output_parts[1] if len(output_parts) > 1 else raw_output

        # print(model_output)
        if remove_sorry:
            if "sorry" in model_output:
                continue
        full_outputs.append((model_output, answer))
        if multi_entites == -1:
            pattern = r'The\s+\w+\s+of\s+[^\"]+\s+is\s+(.+)'
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

def eval_generate(args):
    """Evaluates the generation capabilities of the model using a knowledge base.

    This function orchestrates the generation evaluation process. It loads the dataset,
    prepares the models, and then calls `perform_eval` to run the evaluation.
    The results are saved to files.

    Args:
        args: Command-line arguments parsed by argparse.
    """
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
