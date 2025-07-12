import json
import os
import re
import numpy as np
import torch
import transformers
import evaluate
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.gemma3n_model import KblamGemma3nForConditionalGeneration
from kblam.models.gemma3n_config import Gemma3nConfig
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM
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
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KblamGemma3nForConditionalGeneration,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    encoder_model_spec: str,
    kb_config: KBLaMConfig | Gemma3nConfig,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    topk_size: int = -1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
    fancy_format: bool = False,
    attn_save_dir: str = None,
):
    """Performs evaluation of the model's generation capabilities.

    This function evaluates the model on a generation task using different modes:
    knowledge base (kb), in-context learning (icl), or zero-shot. It computes
    ROUGE and BERT scores for the generated outputs and saves the results.

    Args:
        model (KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KblamGemma3nForConditionalGeneration): The language model to evaluate.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        kb_retriever (KBRetriever): The knowledge base retriever.
        encoder_model_spec (str): The specification of the encoder model.
        kb_config (KBLaMConfig | Gemma3nConfig): The configuration for the knowledge base.
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

    with torch.no_grad():
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

    from .utils import format_question, postprocess_output, get_topk_confidence

    topk_confidences = []
    # attn_save_dir is now passed as an argument
    all_attn_weights = []
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

        Q_fmt = format_question(Q, fancy_format)
        prompt_to_split = Q_fmt
        if eval_mode == "kb":
            # Optionally collect attention weights if answer_question supports it
            result = answer_question(
                tokenizer,
                model,
                Q_fmt,
                kb=kb_embedding,
                topk_size=topk_size,
                kb_config=kb_config,
                output_attentions=True if attn_save_dir else False,
            )
            if isinstance(result, tuple) and len(result) == 2:
                raw_output, attn_weights = result
                if attn_save_dir:
                    all_attn_weights.append(attn_weights)
            else:
                raw_output = result
        elif eval_mode == "icl":
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            prompt_to_split = ins_prompt + prompt_strs + Q_fmt
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
            prompt_to_split = ins_prompt + Q_fmt
            raw_output = answer_question(
                tokenizer, model, prompt_to_split, kb=None, kb_config=kb_config
            )

        output_parts = raw_output.split(prompt_to_split)
        model_output = output_parts[1] if len(output_parts) > 1 else raw_output
        # Post-process output (strip pad/eos tokens, etc.)
        model_output = postprocess_output(model_output, pad_token=getattr(tokenizer, 'pad_token', None))

        # Optionally collect top-k/confidence (if logits available from answer_question)
        # Example: if answer_question returns (output, logits), unpack and use get_topk_confidence
        # logits = ...
        # topk_confidences.append(get_topk_confidence(logits, tokenizer, k=5))

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


    # Save attention weights if requested
    if attn_save_dir is not None and attn_save_dir != "" and all_attn_weights:
        attn_save_path = Path(attn_save_dir)
        attn_save_path.mkdir(exist_ok=True, parents=True)
        attn_file = attn_save_path / "attn_weights.npy"
        np.save(attn_file, np.array(all_attn_weights, dtype=object))
        print(f"Attention weights saved to: {attn_file}")

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
    # --- Argument & Path Setup ---
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))


    # --- Tokenizer Loading (Best Practice) ---
    # For consistency and to prevent mismatches, always load the tokenizer from the base model directory.
    # No new tokens are added during KBLaM fine-tuning, so the base tokenizer is the source of truth.
    if not llm_base_dir:
        raise ValueError("A --llm_base_dir must be provided to load the correct base tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(llm_base_dir, trust_remote_code=True)
    # Set left padding for decoder-only models (important for correct generation)
    tokenizer.padding_side = "left"
    # Try to use '^' as pad token, fallback to eos_token if not in vocab
    try:
        tokenizer.pad_token = "^"
        # If '^' is not in vocab, this will raise an error on encode
        _ = tokenizer.encode("^")
    except Exception:
        print("Warning: '^' not in tokenizer vocab, falling back to eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model Loading ---
    # A mapping of llm_type to the corresponding model class.
    MODEL_CLASS_MAP = {
        "llama": KblamLlamaForCausalLM,
        "phi": KBLaMPhi3ForCausalLM,
        "bitnet": KBLaMBitNetForCausalLM,
        "gemma3n": KblamGemma3nForConditionalGeneration,
    }

    model_class = MODEL_CLASS_MAP.get(llm_type)
    if model_class is None:
        raise ValueError(f"Unsupported llm_type: '{llm_type}'. Supported types are: {list(MODEL_CLASS_MAP.keys())}")

    print(f"Loading fine-tuned '{llm_type}' model from: {model_path}")
    model = model_class.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.cuda()

    # The KBLaM config is loaded with the model itself.
    kb_config = model.config


    # --- KB Retriever Setup ---
    # Support both precomputed and on-the-fly encoder embeddings
    encoder = None
    use_precomputed = precomputed_embed_keys_path is not None and precomputed_embed_values_path is not None
    if not use_precomputed:
        # On-the-fly encoder initialization (matches original logic)
        from kblam.kb_encoder import KBEncoder
        # Use model config to determine output dimension
        out_dim = kb_config.hidden_size * (kb_config.num_hidden_layers // getattr(kb_config, 'kb_layer_frequency', 3) + 1)
        encoder = KBEncoder(
            encoder_name=encoder_model_spec.upper(),
            projector_type="linear",
            endpoint_url="",
            out_dim=out_dim,
            frozen_base_model=True,
            device=torch.device("cuda"),
        )
        # Optionally load encoder weights if path provided
        encoder_dir = getattr(args, "encoder_dir", None)
        if encoder_dir:
            encoder_weights_path = os.path.join(encoder_dir, "encoder.pt")
            if not os.path.exists(encoder_weights_path):
                raise FileNotFoundError(f"Encoder weights not found at {encoder_weights_path}. Please check the path.")
            encoder.load_state_dict(torch.load(encoder_weights_path))

    kb_retriever = KBRetriever(
        encoder=encoder,
        dataset=dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    # --- Perform Evaluation ---
    gen_results, score_results = perform_eval(
        model=model,
        tokenizer=tokenizer,
        kb_retriever=kb_retriever,
        encoder_model_spec=encoder_model_spec,
        kb_config=kb_config,
        eval_mode=eval_mode,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        multi_entites=args.multi_entites,
        fancy_format=getattr(args, 'fancy_format', False),
        attn_save_dir=getattr(args, 'attn_save_dir', None),
    )
    mem_cost = torch.cuda.max_memory_reserved("cuda")
    score_results["mem_cost"] = mem_cost

    save_path = Path(args.save_dir) / exp_config
    save_path.mkdir(exist_ok=True, parents=True)
    write_to_json(score_results, save_path.with_suffix(".json"))
    print(f"Scores saved to: {save_path.with_suffix('.json')}")
    print(score_results)

    results_txt_path = save_path.with_suffix(".txt")
    with open(results_txt_path, "w", encoding="utf-8") as text_file:
        text_file.write(gen_results)
    print(f"Generation results saved to: {results_txt_path}")

    print("\nEvaluation complete.")
