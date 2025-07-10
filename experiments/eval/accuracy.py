import json
import os
import shutil
import numpy as np
import torch
from pathlib import Path

from kblam.models.gemma3n_model import KblamGemma3nForConditionalGeneration
from kblam.models.gemma3n_config import Gemma3nConfig
from kblam.utils.data_utils import augment_row
from kblam.utils.eval_utils import (
    _format_Q_llama,
    _format_Q_phi3,
    _format_Q_bitnet,
    _format_Q_gemma3n,
)

from .retriever import KBRetriever
from .models import _prepare_models
from .utils import write_to_json

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
    """Evaluates the model's accuracy using a knowledge base (KB).

    This function assesses the model's performance by generating answers to questions
    based on a provided knowledge base. It calculates accuracy and top-5 accuracy based
    on the attention weights of the model, saving the results and generated outputs.

    Args:
        tokenizer: The tokenizer for the model.
        kb_retriever (KBRetriever): The knowledge base retriever.
        model: The language model to be evaluated.
        dataset (list): The dataset containing questions and answers.
        exp_config (str): The experiment configuration name.
        fancy_question (bool): Whether to use augmented (fancy) questions.
        kb_config (KBLaMConfig | Gemma3nConfig): The configuration for the knowledge base.
        kb_size (int): The size of the knowledge base to use.
        llm_type (str): The type of the language model (e.g., 'llama3', 'phi3').
        test_batch_size (int): The batch size for testing.
        save_dir (str): The directory to save the evaluation results.
        attn_save_dir (str): The directory to save attention weights.

    Returns:
        list: A list of dictionaries containing accuracy and top-5 accuracy for each layer.
    """

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

    format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3, "bitnet": _format_Q_bitnet, "gemma3n": _format_Q_gemma3n}

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

def eval_accuracy_cli(args):
    """Command-line interface for evaluating model accuracy.

    Parses command-line arguments, prepares the models and data, and then
    calls the `eval_accuracy` function to perform the evaluation.

    Args:
        args: Command-line arguments parsed by argparse.
    """
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

def run_accuracy_evalution(args):
    """Runs the accuracy evaluation across a range of knowledge base sizes.

    This function iterates through a predefined list of knowledge base sizes, running
    the accuracy evaluation for each size. The results are collected and saved to a
    JSON file.

    Args:
        args: Command-line arguments parsed by argparse.
    """
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
