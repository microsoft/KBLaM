import json
import os
import numpy as np
import torch
import evaluate
from pathlib import Path

from kblam.kb_encoder import KBEncoder
from kblam.models.gemma3n_model import KblamGemma3nForConditionalGeneration
from kblam.utils.eval_utils import (
    _format_Q_llama,
    _format_Q_phi3,
    _format_Q_bitnet,
    _format_Q_gemma3n,
    model_prune_format_mapping,
    softmax,
)

from .retriever import KBRetriever
from .models import _prepare_models

rouge = evaluate.load("rouge")

def eval_standard(args):
    """Performs a standard evaluation of the knowledge base model.

    This function runs a comprehensive evaluation, including generation with and
    without the knowledge base, and analyzes attention weights to calculate
    accuracy and confidence scores. The results, including ROUGE scores and
    attention-based metrics, are saved to files.

    Args:
        args: Command-line arguments parsed by argparse.
    """
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

        format_func_map = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3, "bitnet": _format_Q_bitnet, "gemma3n": _format_Q_gemma3n}

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
            if idx % kb_layer_frequency == 0:
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
