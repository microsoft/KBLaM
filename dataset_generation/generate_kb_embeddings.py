import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kblam.gpt_session import GPT
from kblam.utils.data_utils import DataPoint


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="text-embedding-3-large",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings"],
    )
    parser.add_argument("--dataset_name", type=str, default="synthetic_data")
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset in JSON format.")
    parser.add_argument("--output_path", type=str, default="dataset")

    args = parser.parse_args()
    return args


def compute_embeddings(encoder_model_spec: str, dataset: list[DataPoint], part: str, batch_size: int = 100) -> np.array:
    """Compute embeddings for the given dataset in batches using the encoder model spec."""
    embeddings = []
    all_elements = []
    for entity in dataset:
        if part == "key_string":
            all_elements.append(entity.key_string)
        elif part == "description":
            all_elements.append(entity.description)
        else:
            raise ValueError(f"Part {part} not supported.")
    chunks = [all_elements[i : i + batch_size] for i in range(0, len(all_elements), batch_size)]

    model = SentenceTransformer(encoder_model_spec, device='cuda')
    for chunk in tqdm(chunks):
        embd = model.encode(chunk, convert_to_numpy=True)
        embeddings.append(embd)

    embeddings = np.concatenate(embeddings, 0)
    assert len(embeddings) == len(all_elements)
    return embeddings


if __name__ == "__main__":
    args = parser_args()
    with open(args.dataset_path, "r") as file:
        dataset = [DataPoint(**json.loads(line)) for line in file]

    if args.model_name == "all-MiniLM-L6-v2":
        key_embeds = compute_embeddings(args.model_name, dataset, "key_string")
        value_embeds = compute_embeddings(args.model_name, dataset, "description")
    elif args.model_name in ["ada-embeddings", "text-embedding-3-large"]:
        gpt = GPT(args.model_name, args.endpoint_url)

        key_embeds = []
        value_embeds = []

        for entity in tqdm(dataset):
            key_embeds.append(gpt.generate_embedding(entity.key_string))
            value_embeds.append(gpt.generate_embedding(entity.description))
    else:
        raise ValueError(f"Model {args.model_name} not supported.")

    os.makedirs(args.output_path, exist_ok=True)

    if args.model_name == "all-MiniLM-L6-v2":
        save_name = "all-MiniLM-L6-v2"
    elif args.model_name == "ada-embeddings":
        save_name = "OAI"
    else:
        save_name = "BigOAI"

    np.save(f'{args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy', np.array(key_embeds))
    np.save(f'{args.output_path}/{args.dataset_name}_{save_name}_embd_value.npy', np.array(value_embeds))
