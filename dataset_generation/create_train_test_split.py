import argparse
import json
from pathlib import Path

import numpy as np


def _create_train_test_names(orig_path: str) -> tuple[str, str]:
    train = f"train_{Path(orig_path).name}"
    test = f"test_{Path(orig_path).name}"
    return train, test


def _write_json(data: list[dict], save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


def _save_array(arr: np.array, filename: str) -> None:
    np.save(filename, arr)


def create_train_test_split(
    data_path: str, embedding_keys_path: str, embeddings_values_path: str, split_index: int, output_path: str
) -> None:
    """Split data into training and test sets and save the results.

    This function loads data from the specified paths, creates a train-test split at the given index,
    and saves the resulting splits to the output path.

    Parameters
    ----------
    data_path : str
        Path to the main data file to be split
    embedding_keys_path : str
        Path to the file containing embedding keys
    embeddings_values_path : str
        Path to the file containing embedding values
    split_index : int
        Index at which to split the data into train and test sets.
        Data before this index will be training, after will be test.
    output_path : str
        Directory path where the split datasets will be saved

    Returns
    -------
    None
        Function saves the split datasets to disk but does not return any values

    """

    output_p = Path(output_path)
    output_p.mkdir(exist_ok=True, parents=True)

    train_dataset = json.load(open(data_path))[:split_index]

    train_key_embds = np.load(embedding_keys_path).astype("float32")[:split_index]
    train_value_embds = np.load(embeddings_values_path).astype("float32")[:split_index]

    test_dataset = json.load(open(data_path))[split_index:]

    test_key_embds = np.load(embedding_keys_path).astype("float32")[split_index:]
    test_value_embds = np.load(embeddings_values_path).astype("float32")[split_index:]

    train_dataset_name, test_dataset_name = _create_train_test_names(data_path)
    train_keys_name, test_keys_name = _create_train_test_names(embedding_keys_path)
    train_values_name, test_values_name = _create_train_test_names(embeddings_values_path)

    _write_json(train_dataset, output_p / train_dataset_name)
    _write_json(test_dataset, output_p / test_dataset_name)
    _save_array(train_key_embds, output_p / train_keys_name)
    _save_array(train_value_embds, output_p / train_values_name)
    _save_array(test_key_embds, output_p / test_keys_name)
    _save_array(test_value_embds, output_p / test_values_name)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--embedding_keys_path", type=str)
    parser.add_argument("--embeddings_values_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--split_index", type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    create_train_test_split(
        args.data_path,
        args.embedding_keys_path,
        args.embeddings_values_path,
        args.split_index,
        args.output_path,
    )
