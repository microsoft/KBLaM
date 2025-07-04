import numpy as np
import os

def _load_cached_embeddings(encoder_model_spec: str, dataset_dir: str, dataset_name: str, key_embd_src: str):
    """Loads cached key and value embeddings from numpy files.

    This function constructs the file paths for pre-computed key and value embeddings
    based on the encoder model specification, dataset directory, dataset name, and the
    source of the key embeddings. It then loads these embeddings from the corresponding
    .npy files.

    Args:
        encoder_model_spec (str): The specification of the encoder model (e.g., 'OAI').
        dataset_dir (str): The directory where the dataset and embeddings are stored.
        dataset_name (str): The name of the dataset.
        key_embd_src (str): The source of the key embeddings (e.g., 'key', 'answer').

    Returns:
        tuple: A tuple containing the loaded key and value embeddings as numpy arrays.
    """
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
