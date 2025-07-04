import numpy as np
from typing import Dict, List, Optional

from kblam.kb_encoder import KBEncoder
from kblam.utils.train_utils import get_kb_embd

class KBRetriever:
    """A class to retrieve knowledge base embeddings.

    This class handles the retrieval of key and value embeddings from a knowledge base.
    It can either use precomputed embeddings from files or compute them on the fly
    using a provided encoder.

    Attributes:
        encoder (KBEncoder): The knowledge base encoder.
        dataset (List[Dict]): The dataset containing the knowledge base.
        key_embds (Optional[np.ndarray]): Precomputed key embeddings.
        value_embds (Optional[np.ndarray]): Precomputed value embeddings.
    """
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[np.ndarray] = None,
    ):
        """Initializes the KBRetriever.

        Args:
            encoder (KBEncoder): The knowledge base encoder.
            dataset (List[Dict]): The dataset containing the knowledge base.
            precomputed_embed_keys_path (Optional[str], optional): Path to precomputed key embeddings. Defaults to None.
            precomputed_embed_values_path (Optional[np.ndarray], optional): Path to precomputed value embeddings. Defaults to None.
        """
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
        """Checks if precomputed embeddings are available.

        Returns:
            bool: True if both key and value embeddings are loaded, False otherwise.
        """
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices):
        """Retrieves key and value embeddings for a given batch of indices.

        If precomputed embeddings are available, they are used. Otherwise, the embeddings
        are computed using the encoder.

        Args:
            batch_indices (list): A list of indices for which to retrieve embeddings.

        Returns:
            tuple: A tuple containing the key and value embeddings.
        """
        if self._use_cached_embd():
            return get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            return get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)
