import torch
import numpy as np
from typing import Dict, List, Optional

from kblam.kb_encoder import KBEncoder
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd

class KBRetriever:
    """A class to retrieve knowledge base embeddings for training.

    This class manages the retrieval of key and value embeddings from a knowledge base.
    It supports using pre-computed embeddings or computing them on the fly. It also
    handles the creation of context sets for training.

    Attributes:
        encoder (KBEncoder): The knowledge base encoder.
        dataset (List[Dict]): The training dataset.
        key_embds (Optional[np.ndarray]): Precomputed key embeddings.
        value_embds (Optional[np.ndarray]): Precomputed value embeddings.
    """
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray],
        value_embds: Optional[np.ndarray],
    ):
        """Initializes the KBRetriever.

        Args:
            encoder (KBEncoder): The knowledge base encoder.
            dataset (List[Dict]): The training dataset.
            key_embds (Optional[np.ndarray]): Precomputed key embeddings.
            value_embds (Optional[np.ndarray]): Precomputed value embeddings.
        """
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset

    def _use_cached_embd(self):
        """Checks if precomputed embeddings are available.

        Returns:
            bool: True if both key and value embeddings are loaded, False otherwise.
        """
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices, batch_size, step, kb_size):
        """Retrieves key and value embeddings for a training batch.

        This function gets the embeddings for the ground truth entities in the batch
        and also samples a context set of other entities from the knowledge base.
        It combines these to form the final knowledge base embeddings for the batch.

        Args:
            batch_indices (list): The indices of the ground truth entities for the batch.
            batch_size (int): The size of the batch.
            step (int): The current training step.
            kb_size (int or list): The size of the knowledge base or a range for dynamic sizing.

        Returns:
            tuple: A tuple containing the concatenated key and value embeddings for the batch.
        """
        if self._use_cached_embd():
            train_set_key, train_set_val = get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            train_set_key, train_set_val = get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)

        if len(train_set_key.shape) == 2:
            # Add comment on why we need this line
            train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
            train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_index = np.random.choice(len(self.dataset), context_set_size, replace=False)  # type: ignore
        if self._use_cached_embd():
            context_set_key, context_set_val = get_kb_embd(
                self.encoder,
                context_set_index,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            context_set_key, context_set_val = get_kb_embd(self.encoder, context_set_index, kb_dict=self.dataset)
        context_set_key = context_set_key.unsqueeze(0).expand(batch_size, *context_set_key.shape)
        context_set_val = context_set_val.unsqueeze(0).expand(batch_size, *context_set_val.shape)
        # context_set_val = torch.randn_like(context_set_val)
        # Idea: Try torch.randn here context_set_tokens??
        true_kb_copy = 1
        kb_embedding = (
            torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
            torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        )
        return kb_embedding
