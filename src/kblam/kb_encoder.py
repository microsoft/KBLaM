import torch
import torch.nn as nn
from transformers import FeatureExtractionMixin
from sentence_transformers import SentenceTransformer
from .gpt_session import GPT
from typing import Union


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def get_projector(projector_type: str, in_dim: int, out_dim: int, projector_kwargs: dict) -> nn.Module:
    assert isinstance(projector_kwargs, dict)
    if projector_type == "identity":
        return IdentityMap()
    elif projector_type == "linear":
        return nn.Linear(in_dim, out_dim)
    elif projector_type == "mlp":
        mlp_depth, mlp_hidden_dim = (
            projector_kwargs["mlp_depth"],
            projector_kwargs["mlp_hidden_dim"],
        )
        modules = [nn.Linear(in_dim, mlp_hidden_dim)]
        for _ in range(mlp_depth):
            modules.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(mlp_hidden_dim, out_dim))
        return nn.Sequential(*modules)
    else:
        raise NotImplementedError(f"Projector type {projector_type} not found")

# TODO(t-isazawat): Add support for batching here
class KBEncoder(nn.Module, FeatureExtractionMixin):
    kb_special_token = {
        "<KB_BEGIN>": 0,
        "<KB_END>": 1,
        "<KEY_SEP>": 2,
        "<VALUE_SEP>": 3,
        "<ENTITY_SEP>": 4,
        "<KV_SEP>": 5,
    }

    def __init__(
        self,
        encoder_name: str,
        projector_type: str,
        out_dim: int,
        endpoint_url: str,
        projector_kwargs: dict = {},
        frozen_base_model: bool = True,
        device: Union[str, torch.device] = "cuda",
        get_oai_embd_online: bool = False,
    ):
        super().__init__()
        # Define the KB encoder backbone
        self.encoder_spec = encoder_name

        if encoder_name in ['OAI', 'BigOAI']:
            big = 'Big' in encoder_name
            if get_oai_embd_online:
                if big:
                    self.gs = GPT("text-embedding-3-large", endpoint_url)
                else:
                    self.gs = GPT("ada-embeddings", endpoint_url)

                self.base_model_encode = lambda s: torch.tensor(self.gs.generate_embedding(s)).to(self.device)
            else:
                self.base_model_encode = None
            self.in_dim = 3072 if big else 1536
        else:
            self.base_model = SentenceTransformer(encoder_name)
            self.base_model_encode = lambda s: self.base_model.encode(s, convert_to_numpy=False)
            self.frozen_base_model = frozen_base_model
            if frozen_base_model:
                self.base_model.eval()
                for param in self.base_model.parameters():
                    param.requires_grad = False
            else:
                self.base_model.train()
            self.in_dim = self.base_model.get_sentence_embedding_dimension()
        self.out_dim = out_dim
        self.projector_k = get_projector(projector_type, self.in_dim, self.out_dim, projector_kwargs)
        self.projector_v = get_projector(projector_type, self.in_dim, self.out_dim, projector_kwargs)
        self.key_layernorm = nn.LayerNorm(self.out_dim, elementwise_affine=False, bias=False)
        self.embedding = nn.Embedding(len(self.kb_special_token), out_dim)
        self.device = device
        self.to(self.device)

    def freeze_v(self):
        for param in self.projector_v.parameters():
            param.requires_grad = False

    def encode_key(self, S=None, base_emb=None):
        """
        Convert the keys to embedding using the backbone model + adapter
        """
        if S:
            base_embedding = self.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        return self.key_layernorm(self.projector_k(base_embedding)).bfloat16()

    def encode_val(self, S=None, base_emb=None):
        """
        Convert the values to embedding using the backbone model + adapter
        """
        if S:
            base_embedding = self.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        return self.projector_v(base_embedding).bfloat16()

    def encode_key_value(self, key, value):
        key_embd = self.encode_key(S=key)
        value_embd = self.encode_val(S=value)
        return key_embd, value_embd

    def encode_key_value_embeddings(self, key_embd, value_embd):
        key_embd = self.encode_key(base_emb=key_embd)
        value_embd = self.encode_val(base_emb=value_embd)
        return key_embd, value_embd

    def encode_base_embeddings(self, kb: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the knowledge base into embeddings. Assumes that the input KB is given as a tuple of two torch tensors: keys and values
        """
        key_embds, value_embds = [], []
        for key, value in zip(kb[0], kb[1]):
            key_embd, value_embd = self.encode_key_value_embeddings(key, value)
            key_embds.append(key_embd)
            value_embds.append(value_embd)
        return torch.stack(key_embds), torch.stack(value_embds)

    def encode(self, kb: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the knowledge base into embeddings
        """
        key_embds, value_embds = [], []
        for key, value in kb:
            key_embd, value_embd = self.encode_key_value(key, value)
            key_embds.append(key_embd)
            value_embds.append(value_embd)
        return torch.stack(key_embds), torch.stack(value_embds)

    def get_special_token_embd(self, token_type):
        """
        Get the embedding for the special token,
        take in a string, returns a tensor
        """
        idx = torch.tensor(self.kb_special_token[token_type]).to(self.embedding.weight.device)
        return self.embedding(idx).bfloat16()