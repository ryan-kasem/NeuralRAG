"""
models/embedding_model.py — Custom fine-tuned sentence embedding model.

We wrap a HuggingFace transformer and add:
  1. Mean-pooling with attention-mask weighting (better than CLS token)
  2. L2 normalization so cosine similarity == dot product (faster at inference)
  3. SimCSE-style in-batch contrastive loss for fine-tuning

This is the core of what makes NeuralRAG better than off-the-shelf embeddings:
a domain-adapted model that understands *your* data distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
from config import cfg


class MeanPooler(nn.Module):
    """
    Collapses the [seq_len, hidden] token embeddings into a single [hidden]
    sentence vector by averaging over non-padding tokens.

    Why not just use the [CLS] token?  Mean pooling consistently outperforms
    CLS on semantic similarity benchmarks (Reimers & Gurevych, 2019).
    """

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Expand mask to match embedding dimension: [B, L] → [B, L, H]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        # Sum token embeddings (zeroing out padding via mask)
        summed = (token_embeddings * mask_expanded).sum(dim=1)
        # Divide by actual sequence length (not counting padding)
        lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / lengths


class EmbeddingModel(nn.Module):
    """
    Fine-tunable sentence embedding model based on a pre-trained transformer.

    Architecture:
        transformer backbone → mean pooler → L2 norm → 384-d unit vector
    """

    def __init__(self, model_name: str = None, embedding_dim: int = None):
        super().__init__()
        model_name = model_name or cfg.embedding.base_model
        self.embedding_dim = embedding_dim or cfg.embedding.embedding_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooler()

        # Optional projection head — maps backbone hidden size → embedding_dim.
        # Useful if you want a smaller embedding than the backbone's hidden size.
        backbone_hidden = self.backbone.config.hidden_size
        if backbone_hidden != self.embedding_dim:
            self.projection = nn.Linear(backbone_hidden, self.embedding_dim, bias=False)
        else:
            self.projection = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized sentence embeddings of shape [B, embedding_dim]."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: [B, L, H]
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)  # [B, H]
        projected = self.projection(pooled)                               # [B, D]
        # L2 normalize so we can use dot product as cosine similarity
        return F.normalize(projected, p=2, dim=-1)

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Convenience method: tokenize → encode → return numpy-ready tensor.
        Used by the retriever at inference time (no grad needed).
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        self.to(device)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.embedding.max_seq_length,
                return_tensors="pt",
            ).to(device)
            emb = self(encoded["input_ids"], encoded["attention_mask"])
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)  # [N, D]


class SimCSELoss(nn.Module):
    """
    In-batch contrastive loss based on SimCSE (Gao et al., 2021).

    Given a batch of (anchor, positive, hard_negative) triplets:
    - Push anchor close to its positive
    - Push anchor away from all other sentences in the batch (in-batch negatives)
    - Optionally weight hard negatives more heavily

    This is basically cross-entropy over cosine similarities, which is why
    the temperature hyperparameter matters — it controls the "sharpness" of
    the distribution.
    """

    def __init__(self, temperature: float = None):
        super().__init__()
        self.temperature = temperature or cfg.embedding.temperature

    def forward(
        self,
        anchors: torch.Tensor,    # [B, D] — query embeddings
        positives: torch.Tensor,  # [B, D] — relevant document embeddings
        hard_negatives: torch.Tensor = None,  # [B, D] — hard negative embeddings
    ) -> torch.Tensor:
        batch_size = anchors.size(0)

        # Compute similarity matrix between anchors and positives: [B, B]
        # The diagonal is the anchor-positive similarity (what we want to maximize)
        sim_matrix = torch.matmul(anchors, positives.T) / self.temperature

        if hard_negatives is not None:
            # Concatenate positives and hard negatives as the "denominator" pool
            # This forces the model to distinguish true matches from close misses
            all_docs = torch.cat([positives, hard_negatives], dim=0)  # [2B, D]
            sim_matrix = torch.matmul(anchors, all_docs.T) / self.temperature  # [B, 2B]

        # Labels: position i should match document i (the diagonal)
        labels = torch.arange(batch_size, device=anchors.device)

        # Cross-entropy loss: maximize diagonal, minimize off-diagonal
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
