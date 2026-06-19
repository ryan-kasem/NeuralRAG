"""
models/retriever.py — FAISS-backed vector retriever with MMR diversity filtering.

Two-stage retrieval:
  1. FAISS ANN search: fast approximate nearest-neighbor lookup (millions of docs/sec)
  2. MMR re-selection: picks a diverse subset so the LLM doesn't see 5 copies of
     the same paragraph

Maximal Marginal Relevance (MMR) was introduced by Carbonell & Goldstein (1998)
for summarization but works great for RAG — it balances relevance against redundancy.
"""

import faiss
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from models.embedding_model import EmbeddingModel
from config import cfg


@dataclass
class Document:
    """A single chunk of text with its metadata (source, page, etc.)."""
    text: str
    metadata: Dict[str, Any]
    doc_id: int = -1


class FAISSRetriever:
    """
    Manages a FAISS index for dense retrieval over a corpus of Document chunks.

    Usage:
        retriever = FAISSRetriever(embed_model)
        retriever.add_documents(docs)
        results = retriever.retrieve("how does attention work?")
    """

    def __init__(self, embed_model: EmbeddingModel, device: str = "cpu"):
        self.embed_model = embed_model
        self.device = device
        self.dim = cfg.embedding.embedding_dim

        # FAISS index: FlatIP = exact inner product search (= cosine since L2-normed)
        # For >1M docs, swap to faiss.IndexIVFFlat for ~10x speedup
        self.index = faiss.IndexFlatIP(self.dim)

        # Parallel list to map FAISS integer IDs back to Document objects
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document], batch_size: int = 256) -> None:
        """Embed all documents and add them to the FAISS index."""
        print(f"[Retriever] Indexing {len(documents)} documents...")
        texts = [doc.text for doc in documents]

        all_vecs = self.embed_model.encode(texts, batch_size=batch_size, device=self.device)
        # FAISS requires float32 numpy arrays, contiguous in memory
        vecs_np = all_vecs.numpy().astype(np.float32)
        np.ascontiguousarray(vecs_np)

        # Assign sequential IDs starting from current index size
        start_id = len(self.documents)
        for i, doc in enumerate(documents):
            doc.doc_id = start_id + i

        self.index.add(vecs_np)
        self.documents.extend(documents)
        print(f"[Retriever] Index now contains {self.index.ntotal} vectors.")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        top_k_final: int = None,
        mmr_lambda: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the most relevant documents for a query using FAISS + MMR.

        Returns list of (Document, score) tuples sorted by relevance.
        """
        top_k = top_k or cfg.retriever.top_k
        top_k_final = top_k_final or cfg.retriever.top_k_final
        mmr_lambda = mmr_lambda or cfg.retriever.mmr_lambda

        # Step 1: Embed the query
        query_vec = self.embed_model.encode(query, device=self.device).numpy().astype(np.float32)

        # Step 2: FAISS ANN search — returns top_k candidates
        scores, indices = self.index.search(query_vec, top_k)  # [1, top_k] each
        scores, indices = scores[0], indices[0]

        # Filter out FAISS's -1 sentinel (returned when index has fewer than top_k docs)
        valid = [(self.documents[i], float(s)) for i, s in zip(indices, scores) if i != -1]

        # Step 3: MMR diversity filtering over the candidates
        selected = self._mmr_select(query_vec[0], valid, top_k_final, mmr_lambda)
        return selected

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        candidates: List[Tuple[Document, float]],
        k: int,
        lam: float,
    ) -> List[Tuple[Document, float]]:
        """
        Maximal Marginal Relevance selection.

        MMR score = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))

        lam=1 → pure relevance (same as FAISS order)
        lam=0 → pure diversity (random-ish)
        lam=0.6 → good balance for RAG (keeps relevant but avoids duplicate chunks)
        """
        if not candidates:
            return []

        # Pre-compute doc embeddings matrix for fast similarity lookups
        doc_vecs = np.stack([
            self.embed_model.encode(doc.text, device=self.device).numpy()[0]
            for doc, _ in candidates
        ])  # [N, D]

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            if not remaining_indices:
                break

            best_idx = None
            best_score = -np.inf

            for idx in remaining_indices:
                # Relevance: cosine sim to query (already L2-normed → dot product)
                relevance = float(np.dot(query_vec, doc_vecs[idx]))

                # Redundancy: max similarity to any already-selected doc
                if selected_indices:
                    selected_vecs = doc_vecs[selected_indices]
                    redundancy = float(np.max(selected_vecs @ doc_vecs[idx]))
                else:
                    redundancy = 0.0  # First pick: no redundancy penalty

                mmr_score = lam * relevance - (1 - lam) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def save(self, path: Path = None) -> None:
        """Persist the FAISS index and document store to disk."""
        path = path or cfg.retriever.index_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))
        with open(path.with_suffix(".docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        print(f"[Retriever] Saved index to {path}")

    @classmethod
    def load(cls, embed_model: EmbeddingModel, path: Path = None, device: str = "cpu") -> "FAISSRetriever":
        """Load a previously saved FAISS index from disk."""
        path = path or cfg.retriever.index_path
        path = Path(path)

        retriever = cls(embed_model, device)
        retriever.index = faiss.read_index(str(path))
        with open(path.with_suffix(".docs.pkl"), "rb") as f:
            retriever.documents = pickle.load(f)

        print(f"[Retriever] Loaded index with {retriever.index.ntotal} vectors.")
        return retriever
