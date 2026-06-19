"""
config.py — Central configuration for NeuralRAG.
All hyperparameters, model names, and paths live here so nothing is hardcoded.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent


@dataclass
class EmbeddingConfig:
    # Base transformer we fine-tune on top of; MiniLM is fast but still accurate
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Output dimension of our embedding vectors
    embedding_dim: int = 384
    # Maximum tokens per input chunk
    max_seq_length: int = 256
    # Temperature for contrastive (SimCSE) loss — lower → harder negatives matter more
    temperature: float = 0.05
    # Where to save / load our fine-tuned weights
    checkpoint_dir: Path = ROOT / "checkpoints" / "embedding_model"


@dataclass
class RetrieverConfig:
    # Number of candidates to fetch from FAISS before reranking
    top_k: int = 20
    # Final number of documents returned after MMR diversity filtering
    top_k_final: int = 5
    # MMR lambda: 0 = max diversity, 1 = max relevance
    mmr_lambda: float = 0.6
    # FAISS index type — "flat" (exact) or "ivf" (approximate, faster at scale)
    index_type: str = "flat"
    # Path to persisted FAISS index
    index_path: Path = ROOT / "data" / "faiss.index"


@dataclass
class RAGConfig:
    # Whether to use HyDE (generate a fake answer to improve retrieval)
    use_hyde: bool = True
    # Whether to apply cross-encoder reranking after retrieval
    use_reranker: bool = True
    # Cross-encoder model for reranking (much smaller than a generative LLM)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Max tokens to stuff into the LLM context window
    max_context_tokens: int = 3000
    # Local Ollama model for generation — run `ollama pull llama3.2` to download
    # Swap to "llama3.1:8b" or "mistral" etc. depending on what you've pulled
    llm_model: str = "llama3.2"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    # Hard-negative mining ratio: for every positive pair, sample N negatives
    hard_negative_ratio: int = 3
    eval_every_n_steps: int = 200
    # Weights & Biases project name for experiment tracking
    wandb_project: str = "neuralrag-embeddings"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # Max requests per minute per client IP
    rate_limit: int = 60


@dataclass
class NeuralRAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Singleton — import this everywhere
cfg = NeuralRAGConfig()
