"""
pipeline/rag_pipeline.py — End-to-end RAG pipeline.

Pipeline stages:
  1. Query expansion via HyDE (Hypothetical Document Embeddings)
     — generate a fake answer to the question, embed that instead of the raw
       question.  This closes the query-document vocabulary gap.
  2. Dense retrieval (FAISS + MMR, see models/retriever.py)
  3. Cross-encoder reranking — a smaller, slower model that does pairwise
     (query, doc) scoring to re-order the FAISS candidates more precisely
  4. Context assembly — smart truncation so we never exceed the LLM's context window
  5. LLM generation — Ollama (free, local Llama 3 running on your machine)

Setup (one-time):
    brew install ollama          # or download from ollama.com
    ollama serve                 # start the local server
    ollama pull llama3.2         # download the model (~2GB)

Reference: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
"""

import os
from typing import List, Optional, Tuple
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from models.embedding_model import EmbeddingModel
from models.retriever import FAISSRetriever, Document
from config import cfg

# Ollama runs locally and exposes an OpenAI-compatible REST API at port 11434.
# We use the openai library pointed at localhost — no API key needed.
try:
    from openai import OpenAI
    _llm_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama ignores the key but the client requires a non-empty string
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[RAG] openai package not installed. Run: pip install openai")


@dataclass
class RAGResult:
    """Structured output from a RAG query."""
    answer: str
    retrieved_docs: List[Tuple[Document, float]]  # (doc, relevance_score)
    query_used: str           # The (possibly expanded) query that drove retrieval
    num_tokens_in_context: int


class RAGPipeline:
    """
    Full RAG pipeline: query → retrieve → rerank → generate.

    Example:
        pipeline = RAGPipeline.from_checkpoint("checkpoints/embedding_model/best_model.pt")
        pipeline.index_documents(my_docs)
        result = pipeline.query("What is the capital of France?")
        print(result.answer)
    """

    def __init__(
        self,
        embed_model: EmbeddingModel,
        retriever: FAISSRetriever,
        use_hyde: bool = None,
        use_reranker: bool = None,
        device: str = "cpu",
    ):
        self.embed_model = embed_model
        self.retriever = retriever
        self.device = device
        self.use_hyde = use_hyde if use_hyde is not None else cfg.rag.use_hyde
        self.use_reranker = use_reranker if use_reranker is not None else cfg.rag.use_reranker

        # Cross-encoder for reranking (separate from the bi-encoder retriever)
        # Bi-encoder is fast but less precise; cross-encoder is slower but more accurate.
        # Two-stage design gives us the best of both worlds.
        if self.use_reranker:
            print(f"[RAG] Loading cross-encoder reranker: {cfg.rag.reranker_model}")
            self.reranker = CrossEncoder(cfg.rag.reranker_model)

        # Reuse the embedding model's tokenizer to count tokens in the context window.
        # This is a reasonable approximation regardless of which LLM is doing generation.
        self.tokenizer_counter = embed_model.tokenizer

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "RAGPipeline":
        """Build a pipeline by loading a fine-tuned embedding model from disk."""
        import torch
        embed_model = EmbeddingModel()
        ckpt = torch.load(checkpoint_path, map_location=device)
        embed_model.load_state_dict(ckpt["model_state_dict"])
        embed_model.eval()

        retriever = FAISSRetriever(embed_model, device=device)
        return cls(embed_model, retriever, device=device)

    def index_documents(self, documents: List[Document]) -> None:
        """Add documents to the retrieval index."""
        self.retriever.add_documents(documents)

    def query(self, question: str) -> RAGResult:
        """
        Run the full RAG pipeline for a single question.

        This is the main public API. Everything else is implementation detail.
        """
        # ── Stage 1: Query expansion (HyDE) ──────────────────────────────────
        retrieval_query = question
        if self.use_hyde and LLM_AVAILABLE:
            retrieval_query = self._hyde_expand(question)

        # ── Stage 2: Dense retrieval ──────────────────────────────────────────
        candidates = self.retriever.retrieve(retrieval_query)

        # ── Stage 3: Cross-encoder reranking ──────────────────────────────────
        if self.use_reranker and candidates:
            candidates = self._rerank(question, candidates)

        # ── Stage 4: Context assembly ─────────────────────────────────────────
        context, num_tokens = self._assemble_context(candidates)

        # ── Stage 5: LLM generation ────────────────────────────────────────────
        answer = self._generate(question, context)

        return RAGResult(
            answer=answer,
            retrieved_docs=candidates,
            query_used=retrieval_query,
            num_tokens_in_context=num_tokens,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _hyde_expand(self, question: str) -> str:
        """
        Hypothetical Document Embeddings (Gao et al., 2022).

        The idea: embedding a question and embedding a relevant answer live in
        different vector spaces.  HyDE generates a *fake* answer with an LLM
        and embeds that instead, moving us into the document vector space.

        Works surprisingly well — the LLM hallucination is OK here because
        we only use the *embedding*, not the text itself.
        """
        prompt = (
            "Write a short factual paragraph that directly answers the following question. "
            "If you're unsure, make an educated guess.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        try:
            response = _llm_client.chat.completions.create(
                model=cfg.rag.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,  # Deterministic fake answer for reproducibility
            )
            hypothetical_doc = response.choices[0].message.content.strip()
            return hypothetical_doc
        except Exception as e:
            # If HyDE fails (e.g., API quota), fall back to the raw question
            print(f"[RAG] HyDE failed ({e}), using raw question.")
            return question

    def _rerank(
        self,
        question: str,
        candidates: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Re-score (query, doc) pairs with a cross-encoder and re-sort.

        Cross-encoders are more accurate than bi-encoders because they see
        both texts at once and can compute token-level interactions.
        The trade-off is O(k) inference calls instead of O(1) + ANN search.
        """
        pairs = [(question, doc.text) for doc, _ in candidates]
        scores = self.reranker.predict(pairs)  # returns numpy array of floats

        # Zip new scores back onto documents and sort descending
        reranked = sorted(
            zip([doc for doc, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked

    def _assemble_context(
        self,
        docs: List[Tuple[Document, float]],
    ) -> Tuple[str, int]:
        """
        Concatenate retrieved docs into a context string, respecting the token budget.

        We trim from the bottom up: if adding the next doc would overflow the
        context window, we skip it rather than truncating mid-sentence.
        """
        budget = cfg.rag.max_context_tokens
        parts = []
        total_tokens = 0

        for i, (doc, score) in enumerate(docs):
            chunk = f"[Source {i+1}] {doc.text}"
            chunk_tokens = len(self.tokenizer_counter.encode(chunk, add_special_tokens=False))
            if total_tokens + chunk_tokens > budget:
                break  # Stop before overflowing the context window
            parts.append(chunk)
            total_tokens += chunk_tokens

        context = "\n\n".join(parts)
        return context, total_tokens

    def _generate(self, question: str, context: str) -> str:
        """
        Call the local Ollama LLM with a structured RAG prompt.

        Ollama must be running (`ollama serve`) and the model must be pulled
        (`ollama pull llama3.2`) before this will work.
        """
        if not LLM_AVAILABLE:
            return f"[DEMO MODE — install openai package to enable generation]\n\nContext retrieved:\n{context}"

        system_prompt = (
            "You are a precise question-answering assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say 'I don't have enough information to answer that.' "
            "Cite source numbers (e.g., [Source 1]) when referencing specific facts."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        try:
            response = _llm_client.chat.completions.create(
                model=cfg.rag.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temp for factual Q&A
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Ollama not running or model not pulled — return context so demo still works
            return (
                f"[LLM unavailable: {e}]\n"
                f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {cfg.rag.llm_model}`).\n\n"
                f"Retrieved context:\n{context}"
            )
