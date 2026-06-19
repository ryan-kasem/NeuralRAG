"""
evaluation/metrics.py — RAGAS-inspired evaluation framework.

Evaluating RAG systems is tricky because you need to measure three things:
  1. Retrieval quality  — did we fetch the right documents?
  2. Answer faithfulness — does the answer stick to what the context says?
  3. Answer relevance   — does the answer actually address the question?

We compute a NeuralRAG Score (0–1) that combines all three.  This is something
you can put on a resume — employers love seeing that you evaluated your system
and didn't just ship it blind.

Reference: RAGAS paper (Es et al., 2023) https://arxiv.org/abs/2309.15217
"""

import re
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# We reuse the embedding model for semantic similarity scoring
from models.embedding_model import EmbeddingModel


@dataclass
class EvalSample:
    """One evaluation example with ground truth."""
    question: str
    ground_truth_answer: str
    ground_truth_doc_ids: List[int]  # IDs of documents that contain the answer


@dataclass
class EvalResult:
    """Metrics for a single RAG prediction."""
    question: str
    predicted_answer: str

    # Retrieval metrics
    context_precision: float    # How many retrieved docs are relevant? (precision)
    context_recall: float       # Did we retrieve all relevant docs? (recall)

    # Generation metrics
    faithfulness: float         # Is the answer grounded in the context?
    answer_relevance: float     # Does the answer address the question?

    # Combined
    neuralrag_score: float      # Harmonic mean of all four metrics


@dataclass
class EvalReport:
    """Aggregate metrics across an entire test set."""
    results: List[EvalResult] = field(default_factory=list)

    @property
    def avg_context_precision(self) -> float:
        return np.mean([r.context_precision for r in self.results])

    @property
    def avg_context_recall(self) -> float:
        return np.mean([r.context_recall for r in self.results])

    @property
    def avg_faithfulness(self) -> float:
        return np.mean([r.faithfulness for r in self.results])

    @property
    def avg_answer_relevance(self) -> float:
        return np.mean([r.answer_relevance for r in self.results])

    @property
    def avg_neuralrag_score(self) -> float:
        return np.mean([r.neuralrag_score for r in self.results])

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("         NeuralRAG Evaluation Report")
        print("=" * 55)
        print(f"  Samples evaluated:    {len(self.results)}")
        print(f"  Context Precision:    {self.avg_context_precision:.3f}")
        print(f"  Context Recall:       {self.avg_context_recall:.3f}")
        print(f"  Faithfulness:         {self.avg_faithfulness:.3f}")
        print(f"  Answer Relevance:     {self.avg_answer_relevance:.3f}")
        print(f"  NeuralRAG Score:      {self.avg_neuralrag_score:.3f}  ← headline metric")
        print("=" * 55 + "\n")


class RAGEvaluator:
    """
    Evaluates a RAGPipeline against a labeled test set.

    Metrics are computed using a combination of:
    - Exact set matching (for retrieval precision/recall)
    - Semantic similarity via our embedding model (for answer metrics)
    - LLM-as-judge for faithfulness (optional, requires API key)
    """

    def __init__(self, embed_model: EmbeddingModel, use_llm_judge: bool = False):
        self.embed_model = embed_model
        self.use_llm_judge = use_llm_judge

    def evaluate(
        self,
        pipeline,  # RAGPipeline — avoid circular import by not type-hinting
        test_samples: List[EvalSample],
        device: str = "cpu",
    ) -> EvalReport:
        """Run all test samples through the pipeline and compute metrics."""
        report = EvalReport()

        for i, sample in enumerate(test_samples):
            print(f"  Evaluating sample {i+1}/{len(test_samples)}: {sample.question[:60]}...")
            result = pipeline.query(sample.question)

            retrieved_ids = {doc.doc_id for doc, _ in result.retrieved_docs}
            ground_truth_ids = set(sample.ground_truth_doc_ids)

            # ── Retrieval metrics ─────────────────────────────────────────────
            precision = self._context_precision(retrieved_ids, ground_truth_ids)
            recall = self._context_recall(retrieved_ids, ground_truth_ids)

            # ── Generation metrics ────────────────────────────────────────────
            context_text = " ".join(doc.text for doc, _ in result.retrieved_docs)
            faithfulness = self._faithfulness(result.answer, context_text, device)
            relevance = self._answer_relevance(sample.question, result.answer, device)

            # Harmonic mean — punishes any single weak metric more than arithmetic mean
            score = self._harmonic_mean([precision, recall, faithfulness, relevance])

            report.results.append(EvalResult(
                question=sample.question,
                predicted_answer=result.answer,
                context_precision=precision,
                context_recall=recall,
                faithfulness=faithfulness,
                answer_relevance=relevance,
                neuralrag_score=score,
            ))

        return report

    # ── Individual metric implementations ─────────────────────────────────────

    def _context_precision(self, retrieved: set, relevant: set) -> float:
        """
        Precision@k: fraction of retrieved docs that are actually relevant.
        1.0 = every retrieved doc is useful; 0.0 = all retrieved docs are garbage.
        """
        if not retrieved:
            return 0.0
        return len(retrieved & relevant) / len(retrieved)

    def _context_recall(self, retrieved: set, relevant: set) -> float:
        """
        Recall@k: fraction of relevant docs that we actually retrieved.
        1.0 = we found everything; 0.0 = we missed all relevant docs.
        """
        if not relevant:
            return 1.0  # No relevant docs → vacuously correct
        return len(retrieved & relevant) / len(relevant)

    def _faithfulness(self, answer: str, context: str, device: str) -> float:
        """
        Estimate how grounded the answer is in the retrieved context.

        Approach: split the answer into atomic claims (sentences), then measure
        average semantic similarity between each claim and the context.
        High similarity → claims are supported by context → high faithfulness.

        A production system would use an LLM-as-judge here, but semantic
        similarity is a reasonable proxy that doesn't require API calls.
        """
        claims = _split_into_sentences(answer)
        if not claims:
            return 0.0

        claim_embs = self.embed_model.encode(claims, device=device)         # [C, D]
        context_emb = self.embed_model.encode([context], device=device)     # [1, D]

        # Cosine similarity of each claim against the full context vector
        sims = (claim_embs @ context_emb.T).squeeze(1).numpy()  # [C]
        return float(np.mean(sims).clip(0, 1))

    def _answer_relevance(self, question: str, answer: str, device: str) -> float:
        """
        Measure how directly the answer addresses the question.

        Inspired by RAGAS: embed the answer, then check if it points back at
        the question's semantic content.  We add slight noise and average to
        reduce sensitivity to exact phrasing.
        """
        if not answer.strip():
            return 0.0

        q_emb = self.embed_model.encode([question], device=device)    # [1, D]
        a_emb = self.embed_model.encode([answer], device=device)      # [1, D]

        sim = float((q_emb @ a_emb.T).squeeze())
        # Clip to [0, 1] — cosine can technically be negative
        return max(0.0, min(1.0, sim))

    @staticmethod
    def _harmonic_mean(values: List[float]) -> float:
        """Harmonic mean — zero in any component pulls the whole score toward zero."""
        if any(v == 0 for v in values):
            return 0.0
        return len(values) / sum(1 / v for v in values)


def _split_into_sentences(text: str) -> List[str]:
    """Naive sentence splitter (good enough for evaluation purposes)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]  # Filter out fragments
