# NeuralRAG

A production-grade Retrieval-Augmented Generation (RAG) system built from scratch with PyTorch. Fine-tunes sentence embeddings using SimCSE contrastive learning, retrieves with FAISS + cross-encoder reranking, and generates answers via Llama 3.2 running locally.

---

## What it does

Standard RAG systems use off-the-shelf embeddings and basic vector search. NeuralRAG adds:

- **SimCSE fine-tuning** - trains the embedding model with in-batch contrastive loss so it learns domain-specific similarity, not just generic sentence similarity
- **Two-stage retrieval** - FAISS bi-encoder for fast candidate retrieval → cross-encoder reranker for precision scoring
- **MMR diversity filtering** - eliminates redundant chunks from the retrieved context
- **HyDE query expansion** - generates a hypothetical answer to the query and uses *that* as the search vector, bridging the gap between short questions and long document passages
- **RAGAS-style evaluation** - measures context precision, context recall, faithfulness, and answer relevance with a harmonic mean "NeuralRAG Score"

---

## Architecture

```
Query
  │
  ▼
HyDE Expansion (LLM generates hypothetical doc)
  │
  ▼
FAISS Bi-Encoder Retrieval (top-k candidates)
  │
  ▼
Cross-Encoder Reranker (precision scoring)
  │
  ▼
MMR Diversity Filter (remove redundant chunks)
  │
  ▼
Context Assembly (token budget)
  │
  ▼
Llama 3.2 Generation (via Ollama)
  │
  ▼
Answer
```

---

## Stack

| Component | Technology |
|---|---|
| Embeddings | PyTorch + HuggingFace Transformers |
| Vector Search | FAISS (IndexFlatIP) |
| Reranker | sentence-transformers CrossEncoder |
| LLM | Llama 3.2 via Ollama |
| API | FastAPI + Uvicorn |
| Training | AdamW + cosine LR warmup + W&B logging |

---

## Quick Start

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) installed

```bash
# 1. Clone
git clone https://github.com/ryan-kasem/NeuralRAG.git
cd NeuralRAG

# 2. Install deps
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Pull the LLM (free, runs locally)
ollama pull llama3.2
ollama serve  # run in a separate terminal tab

# 4. Run the demo
python main.py
```

---

## Project Structure

```
NeuralRAG/
├── config.py                  # all hyperparameters in one place
├── models/
│   ├── embedding_model.py     # transformer + SimCSE contrastive loss
│   └── retriever.py           # FAISS index + MMR selection
├── pipeline/
│   └── rag_pipeline.py        # HyDE, reranking, generation
├── training/
│   └── train.py               # fine-tuning loop with W&B logging
├── evaluation/
│   └── metrics.py             # RAGAS-style eval framework
├── data/
│   └── document_processor.py  # sliding window chunking
├── api/
│   └── app.py                 # FastAPI endpoints
└── main.py                    # demo + evaluation runner
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Context Precision | Of retrieved chunks, how many were actually relevant? |
| Context Recall | Were all relevant chunks retrieved? |
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevance | Does the answer actually address the question? |
| **NeuralRAG Score** | Harmonic mean of all four |

---

## References

- Lewis et al. (2020) - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Gao et al. (2021) - [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
- Gao et al. (2022) - [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496)
- Es et al. (2023) - [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
