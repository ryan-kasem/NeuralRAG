"""
main.py — Demo runner for NeuralRAG.

This script shows the full system working end-to-end without needing an API key:
  1. Builds a small in-memory corpus about ML topics
  2. Indexes all documents into FAISS
  3. Runs a few example queries and prints the retrieved context + answer
  4. Runs the evaluation suite and prints a metrics report

Run:
    python main.py
    python main.py --query "What is the transformer architecture?"
    python main.py --eval    # Run evaluation suite
"""

import argparse
from models.embedding_model import EmbeddingModel
from models.retriever import FAISSRetriever
from pipeline.rag_pipeline import RAGPipeline
from data.document_processor import load_from_strings
from evaluation.metrics import RAGEvaluator, EvalSample

# ── Demo corpus: ML topics (swap this out for any domain) ────────────────────

DEMO_DOCUMENTS = [
    (
        "The Transformer architecture was introduced in 'Attention Is All You Need' (Vaswani et al., 2017). "
        "It relies entirely on self-attention mechanisms, discarding recurrence and convolutions. "
        "The key innovation is multi-head attention, which allows the model to jointly attend to "
        "information from different representation subspaces at different positions. "
        "Transformers have become the foundation of modern NLP and are the basis for BERT, GPT, and T5.",
        "transformers_overview"
    ),
    (
        "Retrieval-Augmented Generation (RAG) was proposed by Lewis et al. (2020) to address the "
        "knowledge limitations of parametric language models. RAG combines a parametric memory "
        "(a pre-trained seq2seq model) with a non-parametric memory (a dense vector index of Wikipedia). "
        "The retriever fetches relevant documents given a query, and the generator conditions on both "
        "the query and retrieved documents to produce an answer. This allows the model to access "
        "up-to-date information without retraining.",
        "rag_paper"
    ),
    (
        "Contrastive learning trains embeddings by pulling similar examples together and pushing "
        "dissimilar ones apart in the representation space. SimCSE (Gao et al., 2021) applies this "
        "to sentence embeddings using dropout as minimal data augmentation: the same sentence is fed "
        "twice with different dropout masks to create positive pairs, while other sentences in the "
        "batch serve as negatives. SimCSE achieves state-of-the-art performance on semantic textual "
        "similarity benchmarks while being simple to implement.",
        "simcse_paper"
    ),
    (
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over "
        "dense vectors. It supports exact search (IndexFlatIP) and approximate nearest neighbor search "
        "(IndexIVFFlat, IndexHNSW) with different speed/accuracy trade-offs. FAISS can search billions "
        "of vectors in milliseconds by using quantization and inverted file structures. "
        "It runs on both CPU and GPU.",
        "faiss_overview"
    ),
    (
        "Fine-tuning is the process of taking a pre-trained model and continuing training on a "
        "task-specific dataset. For language models, this typically uses a much lower learning rate "
        "(2e-5 to 5e-5) than pre-training, a linear warmup schedule, and a small number of epochs "
        "(2-5). Common fine-tuning objectives include cross-entropy for classification and contrastive "
        "loss for embedding tasks. AdamW optimizer is preferred over Adam because it correctly "
        "decouples weight decay from the gradient update.",
        "finetuning_guide"
    ),
    (
        "Maximal Marginal Relevance (MMR) was introduced by Carbonell and Goldstein (1998) for "
        "document summarization and has since been applied to information retrieval and RAG. "
        "MMR balances relevance to a query against redundancy with already-selected documents. "
        "The MMR score for a document is: λ * sim(query, doc) - (1-λ) * max(sim(doc, selected)). "
        "Setting λ=0.6 typically provides a good balance between relevance and diversity.",
        "mmr_paper"
    ),
    (
        "BERT (Bidirectional Encoder Representations from Transformers) was introduced by Devlin et al. "
        "(2018). It is pre-trained using two objectives: Masked Language Modeling (MLM), where 15% of "
        "tokens are masked and the model predicts them; and Next Sentence Prediction (NSP), where the "
        "model predicts whether two sentences are consecutive. BERT's bidirectional attention allows "
        "each token to attend to all other tokens in both directions, unlike GPT which is autoregressive.",
        "bert_overview"
    ),
    (
        "Evaluation of RAG systems requires measuring both retrieval quality and generation quality. "
        "RAGAS (Es et al., 2023) proposes four metrics: Context Precision (fraction of retrieved "
        "chunks that are relevant), Context Recall (fraction of relevant chunks that were retrieved), "
        "Faithfulness (whether the answer is supported by the context), and Answer Relevance "
        "(whether the answer addresses the question). These metrics can be computed with an LLM judge "
        "or using semantic similarity as a proxy.",
        "ragas_eval"
    ),
]


def build_pipeline(device: str = "cpu") -> RAGPipeline:
    """Initialize the full pipeline with our demo corpus."""
    print("[Demo] Initializing embedding model...")
    # Use the base model (no fine-tuning checkpoint required for the demo)
    embed_model = EmbeddingModel()
    embed_model.eval()

    retriever = FAISSRetriever(embed_model, device=device)

    # Pipeline with HyDE and reranker disabled in demo mode (no API key needed)
    pipeline = RAGPipeline(
        embed_model=embed_model,
        retriever=retriever,
        use_hyde=False,   # Set to True once OPENAI_API_KEY is set
        use_reranker=True,
        device=device,
    )

    print("[Demo] Indexing demo corpus...")
    texts, names = zip(*DEMO_DOCUMENTS)
    docs = load_from_strings(list(texts), list(names))
    pipeline.index_documents(docs)

    return pipeline


def run_demo_queries(pipeline: RAGPipeline) -> None:
    """Run a few showcase queries and pretty-print the results."""
    queries = [
        "What is the Transformer architecture and why is it important?",
        "How does contrastive learning improve sentence embeddings?",
        "What metrics should I use to evaluate a RAG system?",
        "How does FAISS handle large-scale vector search?",
    ]

    print("\n" + "=" * 60)
    print("              NeuralRAG Demo")
    print("=" * 60)

    for q in queries:
        print(f"\nQ: {q}")
        print("-" * 60)
        result = pipeline.query(q)
        print(f"A: {result.answer}")
        print(f"\nSources retrieved ({len(result.retrieved_docs)}):")
        for doc, score in result.retrieved_docs:
            src = doc.metadata.get("source", "?")
            print(f"  [{score:.3f}] {src}: {doc.text[:80]}...")


def run_evaluation(pipeline: RAGPipeline) -> None:
    """Evaluate retrieval quality on a small labeled test set."""
    # Look up doc IDs by source name so they match what FAISS actually assigned
    def ids_for_source(source_name: str):
        return [d.doc_id for d in pipeline.retriever.documents if d.metadata.get("source") == source_name]

    test_samples = [
        EvalSample(
            question="What is RAG and who proposed it?",
            ground_truth_answer="RAG was proposed by Lewis et al. (2020). It combines a retriever with a generative model.",
            ground_truth_doc_ids=ids_for_source("rag_paper"),
        ),
        EvalSample(
            question="What is SimCSE?",
            ground_truth_answer="SimCSE is a contrastive learning method for sentence embeddings using dropout augmentation.",
            ground_truth_doc_ids=ids_for_source("simcse_paper"),
        ),
        EvalSample(
            question="What does MMR stand for and what does it do?",
            ground_truth_answer="MMR stands for Maximal Marginal Relevance. It balances relevance and diversity in retrieval.",
            ground_truth_doc_ids=ids_for_source("mmr_paper"),
        ),
    ]

    evaluator = RAGEvaluator(pipeline.embed_model)
    print("\n[Demo] Running evaluation suite...")
    report = evaluator.evaluate(pipeline, test_samples)
    report.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralRAG demo")
    parser.add_argument("--query", type=str, default=None, help="Run a single custom query")
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite")
    parser.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    pipeline = build_pipeline(device=args.device)

    if args.query:
        result = pipeline.query(args.query)
        print(f"\nQ: {args.query}")
        print(f"A: {result.answer}")
    elif args.eval:
        run_evaluation(pipeline)
    else:
        run_demo_queries(pipeline)
        run_evaluation(pipeline)
