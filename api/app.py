"""
api/app.py — FastAPI REST endpoint for NeuralRAG.

Exposes two endpoints:
  POST /query       — Ask a question, get a RAG-generated answer
  POST /index       — Add new documents to the live index (no restart required)
  GET  /health      — Uptime check for load balancers / k8s probes

Rate limiting, request validation, and structured error responses are included
because production ML systems live and die by their API quality.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.rag_pipeline import RAGPipeline
from data.document_processor import load_from_strings
from models.embedding_model import EmbeddingModel
from models.retriever import FAISSRetriever
from config import cfg


# ── Global state (loaded once at startup) ────────────────────────────────────

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: runs once on startup, once on shutdown.
    We load the model here so the first request doesn't pay the load latency.
    """
    global pipeline
    print("[API] Loading NeuralRAG pipeline...")
    embed_model = EmbeddingModel()
    retriever = FAISSRetriever(embed_model)
    pipeline = RAGPipeline(embed_model, retriever)
    print("[API] Pipeline ready.")
    yield  # Server runs here
    print("[API] Shutting down.")


app = FastAPI(
    title="NeuralRAG API",
    description="Production-ready Retrieval-Augmented Generation with custom embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any origin — lock this down in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Rate limiter (simple in-memory sliding window) ────────────────────────────

# Maps IP → list of request timestamps
_rate_limit_store: dict = defaultdict(list)


def check_rate_limit(request: Request, limit: int = cfg.api.rate_limit) -> None:
    """
    Sliding window rate limiter: max `limit` requests per 60 seconds per IP.

    In production you'd use Redis for this so it works across multiple workers.
    This in-process version is fine for a single-node deployment.
    """
    ip = request.client.host
    now = time.time()
    window_start = now - 60.0

    # Evict requests older than the window
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > window_start]

    if len(_rate_limit_store[ip]) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {limit} requests/min",
        )

    _rate_limit_store[ip].append(now)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="What is machine learning?")
    use_hyde: Optional[bool] = Field(None, description="Override HyDE setting for this request")
    use_reranker: Optional[bool] = Field(None, description="Override reranker setting for this request")


class SourceDoc(BaseModel):
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: List[SourceDoc]
    latency_ms: float
    tokens_in_context: int


class IndexRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    source_names: Optional[List[str]] = None


class IndexResponse(BaseModel):
    indexed: int
    total_in_index: int


class HealthResponse(BaseModel):
    status: str
    index_size: int
    version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness + readiness probe in one endpoint."""
    return HealthResponse(
        status="ok",
        index_size=pipeline.retriever.index.ntotal if pipeline else 0,
        version=app.version,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(req: QueryRequest, request: Request):
    """
    Ask a question. Returns a grounded answer + source documents.

    The pipeline:
      1. Optionally expands the query via HyDE
      2. Retrieves top-k documents from the FAISS index
      3. Reranks with a cross-encoder
      4. Generates an answer grounded in the retrieved context
    """
    check_rate_limit(request)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if pipeline.retriever.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. POST documents to /index first.",
        )

    # Allow per-request overrides of pipeline settings
    original_hyde = pipeline.use_hyde
    original_reranker = pipeline.use_reranker
    if req.use_hyde is not None:
        pipeline.use_hyde = req.use_hyde
    if req.use_reranker is not None:
        pipeline.use_reranker = req.use_reranker

    t0 = time.perf_counter()
    try:
        result = pipeline.query(req.question)
    finally:
        # Always restore original settings, even if the request fails
        pipeline.use_hyde = original_hyde
        pipeline.use_reranker = original_reranker

    latency = (time.perf_counter() - t0) * 1000

    sources = [
        SourceDoc(
            text=doc.text[:300] + ("..." if len(doc.text) > 300 else ""),
            source=doc.metadata.get("source", "unknown"),
            score=round(float(score), 4),
        )
        for doc, score in result.retrieved_docs
    ]

    return QueryResponse(
        request_id=str(uuid.uuid4()),
        answer=result.answer,
        sources=sources,
        latency_ms=round(latency, 2),
        tokens_in_context=result.num_tokens_in_context,
    )


@app.post("/index", response_model=IndexResponse, tags=["rag"])
async def index_documents(req: IndexRequest, request: Request):
    """
    Add new documents to the retrieval index at runtime.

    Documents are chunked automatically.  The index updates are immediate —
    new documents are searchable on the very next /query call.
    """
    check_rate_limit(request)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    docs = load_from_strings(req.texts, req.source_names)
    pipeline.index_documents(docs)

    return IndexResponse(
        indexed=len(docs),
        total_in_index=pipeline.retriever.index.ntotal,
    )
