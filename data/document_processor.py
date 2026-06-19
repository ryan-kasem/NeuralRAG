"""
data/document_processor.py — Load raw documents and split them into chunks.

RAG systems work best with chunks of ~200–500 tokens.  Too small → not enough
context per chunk.  Too large → dilutes the relevant signal and wastes the LLM's
context window.

We use a sliding-window chunker with overlap so that sentences spanning chunk
boundaries don't get cut in half and lost.
"""

import re
from pathlib import Path
from typing import List, Iterator
from models.retriever import Document


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[str]:
    """
    Split text into overlapping chunks by word count.

    overlap: how many words from the end of one chunk to repeat at the start
    of the next.  Prevents information loss at boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap  # Slide forward with overlap

    return chunks


def load_text_files(directory: str, extensions: tuple = (".txt", ".md")) -> List[Document]:
    """
    Recursively load all text files from a directory and chunk them.

    Each chunk becomes one Document with metadata pointing back to its source file.
    """
    docs = []
    doc_id = 0
    root = Path(directory)

    for ext in extensions:
        for filepath in sorted(root.rglob(f"*{ext}")):
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            # Strip markdown headers/syntax for cleaner embedding
            clean_text = _clean_markdown(text)
            chunks = chunk_text(clean_text)

            for chunk_idx, chunk in enumerate(chunks):
                docs.append(Document(
                    text=chunk,
                    metadata={
                        "source": str(filepath.relative_to(root)),
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    },
                    doc_id=doc_id,
                ))
                doc_id += 1

    print(f"[Processor] Loaded {len(docs)} chunks from {directory}")
    return docs


def load_from_strings(texts: List[str], source_names: List[str] = None) -> List[Document]:
    """
    Convenience function: load from a list of strings (e.g., web-scraped content).
    Used in the demo / notebook.
    """
    if source_names is None:
        source_names = [f"doc_{i}" for i in range(len(texts))]

    docs = []
    doc_id = 0

    for text, name in zip(texts, source_names):
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            docs.append(Document(
                text=chunk,
                metadata={"source": name, "chunk_idx": chunk_idx},
                doc_id=doc_id,
            ))
            doc_id += 1

    return docs


def _clean_markdown(text: str) -> str:
    """Strip common markdown syntax that adds noise to embeddings."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove markdown headers (keep the text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown links — keep display text, drop URL
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
