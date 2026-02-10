"""
Mnemosyne -- Vector Store
Handles all ChromaDB interactions: adding documents, querying,
and resetting the collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chromadb
from rich.console import Console

from src.core.config import get_config
from src.core.ingester import DocumentChunk

console = Console()


@dataclass
class QueryResult:
    """A single result returned from a similarity search."""
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Client / Collection helpers  (cached singletons)
# ─────────────────────────────────────────────────────────────────────────

_client_cache: dict[str, chromadb.ClientAPI] = {}


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a cached persistent ChromaDB client (one per db path)."""
    config = get_config()
    db_path = config.chroma_db_path
    if db_path not in _client_cache:
        _client_cache[db_path] = chromadb.PersistentClient(path=db_path)
    return _client_cache[db_path]


def get_or_create_collection(
    client: chromadb.ClientAPI | None = None,
) -> chromadb.Collection:
    """Get (or create) the Mnemosyne collection with cosine similarity."""
    config = get_config()
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 128


def add_documents(chunks: list[DocumentChunk]) -> int:
    """
    Upsert *chunks* into ChromaDB in batches.

    Returns the number of chunks written.
    """
    if not chunks:
        return 0

    collection = get_or_create_collection()
    total = 0

    for start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[start : start + _BATCH_SIZE]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.content for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        total += len(batch)

    return total


# ─────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────

def query(
    text: str,
    n_results: int = 8,
    filter_meta: dict[str, Any] | None = None,
) -> list[QueryResult]:
    """
    Run a similarity search against the collection.

    Returns up to *n_results* ``QueryResult`` objects sorted by relevance.
    """
    collection = get_or_create_collection()

    if collection.count() == 0:
        console.print(
            "[yellow][!][/yellow] The knowledge base is empty. "
            "Run [green]mnemosyne ingest <path>[/green] first."
        )
        return []

    try:
        results = collection.query(
            query_texts=[text],
            n_results=min(n_results, collection.count()),
            where=filter_meta,
        )
    except Exception as exc:
        console.print(f"[red][-][/red] Query failed: {exc}")
        return []

    items: list[QueryResult] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        items.append(
            QueryResult(
                content=doc,
                source=meta.get("source", "unknown"),
                score=round(1 - dist, 4),     # cosine -> similarity
                metadata=meta,
            )
        )

    return items


# ─────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────

def reset_collection() -> None:
    """Delete and recreate the collection (the 'forget' operation)."""
    config = get_config()
    client = get_chroma_client()
    try:
        client.delete_collection(name=config.collection_name)
        console.print("[green][+][/green] Collection wiped successfully.")
    except Exception:
        console.print("[yellow][!][/yellow] No collection to delete -- already clean.")
