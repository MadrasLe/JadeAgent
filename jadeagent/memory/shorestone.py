"""
ShoreStone v2 — Persistent vector memory with RFR-Score curation.

Evolution of JADE's ShoreStone memory system:
- ChromaDB-backed persistent vector store
- Semantic search via sentence embeddings
- RFR-Score curation (Relevance, Frequency, Recency)
- Automatic maintenance (prune low-quality memories)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import BaseMemory

logger = logging.getLogger("jadeagent.memory.shorestone")


class ShoreStoneMemory(BaseMemory):
    """
    Persistent vector memory backed by ChromaDB.

    Uses sentence-transformers for embeddings and ChromaDB for
    persistent storage + semantic similarity search.

    RFR-Score: Each memory has a relevance score based on:
    - Relevance: cosine similarity to queries
    - Frequency: how often it's been retrieved
    - Recency: when it was last accessed

    Example:
        memory = ShoreStoneMemory(collection="my_agent")
        memory.memorize("User prefers Python for ML projects")
        memory.memorize("Project uses PyTorch and custom CUDA kernels")

        results = memory.remember("what framework does the user like?")
        # → ["User prefers Python for ML projects"]
    """

    def __init__(
        self,
        collection: str = "jade_default",
        persist_dir: str = "./.shorestone",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ShoreStone requires chromadb: pip install chromadb"
            )

        self._persist_dir = persist_dir
        self._embedding_model_name = embedding_model
        self._embedder = None  # Lazy load

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

        # RFR tracking
        self._access_count: dict[str, int] = {}
        self._last_access: dict[str, float] = {}

    def _get_embedder(self):
        """Lazy-load sentence transformer."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "ShoreStone requires sentence-transformers: "
                    "pip install sentence-transformers"
                )
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence transformer."""
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def remember(self, query: str, k: int = 5) -> list[str]:
        """
        Retrieve semantically similar memories.

        Args:
            query: Natural language search query.
            k: Number of memories to retrieve.

        Returns:
            List of relevant memory strings, ordered by relevance.
        """
        if self._collection.count() == 0:
            return []

        # Embed query
        query_embedding = self._embed([query])[0]

        # Search ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
        )

        if not results or not results["documents"]:
            return []

        memories = results["documents"][0]
        ids = results["ids"][0]

        # Update RFR tracking
        now = time.time()
        for mem_id in ids:
            self._access_count[mem_id] = self._access_count.get(mem_id, 0) + 1
            self._last_access[mem_id] = now

        return memories

    def memorize(self, content: str, metadata: dict | None = None):
        """
        Store a new memory with embedding.

        Args:
            content: Text content to memorize.
            metadata: Optional metadata dict.
        """
        mem_id = f"mem_{int(time.time() * 1000)}_{self._collection.count()}"
        embedding = self._embed([content])[0]

        meta = metadata or {}
        meta["created_at"] = time.time()
        meta["rfr_score"] = 1.0  # Initial score

        self._collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )

        logger.debug(f"Memorized: {content[:50]}... (id={mem_id})")

    def run_maintenance(self, min_score: float = 0.3):
        """
        RFR-Score curation: prune low-quality memories.

        Score = relevance_weight * frequency + recency_weight * recency
        Memories below min_score are removed.
        """
        if self._collection.count() == 0:
            return

        all_data = self._collection.get(include=["metadatas"])
        ids_to_remove = []
        now = time.time()

        for mem_id in all_data["ids"]:
            freq = self._access_count.get(mem_id, 0)
            last = self._last_access.get(mem_id, 0)

            # Recency score: exponential decay (half-life = 24 hours)
            if last > 0:
                hours_since = (now - last) / 3600
                recency = 2 ** (-hours_since / 24)
            else:
                recency = 0.1  # Never accessed

            # Combined RFR score
            rfr = 0.4 * min(freq / 10, 1.0) + 0.6 * recency

            if rfr < min_score:
                ids_to_remove.append(mem_id)

        if ids_to_remove:
            self._collection.delete(ids=ids_to_remove)
            logger.info(f"Maintenance: pruned {len(ids_to_remove)} low-quality memories")

    def clear(self):
        """Clear all memories."""
        # ChromaDB doesn't have a clear method, so we delete and recreate
        name = self._collection.name
        meta = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata=meta,
        )
        self._access_count.clear()
        self._last_access.clear()

    @property
    def size(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return f"<ShoreStoneMemory(collection='{self._collection.name}', size={self.size})>"
