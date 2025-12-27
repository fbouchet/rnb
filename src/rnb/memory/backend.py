"""
ChromaDB backend for vector-based memory storage.

Provides semantic search over memories using embeddings.
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings


class ChromaBackend:
    """
    ChromaDB backend for RnB memory system.

    Uses ChromaDB for:
    - Vector embeddings of memory content
    - Semantic similarity search
    - Efficient retrieval of relevant memories

    Collections:
    - memories_{agent_id}: Per-agent memory storage

    Metadata filtering supports:
    - Memory type (user/session/task)
    - Custom metadata fields
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_prefix: str = "memories",
    ):
        """
        Initialize ChromaDB backend.

        Args:
            persist_directory: Directory for persistent storage
            collection_prefix: Prefix for collection names
        """
        self.persist_directory = persist_directory
        self.collection_prefix = collection_prefix

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    def get_collection(self, agent_id: str):
        """
        Get or create collection for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            ChromaDB collection
        """
        collection_name = f"{self.collection_prefix}_{agent_id}"
        return self.client.get_or_create_collection(
            name=collection_name, metadata={"agent_id": agent_id}
        )

    def add_memory(
        self, agent_id: str, memory_id: str, content: str, metadata: dict[str, Any]
    ):
        """
        Add memory to agent's collection.

        Args:
            agent_id: Agent identifier
            memory_id: Unique memory identifier
            content: Text content for embedding
            metadata: Additional metadata for filtering
        """
        collection = self.get_collection(agent_id)

        # ChromaDB requires non-empty metadata
        # Add a default field if metadata is empty
        if not metadata:
            metadata = {"_placeholder": "true"}

        # Add to collection (ChromaDB handles embedding)
        collection.add(ids=[memory_id], documents=[content], metadatas=[metadata])

    def search_memories(
        self,
        agent_id: str,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search for relevant memories.

        Args:
            agent_id: Agent identifier
            query: Search query (will be embedded)
            n_results: Number of results to return
            where: Metadata filters (e.g., {"type": "user"})

        Returns:
            List of matching memories with scores
        """
        collection = self.get_collection(agent_id)

        results = collection.query(
            query_texts=[query], n_results=n_results, where=where
        )

        # Format results
        memories = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                memories.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                    }
                )

        return memories

    def get_memory(self, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        """
        Retrieve specific memory by ID.

        Args:
            agent_id: Agent identifier
            memory_id: Memory identifier

        Returns:
            Memory dict or None if not found
        """
        collection = self.get_collection(agent_id)

        try:
            result = collection.get(ids=[memory_id])

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception:
            return None

        return None

    def delete_memory(self, agent_id: str, memory_id: str):
        """
        Delete specific memory.

        Args:
            agent_id: Agent identifier
            memory_id: Memory identifier
        """
        collection = self.get_collection(agent_id)
        collection.delete(ids=[memory_id])

    def clear_memories(self, agent_id: str, where: dict[str, Any] | None = None):
        """
        Clear memories for agent (optionally filtered).

        Args:
            agent_id: Agent identifier
            where: Optional metadata filter
        """
        if where:
            collection = self.get_collection(agent_id)
            # Get IDs matching filter
            results = collection.get(where=where)
            if results["ids"]:
                collection.delete(ids=results["ids"])
        else:
            # Delete entire collection
            collection_name = f"{self.collection_prefix}_{agent_id}"
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                logging.warning(
                    f"ChromaBackend: Failed to delete collection {collection_name}"
                )
                pass

    def count_memories(self, agent_id: str, where: dict[str, Any] | None = None) -> int:
        """
        Count memories for agent.

        Args:
            agent_id: Agent identifier
            where: Optional metadata filter

        Returns:
            Number of memories
        """
        collection = self.get_collection(agent_id)

        if where:
            results = collection.get(where=where)
            return len(results["ids"])
        else:
            return collection.count()

    def close(self):
        """Close backend connection"""
        # ChromaDB handles cleanup automatically
        pass
