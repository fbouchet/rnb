"""
High-level memory store interface.

Provides convenient access to M.U, M.S, and M.T memories.
"""

import uuid
from typing import Any

from .backend import ChromaBackend
from .types import MemoryType, SessionMemory, TaskMemory, UserMemory


class MemoryStore:
    """
    High-level interface for RnB memory system.

    Provides methods for:
    - Adding memories (user, session, task)
    - Searching memories semantically
    - Retrieving memories by type
    - Managing memory lifecycle

    Example:
        store = MemoryStore()

        # Add user preference
        store.add_user_memory(
            agent_id="tutor_001",
            content="User prefers Python over Java",
            metadata={"category": "preference"}
        )

        # Search relevant memories
        memories = store.search_memories(
            agent_id="tutor_001",
            query="programming language preference",
            memory_type=MemoryType.USER
        )
    """

    def __init__(
        self,
        backend: ChromaBackend | None = None,
        persist_directory: str = "./data/chroma",
    ):
        """
        Initialize memory store.

        Args:
            backend: Optional ChromaDB backend instance
            persist_directory: Directory for ChromaDB persistence
        """
        self.backend = backend or ChromaBackend(persist_directory=persist_directory)

    def add_user_memory(
        self, agent_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> UserMemory:
        """
        Add memory to M.U (User Model).

        Args:
            agent_id: Agent identifier
            content: Memory content
            metadata: Additional metadata

        Returns:
            Created UserMemory
        """
        memory_id = f"user_{uuid.uuid4().hex[:8]}"
        metadata = metadata or {}
        metadata["type"] = MemoryType.USER.value

        self.backend.add_memory(
            agent_id=agent_id, memory_id=memory_id, content=content, metadata=metadata
        )

        return UserMemory(id=memory_id, content=content, metadata=metadata)

    def add_session_memory(
        self, agent_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> SessionMemory:
        """
        Add memory to M.S (Session Model).

        Args:
            agent_id: Agent identifier
            content: Memory content
            metadata: Additional metadata (e.g., turn, session_id)

        Returns:
            Created SessionMemory
        """
        memory_id = f"session_{uuid.uuid4().hex[:8]}"
        metadata = metadata or {}
        metadata["type"] = MemoryType.SESSION.value

        self.backend.add_memory(
            agent_id=agent_id, memory_id=memory_id, content=content, metadata=metadata
        )

        return SessionMemory(id=memory_id, content=content, metadata=metadata)

    def add_task_memory(
        self, agent_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> TaskMemory:
        """
        Add memory to M.T (Task Model).

        Args:
            agent_id: Agent identifier
            content: Memory content
            metadata: Additional metadata (e.g., status, steps)

        Returns:
            Created TaskMemory
        """
        memory_id = f"task_{uuid.uuid4().hex[:8]}"
        metadata = metadata or {}
        metadata["type"] = MemoryType.TASK.value

        self.backend.add_memory(
            agent_id=agent_id, memory_id=memory_id, content=content, metadata=metadata
        )

        return TaskMemory(id=memory_id, content=content, metadata=metadata)

    def search_memories(
        self,
        agent_id: str,
        query: str,
        memory_type: MemoryType | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search memories semantically.

        Args:
            agent_id: Agent identifier
            query: Search query
            memory_type: Optional filter by memory type
            n_results: Number of results to return

        Returns:
            List of matching memories
        """
        where = None
        if memory_type:
            where = {"type": memory_type.value}

        return self.backend.search_memories(
            agent_id=agent_id, query=query, n_results=n_results, where=where
        )

    def get_user_memories(
        self, agent_id: str, query: str | None = None, n_results: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get M.U (User Model) memories.

        Args:
            agent_id: Agent identifier
            query: Optional semantic search query
            n_results: Number of results

        Returns:
            List of user memories
        """
        if query:
            return self.search_memories(
                agent_id=agent_id,
                query=query,
                memory_type=MemoryType.USER,
                n_results=n_results,
            )
        else:
            # Return all user memories (not semantic search)
            # Note: ChromaDB doesn't have direct "get all" without query
            # Using empty query as fallback
            return self.backend.search_memories(
                agent_id=agent_id,
                query="",
                n_results=n_results,
                where={"type": MemoryType.USER.value},
            )

    def get_session_memories(
        self, agent_id: str, session_id: str | None = None, n_results: int = 20
    ) -> list[dict[str, Any]]:
        """
        Get M.S (Session Model) memories.

        Args:
            agent_id: Agent identifier
            session_id: Optional filter by session ID
            n_results: Number of results

        Returns:
            List of session memories
        """
        # ChromaDB requires $and for multiple filters
        if session_id:
            where = {
                "$and": [{"type": MemoryType.SESSION.value}, {"session_id": session_id}]
            }
        else:
            where = {"type": MemoryType.SESSION.value}

        return self.backend.search_memories(
            agent_id=agent_id,
            query="",  # Empty query to get all
            n_results=n_results,
            where=where,
        )

    def get_task_memories(
        self, agent_id: str, status: str | None = None, n_results: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get M.T (Task Model) memories.

        Args:
            agent_id: Agent identifier
            status: Optional filter by task status
            n_results: Number of results

        Returns:
            List of task memories
        """
        # ChromaDB requires $and for multiple filters
        if status:
            where = {"$and": [{"type": MemoryType.TASK.value}, {"status": status}]}
        else:
            where = {"type": MemoryType.TASK.value}

        return self.backend.search_memories(
            agent_id=agent_id, query="", n_results=n_results, where=where
        )

    def delete_memory(self, agent_id: str, memory_id: str):
        """Delete specific memory"""
        self.backend.delete_memory(agent_id, memory_id)

    def clear_memories(self, agent_id: str, memory_type: MemoryType | None = None):
        """
        Clear memories for agent.

        Args:
            agent_id: Agent identifier
            memory_type: Optional filter by memory type
        """
        where = None
        if memory_type:
            where = {"type": memory_type.value}

        self.backend.clear_memories(agent_id, where=where)

    def count_memories(
        self, agent_id: str, memory_type: MemoryType | None = None
    ) -> int:
        """
        Count memories for agent.

        Args:
            agent_id: Agent identifier
            memory_type: Optional filter by memory type

        Returns:
            Number of memories
        """
        where = None
        if memory_type:
            where = {"type": memory_type.value}

        return self.backend.count_memories(agent_id, where=where)

    def close(self):
        """Close store and backend"""
        self.backend.close()
