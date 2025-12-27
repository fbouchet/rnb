"""
Memory System - RnB Model M Extensions

This module implements vector-based memory storage extending RnB Model M.A
to include M.U (User Model), M.S (Session Model), and M.T (Task Model).

From RnB papers (Bouchet & Sansonnet):
"Model M encompasses:
- M.A: Agent mental model (personality state) [Already implemented]
- M.U: User model (agent's knowledge about the user)
- M.S: Session model (conversation history, context)
- M.T: Task model (current task state, goals, progress)"

Memory Components:
- ChromaBackend: Vector database for semantic memory storage
- MemoryStore: High-level interface for storing/retrieving memories
- UserMemory: M.U implementation - user preferences, traits, history
- SessionMemory: M.S implementation - conversation context
- TaskMemory: M.T implementation - task state and goals

Reference: Bouchet & Sansonnet (2009-2013), RnB Model M
"""

from .backend import ChromaBackend
from .store import MemoryStore
from .types import Memory, MemoryType, SessionMemory, TaskMemory, UserMemory

__all__ = [
    "ChromaBackend",
    "MemoryStore",
    "MemoryType",
    "Memory",
    "UserMemory",
    "SessionMemory",
    "TaskMemory",
]
