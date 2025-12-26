"""
Memory types for RnB Model M extensions.

Defines structured memory objects for M.U, M.S, and M.T.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    """Memory categories in RnB Model M"""
    USER = "user"        # M.U: User model
    SESSION = "session"  # M.S: Session model
    TASK = "task"        # M.T: Task model


class Memory(BaseModel):
    """
    Base memory structure.
    
    All memories have:
    - id: Unique identifier
    - type: Memory category (user/session/task)
    - content: Text content for semantic search
    - metadata: Additional structured information
    - timestamp: When memory was created
    """
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class UserMemory(Memory):
    """
    M.U: User Model Memory
    
    Stores agent's knowledge about the user:
    - Preferences (likes, dislikes, interests)
    - Observed traits (inferred personality, communication style)
    - Personal facts (background, expertise, context)
    - Interaction patterns (frequency, topics, sentiment)
    
    Example:
        UserMemory(
            id="user_pref_001",
            type=MemoryType.USER,
            content="User prefers concise explanations with examples",
            metadata={
                "category": "preference",
                "confidence": 0.8,
                "observed_interactions": 5
            }
        )
    
    Reference: RnB M.U - enables personalization
    """
    type: MemoryType = Field(default=MemoryType.USER)
    
    def __init__(self, **data):
        if 'type' not in data:
            data['type'] = MemoryType.USER
        super().__init__(**data)


class SessionMemory(Memory):
    """
    M.S: Session Model Memory
    
    Stores conversation history and context:
    - Dialogue turns (user/agent exchanges)
    - Conversational context (topics discussed, references)
    - Session state (objectives achieved, pending questions)
    - Discourse markers (topic shifts, clarifications)
    
    Example:
        SessionMemory(
            id="session_turn_003",
            type=MemoryType.SESSION,
            content="User asked about recursion; agent provided definition and example",
            metadata={
                "turn": 3,
                "user_message": "Can you explain recursion?",
                "agent_response": "Recursion is when...",
                "topics": ["recursion", "programming"],
                "session_id": "conv_20250101_001"
            }
        )
    
    Reference: RnB M.S - enables context-aware responses
    """
    type: MemoryType = Field(default=MemoryType.SESSION)
    
    def __init__(self, **data):
        if 'type' not in data:
            data['type'] = MemoryType.SESSION
        super().__init__(**data)


class TaskMemory(Memory):
    """
    M.T: Task Model Memory
    
    Stores current task state and goals:
    - Task definition (objective, constraints, success criteria)
    - Progress tracking (steps completed, remaining work)
    - Sub-goals (decomposed objectives, dependencies)
    - Resources (relevant information, tools available)
    
    Example:
        TaskMemory(
            id="task_goal_001",
            type=MemoryType.TASK,
            content="Help user implement binary search algorithm in Python",
            metadata={
                "status": "in_progress",
                "steps_completed": ["explain_concept", "provide_pseudocode"],
                "steps_remaining": ["write_code", "test_examples"],
                "success_criteria": ["working_implementation", "user_understanding"]
            }
        )
    
    Reference: RnB M.T - enables goal-directed behavior
    """
    type: MemoryType = Field(default=MemoryType.TASK)
    
    def __init__(self, **data):
        if 'type' not in data:
            data['type'] = MemoryType.TASK
        super().__init__(**data)