"""Personality state management for RnB agents"""

from .backend import RedisBackend
from .exceptions import (
    AgentAlreadyExistsError,
    AgentNotFoundError,
    InvalidValueError,
    RnBException,
    StorageConnectionError,
)
from .manager import AgentManager
from .state import AffectDimension, MoodDimension, PersonalityState
from .store import PersonalityStateStore
from .taxonomy import Trait

__all__ = [
    # Exceptions
    "RnBException",
    "AgentNotFoundError",
    "AgentAlreadyExistsError",
    "InvalidValueError",
    "StorageConnectionError",
    # Data models
    "PersonalityState",
    "Trait",
    "MoodDimension",
    "AffectDimension",
    # Core classes
    "RedisBackend",
    "PersonalityStateStore",
    "AgentManager",
]
