"""Personality state management for RnB agents"""

from .exceptions import (
    RnBException,
    AgentNotFoundError,
    AgentAlreadyExistsError,
    InvalidValueError,
    StorageConnectionError
)
from .state import (
    PersonalityState,
    MoodDimension,
    AffectDimension
)
from .taxonomy import Trait
from .backend import RedisBackend
from .store import PersonalityStateStore
from .manager import AgentManager

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