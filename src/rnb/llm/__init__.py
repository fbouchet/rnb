"""
LLM Integration - Model-Agnostic Client

This module provides a unified interface for interacting with various LLM providers:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3.5 Sonnet, etc.)
- Local models via Ollama (Llama 3.2, Mistral, etc.)

Uses Instructor for structured outputs and consistent error handling.
"""

from .client import LLMClient, ModelProvider
from .exceptions import (
    ContextLengthExceededError,
    LLMException,
    ModelNotAvailableError,
    RateLimitError,
)

__all__ = [
    "LLMClient",
    "ModelProvider",
    "LLMException",
    "ModelNotAvailableError",
    "ContextLengthExceededError",
    "RateLimitError",
]
