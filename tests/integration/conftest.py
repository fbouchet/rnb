"""Pytest fixtures for integration tests"""

import pytest
from rnb.personality.backend import RedisBackend
from rnb.personality.store import PersonalityStateStore
from rnb.personality.manager import AgentManager
from rnb.influence.registry import OperatorRegistry
from rnb.llm import LLMClient, ModelProvider


@pytest.fixture(scope="session")
def llm_client():
    """
    Local LLM client for integration tests.
    
    Uses llama3.2:3b via Ollama (must be running).
    Session-scoped to reuse across tests.
    """
    return LLMClient(
        provider=ModelProvider.LOCAL,
        model_name="llama3.2:3b",
        timeout=120.0  # Longer timeout for local models
    )


@pytest.fixture(scope="function")
def redis_backend():
    """Redis backend for each test (clean state)"""
    backend = RedisBackend()
    yield backend
    # Cleanup after test
    backend.flush_db()
    backend.close()


@pytest.fixture
def personality_store(redis_backend):
    """Personality store with clean Redis backend"""
    return PersonalityStateStore(redis_backend)


@pytest.fixture
def agent_manager(personality_store):
    """Agent manager for test"""
    return AgentManager(personality_store)


@pytest.fixture
def operator_registry():
    """Empty operator registry for each test"""
    return OperatorRegistry()


@pytest.fixture
def base_test_prompt():
    """Standard prompt for consistency testing"""
    return "Explain how recursion works in programming."