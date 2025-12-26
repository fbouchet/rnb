"""Unit tests for LLM client"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from rnb.llm.client import LLMClient, ModelProvider
from rnb.llm.exceptions import (
    ModelNotAvailableError,
    RateLimitError,
    LLMException
)


# ===== Mock Response Models =====

class MockChatCompletion:
    """Mock OpenAI chat completion response"""
    def __init__(self, content: str):
        self.choices = [Mock(message=Mock(content=content))]


class MockAnthropicMessage:
    """Mock Anthropic message response"""
    def __init__(self, text: str):
        self.content = [Mock(text=text)]


class SampleResponse(BaseModel):
    """Sample Pydantic model for structured outputs"""
    answer: str
    confidence: float

# ===== Fixtures =====

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    with patch('rnb.llm.client.OpenAI') as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    with patch('rnb.llm.client.Anthropic') as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_instructor():
    """Mock instructor wrapper"""
    with patch('rnb.llm.client.instructor') as mock:
        yield mock


# ===== Initialization Tests =====

def test_openai_client_initialization(mock_openai_client, mock_instructor):
    """Test OpenAI client initialization"""
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    assert client.provider == ModelProvider.OPENAI
    assert client.model_name == "gpt-4"


def test_anthropic_client_initialization(mock_anthropic_client, mock_instructor):
    """Test Anthropic client initialization"""
    client = LLMClient(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        api_key="test-key"
    )
    
    assert client.provider == ModelProvider.ANTHROPIC
    assert client.model_name == "claude-3-5-sonnet-20241022"


def test_local_client_initialization(mock_openai_client, mock_instructor):
    """Test local (Ollama) client initialization"""
    client = LLMClient(
        provider=ModelProvider.LOCAL,
        model_name="llama3.2:3b"
    )
    
    assert client.provider == ModelProvider.LOCAL
    assert client.model_name == "llama3.2:3b"


def test_local_client_custom_base_url(mock_openai_client, mock_instructor):
    """Test local client with custom base URL"""
    with patch('rnb.llm.client.OpenAI') as mock_openai:
        client = LLMClient(
            provider=ModelProvider.LOCAL,
            model_name="llama3.2:3b",
            base_url="http://custom:8000/v1"
        )
        
        # Verify OpenAI was called with custom base_url
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs['base_url'] == "http://custom:8000/v1"


def test_unsupported_provider_raises_error():
    """Test that unsupported provider raises error"""
    with pytest.raises(ModelNotAvailableError):
        LLMClient(
            provider="unsupported",  # Invalid provider
            model_name="test"
        )


# ===== Query Tests =====

def test_query_openai(mock_openai_client, mock_instructor):
    """Test simple query with OpenAI"""
    # Setup mock
    mock_instructor.from_openai.return_value.chat.completions.create.return_value = \
        MockChatCompletion("This is the response.")
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    response = client.query("Test prompt")
    
    assert response == "This is the response."


def test_query_with_system_prompt(mock_openai_client, mock_instructor):
    """Test query with system prompt"""
    mock_create = Mock(return_value=MockChatCompletion("Response"))
    mock_instructor.from_openai.return_value.chat.completions.create = mock_create
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    client.query(
        "User prompt",
        system_prompt="You are a helpful assistant."
    )
    
    # Verify system prompt was included
    call_args = mock_create.call_args
    messages = call_args[1]['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'


def test_query_with_temperature(mock_openai_client, mock_instructor):
    """Test query with custom temperature"""
    mock_create = Mock(return_value=MockChatCompletion("Response"))
    mock_instructor.from_openai.return_value.chat.completions.create = mock_create
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    client.query("Test", temperature=0.3)
    
    call_args = mock_create.call_args
    assert call_args[1]['temperature'] == 0.3


def test_query_with_max_tokens(mock_openai_client, mock_instructor):
    """Test query with max tokens"""
    mock_create = Mock(return_value=MockChatCompletion("Response"))
    mock_instructor.from_openai.return_value.chat.completions.create = mock_create
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    client.query("Test", max_tokens=500)
    
    call_args = mock_create.call_args
    assert call_args[1]['max_tokens'] == 500


# ===== Structured Query Tests =====

def test_query_structured(mock_openai_client, mock_instructor):
    """Test structured query with Pydantic model"""
    # Setup mock to return SampleResponse instance
    mock_response = SampleResponse(answer="Structured answer", confidence=0.95)
    mock_instructor.from_openai.return_value.chat.completions.create.return_value = mock_response
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    result = client.query_structured(
        "Test prompt",
        response_model=SampleResponse
    )
    
    assert isinstance(result, SampleResponse)
    assert result.answer == "Structured answer"
    assert result.confidence == 0.95


# ===== Error Handling Tests =====

def test_rate_limit_error_handling(mock_openai_client, mock_instructor):
    """Test rate limit error is properly caught and re-raised"""
    from openai import RateLimitError as OpenAIRateLimitError
    
    mock_instructor.from_openai.return_value.chat.completions.create.side_effect = \
        OpenAIRateLimitError("Rate limit exceeded", response=Mock(), body={})
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    with pytest.raises(RateLimitError) as exc_info:
        client.query("Test")
    
    assert "openai" in str(exc_info.value).lower()


def test_api_error_handling(mock_openai_client, mock_instructor):
    """Test generic API errors are caught"""
    from openai import APIError
    
    mock_instructor.from_openai.return_value.chat.completions.create.side_effect = \
        APIError("API error", request=Mock(), body={})
    
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    with pytest.raises(LLMException):
        client.query("Test")


# ===== Representation Tests =====

def test_client_repr(mock_openai_client, mock_instructor):
    """Test client string representation"""
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key"
    )
    
    repr_str = repr(client)
    assert "LLMClient" in repr_str
    assert "openai" in repr_str
    assert "gpt-4" in repr_str


# ===== Provider-Specific Tests =====

def test_anthropic_query_format(mock_anthropic_client, mock_instructor):
    """Test Anthropic uses correct message format"""
    mock_create = Mock(return_value=MockAnthropicMessage("Response"))
    mock_instructor.from_anthropic.return_value.messages.create = mock_create
    
    client = LLMClient(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        api_key="test-key"
    )
    
    client.query(
        "User prompt",
        system_prompt="System instruction"
    )
    
    # Verify Anthropic-specific format
    call_args = mock_create.call_args[1]
    assert 'system' in call_args
    assert 'messages' in call_args
    assert call_args['system'] == "System instruction"