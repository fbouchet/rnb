"""
Model-agnostic LLM client using Instructor for structured outputs.

Supports:
- OpenAI API (GPT-4, GPT-3.5, etc.)
- Anthropic API (Claude 3.5 Sonnet, etc.)
- Local models via Ollama (Llama 3.2, Mistral, etc.)

Example usage:
    # OpenAI
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="your-key"
    )
    
    # Anthropic
    client = LLMClient(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        api_key="your-key"
    )
    
    # Local (Ollama)
    client = LLMClient(
        provider=ModelProvider.LOCAL,
        model_name="llama3.2:3b"
    )
    
    # Simple query
    response = client.query("Explain recursion.")
    
    # Structured query
    class Answer(BaseModel):
        explanation: str
        confidence: float
    
    structured = client.query_structured(
        "Explain recursion.",
        response_model=Answer
    )
"""

import instructor
from openai import OpenAI, APIError, RateLimitError as OpenAIRateLimitError
from anthropic import Anthropic, APIError as AnthropicAPIError
from pydantic import BaseModel
from typing import Optional, Union
from enum import Enum

from .exceptions import (
    ModelNotAvailableError,
    ContextLengthExceededError,
    RateLimitError,
    LLMException
)


class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Ollama or other OpenAI-compatible local servers


class LLMClient:
    """
    Model-agnostic LLM client with structured output support.
    
    Provides unified interface across OpenAI, Anthropic, and local models.
    Uses Instructor library for reliable structured outputs.
    """
    
    def __init__(
        self,
        provider: ModelProvider,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: Model provider (openai, anthropic, local)
            model_name: Specific model identifier
            api_key: API key (not needed for local models)
            base_url: Custom base URL (for local models or proxies)
            timeout: Request timeout in seconds
            
        Raises:
            ModelNotAvailableError: If provider/model combination unavailable
        """
        self.provider = provider
        self.model_name = model_name
        self.timeout = timeout
        
        try:
            if provider == ModelProvider.OPENAI:
                # Keep both raw and instructor-wrapped clients
                self.raw_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout
                )
                self.instructor_client = instructor.from_openai(self.raw_client)
                self._query_method = self._query_openai
                self._query_structured_method = self._query_structured_openai
            
            elif provider == ModelProvider.ANTHROPIC:
                # Keep both raw and instructor-wrapped clients
                self.raw_client = Anthropic(
                    api_key=api_key,
                    timeout=timeout
                )
                self.instructor_client = instructor.from_anthropic(self.raw_client)
                self._query_method = self._query_anthropic
                self._query_structured_method = self._query_structured_anthropic
            
            elif provider == ModelProvider.LOCAL:
                # Ollama or other OpenAI-compatible local servers
                if base_url is None:
                    base_url = "http://localhost:11434/v1"  # Ollama default
                
                self.raw_client = OpenAI(
                    api_key="ollama",  # Placeholder (not used by Ollama)
                    base_url=base_url,
                    timeout=timeout
                )
                self.instructor_client = instructor.from_openai(self.raw_client)
                self._query_method = self._query_openai
                self._query_structured_method = self._query_structured_openai
            
            else:
                raise ModelNotAvailableError(
                    provider=provider,
                    model=model_name,
                    details=f"Unsupported provider: {provider}"
                )
        
        except Exception as e:
            if isinstance(e, ModelNotAvailableError):
                raise
            raise ModelNotAvailableError(
                provider=provider,
                model=model_name,
                details=str(e)
            )
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Simple text completion.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt (instructions)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Model's text response
            
        Raises:
            ContextLengthExceededError: If prompt exceeds context limit
            RateLimitError: If rate limit exceeded
            LLMException: For other API errors
        """
        try:
            return self._query_method(prompt, system_prompt, temperature, max_tokens)
        except OpenAIRateLimitError:
            raise RateLimitError(provider=self.provider.value)
        except (APIError, AnthropicAPIError) as e:
            raise LLMException(f"API error: {str(e)}")
    
    def query_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> BaseModel:
        """
        Structured output using Pydantic models.
        
        Args:
            prompt: User prompt/question
            response_model: Pydantic model class for response structure
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Instance of response_model with structured data
        """
        try:
            return self._query_structured_method(
                prompt, response_model, system_prompt, temperature, max_tokens
            )
        except OpenAIRateLimitError:
            raise RateLimitError(provider=self.provider.value)
        except (APIError, AnthropicAPIError) as e:
            raise LLMException(f"API error: {str(e)}")
    
    def _query_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """OpenAI-specific query implementation (uses raw client)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use raw client for regular text queries
        response = self.raw_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _query_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Anthropic-specific query implementation (uses raw client)"""
        # Use raw client for regular text queries
        response = self.raw_client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _query_structured_openai(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> BaseModel:
        """OpenAI-specific structured query (uses instructor client)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use instructor client for structured outputs
        return self.instructor_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model
        )
    
    def _query_structured_anthropic(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> BaseModel:
        """Anthropic-specific structured query (uses instructor client)"""
        # Use instructor client for structured outputs
        return self.instructor_client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model
        )
    
    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider.value}, model={self.model_name})"