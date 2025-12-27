"""Custom exceptions for LLM client operations"""


class LLMException(Exception):
    """Base exception for LLM-related errors"""

    pass


class ModelNotAvailableError(LLMException):
    """Raised when requested model is not available"""

    def __init__(self, provider: str, model: str, details: str = ""):
        self.provider = provider
        self.model = model
        self.details = details
        message = f"Model '{model}' not available for provider '{provider}'"
        if details:
            message += f": {details}"
        super().__init__(message)


class ContextLengthExceededError(LLMException):
    """Raised when input exceeds model's context length"""

    def __init__(self, model: str, max_tokens: int, provided_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.provided_tokens = provided_tokens
        super().__init__(
            f"Context length exceeded for {model}: "
            f"{provided_tokens} tokens provided, max is {max_tokens}"
        )


class RateLimitError(LLMException):
    """Raised when API rate limit is hit"""

    def __init__(self, provider: str, retry_after: int = None):
        self.provider = provider
        self.retry_after = retry_after
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(message)
