"""
Basic LLM client usage examples.

Demonstrates:
1. Simple text queries
2. Structured outputs
3. Different providers (OpenAI, Anthropic, local)
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel

from rnb.console import blank, rule, say_agent, say_system, say_user
from rnb.llm import LLMClient, ModelProvider
from rnb.logging import configure_logging

# Load environment variables from .env file
load_dotenv()


# ===== Example 1: Simple Query with Local Model =====
def example_local_simple() -> None:
    """Simple query using local Llama model via Ollama"""
    rule("Example 1: Local Model (Ollama)")

    client = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")

    prompt = "Explain recursion in one sentence."
    say_user(prompt)

    response = client.query(prompt, temperature=0.7)

    say_agent(f"Response: {response}")


# ===== Example 2: Structured Output =====
class Explanation(BaseModel):
    """Structured explanation response"""

    summary: str
    key_points: list[str]
    difficulty: str  # "beginner", "intermediate", "advanced"


def example_structured_output() -> None:
    """Structured output using Pydantic models"""
    rule("Example 2: Structured Output")

    client = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")

    prompt = "Explain photosynthesis briefly"
    say_user(prompt)

    result = client.query_structured(
        prompt=prompt,
        response_model=Explanation,
        temperature=0.5,
    )

    say_agent(f"Summary: {result.summary}")
    say_agent(f"Key points: {', '.join(result.key_points)}")
    say_agent(f"Difficulty: {result.difficulty}")


# ===== Example 3: With System Prompt =====
def example_with_system_prompt() -> None:
    """Query with system prompt (persona)"""
    rule("Example 3: With System Prompt")

    client = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")

    prompt = "What is machine learning?"
    system_prompt = "You are a patient teacher explaining concepts to a 10-year-old."

    say_user(prompt)
    say_system(f"System prompt: {system_prompt}")

    response = client.query(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.7,
    )

    say_agent(f"Response: {response}")


# ===== Example 4: OpenAI (if API key available) =====
def example_openai() -> None:
    """Query using OpenAI API"""
    rule("Example 4: OpenAI API")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        say_system("Skipped: OPENAI_API_KEY not set")
        return

    client = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key=api_key,
    )

    prompt = "Explain recursion in one sentence."
    say_user(prompt)

    response = client.query(prompt, temperature=0.7)

    say_agent(f"Response: {response}")


# ===== Example 5: Anthropic (if API key available) =====
def example_anthropic() -> None:
    """Query using Anthropic API"""
    rule("Example 5: Anthropic API")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        say_system("Skipped: ANTHROPIC_API_KEY not set")
        return

    client = LLMClient(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        api_key=api_key,
    )

    prompt = "Explain recursion in one sentence."
    say_user(prompt)

    response = client.query(prompt, temperature=0.7)

    say_agent(f"Response: {response}")


if __name__ == "__main__":
    # Configure demo logging once.
    # Optional file logging: set env RNB_LOG_FILE=./rnb.log
    configure_logging(level="INFO", rich=True)

    blank(1)
    rule("Basic LLM Client Usage Examples")
    # Run examples
    # Note: Requires Ollama running locally with llama3.2:3b model
    # Install: ollama pull llama3.2:3b

    example_local_simple()
    example_structured_output()
    example_with_system_prompt()
    example_openai()
    example_anthropic()
