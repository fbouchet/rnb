"""
RnB Validation Integration

Bridges the validation framework to the actual RnB codebase:
- LLMClientAdapter: Wraps rnb.llm.LLMClient to match validation protocol
- RnBAgent: Wraps PersonalityState + OperatorRegistry for validation
- RnBAgentFactory: Creates agents from archetypes for validation experiments

Usage:
    from rnb.validation.integration import (
        LLMClientAdapter,
        RnBAgentFactory,
        create_validation_runner
    )

    # Quick setup
    runner = create_validation_runner(
        provider="local",
        model_name="llama3.2:3b"
    )

    # Run conformity test
    result = runner.test_conformity("resilient", instrument="tipi")
    print(f"Correlation: {result.conformity.correlation:.2f}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rnb.influence.context import InfluenceContext
from rnb.influence.registry import OperatorRegistry

# RnB imports
from rnb.llm import LLMClient, ModelProvider
from rnb.personality import (
    AgentManager,
    PersonalityState,
    PersonalityStateStore,
    RedisBackend,
    Trait,
)

# Try to import gloss-based influence if available
try:
    from ..influence.gloss_operator import GlossBasedInfluence

    HAS_GLOSS_INFLUENCE = True
except ImportError:
    HAS_GLOSS_INFLUENCE = False

# Try to import trait-based operators
try:
    from rnb.influence.trait_based import (
        DetailOrientedInfluence,
        EnthusiasmInfluence,
        StructureInfluence,
    )

    HAS_TRAIT_OPERATORS = True
except ImportError:
    HAS_TRAIT_OPERATORS = False

# Validation imports
from .assessor import (
    get_archetypes_path,
    load_archetypes,
)
from .runner import ValidationRunner

# =============================================================================
# LLM Client Adapter
# =============================================================================


class LLMClientAdapter:
    """
    Adapts rnb.llm.LLMClient to the validation framework's LLMClient protocol.

    The validation framework expects:
        generate(prompt: str, system: Optional[str] = None) -> str

    The RnB LLMClient provides:
        query(prompt: str, system_prompt: Optional[str] = None, ...) -> str
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize adapter.

        Args:
            client: RnB LLMClient instance
            temperature: Default temperature for queries
            max_tokens: Default max tokens for queries
        """
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Generate response (validation protocol).

        Maps to LLMClient.query().
        """
        return self.client.query(
            prompt=prompt,
            system_prompt=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_provider(
        cls,
        provider: str | ModelProvider = "local",
        model_name: str = "llama3.2:3b",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: float = 120.0,
    ) -> LLMClientAdapter:
        """
        Create adapter from provider specification.

        Args:
            provider: "local", "openai", or "anthropic"
            model_name: Model identifier
            api_key: API key (reads from env if not provided)
            temperature: Default temperature
            max_tokens: Default max tokens
            timeout: Request timeout

        Returns:
            Configured LLMClientAdapter
        """
        # Convert string to enum if needed
        if isinstance(provider, str):
            provider = ModelProvider(provider.lower())

        # Get API key from environment if not provided
        if api_key is None:
            if provider == ModelProvider.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider == ModelProvider.ANTHROPIC:
                api_key = os.getenv("ANTHROPIC_API_KEY")

        client = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
        )

        return cls(
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# =============================================================================
# RnB Agent Wrapper
# =============================================================================


@dataclass
class RnBAgent:
    """
    Wraps PersonalityState + OperatorRegistry to match validation AgentProtocol.

    The validation framework expects:
        - agent_id: str
        - respond(user_message: str) -> str
        - get_system_prompt() -> str
    """

    personality: PersonalityState
    registry: OperatorRegistry
    llm_adapter: LLMClientAdapter
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # Optional: base system prompt to prepend
    base_system_prompt: str = ""

    @property
    def agent_id(self) -> str:
        """Agent identifier."""
        return self.personality.agent_id

    def respond(self, user_message: str) -> str:
        """
        Generate response to user message with personality influence.

        This implements the RnB dual-engine architecture:
        1. Rational prompt (the user message)
        2. Behavioral modifications (from personality via operators)
        3. LLM query with combined prompt
        """
        # Create influence context from personality
        context = InfluenceContext.from_personality(self.personality)

        # Apply behavioral influences to the prompt
        behavioral_prompt = self.registry.apply_all(user_message, context)

        # Get system prompt with personality context
        system_prompt = self.get_system_prompt()

        # Query LLM
        response = self.llm_adapter.generate(behavioral_prompt, system=system_prompt)

        # Track conversation
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Increment interaction count
        self.personality.interaction_count += 1

        return response

    def get_system_prompt(self) -> str:
        """
        Get system prompt with personality context.

        Combines base system prompt with personality-derived behavioral context.
        """
        parts = []

        if self.base_system_prompt:
            parts.append(self.base_system_prompt)

        # Add personality context
        personality_context = self._generate_personality_context()
        if personality_context:
            parts.append(personality_context)

        return "\n\n".join(parts)

    def _generate_personality_context(self) -> str:
        """Generate personality context for system prompt."""
        lines = ["You have the following personality characteristics:"]

        # Add trait descriptions
        trait_descriptions = {
            Trait.OPENNESS: ("open to experience", "conventional"),
            Trait.CONSCIENTIOUSNESS: (
                "organized and thorough",
                "spontaneous and flexible",
            ),
            Trait.EXTRAVERSION: ("outgoing and energetic", "reserved and reflective"),
            Trait.AGREEABLENESS: ("cooperative and helpful", "direct and challenging"),
            Trait.NEUROTICISM: ("emotionally reactive", "emotionally stable"),
        }

        for trait, (high_desc, low_desc) in trait_descriptions.items():
            value = self.personality.get_trait(trait)
            if abs(value) > 0.3:  # Only mention notable traits
                if value > 0:
                    lines.append(f"- You tend to be {high_desc}")
                else:
                    lines.append(f"- You tend to be {low_desc}")

        if len(lines) == 1:
            return ""  # No notable traits

        return "\n".join(lines)

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()


# =============================================================================
# Agent Factory
# =============================================================================


class RnBAgentFactory:
    """
    Factory for creating RnB agents from archetypes or trait specifications.

    Implements the validation framework's AgentFactory protocol.
    """

    def __init__(
        self,
        llm_adapter: LLMClientAdapter,
        store: PersonalityStateStore | None = None,
        archetypes_path: Path | None = None,
        base_system_prompt: str = "",
        register_default_operators: bool = True,
    ):
        """
        Initialize factory.

        Args:
            llm_adapter: LLM client adapter for agent responses
            store: Personality state store (creates default if None)
            archetypes_path: Path to archetypes YAML
            base_system_prompt: Base system prompt for all agents
            register_default_operators: Whether to register default influence operators
        """
        self.llm_adapter = llm_adapter
        self.base_system_prompt = base_system_prompt
        self.register_default_operators = register_default_operators

        # Set up store
        if store is None:
            backend = RedisBackend()
            store = PersonalityStateStore(backend)
        self.store = store
        self.manager = AgentManager(store)

        # Load archetypes
        if archetypes_path is None:
            archetypes_path = get_archetypes_path()
        self.archetypes = load_archetypes(archetypes_path)

        # Track created agents for cleanup
        self._created_agents: list[str] = []

    def _create_registry(self) -> OperatorRegistry:
        """Create and configure operator registry."""
        registry = OperatorRegistry()

        if self.register_default_operators:
            # Register trait-based operators if available
            if HAS_TRAIT_OPERATORS:
                registry.register(StructureInfluence())
                registry.register(DetailOrientedInfluence())
                registry.register(EnthusiasmInfluence())

            # Register gloss-based influence if available
            if HAS_GLOSS_INFLUENCE:
                try:
                    gloss_op = GlossBasedInfluence.from_default_resources()
                    registry.register(gloss_op)
                except Exception as e:
                    logging.debug(f"Gloss resources not available: {e}")

        return registry

    def _traits_to_rnb_traits(self, traits: dict[str, float]) -> dict[Trait, float]:
        """Convert string trait names to Trait enum."""
        trait_map = {
            "openness": Trait.OPENNESS,
            "conscientiousness": Trait.CONSCIENTIOUSNESS,
            "extraversion": Trait.EXTRAVERSION,
            "agreeableness": Trait.AGREEABLENESS,
            "neuroticism": Trait.NEUROTICISM,
        }

        return {
            trait_map[name.lower()]: value
            for name, value in traits.items()
            if name.lower() in trait_map
        }

    def from_archetype(self, archetype_name: str) -> RnBAgent:
        """
        Create agent with specified archetype personality.

        Args:
            archetype_name: Name of archetype (e.g., "resilient", "overcontrolled")

        Returns:
            RnBAgent configured with archetype personality
        """
        if archetype_name not in self.archetypes:
            raise ValueError(f"Unknown archetype: {archetype_name}")

        archetype = self.archetypes[archetype_name]
        return self.from_traits(archetype.traits)

    def from_traits(self, traits: dict[str, float]) -> RnBAgent:
        """
        Create agent with specified trait values.

        Args:
            traits: Dict mapping trait names to values on RnB scale [-1, 1]

        Returns:
            RnBAgent configured with specified traits
        """
        # Generate unique agent ID
        import uuid

        agent_id = f"validation_{uuid.uuid4().hex[:8]}"

        # Convert traits to enum
        rnb_traits = self._traits_to_rnb_traits(traits)

        # Create agent via manager
        try:
            state = self.manager.create_agent(
                agent_id=agent_id,
                traits=rnb_traits,
            )
        except Exception:
            # Agent might already exist, try to get it
            state = self.store.get_state(agent_id)
            if state is None:
                raise

        self._created_agents.append(agent_id)

        # Create registry
        registry = self._create_registry()

        return RnBAgent(
            personality=state,
            registry=registry,
            llm_adapter=self.llm_adapter,
            base_system_prompt=self.base_system_prompt,
        )

    def cleanup(self) -> None:
        """Delete all agents created by this factory."""
        for agent_id in self._created_agents:
            try:
                self.manager.delete_agent(agent_id)
            except Exception as e:
                logging.debug(f"Could not delete agent {agent_id}: {e}")
        self._created_agents.clear()

    def __enter__(self) -> RnBAgentFactory:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_validation_runner(
    provider: str | ModelProvider = "local",
    model_name: str = "llama3.2:3b",
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    instruments_dir: Path | None = None,
    archetypes_path: Path | None = None,
    conformity_threshold: float = 0.7,
    consistency_threshold: float = 0.85,
    tolerance: float = 0.25,
    base_system_prompt: str = "",
) -> ValidationRunner:
    """
    Create a fully configured ValidationRunner with RnB integration.

    This is the main entry point for running validation experiments.

    Args:
        provider: LLM provider ("local", "openai", "anthropic")
        model_name: Model identifier
        api_key: API key (reads from env if not provided)
        temperature: LLM temperature
        max_tokens: Max tokens for LLM responses
        instruments_dir: Path to instruments directory
        archetypes_path: Path to archetypes YAML
        conformity_threshold: Minimum correlation for conformity pass
        consistency_threshold: Minimum ICC for consistency pass
        tolerance: Tolerance for trait deviation
        base_system_prompt: Base system prompt for agents

    Returns:
        Configured ValidationRunner

    Example:
        runner = create_validation_runner(
            provider="local",
            model_name="llama3.2:3b"
        )

        result = runner.test_conformity("resilient", instrument="tipi")
        print(f"Correlation: {result.conformity.correlation:.2f}")
        print(f"Passed: {result.passed}")
    """
    # Create LLM adapter
    llm_adapter = LLMClientAdapter.from_provider(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Create agent factory
    agent_factory = RnBAgentFactory(
        llm_adapter=llm_adapter,
        archetypes_path=archetypes_path,
        base_system_prompt=base_system_prompt,
    )

    # Create validation runner
    return ValidationRunner(
        llm_client=llm_adapter,
        agent_factory=agent_factory,
        instruments_dir=instruments_dir,
        archetypes_path=archetypes_path,
        conformity_threshold=conformity_threshold,
        consistency_threshold=consistency_threshold,
        tolerance=tolerance,
    )


def run_quick_validation(
    archetype: str = "resilient",
    instrument: str = "tipi",
    provider: str = "local",
    model_name: str = "llama3.2:3b",
    tolerance: float = 0.25,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a quick validation test and return results.

    Convenience function for rapid testing.

    Args:
        archetype: Archetype to test
        instrument: Instrument to use ("tipi" or "bfi2s")
        provider: LLM provider
        model_name: Model name
        tolerance: Tolerance for trait deviation
        verbose: Print results

    Returns:
        Dict with test results

    Example:
        results = run_quick_validation("resilient", verbose=True)
    """
    runner = create_validation_runner(
        provider=provider,
        model_name=model_name,
        tolerance=tolerance,
    )

    result = runner.test_conformity(
        archetype_name=archetype,
        instrument=instrument,
        tolerance=tolerance,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validation: {archetype} with {instrument.upper()}")
        print(f"{'='*60}")
        print(f"Correlation:        {result.conformity.correlation:.3f}")
        print(f"All within tol:     {result.conformity.all_within_tolerance}")
        print(f"Tolerance used:     ±{result.tolerance}")
        print(f"Passed:             {'✓' if result.passed else '✗'}")
        print()
        print("Designed vs Measured (RnB scale):")
        for trait in sorted(result.conformity.designed_traits.keys()):
            designed = result.conformity.designed_traits[trait]
            measured = result.conformity.measured_scores.get(trait, 0)
            within = result.conformity.within_tolerance.get(trait, False)
            status = "✓" if within else "✗"
            print(
                f"  {trait:20} designed={designed:+.2f}  measured={measured:+.2f}  {status}"
            )
        print(f"{'='*60}\n")

    return result.to_dict()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LLMClientAdapter",
    "RnBAgent",
    "RnBAgentFactory",
    "create_validation_runner",
    "run_quick_validation",
]
