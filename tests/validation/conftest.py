"""
Pytest fixtures for RnB validation tests.

Provides fixtures for:
- Loading instruments and archetypes
- Creating assessors
- Mock LLM clients for testing

Usage:
    pytest tests/validation/ -v
    pytest tests/validation/test_conformity.py -v --archetype=resilient
    pytest -m validation  # Run all validation tests
"""

import json

# Import validation components
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rnb.validation import (
    ArchetypeConfig,
    Instrument,
    PersonalityAssessor,
    load_archetypes,
    load_instrument,
    set_resources_dir,
)

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--archetype",
        action="store",
        default=None,
        help="Specific archetype to test (default: all)",
    )
    parser.addoption(
        "--instrument",
        action="store",
        default="tipi",
        help="Instrument to use: tipi or bfi2s (default: tipi)",
    )
    parser.addoption(
        "--conformity-threshold",
        action="store",
        type=float,
        default=0.7,
        help="Minimum correlation for conformity pass (default: 0.7)",
    )
    parser.addoption(
        "--consistency-threshold",
        action="store",
        type=float,
        default=0.85,
        help="Minimum ICC for consistency pass (default: 0.85)",
    )
    parser.addoption(
        "--tolerance",
        action="store",
        type=float,
        default=0.25,
        help="Tolerance for trait deviation (default: 0.25)",
    )
    parser.addoption(
        "--resources-dir",
        action="store",
        default=None,
        help="Path to resources directory containing instruments/ and data/",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "validation: mark test as validation test")
    config.addinivalue_line("markers", "conformity: mark test as conformity test")
    config.addinivalue_line("markers", "consistency: mark test as consistency test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def resources_dir(request) -> Path:
    """Path to resources directory (configurable via --resources-dir)."""
    custom_path = request.config.getoption("--resources-dir")
    if custom_path:
        path = Path(custom_path)
    else:
        # Default: look for resources in parent directories
        # This assumes tests are run from the repo root
        path = Path(__file__).parent.parent.parent / "src" / "rnb" / "resources"
        if not path.exists():
            # Fallback: create test resources in temp location
            path = Path(__file__).parent / "test_resources"

    # Set global resources dir
    if path.exists():
        set_resources_dir(path)

    return path


@pytest.fixture(scope="session")
def instruments_dir(resources_dir) -> Path:
    """Path to instruments directory."""
    return resources_dir / "instruments"


@pytest.fixture(scope="session")
def archetypes_path(resources_dir) -> Path:
    """Path to archetypes YAML file."""
    return resources_dir / "data" / "archetypes.yaml"


# =============================================================================
# Instrument Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tipi_instrument(instruments_dir) -> Instrument | None:
    """Loaded TIPI instrument."""
    path = instruments_dir / "tipi.yaml"
    if not path.exists():
        pytest.skip(f"TIPI instrument not found at {path}")
    return load_instrument(path)


@pytest.fixture(scope="session")
def bfi2s_instrument(instruments_dir) -> Instrument | None:
    """Loaded BFI-2-S instrument."""
    path = instruments_dir / "bfi2s.yaml"
    if not path.exists():
        pytest.skip(f"BFI-2-S instrument not found at {path}")
    return load_instrument(path)


@pytest.fixture
def instrument(request, tipi_instrument, bfi2s_instrument) -> Instrument:
    """Get instrument based on command line option."""
    instrument_name = request.config.getoption("--instrument")
    if instrument_name == "bfi2s":
        return bfi2s_instrument
    return tipi_instrument


@pytest.fixture
def tipi_assessor(instruments_dir) -> PersonalityAssessor:
    """TIPI assessor."""
    return PersonalityAssessor.from_instrument_name("tipi", instruments_dir)


@pytest.fixture
def bfi2s_assessor(instruments_dir) -> PersonalityAssessor:
    """BFI-2-S assessor."""
    return PersonalityAssessor.from_instrument_name("bfi2s", instruments_dir)


@pytest.fixture
def assessor(request, instruments_dir) -> PersonalityAssessor:
    """Get assessor based on command line option."""
    instrument_name = request.config.getoption("--instrument")
    return PersonalityAssessor.from_instrument_name(instrument_name, instruments_dir)


# =============================================================================
# Archetype Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def archetypes(archetypes_path) -> dict[str, ArchetypeConfig]:
    """All loaded archetypes."""
    if not archetypes_path.exists():
        pytest.skip(f"Archetypes file not found at {archetypes_path}")
    return load_archetypes(archetypes_path)


@pytest.fixture
def archetype_names(request, archetypes) -> list[str]:
    """Get archetype names to test based on command line."""
    specific = request.config.getoption("--archetype")
    if specific:
        if specific not in archetypes:
            pytest.skip(f"Unknown archetype: {specific}")
        return [specific]
    return list(archetypes.keys())


@pytest.fixture(params=["resilient", "overcontrolled", "undercontrolled", "average"])
def ruo_archetype(request, archetypes) -> ArchetypeConfig:
    """Parametrized fixture for RUO archetypes."""
    if request.param not in archetypes:
        pytest.skip(f"Archetype {request.param} not found")
    return archetypes[request.param]


# =============================================================================
# Threshold and Tolerance Fixtures
# =============================================================================


@pytest.fixture
def conformity_threshold(request) -> float:
    """Conformity correlation threshold."""
    return request.config.getoption("--conformity-threshold")


@pytest.fixture
def consistency_threshold(request) -> float:
    """Consistency ICC threshold."""
    return request.config.getoption("--consistency-threshold")


@pytest.fixture
def tolerance(request) -> float:
    """Tolerance for trait deviation."""
    return request.config.getoption("--tolerance")


# =============================================================================
# Mock LLM Fixtures
# =============================================================================


class MockLLMClient:
    """
    Mock LLM client for testing.

    Can be configured to return specific responses for testing parsing,
    or to simulate personality-consistent responses.
    """

    def __init__(self, archetype: ArchetypeConfig | None = None):
        self.archetype = archetype
        self.call_count = 0
        self.last_prompt = None
        self.fixed_response: str | None = None

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate mock response."""
        self.call_count += 1
        self.last_prompt = prompt

        if self.fixed_response:
            return self.fixed_response

        if self.archetype:
            return self._generate_archetype_response(prompt)

        # Default: return neutral responses
        return self._generate_neutral_response(prompt)

    def _generate_archetype_response(self, prompt: str) -> str:
        """Generate response consistent with archetype."""
        # Detect if this is a batch assessment
        if "JSON format" in prompt:
            return self._generate_batch_response()

        # Single item response
        return self._generate_single_response()

    def _generate_batch_response(self) -> str:
        """Generate batch JSON response."""
        if self.archetype is None:
            # Neutral: all 4s
            responses = {str(i): 4 for i in range(1, 31)}
        else:
            responses = self._simulate_responses()

        return json.dumps(responses)

    def _generate_single_response(self) -> str:
        """Generate single item response."""
        return "4"  # Neutral

    def _generate_neutral_response(self, prompt: str) -> str:
        """Generate neutral response."""
        if "JSON format" in prompt:
            # Assume TIPI (10 items) or BFI-2-S (30 items)
            if "30" in prompt or len(prompt) > 2000:
                responses = {str(i): 3 for i in range(1, 31)}
            else:
                responses = {str(i): 4 for i in range(1, 11)}
            return json.dumps(responses)
        return "4"

    def _simulate_responses(self) -> dict[str, int]:
        """Simulate responses based on archetype traits."""
        # Simple simulation: convert RnB trait to scale response
        # This is a basic simulation - real tests would use actual LLM

        responses = {}

        # Assume we need to generate for all possible items
        for i in range(1, 31):
            # Map item to trait (simplified)
            item_to_trait = {
                1: "extraversion",
                6: "extraversion",
                11: "extraversion",
                16: "extraversion",
                21: "extraversion",
                26: "extraversion",
                2: "agreeableness",
                7: "agreeableness",
                12: "agreeableness",
                17: "agreeableness",
                22: "agreeableness",
                27: "agreeableness",
                3: "conscientiousness",
                8: "conscientiousness",
                13: "conscientiousness",
                18: "conscientiousness",
                23: "conscientiousness",
                28: "conscientiousness",
                4: "neuroticism",
                9: "neuroticism",
                14: "neuroticism",
                19: "neuroticism",
                24: "neuroticism",
                29: "neuroticism",
                5: "openness",
                10: "openness",
                15: "openness",
                20: "openness",
                25: "openness",
                30: "openness",
            }

            trait = item_to_trait.get(i, "openness")
            trait_value = self.archetype.traits.get(trait, 0.0)

            # Convert to 1-5 scale (BFI-2-S)
            # RnB: -1 to 1, BFI: 1 to 5, midpoint 3
            score = 3.0 + (trait_value * 2.0)

            # Add some noise
            import random

            score += random.uniform(-0.5, 0.5)  # noqa: S311

            # Clamp and round
            score = max(1, min(5, round(score)))
            responses[str(i)] = int(score)

        return responses


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Basic mock LLM client with neutral responses."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_for_archetype(ruo_archetype) -> MockLLMClient:
    """Mock LLM client configured for specific archetype."""
    return MockLLMClient(archetype=ruo_archetype)


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def real_llm_client():
    """
    Real LLM client for integration tests.

    Only available if OLLAMA_HOST is set or ollama is running locally.
    """
    try:
        # Try to import and create real client
        from rnb.llm import LLMClient, ModelProvider

        client = LLMClient(
            provider=ModelProvider.LOCAL,
            model_name="llama3.2:3b",
            timeout=120.0,
        )

        # Quick test
        response = client.generate("Say 'test'")
        if not response:
            pytest.skip("LLM client not responding")

        return client
    except Exception as e:
        pytest.skip(f"LLM client not available: {e}")
