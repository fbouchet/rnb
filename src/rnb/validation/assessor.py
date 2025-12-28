"""
RnB Personality Validation Module

Provides personality assessment using standardized instruments (TIPI, BFI-2-S)
for validating RnB agent personality conformity and consistency.

Example usage:
    from rnb.validation import PersonalityAssessor, load_archetypes

    assessor = PersonalityAssessor.from_instrument("tipi")
    result = assessor.assess(agent, method="batch")

    archetypes = load_archetypes()
    conformity = assessor.check_conformity(result, archetypes["resilient"])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ScaleConfig:
    """Configuration for a rating scale."""

    min: int
    max: int
    midpoint: float
    labels: dict[int, str]

    @property
    def half_range(self) -> float:
        return (self.max - self.min) / 2.0


@dataclass
class Item:
    """A single assessment item."""

    id: int
    text: str
    domain: str
    reversed: bool
    facet: str | None = None


@dataclass
class ScoringConfig:
    """Scoring configuration for an instrument."""

    method: str  # "mean" or "sum"
    domains: dict[str, dict[str, Any]]
    facets: dict[str, dict[str, Any]] | None = None
    rnb_conversion: dict[str, float] | None = None
    neuroticism_inversion: bool = False


@dataclass
class Instrument:
    """A personality assessment instrument."""

    name: str
    acronym: str
    items_count: int
    instructions: str
    scale: ScaleConfig
    stem: str
    items: list[Item]
    scoring: ScoringConfig
    citation: str
    notes: str | None = None

    def get_items_for_domain(self, domain: str) -> list[Item]:
        """Get all items for a specific domain."""
        return [item for item in self.items if item.domain == domain]

    def get_items_for_facet(self, facet: str) -> list[Item]:
        """Get all items for a specific facet."""
        return [item for item in self.items if item.facet == facet]


@dataclass
class AssessmentResult:
    """Result of a personality assessment."""

    instrument: str
    raw_responses: dict[int, int]  # item_id -> response
    domain_scores: dict[str, float]  # domain -> score on instrument scale
    facet_scores: dict[str, float] | None = None  # facet -> score
    rnb_scores: dict[str, float] | None = None  # domain -> score on RnB scale [-1, 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "raw_responses": self.raw_responses,
            "domain_scores": self.domain_scores,
            "facet_scores": self.facet_scores,
            "rnb_scores": self.rnb_scores,
        }


@dataclass
class ConformityResult:
    """Result of conformity check against an archetype."""

    archetype_name: str
    correlation: float
    within_tolerance: dict[str, bool]  # domain -> whether score is within tolerance
    all_within_tolerance: bool
    designed_traits: dict[str, float]
    measured_scores: dict[str, float]
    deviations: dict[str, float]  # domain -> (measured - designed)
    tolerance: float  # tolerance used for range check


@dataclass
class ArchetypeConfig:
    """Configuration for a personality archetype."""

    name: str
    description: str
    traits: dict[str, float]  # RnB scale [-1, 1]
    behavioral_markers: list[str]
    references: list[str]


# =============================================================================
# Instrument Loading
# =============================================================================


def load_instrument(path: str | Path) -> Instrument:
    """Load an instrument from a YAML file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Parse scale config
    scale_data = data["scale"]
    scale = ScaleConfig(
        min=scale_data["min"],
        max=scale_data["max"],
        midpoint=scale_data["midpoint"],
        labels=scale_data["labels"],
    )

    # Parse items
    items = []
    for item_data in data["items"]:
        items.append(
            Item(
                id=item_data["id"],
                text=item_data["text"],
                domain=item_data["domain"],
                reversed=item_data["reversed"],
                facet=item_data.get("facet"),
            )
        )

    # Parse scoring config
    scoring_data = data["scoring"]
    scoring = ScoringConfig(
        method=scoring_data["method"],
        domains=scoring_data["domains"],
        facets=scoring_data.get("facets"),
        rnb_conversion=scoring_data.get("rnb_conversion"),
        neuroticism_inversion=scoring_data.get("neuroticism_inversion", False),
    )

    # Build instrument
    inst_data = data["instrument"]
    return Instrument(
        name=inst_data["name"],
        acronym=inst_data["acronym"],
        items_count=inst_data["items_count"],
        instructions=data["instructions"],
        scale=scale,
        stem=data["stem"],
        items=items,
        scoring=scoring,
        citation=inst_data["citation"],
        notes=inst_data.get("notes"),
    )


def load_archetypes(path: str | Path) -> dict[str, ArchetypeConfig]:
    """Load archetypes from a YAML file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    archetypes = {}
    for name, arch_data in data["archetypes"].items():
        archetypes[name] = ArchetypeConfig(
            name=name,
            description=arch_data["description"],
            traits=arch_data["traits"],
            behavioral_markers=arch_data["behavioral_markers"],
            references=arch_data["references"],
        )

    return archetypes


# =============================================================================
# Scoring Functions
# =============================================================================


def reverse_score(response: int, scale: ScaleConfig) -> int:
    """Reverse a response on the given scale."""
    return scale.max + scale.min - response


def calculate_domain_score(
    responses: dict[int, int],
    domain_config: dict[str, Any],
    scale: ScaleConfig,
    method: str = "mean",
) -> float:
    """Calculate domain score from item responses."""
    item_ids = domain_config["items"]
    reversed_ids = set(domain_config.get("reversed_items", []))

    scores = []
    for item_id in item_ids:
        if item_id not in responses:
            raise ValueError(f"Missing response for item {item_id}")

        response = responses[item_id]
        if item_id in reversed_ids:
            response = reverse_score(response, scale)
        scores.append(response)

    if method == "mean":
        return sum(scores) / len(scores)
    elif method == "sum":
        return sum(scores)
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def convert_to_rnb_scale(
    score: float, midpoint: float, half_range: float, invert: bool = False
) -> float:
    """
    Convert instrument score to RnB scale [-1, 1].

    Formula: rnb = (score - midpoint) / half_range
    If invert=True: rnb = -1 * rnb
    """
    rnb = (score - midpoint) / half_range
    if invert:
        rnb = -1 * rnb
    return max(-1.0, min(1.0, rnb))  # Clamp to [-1, 1]


def convert_from_rnb_scale(
    rnb_score: float, midpoint: float, half_range: float, invert: bool = False
) -> float:
    """
    Convert RnB scale [-1, 1] to instrument score.

    Formula: score = midpoint + (rnb * half_range)
    If invert=True: first rnb = -1 * rnb
    """
    if invert:
        rnb_score = -1 * rnb_score
    return midpoint + (rnb_score * half_range)


# =============================================================================
# Prompt Generation
# =============================================================================


def generate_batch_prompt(instrument: Instrument) -> str:
    """
    Generate a batch assessment prompt with all items.

    Returns a prompt that asks the agent to respond to all items at once,
    formatted for easy parsing.
    """
    scale_min = instrument.scale.min
    scale_max = instrument.scale.max

    scale_description = "\n".join(
        f"  {k}: {v}" for k, v in sorted(instrument.scale.labels.items())
    )

    items_text = "\n".join(
        f"{item.id}. {item.text}"
        for item in sorted(instrument.items, key=lambda x: x.id)
    )

    prompt = f"""{instrument.instructions}

Rating Scale (use ONLY integers from {scale_min} to {scale_max}):
{scale_description}

IMPORTANT: Each rating MUST be an integer between {scale_min} and {scale_max} inclusive.
Do NOT use any number outside this range.

{instrument.stem}

{items_text}

Respond with your ratings as a JSON object. Each value MUST be {scale_min}, {scale_max}, or any integer in between.
Format: {{"1": <rating>, "2": <rating>, ..., "{instrument.items_count}": <rating>}}

Respond ONLY with the JSON object, no additional text."""

    return prompt


def generate_single_item_prompt(instrument: Instrument, item: Item) -> str:
    """Generate a prompt for a single item assessment."""
    scale_description = ", ".join(
        f"{k}={v}" for k, v in sorted(instrument.scale.labels.items())
    )

    prompt = f"""{instrument.stem} {item.text}

Rate on a scale of {instrument.scale.min}-{instrument.scale.max}:
{scale_description}

Respond with just the number."""

    return prompt


# =============================================================================
# Response Parsing
# =============================================================================


def parse_batch_response(response: str, instrument: Instrument) -> dict[int, int]:
    """
    Parse a batch response into item ratings.

    Handles various formats:
    - JSON: {"1": 5, "2": 3, ...}
    - Numbered list: 1. 5\n2. 3\n...
    - Simple list: 5, 3, 4, ...
    """
    response = response.strip()

    # Try JSON format first
    try:
        # Handle markdown code blocks
        if "```" in response:
            json_match = re.search(
                r"```(?:json)?\s*(\{[^`]+\})\s*```", response, re.DOTALL
            )
            if json_match:
                response = json_match.group(1)

        data = json.loads(response)
        return {int(k): int(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Try numbered list format: "1. 5" or "1: 5" or "1) 5"
    pattern = r"(\d+)[.):]\s*(\d+)"
    matches = re.findall(pattern, response)
    if len(matches) >= instrument.items_count:
        return {int(item_id): int(rating) for item_id, rating in matches}

    # Try simple comma/newline separated list
    numbers = re.findall(r"\b(\d+)\b", response)
    if len(numbers) >= instrument.items_count:
        return {i + 1: int(numbers[i]) for i in range(instrument.items_count)}

    raise ValueError(
        f"Could not parse response. Expected {instrument.items_count} ratings. "
        f"Got: {response[:200]}..."
    )


def parse_single_response(response: str, scale: ScaleConfig) -> int:
    """Parse a single item response."""
    response = response.strip()

    # Find first number in response
    match = re.search(r"\b(\d+)\b", response)
    if match:
        rating = int(match.group(1))
        if scale.min <= rating <= scale.max:
            return rating

    raise ValueError(
        f"Could not parse response as rating {scale.min}-{scale.max}: {response}"
    )


# =============================================================================
# Personality Assessor
# =============================================================================


class LLMClient(Protocol):
    """Protocol for LLM client interface."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response to the given prompt."""
        ...


class PersonalityAssessor:
    """
    Assesses agent personality using standardized instruments.

    Supports both batch assessment (all items at once) and
    item-by-item assessment.
    """

    def __init__(self, instrument: Instrument):
        self.instrument = instrument

    @classmethod
    def from_yaml(cls, path: str | Path) -> PersonalityAssessor:
        """Create assessor from instrument YAML file."""
        instrument = load_instrument(path)
        return cls(instrument)

    @classmethod
    def from_instrument_name(
        cls, name: str, instruments_dir: Path | None = None
    ) -> PersonalityAssessor:
        """Create assessor by instrument name (tipi, bfi2s)."""
        if instruments_dir is None:
            instruments_dir = Path(__file__).parent / "instruments"

        name_lower = name.lower()
        if name_lower in ("tipi", "ten-item", "tenitem"):
            path = instruments_dir / "tipi.yaml"
        elif name_lower in ("bfi2s", "bfi-2-s", "bfi2-s"):
            path = instruments_dir / "bfi2s.yaml"
        else:
            raise ValueError(f"Unknown instrument: {name}")

        return cls.from_yaml(path)

    def assess_batch(
        self, llm_client: LLMClient, system_prompt: str | None = None
    ) -> AssessmentResult:
        """
        Assess personality using batch method (all items at once).

        This is the recommended method for efficiency.
        """
        prompt = generate_batch_prompt(self.instrument)

        response = llm_client.generate(prompt, system=system_prompt)
        responses = parse_batch_response(response, self.instrument)

        return self._calculate_scores(responses)

    def assess_itemwise(
        self, llm_client: LLMClient, system_prompt: str | None = None
    ) -> AssessmentResult:
        """
        Assess personality item by item.

        More expensive but may be more robust for some models.
        """
        responses = {}

        for item in sorted(self.instrument.items, key=lambda x: x.id):
            prompt = generate_single_item_prompt(self.instrument, item)
            response = llm_client.generate(prompt, system=system_prompt)
            responses[item.id] = parse_single_response(response, self.instrument.scale)

        return self._calculate_scores(responses)

    def assess(
        self,
        llm_client: LLMClient,
        system_prompt: str | None = None,
        method: str = "batch",
    ) -> AssessmentResult:
        """
        Assess personality using specified method.

        Args:
            llm_client: Client for generating responses
            system_prompt: Optional system prompt for the agent
            method: "batch" (default) or "itemwise"
        """
        if method == "batch":
            return self.assess_batch(llm_client, system_prompt)
        elif method == "itemwise":
            return self.assess_itemwise(llm_client, system_prompt)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_scores(self, responses: dict[int, int]) -> AssessmentResult:
        """Calculate domain and facet scores from responses."""
        inst = self.instrument
        scoring = inst.scoring

        # Validate responses
        for item in inst.items:
            if item.id not in responses:
                raise ValueError(f"Missing response for item {item.id}")
            rating = responses[item.id]
            if not (inst.scale.min <= rating <= inst.scale.max):
                raise ValueError(
                    f"Invalid rating {rating} for item {item.id}. "
                    f"Expected {inst.scale.min}-{inst.scale.max}"
                )

        # Calculate domain scores
        domain_scores = {}
        for domain, config in scoring.domains.items():
            domain_scores[domain] = calculate_domain_score(
                responses, config, inst.scale, scoring.method
            )

        # Calculate facet scores if available
        facet_scores = None
        if scoring.facets:
            facet_scores = {}
            for facet, config in scoring.facets.items():
                facet_scores[facet] = calculate_domain_score(
                    responses, config, inst.scale, scoring.method
                )

        # Convert to RnB scale
        rnb_scores = None
        if scoring.rnb_conversion:
            midpoint = scoring.rnb_conversion["midpoint"]
            half_range = scoring.rnb_conversion["half_range"]

            rnb_scores = {}
            for domain, score in domain_scores.items():
                # Special handling for emotional_stability -> neuroticism inversion
                invert = (
                    scoring.neuroticism_inversion and domain == "emotional_stability"
                )

                # Map to standardized domain name
                rnb_domain = (
                    "neuroticism" if domain == "emotional_stability" else domain
                )

                rnb_scores[rnb_domain] = convert_to_rnb_scale(
                    score, midpoint, half_range, invert=invert
                )

        return AssessmentResult(
            instrument=inst.acronym,
            raw_responses=responses,
            domain_scores=domain_scores,
            facet_scores=facet_scores,
            rnb_scores=rnb_scores,
        )

    def check_conformity(
        self,
        result: AssessmentResult,
        archetype: ArchetypeConfig,
        tolerance: float = 0.25,
    ) -> ConformityResult:
        """
        Check how well assessment results conform to an archetype.

        Args:
            result: Assessment result to check
            archetype: Archetype to check against
            tolerance: Acceptable deviation from designed trait values on RnB scale
                       Default 0.25 means ±0.25 on [-1, 1] scale

        Returns:
            ConformityResult with correlation and tolerance check results
        """
        from .metrics import check_range_conformity, pearson_correlation

        if result.rnb_scores is None:
            raise ValueError("Assessment result has no RnB scores")

        # Check tolerance using RnB scores
        all_within, within_tolerance, expected_ranges = check_range_conformity(
            result.rnb_scores, archetype.traits, tolerance=tolerance
        )

        # Calculate deviations (measured - designed)
        deviations = {
            trait: result.rnb_scores.get(trait, 0) - archetype.traits.get(trait, 0)
            for trait in archetype.traits.keys()
            if trait in result.rnb_scores
        }

        # Calculate correlation between designed traits and measured scores
        common_traits = set(archetype.traits.keys()) & set(result.rnb_scores.keys())
        if len(common_traits) >= 2:
            designed = [archetype.traits[t] for t in sorted(common_traits)]
            measured = [result.rnb_scores[t] for t in sorted(common_traits)]
            correlation = pearson_correlation(designed, measured)
        else:
            correlation = 0.0

        return ConformityResult(
            archetype_name=archetype.name,
            correlation=correlation,
            within_tolerance=within_tolerance,
            all_within_tolerance=all_within,
            designed_traits=archetype.traits,
            measured_scores=result.rnb_scores,
            deviations=deviations,
            tolerance=tolerance,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

# Default paths - can be overridden by passing explicit paths to functions
_DEFAULT_RESOURCES_DIR: Path | None = None


def set_resources_dir(path: str | Path) -> None:
    """Set the default resources directory for instruments and archetypes."""
    global _DEFAULT_RESOURCES_DIR
    _DEFAULT_RESOURCES_DIR = Path(path)


def get_resources_dir() -> Path:
    """Get the resources directory."""
    if _DEFAULT_RESOURCES_DIR is not None:
        return _DEFAULT_RESOURCES_DIR
    # Default: assume we're in src/rnb/validation, resources is at src/rnb/resources
    return Path(__file__).parent.parent / "resources"


def get_instruments_dir() -> Path:
    """Get the default instruments directory."""
    return get_resources_dir() / "instruments"


def get_archetypes_path() -> Path:
    """Get the default archetypes file path."""
    return get_resources_dir() / "data" / "archetypes.yaml"


def list_available_instruments(instruments_dir: Path | None = None) -> list[str]:
    """List available instruments."""
    import logging

    logger = logging.getLogger(__name__)

    if instruments_dir is None:
        instruments_dir = get_instruments_dir()

    instruments = []
    for path in instruments_dir.glob("*.yaml"):
        try:
            inst = load_instrument(path)
            instruments.append(inst.acronym)
        except Exception as e:
            logger.debug(f"Could not load instrument from {path}: {e}")

    return instruments


def list_available_archetypes(archetypes_path: Path | None = None) -> list[str]:
    """List available archetypes."""
    if archetypes_path is None:
        archetypes_path = get_archetypes_path()

    archetypes = load_archetypes(archetypes_path)
    return list(archetypes.keys())
