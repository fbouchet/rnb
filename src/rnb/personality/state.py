"""
Personality State with Scheme-Level Storage

This module provides the core personality state representation following
the full RnB taxonomy: Trait → Facet → Scheme.

Key features:
- Scheme-level storage (70 schemes) for full RnB fidelity
- Computed facet and trait values (aggregated from schemes)
- Dynamic moods and affects (separate from stable personality)
- Source adjective traceability

The state supports the bipolar nature of schemes:
- Positive values (+1) indicate positive pole activation
- Negative values (-1) indicate negative pole activation
- Zero indicates neutral/default

Reference: Bouchet & Sansonnet (2013), "Influence of FFM/NEO PI-R
personality traits on the rational process of autonomous agents"
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .taxonomy import FACETS_BY_TRAIT, Trait


class MoodDimension(str, Enum):
    """
    Emotional mood dimensions (RnB Model M.A).

    Moods are transient emotional states that evolve over interactions.
    Unlike personality traits, moods change dynamically.
    """

    HAPPINESS = "happiness"
    SATISFACTION = "satisfaction"
    ENERGY = "energy"
    CALMNESS = "calmness"


class AffectDimension(str, Enum):
    """
    Interpersonal affect dimensions (RnB Model M.A).

    Affects represent the agent's stance toward the current interlocutor.
    They are relationship-specific and can change based on interaction.
    """

    DOMINANCE = "dominance"
    COOPERATION = "cooperation"
    TRUST = "trust"
    FAMILIARITY = "familiarity"


@dataclass
class PersonalityState:
    """
    Complete personality state with scheme-level granularity.

    Stores personality at the scheme level (70 values) while providing
    computed aggregations at facet (30) and trait (5) levels.

    Structure:
        - schemes: dict[str, float] - Core storage, scheme_key → value [-1, 1]
        - moods: dict[MoodDimension, float] - Transient emotional states
        - affects: dict[AffectDimension, float] - Interpersonal stances
        - source_adjectives: list[str] - Input adjectives for traceability

    Values are bipolar:
        - +1.0 = Strong positive pole (e.g., IDEALISTIC)
        - 0.0 = Neutral/default
        - -1.0 = Strong negative pole (e.g., PRACTICAL)

    Example:
        state = PersonalityState(agent_id="assistant-1")
        state.set_scheme("Openness_Fantasy_IDEALISTICNESS", 0.8)

        # Get aggregated values
        fantasy_value = state.get_facet("Openness", "Fantasy")
        openness_value = state.get_trait(Trait.OPENNESS)
    """

    agent_id: str

    # Core scheme-level storage: scheme_key → value in [-1, 1]
    # Keys are in format: "Trait_Facet_SCHEMENAME"
    schemes: dict[str, float] = field(default_factory=dict)

    # Dynamic moods (transient, evolve over interactions)
    moods: dict[MoodDimension, float] = field(
        default_factory=lambda: {m: 0.0 for m in MoodDimension}
    )

    # Interpersonal affects (relationship-specific)
    affects: dict[AffectDimension, float] = field(
        default_factory=lambda: {a: 0.0 for a in AffectDimension}
    )

    # Traceability: which adjectives were used to create this state
    source_adjectives: list[str] = field(default_factory=list)

    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0

    def __post_init__(self):
        """Validate values are in range"""
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        """Ensure all values are in [-1, 1] range"""
        for key, value in self.schemes.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"Scheme {key} value must be in [-1, 1], got {value}")
        for key, value in self.moods.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"Mood {key.value} value must be in [-1, 1], got {value}"
                )
        for key, value in self.affects.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"Affect {key.value} value must be in [-1, 1], got {value}"
                )

    # ===== Scheme Operations =====

    def get_scheme(self, scheme_key: str, default: float = 0.0) -> float:
        """
        Get value for a specific scheme.

        Args:
            scheme_key: Full scheme key (e.g., "Openness_Fantasy_IDEALISTICNESS")
            default: Value to return if scheme not set

        Returns:
            Scheme value in [-1, 1]
        """
        return self.schemes.get(scheme_key, default)

    def set_scheme(self, scheme_key: str, value: float) -> None:
        """
        Set value for a specific scheme.

        Args:
            scheme_key: Full scheme key
            value: Value in [-1, 1]

        Raises:
            ValueError: If value out of range
        """
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"Value must be in [-1, 1], got {value}")
        self.schemes[scheme_key] = value
        self.last_updated = datetime.now()

    def get_schemes_for_facet(self, trait: str | Trait, facet: str) -> dict[str, float]:
        """
        Get all scheme values under a facet.

        Args:
            trait: Trait name or Trait enum
            facet: Facet name

        Returns:
            Dict of scheme_key → value for matching schemes
        """
        trait_name = trait.value if isinstance(trait, Trait) else trait
        prefix = f"{trait_name}_{facet}_"

        return {k: v for k, v in self.schemes.items() if k.startswith(prefix)}

    def get_schemes_for_trait(self, trait: str | Trait) -> dict[str, float]:
        """
        Get all scheme values under a trait.

        Args:
            trait: Trait name or Trait enum

        Returns:
            Dict of scheme_key → value for matching schemes
        """
        trait_name = trait.value if isinstance(trait, Trait) else trait
        prefix = f"{trait_name}_"

        return {k: v for k, v in self.schemes.items() if k.startswith(prefix)}

    # ===== Aggregated Values =====

    def get_facet(self, trait: str | Trait, facet: str, default: float = 0.0) -> float:
        """
        Get aggregated value for a facet.

        Computes mean of all schemes under the facet.

        Args:
            trait: Trait name or Trait enum
            facet: Facet name
            default: Value if no schemes set for this facet

        Returns:
            Mean scheme value for the facet
        """
        schemes = self.get_schemes_for_facet(trait, facet)
        if not schemes:
            return default
        return sum(schemes.values()) / len(schemes)

    def get_trait(self, trait: str | Trait, default: float = 0.0) -> float:
        """
        Get aggregated value for a trait.

        Computes mean of all schemes under the trait.

        Args:
            trait: Trait name or Trait enum
            default: Value if no schemes set for this trait

        Returns:
            Mean scheme value for the trait
        """
        schemes = self.get_schemes_for_trait(trait)
        if not schemes:
            return default
        return sum(schemes.values()) / len(schemes)

    def get_all_facet_values(self) -> dict[str, float]:
        """
        Get aggregated values for all facets.

        Returns:
            Dict of facet_key → mean value
        """
        facet_values = {}

        for trait in Trait:
            for facet in FACETS_BY_TRAIT.get(trait, []):
                facet_key = f"{trait.value}_{facet}"
                facet_values[facet_key] = self.get_facet(trait, facet)

        return facet_values

    def get_all_trait_values(self) -> dict[Trait, float]:
        """
        Get aggregated values for all traits.

        Returns:
            Dict of Trait → mean value
        """
        return {trait: self.get_trait(trait) for trait in Trait}

    # ===== Mood Operations =====

    def get_mood(self, mood: MoodDimension) -> float:
        """Get current mood value"""
        return self.moods.get(mood, 0.0)

    def set_mood(self, mood: MoodDimension, value: float) -> None:
        """Set mood value (clamped to [-1, 1])"""
        self.moods[mood] = max(-1.0, min(1.0, value))
        self.last_updated = datetime.now()

    def update_mood(self, mood: MoodDimension, delta: float) -> None:
        """Update mood by delta (result clamped to [-1, 1])"""
        current = self.moods.get(mood, 0.0)
        self.moods[mood] = max(-1.0, min(1.0, current + delta))
        self.last_updated = datetime.now()

    # ===== Affect Operations =====

    def get_affect(self, affect: AffectDimension) -> float:
        """Get current affect value"""
        return self.affects.get(affect, 0.0)

    def set_affect(self, affect: AffectDimension, value: float) -> None:
        """Set affect value (clamped to [-1, 1])"""
        self.affects[affect] = max(-1.0, min(1.0, value))
        self.last_updated = datetime.now()

    def update_affect(self, affect: AffectDimension, delta: float) -> None:
        """Update affect by delta (result clamped to [-1, 1])"""
        current = self.affects.get(affect, 0.0)
        self.affects[affect] = max(-1.0, min(1.0, current + delta))
        self.last_updated = datetime.now()

    # ===== Serialization =====

    def to_dict(self) -> dict:
        """
        Serialize state to dictionary for storage.

        Returns:
            Dictionary representation suitable for JSON/Redis storage
        """
        return {
            "agent_id": self.agent_id,
            "schemes": self.schemes,
            "moods": {k.value: v for k, v in self.moods.items()},
            "affects": {k.value: v for k, v in self.affects.items()},
            "source_adjectives": self.source_adjectives,
            "last_updated": self.last_updated.isoformat(),
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonalityState":
        """
        Deserialize state from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            PersonalityState instance
        """
        return cls(
            agent_id=data["agent_id"],
            schemes=data.get("schemes", {}),
            moods={MoodDimension(k): v for k, v in data.get("moods", {}).items()},
            affects={AffectDimension(k): v for k, v in data.get("affects", {}).items()},
            source_adjectives=data.get("source_adjectives", []),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            interaction_count=data.get("interaction_count", 0),
        )

    # ===== Utility Methods =====

    def clone(self, new_agent_id: str | None = None) -> "PersonalityState":
        """
        Create a deep copy of this state.

        Args:
            new_agent_id: Optional new agent ID (uses original if None)

        Returns:
            New PersonalityState with copied values
        """
        return PersonalityState(
            agent_id=new_agent_id or self.agent_id,
            schemes=dict(self.schemes),
            moods=dict(self.moods),
            affects=dict(self.affects),
            source_adjectives=list(self.source_adjectives),
            last_updated=datetime.now(),
            interaction_count=0,
        )

    def summary(self) -> str:
        """
        Generate human-readable summary of the state.

        Returns:
            Multi-line string summarizing the personality
        """
        lines = [
            f"PersonalityState: {self.agent_id}",
            f"  Schemes set: {len(self.schemes)}",
            f"  Source adjectives: {self.source_adjectives}",
            "",
            "  Trait values (aggregated):",
        ]

        for trait in Trait:
            value = self.get_trait(trait)
            bar = "+" * int(abs(value) * 5) if value > 0 else "-" * int(abs(value) * 5)
            lines.append(f"    {trait.value}: {value:+.2f} [{bar:5}]")

        # Show non-zero moods
        non_zero_moods = [(m, v) for m, v in self.moods.items() if abs(v) > 0.1]
        if non_zero_moods:
            lines.append("")
            lines.append("  Active moods:")
            for mood, value in non_zero_moods:
                lines.append(f"    {mood.value}: {value:+.2f}")

        return "\n".join(lines)

    @property
    def num_schemes_set(self) -> int:
        """Number of schemes with non-zero values"""
        return sum(1 for v in self.schemes.values() if v != 0.0)

    def __repr__(self) -> str:
        return (
            f"PersonalityState(agent_id='{self.agent_id}', "
            f"schemes={len(self.schemes)}, "
            f"adjectives={self.source_adjectives})"
        )
