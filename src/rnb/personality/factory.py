"""
Personality State Factory

Creates PersonalityState instances from various input levels:
1. Adjectives (via PersonalitySpecification from Step 1)
2. Scheme-level values (direct)
3. Facet-level values (propagated to schemes)
4. Trait-level values (propagated to facets, then schemes)

The factory handles the conversion between input levels and ensures
proper propagation of default values through the hierarchy.

Example:
    factory = PersonalityStateFactory(resolver, taxonomy)

    # From adjectives
    state = factory.from_adjectives("agent-1", ["romantic", "organized"])

    # From traits (high-level)
    state = factory.from_traits("agent-2", {Trait.OPENNESS: 0.7})

    # From facets (mid-level)
    state = factory.from_facets("agent-3", {"Openness_Fantasy": 0.8})
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..resources import (
    ModifiedAdjective,
    PersonalityResolver,
    PersonalitySpecification,
    PhraseParser,
)
from .state import PersonalityState
from .taxonomy import Taxonomy, Trait, normalize_facet, normalize_trait

logger = logging.getLogger(__name__)


@dataclass
class ArchetypeDefinition:
    """
    Definition of a personality archetype.

    Attributes:
        name: Archetype identifier
        description: Human-readable description
        traits: FFM trait values
        adjectives: Representative adjectives
    """

    name: str
    description: str
    traits: dict[str, float]
    adjectives: list[str] = field(default_factory=list)


class ArchetypeRegistry:
    """
    Registry of personality archetypes loaded from YAML.

    Archetypes are predefined personality profiles that can be used
    as starting points for agent personalities.
    """

    def __init__(self):
        self._archetypes: dict[str, ArchetypeDefinition] = {}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ArchetypeRegistry":
        """Load archetypes from YAML file."""
        path = Path(path)
        logger.info(f"Loading archetypes from {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        registry = cls()

        archetypes_data = data.get("archetypes", {})
        for name, archetype_data in archetypes_data.items():
            archetype = ArchetypeDefinition(
                name=name,
                description=archetype_data.get("description", ""),
                traits=archetype_data.get("traits", {}),
                adjectives=archetype_data.get("adjectives", []),
            )
            registry._archetypes[name] = archetype

        logger.info(f"Loaded {len(registry._archetypes)} archetypes")
        return registry

    def get(self, name: str) -> ArchetypeDefinition | None:
        """Get archetype by name."""
        return self._archetypes.get(name)

    def list_names(self) -> list[str]:
        """List all archetype names."""
        return list(self._archetypes.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._archetypes

    def __len__(self) -> int:
        return len(self._archetypes)


class PersonalityStateFactory:
    """
    Factory for creating PersonalityState instances.

    Supports multiple input levels with automatic propagation:
    - Adjectives: Resolved via PersonalityResolver, mapped to schemes
    - Schemes: Direct scheme-level specification
    - Facets: Propagated equally to all schemes under the facet
    - Traits: Propagated equally to all facets, then schemes

    Attributes:
        resolver: PersonalityResolver for adjective lookup
        taxonomy: Taxonomy for structure navigation
        archetypes: Optional ArchetypeRegistry for predefined personalities

    Example:
        factory = PersonalityStateFactory.from_default_resources()

        # Create from adjectives
        state = factory.from_adjectives("agent-1", ["romantic", "shy"])

        # Create from trait-level (propagates down)
        state = factory.from_traits("agent-2", {
            Trait.OPENNESS: 0.8,
            Trait.CONSCIENTIOUSNESS: 0.6
        })
    """

    def __init__(
        self,
        resolver: PersonalityResolver,
        taxonomy: Taxonomy,
        archetypes: ArchetypeRegistry | None = None,
        phrase_parser: PhraseParser | None = None,
    ):
        """
        Initialize factory with resolver and taxonomy.

        Args:
            resolver: PersonalityResolver for adjective lookup
            taxonomy: Taxonomy for structure navigation
            archetypes: Optional ArchetypeRegistry for predefined personalities
        """
        self.resolver = resolver
        self.taxonomy = taxonomy
        self.archetypes = archetypes
        self.phrase_parser = phrase_parser or PhraseParser.from_default_resources()

    @classmethod
    def from_default_resources(cls) -> "PersonalityStateFactory":
        """
        Create factory using default resource locations.

        Returns:
            Configured PersonalityStateFactory
        """
        from ..resources import PersonalityResolver

        resources_dir = Path(__file__).parent.parent / "resources" / "data"

        resolver = PersonalityResolver.from_yaml(
            neopiradj_path=resources_dir / "neopiradj.yaml",
            schemes_path=resources_dir / "schemes.yaml",
            warn_unresolved=True,
        )

        # Build taxonomy from scheme registry
        taxonomy = Taxonomy.from_scheme_registry(resolver.schemes)

        # Load archetypes
        archetypes_path = resources_dir / "archetypes.yaml"
        archetypes = None
        if archetypes_path.exists():
            archetypes = ArchetypeRegistry.from_yaml(archetypes_path)

        from ..resources.phrase_parser import PhraseParser

        phrase_parser = PhraseParser.from_default_resources()

        return cls(resolver, taxonomy, archetypes, phrase_parser)

    # ===== From Adjectives =====

    def from_adjectives(
        self,
        agent_id: str,
        adjectives: list[str],  # Can be "lazy" or "very lazy"
        weight_scaling: bool = True,
        default_intensity: float = 0.7,
    ) -> PersonalityState:
        """
        Create state from personality adjectives.

        Resolves adjectives through the RnB resource and maps them
        to scheme-level values based on their weights and poles.
        Supports both simple adjectives ("lazy") and modified phrases
        ("very lazy", "slightly ambitious").

        Args:
            agent_id: Unique identifier for the agent
            adjectives: List of personality adjectives, optionally with modifiers
            weight_scaling: If True, scale intensity by adjective weight
            default_intensity: Base intensity when weight_scaling is False

        Returns:
            PersonalityState with schemes populated from adjectives

        Example:
        state = factory.from_adjectives("agent", [
            "lazy",                # No modifier
            "very organized",      # Amplified
            "slightly ambitious",  # Downtoned
            "extremely creative"   # Strongly amplified
        ])
        """
        # Parse all adjectives to extract modifiers
        parsed = self.phrase_parser.parse_list(adjectives)

        # Resolve base adjectives (without modifiers)
        base_adjectives = [p.adjective for p in parsed]
        spec = self.resolver.resolve(base_adjectives)

        if spec.unresolved:
            logger.warning(f"Unresolved adjectives: {spec.unresolved}")

        # Create mapping from adjective → ModifiedAdjective for lookup
        modifier_map = {p.adjective: p for p in parsed}

        return self._from_specification_with_modifiers(
            agent_id,
            spec,
            modifier_map,
            weight_scaling=weight_scaling,
            default_intensity=default_intensity,
        )

    def from_specification(
        self,
        agent_id: str,
        spec: PersonalitySpecification,
        weight_scaling: bool = True,
        default_intensity: float = 0.7,
    ) -> PersonalityState:
        """
        Create state from a PersonalitySpecification.

        This is the core conversion from Step 1 output to Step 2 state.

        Args:
            agent_id: Unique identifier for the agent
            spec: Resolved PersonalitySpecification from PersonalityResolver
            weight_scaling: If True, scale intensity by adjective weight
            default_intensity: Base intensity when weight_scaling is False

        Returns:
            PersonalityState with schemes populated
        """
        state = PersonalityState(
            agent_id=agent_id, source_adjectives=spec.input_adjectives
        )

        # Track scheme values for averaging when multiple adjectives hit same scheme
        scheme_values: dict[str, list[float]] = {}

        for resolved in spec.resolved:
            if not resolved.has_scheme:
                # Skip non-FFM mappings (e.g., "Others" category)
                continue

            pos = resolved.position

            # Normalize trait name
            trait = normalize_trait(pos.trait)
            if trait is None:
                continue

            # Normalize facet name
            facet = normalize_facet(pos.facet)

            # Build scheme key
            scheme_key = f"{trait.value}_{facet}_{pos.scheme}"

            # Calculate intensity
            if weight_scaling:
                # Weight is 1-9, scale to 0.3-1.0 range
                weight = resolved.adjective.weight
                intensity = 0.3 + (weight / 9.0) * 0.7
            else:
                intensity = default_intensity

            # Apply polarity (positive pole = positive value, negative = negative)
            value = intensity if pos.pole == "pos" else -intensity

            # Collect values for averaging
            if scheme_key not in scheme_values:
                scheme_values[scheme_key] = []
            scheme_values[scheme_key].append(value)

        # Average values when multiple adjectives hit same scheme
        for scheme_key, values in scheme_values.items():
            avg_value = sum(values) / len(values)
            # Clamp to [-1, 1]
            avg_value = max(-1.0, min(1.0, avg_value))
            state.set_scheme(scheme_key, avg_value)

        logger.info(
            f"Created state for {agent_id} from {len(spec.input_adjectives)} adjectives, "
            f"{len(state.schemes)} schemes set"
        )

        return state

    def _from_specification_with_modifiers(
        self,
        agent_id: str,
        spec: "PersonalitySpecification",
        modifier_map: dict[str, "ModifiedAdjective"],
        weight_scaling: bool = True,
        default_intensity: float = 0.7,
    ) -> "PersonalityState":
        """
        Create state with modifier adjustments.

        Core conversion that applies SO-CAL modifier formula:
            modified_intensity = base_intensity × (1 + modifier_value)
        """
        from ..personality import PersonalityState
        from ..personality.taxonomy import normalize_facet, normalize_trait

        state = PersonalityState(
            agent_id=agent_id, source_adjectives=spec.input_adjectives
        )

        scheme_values: dict[str, list[float]] = {}

        for resolved in spec.resolved:
            if not resolved.has_scheme:
                continue

            pos = resolved.position
            trait = normalize_trait(pos.trait)
            if trait is None:
                continue

            facet = normalize_facet(pos.facet)
            scheme_key = f"{trait.value}_{facet}_{pos.scheme}"

            # Calculate BASE intensity from weight
            if weight_scaling:
                weight = resolved.adjective.weight
                base_intensity = 0.3 + (weight / 9.0) * 0.7
            else:
                base_intensity = default_intensity

            # APPLY MODIFIER if present
            adj_word = resolved.adjective.word
            if adj_word in modifier_map:
                modified_adj = modifier_map[adj_word]
                intensity = modified_adj.apply_to_intensity(base_intensity)

                if modified_adj.is_modified:
                    logger.debug(
                        f"Applied modifier '{modified_adj.modifier}' to '{adj_word}': "
                        f"{base_intensity:.2f} → {intensity:.2f}"
                    )
            else:
                intensity = base_intensity

            # Apply polarity
            value = intensity if pos.pole == "pos" else -intensity

            if scheme_key not in scheme_values:
                scheme_values[scheme_key] = []
            scheme_values[scheme_key].append(value)

        # Average and clamp
        for scheme_key, values in scheme_values.items():
            avg_value = sum(values) / len(values)
            avg_value = max(-1.0, min(1.0, avg_value))
            state.set_scheme(scheme_key, avg_value)

        logger.info(
            f"Created state for {agent_id} from {len(spec.input_adjectives)} adjectives, "
            f"{len(state.schemes)} schemes set (with modifier support)"
        )

        return state

    # ===== From Scheme-Level =====

    def from_schemes(
        self,
        agent_id: str,
        schemes: dict[str, float],
        source_adjectives: list[str] | None = None,
    ) -> PersonalityState:
        """
        Create state from direct scheme-level values.

        Args:
            agent_id: Unique identifier for the agent
            schemes: Dict of scheme_key → value
            source_adjectives: Optional list of source adjectives

        Returns:
            PersonalityState with specified schemes
        """
        state = PersonalityState(
            agent_id=agent_id,
            schemes=dict(schemes),
            source_adjectives=source_adjectives or [],
        )

        return state

    # ===== From Facet-Level =====

    def from_facets(
        self,
        agent_id: str,
        facets: dict[str, float],
        source_adjectives: list[str] | None = None,
    ) -> PersonalityState:
        """
        Create state from facet-level values.

        Propagates facet values to all schemes under each facet.

        Args:
            agent_id: Unique identifier for the agent
            facets: Dict of facet_key (e.g., "Openness_Fantasy") → value
            source_adjectives: Optional list of source adjectives

        Returns:
            PersonalityState with schemes populated from facet values
        """
        state = PersonalityState(
            agent_id=agent_id, source_adjectives=source_adjectives or []
        )

        for facet_key, value in facets.items():
            # Parse facet key
            parts = facet_key.split("_", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid facet key format: {facet_key}")
                continue

            trait_name, facet_name = parts
            trait = normalize_trait(trait_name)
            if trait is None:
                logger.warning(f"Unknown trait in facet key: {facet_key}")
                continue

            facet_name = normalize_facet(facet_name)

            # Get all schemes for this facet and set them to the facet value
            schemes = self.taxonomy.get_schemes_for_facet(trait, facet_name)
            for scheme in schemes:
                state.set_scheme(scheme.scheme_key, value)

        return state

    # ===== From Trait-Level =====

    def from_traits(
        self,
        agent_id: str,
        traits: dict[Trait | str, float],
        source_adjectives: list[str] | None = None,
    ) -> PersonalityState:
        """
        Create state from trait-level values.

        Propagates trait values to all facets, then to all schemes.
        This is the highest-level, coarsest specification.

        Args:
            agent_id: Unique identifier for the agent
            traits: Dict of Trait (or trait name) → value
            source_adjectives: Optional list of source adjectives

        Returns:
            PersonalityState with all schemes under each trait set
        """
        state = PersonalityState(
            agent_id=agent_id, source_adjectives=source_adjectives or []
        )

        for trait_input, value in traits.items():
            # Normalize trait
            if isinstance(trait_input, Trait):
                trait = trait_input
            else:
                trait = normalize_trait(trait_input)
                if trait is None:
                    logger.warning(f"Unknown trait: {trait_input}")
                    continue

            # Get all schemes for this trait and set them
            schemes = self.taxonomy.get_schemes_for_trait(trait)
            for scheme in schemes:
                state.set_scheme(scheme.scheme_key, value)

        return state

    # ===== Mixed-Level Specification =====

    def from_mixed(
        self,
        agent_id: str,
        traits: dict[Trait | str, float] | None = None,
        facets: dict[str, float] | None = None,
        schemes: dict[str, float] | None = None,
        adjectives: list[str] | None = None,
    ) -> PersonalityState:
        """
        Create state from mixed-level specification.

        Applies values in order of specificity:
        1. Traits (most general, applied first)
        2. Facets (override trait-level for specific facets)
        3. Schemes (override facet-level for specific schemes)
        4. Adjectives (override based on resolved positions)

        This allows coarse defaults with fine-grained overrides.

        Args:
            agent_id: Unique identifier for the agent
            traits: Optional trait-level values
            facets: Optional facet-level values
            schemes: Optional scheme-level values
            adjectives: Optional adjectives to resolve

        Returns:
            PersonalityState with combined specification
        """
        # Start with empty state
        state = PersonalityState(agent_id=agent_id)

        # Layer 1: Trait-level defaults
        if traits:
            for trait_input, value in traits.items():
                trait = (
                    trait_input
                    if isinstance(trait_input, Trait)
                    else normalize_trait(trait_input)
                )
                if trait is None:
                    continue
                for scheme in self.taxonomy.get_schemes_for_trait(trait):
                    state.set_scheme(scheme.scheme_key, value)

        # Layer 2: Facet-level overrides
        if facets:
            for facet_key, value in facets.items():
                parts = facet_key.split("_", 1)
                if len(parts) != 2:
                    continue
                trait = normalize_trait(parts[0])
                if trait is None:
                    continue
                facet = normalize_facet(parts[1])

                for scheme in self.taxonomy.get_schemes_for_facet(trait, facet):
                    state.set_scheme(scheme.scheme_key, value)

        # Layer 3: Scheme-level overrides
        if schemes:
            for scheme_key, value in schemes.items():
                state.set_scheme(scheme_key, value)

        # Layer 4: Adjective-based overrides
        if adjectives:
            spec = self.resolver.resolve(adjectives)
            state.source_adjectives = adjectives

            for resolved in spec.resolved:
                if not resolved.has_scheme:
                    continue

                pos = resolved.position
                trait = normalize_trait(pos.trait)
                if trait is None:
                    continue

                facet = normalize_facet(pos.facet)
                scheme_key = f"{trait.value}_{facet}_{pos.scheme}"

                # Calculate intensity from weight
                weight = resolved.adjective.weight
                intensity = 0.3 + (weight / 9.0) * 0.7
                value = intensity if pos.pole == "pos" else -intensity

                state.set_scheme(scheme_key, value)

        return state

    # ===== Utility =====

    def create_neutral(self, agent_id: str) -> PersonalityState:
        """
        Create a neutral state with all schemes at zero.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            PersonalityState with no schemes set (all default to 0)
        """
        return PersonalityState(agent_id=agent_id)

    def create_from_archetype(self, agent_id: str, archetype: str) -> PersonalityState:
        """
        Create state from a predefined archetype.

        Archetypes are common personality profiles loaded from archetypes.yaml.

        Args:
            agent_id: Unique identifier for the agent
            archetype: Archetype name (use list_archetypes() to see available)

        Returns:
            PersonalityState for the archetype

        Raises:
            ValueError: If archetype not found or archetypes not loaded
        """
        if self.archetypes is None:
            raise ValueError(
                "No archetypes loaded. Ensure archetypes.yaml exists in resources/data/"
            )

        archetype_def = self.archetypes.get(archetype)
        if archetype_def is None:
            available = ", ".join(self.archetypes.list_names())
            raise ValueError(f"Unknown archetype '{archetype}'. Available: {available}")

        # Convert trait names to Trait enum
        traits = {}
        for trait_name, value in archetype_def.traits.items():
            trait = normalize_trait(trait_name)
            if trait is not None:
                traits[trait] = value

        return self.from_traits(
            agent_id=agent_id, traits=traits, source_adjectives=archetype_def.adjectives
        )

    def list_archetypes(self) -> list[str]:
        """
        List available archetype names.

        Returns:
            List of archetype names, or empty list if none loaded
        """
        if self.archetypes is None:
            return []
        return self.archetypes.list_names()


# For backwards compatibility, provide a function to get archetypes dict
def get_default_archetypes() -> dict[str, dict]:
    """
    Load archetypes from default YAML location.

    Returns:
        Dictionary of archetype_name → archetype_definition
    """
    resources_dir = Path(__file__).parent.parent / "resources" / "data"
    archetypes_path = resources_dir / "archetypes.yaml"

    if not archetypes_path.exists():
        return {}

    registry = ArchetypeRegistry.from_yaml(archetypes_path)

    # Convert to dict format for backwards compatibility
    result = {}
    for name in registry.list_names():
        archetype = registry.get(name)
        if archetype:
            result[name] = {
                "description": archetype.description,
                "traits": archetype.traits,
                "adjectives": archetype.adjectives,
            }

    return result


# =============================================================================
# USAGE EXAMPLE
# =============================================================================


def example_usage():
    """Demonstrate modifier support in factory."""
    from rnb.personality import PersonalityStateFactory

    factory = PersonalityStateFactory.from_default_resources()

    # Without modifiers (existing behavior)
    state1 = factory.from_adjectives("agent1", ["lazy", "organized"])

    # With modifiers (new behavior)
    state2 = factory.from_adjectives(
        "agent2",
        [
            "very lazy",  # Amplified: 0.7 * 1.2 = 0.84
            "slightly organized",  # Downtoned: 0.7 * 0.5 = 0.35
            "extremely creative",  # Strongly amplified: 0.7 * 1.4 = 0.98
            "somewhat ambitious",  # Moderately downtoned: 0.7 * 0.7 = 0.49
        ],
    )

    # Compare scheme values
    # state2's "lazy" scheme should be higher than state1's
    # state2's "organized" scheme should be lower than state1's

    print("Without modifiers:")
    for key, val in sorted(state1.schemes.items()):
        if abs(val) > 0.01:
            print(f"  {key}: {val:.3f}")

    print("\nWith modifiers:")
    for key, val in sorted(state2.schemes.items()):
        if abs(val) > 0.01:
            print(f"  {key}: {val:.3f}")


# =============================================================================
# NEGATION SUPPORT (Future Enhancement)
# =============================================================================

"""
The SO-CAL paper recommends SHIFT negation over FLIP negation.

For "not lazy", instead of flipping polarity:
    FLIP: lazy = -0.7 → not lazy = +0.7 (too strong!)
    
Use shift toward neutral:
    SHIFT: lazy = -0.7 → not lazy = -0.7 + 0.4 = -0.3 (mildly less lazy)

Implementation would add handling for modifiers with value <= -1.0
which act as near-negators (barely, hardly, almost).

For explicit "not X" patterns, a separate negation handler would:
1. Detect "not" prefix
2. Apply shift of 0.4 (on [-1,1] scale) toward opposite pole
3. This matches SO-CAL's shift=4 on their [-5,5] scale

Example:
    "not lazy" → lazy maps to -0.7, shift by +0.4 → -0.3
    "not ambitious" → ambitious maps to +0.8, shift by -0.4 → +0.4
"""


if __name__ == "__main__":
    example_usage()
