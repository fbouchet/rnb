"""
Data models for RnB personality resources.

This module defines the core data structures for representing:
- Taxonomy positions (trait → facet → scheme → pole)
- Glosses (behavioral definitions from WordNet/Goldberg)
- Adjective entries (personality adjectives with weights)
- Scheme information (behavioral scheme metadata)

Reference: Bouchet & Sansonnet, "Implementing WordNet Personality
Adjectives as Influences on Rational Agents" (IJCISM 2010)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TraitName(str, Enum):
    """
    Five Factor Model personality traits - normalized names.

    These are the top level of the RnB taxonomy (5 traits).
    Each trait contains 6 facets for a total of 30 facets.
    """

    OPENNESS = "Openness"
    CONSCIENTIOUSNESS = "Conscientiousness"
    EXTRAVERSION = "Extraversion"
    AGREEABLENESS = "Agreeableness"
    NEUROTICISM = "Neuroticism"


@dataclass(frozen=True)
class TaxonomyPosition:
    """
    Unique position in the 4-level RnB taxonomy.

    Hierarchy: Trait (5) → Facet (30) → Scheme (69) → Pole (138)

    Immutable (frozen) for use as dictionary keys and set members.

    Attributes:
        trait: FFM trait name (e.g., "Openness")
        facet: NEO PI-R facet name (e.g., "Fantasy")
        scheme: Behavioral scheme name (e.g., "IDEALISTICNESS")
        pole: Positive or negative pole ("pos" or "neg")

    Example:
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        print(pos)  # "Openness/Fantasy/IDEALISTICNESS/+"
    """

    trait: str
    facet: str
    scheme: str
    pole: Literal["pos", "neg"]

    @property
    def facet_key(self) -> str:
        """
        Shorthand key for trait_facet combination.

        Returns:
            String like "Openness_Fantasy"
        """
        return f"{self.trait}_{self.facet}"

    @property
    def scheme_key(self) -> str:
        """
        Full path key including scheme and pole.

        Returns:
            String like "Openness_Fantasy_IDEALISTICNESS_pos"
        """
        return f"{self.trait}_{self.facet}_{self.scheme}_{self.pole}"

    @property
    def valence(self) -> Literal["+", "-"]:
        """Return pole as +/- symbol"""
        return "+" if self.pole == "pos" else "-"

    def __str__(self) -> str:
        """Human-readable representation"""
        return f"{self.trait}/{self.facet}/{self.scheme}/{self.valence}"


@dataclass
class GlossEntry:
    """
    A single gloss from WordNet or Goldberg questionnaire.

    Glosses are the atomic semantic units in RnB — they define
    the actual behavioral meaning of personality positions.

    Attributes:
        id: Unique identifier ("218" for WordNet, "Q1" for Goldberg)
        text: The gloss content (behavioral description)
        position: Full taxonomy position for this gloss

    Example:
        gloss = GlossEntry(
            id="218",
            text="not sensible about practical matters; idealistic and unrealistic",
            position=TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        )
    """

    id: str
    text: str
    position: TaxonomyPosition

    @property
    def source(self) -> Literal["wordnet", "goldberg"]:
        """
        Infer source from ID format.

        Returns:
            "goldberg" for Q-prefixed IDs, "wordnet" otherwise
        """
        return "goldberg" if self.id.startswith("Q") else "wordnet"


@dataclass
class AdjectiveEntry:
    """
    An adjective with its WordNet synset and salience weight.

    From neopiradj resource - links natural language words to glosses.
    Weight (1-9) indicates cognitive salience from corpus frequency
    (how often the adjective appeared across 10 thesaurus sources).

    Attributes:
        word: The adjective lemma (lowercase)
        synset: WordNet synset name (e.g., "WildEyed")
        weight: Salience weight 1-9 (9 = most salient)
        gloss_id: ID of the associated gloss
        gloss_text: Text of the associated gloss
        position: Taxonomy position this adjective maps to

    Example:
        entry = AdjectiveEntry(
            word="romantic",
            synset="WildEyed",
            weight=5,
            gloss_id="218",
            gloss_text="not sensible about practical matters...",
            position=TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        )
    """

    word: str
    synset: str
    weight: int
    gloss_id: str
    gloss_text: str
    position: TaxonomyPosition


@dataclass
class AdjectiveResolution:
    """
    Result of resolving a user-provided adjective.

    A single adjective may map to multiple taxonomy positions
    (different senses of the word). All mappings are returned
    for downstream processing.

    Attributes:
        input_word: Original word as provided by user
        normalized_word: Lowercase, stripped version
        mappings: All AdjectiveEntry matches found

    Example:
        resolution = resolver.resolve("spiritual")
        if resolution.ambiguous:
            print(f"'{resolution.input_word}' has multiple meanings")
            for m in resolution.mappings:
                print(f"  - {m.position}")
    """

    input_word: str
    normalized_word: str
    mappings: list[AdjectiveEntry] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """True if at least one mapping was found"""
        return len(self.mappings) > 0

    @property
    def ambiguous(self) -> bool:
        """
        True if adjective maps to multiple distinct facets.

        Multiple mappings within the same facet are not considered
        ambiguous (just different synsets for same meaning).
        """
        if len(self.mappings) <= 1:
            return False
        facets = {m.position.facet_key for m in self.mappings}
        return len(facets) > 1

    @property
    def total_weight(self) -> int:
        """Sum of weights across all mappings"""
        return sum(m.weight for m in self.mappings)

    @property
    def max_weight(self) -> int:
        """Highest weight among mappings (0 if none)"""
        return max((m.weight for m in self.mappings), default=0)


@dataclass
class PoleInfo:
    """
    One pole (positive or negative) of a behavioral scheme.

    Attributes:
        name: Pole name in uppercase (e.g., "IDEALISTIC", "PRACTICAL")
        glosses: Dictionary mapping gloss_id to gloss_text
    """

    name: str
    glosses: dict[str, str] = field(default_factory=dict)


@dataclass
class SchemeInfo:
    """
    Complete information about a behavioral scheme.

    Schemes are the third level of the taxonomy, grouping
    related glosses into coherent behavioral patterns.
    Each scheme is bipolar (positive and negative poles).

    Attributes:
        name: Scheme name in uppercase (e.g., "IDEALISTICNESS")
        operator_hint: Operator category hint from resource (e.g., "fact", "cooperation")
        trait: Parent trait name
        facet: Parent facet name
        poles: Dictionary with "pos" and "neg" PoleInfo
        formal_expression: Optional F(P(a)) notation for future use

    Example:
        scheme = registry.get_scheme("Openness", "Fantasy", "IDEALISTICNESS")
        print(scheme.poles["pos"].name)  # "IDEALISTIC"
        print(scheme.poles["neg"].name)  # "PRACTICAL"
    """

    name: str
    operator_hint: str
    trait: str
    facet: str
    poles: dict[str, PoleInfo] = field(default_factory=dict)
    formal_expression: str | None = None

    @property
    def facet_key(self) -> str:
        """Shorthand for trait_facet"""
        return f"{self.trait}_{self.facet}"

    @property
    def scheme_key(self) -> str:
        """Full key for this scheme (without pole)"""
        return f"{self.trait}_{self.facet}_{self.name}"

    def get_pole_name(self, pole: Literal["pos", "neg"]) -> str:
        """Get the name of a specific pole"""
        if pole in self.poles:
            return self.poles[pole].name
        return ""

    def get_glosses(self, pole: Literal["pos", "neg"] | None = None) -> dict[str, str]:
        """
        Get glosses, optionally filtered by pole.

        Args:
            pole: "pos" or "neg" to filter, None for all glosses

        Returns:
            Dictionary mapping gloss_id to gloss_text
        """
        if pole is not None:
            return self.poles.get(pole, PoleInfo("")).glosses

        # Combine both poles
        all_glosses = {}
        for pole_info in self.poles.values():
            all_glosses.update(pole_info.glosses)
        return all_glosses
