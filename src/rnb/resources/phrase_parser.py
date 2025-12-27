"""
Phrase Parser - Parse Modified Adjective Phrases

This module parses personality adjective phrases that may include
intensity modifiers, producing structured ModifiedAdjective objects.

Handles:
- Simple adjectives: "lazy" → ModifiedAdjective(adjective="lazy")
- Modified adjectives: "very lazy" → ModifiedAdjective(adjective="lazy", modifier="very", ...)
- Multi-word modifiers: "a little bit lazy" → ModifiedAdjective(adjective="lazy", modifier="a little bit", ...)
- Lists: ["lazy", "very organized", "somewhat ambitious"]

The parser uses the SO-CAL ModifierLexicon for modifier recognition
and SpaCy for optional POS-based validation (when available).

Example:
    parser = PhraseParser.from_default_resources()

    # Parse single phrase
    result = parser.parse("very lazy")
    # ModifiedAdjective(adjective="lazy", modifier="very", modifier_value=0.2)

    # Parse list
    results = parser.parse_list(["lazy", "extremely organized"])
    # [ModifiedAdjective("lazy"), ModifiedAdjective("organized", "extremely", 0.4)]
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .modifier_lexicon import ModifierCategory, ModifierEntry, ModifierLexicon

logger = logging.getLogger(__name__)


# Optional SpaCy support - only used if available
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


@dataclass
class ModifiedAdjective:
    """
    A personality adjective with optional intensity modifier.

    Represents the parsed form of phrases like "very lazy" or
    "slightly ambitious".

    Attributes:
        adjective: The base personality adjective
        modifier: Optional modifier word/phrase (None if unmodified)
        modifier_value: SO-CAL modifier value (0.0 if unmodified)
        modifier_entry: Full ModifierEntry if modifier was found
        source_phrase: Original input phrase

    The modifier_value uses SO-CAL's multiplicative semantics:
        modified_intensity = base_intensity × (1 + modifier_value)

    Example:
        # Unmodified
        ModifiedAdjective(adjective="lazy")
        # → modifier=None, modifier_value=0.0

        # With amplifier
        ModifiedAdjective(adjective="lazy", modifier="very", modifier_value=0.2)
        # → base 0.7 becomes 0.7 × 1.2 = 0.84

        # With downtoner
        ModifiedAdjective(adjective="ambitious", modifier="slightly", modifier_value=-0.5)
        # → base 0.8 becomes 0.8 × 0.5 = 0.4
    """

    adjective: str
    modifier: str | None = None
    modifier_value: float = 0.0
    modifier_entry: ModifierEntry | None = field(default=None, repr=False)
    source_phrase: str = ""

    def __post_init__(self):
        # Set source_phrase if not provided
        if not self.source_phrase:
            if self.modifier:
                self.source_phrase = f"{self.modifier} {self.adjective}"
            else:
                self.source_phrase = self.adjective

    @property
    def is_modified(self) -> bool:
        """True if this adjective has a modifier."""
        return self.modifier is not None

    @property
    def multiplier(self) -> float:
        """
        The multiplicative factor to apply to base intensity.

        Returns 1.0 for unmodified adjectives.
        """
        return 1.0 + self.modifier_value

    def apply_to_intensity(self, base_intensity: float, clamp: bool = True) -> float:
        """
        Apply the modifier to a base intensity value.

        Args:
            base_intensity: Original intensity (typically 0.3-1.0)
            clamp: If True, clamp result to [-1.0, 1.0]

        Returns:
            Modified intensity
        """
        result = base_intensity * self.multiplier
        if clamp:
            result = max(-1.0, min(1.0, result))
        return result

    def __repr__(self) -> str:
        if self.modifier:
            sign = "+" if self.modifier_value >= 0 else ""
            return f"ModifiedAdjective('{self.source_phrase}', mod={sign}{self.modifier_value})"
        return f"ModifiedAdjective('{self.adjective}')"


class PhraseParser:
    """
    Parser for modified personality adjective phrases.

    Extracts modifiers from phrases like "very lazy" and produces
    structured ModifiedAdjective objects for downstream processing.

    The parser:
    1. Attempts to extract a leading modifier using the SO-CAL lexicon
    2. Returns the remaining text as the adjective
    3. Optionally validates using SpaCy POS tagging

    Attributes:
        lexicon: ModifierLexicon for modifier lookup
        use_spacy: Whether to use SpaCy for POS validation

    Example:
        parser = PhraseParser.from_default_resources()

        # Single phrase
        result = parser.parse("extremely organized")
        assert result.adjective == "organized"
        assert result.modifier == "extremely"
        assert result.modifier_value == 0.4

        # Batch parsing
        results = parser.parse_list([
            "lazy",
            "very ambitious",
            "somewhat creative",
            "a little bit nervous"
        ])
    """

    def __init__(
        self,
        lexicon: ModifierLexicon,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize the parser.

        Args:
            lexicon: ModifierLexicon for modifier extraction
            use_spacy: Whether to use SpaCy for POS validation
            spacy_model: SpaCy model name (default: en_core_web_sm)
        """
        self.lexicon = lexicon
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self._nlp = None
        self._spacy_model = spacy_model

        if use_spacy and not SPACY_AVAILABLE:
            logger.warning(
                "SpaCy not available, POS validation disabled. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )

    @property
    def nlp(self):
        """Lazy-load SpaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                self._nlp = spacy.load(self._spacy_model)
                logger.debug(f"Loaded SpaCy model: {self._spacy_model}")
            except OSError:
                logger.warning(
                    f"SpaCy model '{self._spacy_model}' not found. "
                    f"Download with: python -m spacy download {self._spacy_model}"
                )
                self.use_spacy = False
        return self._nlp

    @classmethod
    def from_default_resources(cls, use_spacy: bool = False) -> "PhraseParser":
        """
        Create parser with default lexicon.

        Args:
            use_spacy: Whether to enable SpaCy POS validation

        Returns:
            Configured PhraseParser
        """
        lexicon = ModifierLexicon.from_default_resources()
        return cls(lexicon, use_spacy=use_spacy)

    @classmethod
    def from_lexicon_path(cls, path: Path, use_spacy: bool = False) -> "PhraseParser":
        """
        Create parser with lexicon from specified path.

        Args:
            path: Path to modifiers.yaml
            use_spacy: Whether to enable SpaCy POS validation

        Returns:
            Configured PhraseParser
        """
        lexicon = ModifierLexicon.from_yaml(path)
        return cls(lexicon, use_spacy=use_spacy)

    def parse(self, phrase: str) -> ModifiedAdjective:
        """
        Parse a single adjective phrase.

        Extracts any leading modifier and returns a ModifiedAdjective
        with the base adjective and modifier information.

        Args:
            phrase: Input phrase like "very lazy" or "lazy"

        Returns:
            ModifiedAdjective with parsed components

        Example:
            >>> parser.parse("very lazy")
            ModifiedAdjective('very lazy', mod=+0.2)

            >>> parser.parse("lazy")
            ModifiedAdjective('lazy')

            >>> parser.parse("a little bit nervous")
            ModifiedAdjective('a little bit nervous', mod=-0.5)
        """
        phrase = phrase.strip()
        original_phrase = phrase

        # Try to extract modifier (prioritize adverb intensifiers)
        entry, remainder = self.lexicon.extract_modifier(
            phrase, category_filter=ModifierCategory.ADVERB
        )

        # If no adverb found, try any category
        if entry is None:
            entry, remainder = self.lexicon.extract_modifier(phrase)

        if entry is not None and remainder:
            # Found modifier, remainder is the adjective
            adjective = self._normalize_adjective(remainder)

            return ModifiedAdjective(
                adjective=adjective,
                modifier=entry.phrase,
                modifier_value=entry.value,
                modifier_entry=entry,
                source_phrase=original_phrase,
            )
        else:
            # No modifier found, entire phrase is the adjective
            adjective = self._normalize_adjective(phrase)

            return ModifiedAdjective(
                adjective=adjective,
                modifier=None,
                modifier_value=0.0,
                modifier_entry=None,
                source_phrase=original_phrase,
            )

    def parse_list(self, phrases: list[str]) -> list[ModifiedAdjective]:
        """
        Parse a list of adjective phrases.

        Args:
            phrases: List of phrases to parse

        Returns:
            List of ModifiedAdjective objects
        """
        return [self.parse(phrase) for phrase in phrases]

    def parse_mixed_input(
        self, items: list[str | ModifiedAdjective]
    ) -> list[ModifiedAdjective]:
        """
        Parse input that may contain strings or ModifiedAdjective objects.

        Strings are parsed, ModifiedAdjective objects pass through unchanged.

        Args:
            items: Mixed list of strings and ModifiedAdjective objects

        Returns:
            List of ModifiedAdjective objects
        """
        results = []
        for item in items:
            if isinstance(item, ModifiedAdjective):
                results.append(item)
            else:
                results.append(self.parse(str(item)))
        return results

    def _normalize_adjective(self, adj: str) -> str:
        """
        Normalize an adjective string.

        - Lowercase
        - Strip whitespace
        - Handle common variations

        Args:
            adj: Raw adjective string

        Returns:
            Normalized adjective
        """
        adj = adj.lower().strip()

        # Remove trailing punctuation
        adj = adj.rstrip(".,!?;:")

        return adj

    def _validate_with_spacy(self, phrase: str) -> bool:
        """
        Use SpaCy to validate that phrase contains an adjective.

        This is optional additional validation to catch parsing errors.

        Args:
            phrase: Phrase to validate

        Returns:
            True if phrase appears to contain an adjective
        """
        if not self.use_spacy or self.nlp is None:
            return True

        doc = self.nlp(phrase)
        # Check if any token is an adjective
        return any(token.pos_ == "ADJ" for token in doc)


def parse_adjective_phrases(
    phrases: list[str], use_spacy: bool = False
) -> list[ModifiedAdjective]:
    """
    Convenience function to parse a list of adjective phrases.

    Creates a parser with default resources and parses all phrases.

    Args:
        phrases: List of adjective phrases to parse
        use_spacy: Whether to use SpaCy validation

    Returns:
        List of ModifiedAdjective objects

    Example:
        results = parse_adjective_phrases([
            "lazy",
            "very organized",
            "extremely creative"
        ])
    """
    parser = PhraseParser.from_default_resources(use_spacy=use_spacy)
    return parser.parse_list(phrases)
