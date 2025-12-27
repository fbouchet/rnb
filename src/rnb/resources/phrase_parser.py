"""
Phrase Parser - Parse Modified Adjective Phrases with Negation

This module parses personality adjective phrases that may include
intensity modifiers and/or negation, producing structured ParsedPhrase objects.

Features:
- Intensity modifiers: "very lazy", "slightly ambitious", "extremely creative"
- Negation with SO-CAL shift semantics: "not lazy", "never happy", "hardly confident"
- SpaCy integration for accurate POS tagging and dependency parsing

Negation Semantics (Taboada et al., 2011, Section 2.4):
Instead of flipping polarity (not good = bad), we SHIFT toward neutral:
- "not excellent" (+0.9) → +0.5 (still positive, just less so)
- "not terrible" (-0.9) → -0.5 (still negative, just less so)

This is empirically validated (MTurk: shift 45.2% vs flip 33.4%).

Example:
    parser = PhraseParser.from_default_resources()

    result = parser.parse("not very lazy")
    # result.adjective = "lazy"
    # result.modifier = "very"
    # result.is_negated = True

    # Compute final intensity
    base = 0.7
    polarity = -1  # lazy is negative pole
    final = result.compute_intensity(base, polarity)
    # = ((0.7 × 1.2) × -1) + 0.4 = -0.44
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import spacy

from .modifier_lexicon import ModifierCategory, ModifierEntry, ModifierLexicon

logger = logging.getLogger(__name__)


# =============================================================================
# SpaCy Management
# =============================================================================

_nlp_cache: dict[str, spacy.Language] = {}


def get_spacy_nlp(model: str = "en_core_web_sm") -> spacy.Language:
    """
    Get SpaCy NLP model (cached singleton per model name).

    Args:
        model: SpaCy model name (default: en_core_web_sm)

    Returns:
        Loaded SpaCy Language model

    Raises:
        OSError: If model not found
    """
    if model not in _nlp_cache:
        _nlp_cache[model] = spacy.load(model)
        logger.info(f"Loaded SpaCy model: {model}")

    return _nlp_cache[model]


# =============================================================================
# Negation Configuration
# =============================================================================


class NegationType(str, Enum):
    """Type of negation detected."""

    NONE = "none"  # No negation
    STANDARD = "standard"  # "not lazy", "no good"
    CONTRACTED = "contracted"  # "isn't lazy", "aren't good"
    LEXICAL = "lexical"  # "never", "without", "lack"


# Standard negation words
NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "without",
        "lack",
        "lacking",
        "hardly",
        "barely",
        "scarcely",  # Near-negators
    }
)

# Contracted negations (SpaCy tokenizes "don't" → "do" + "n't")
NEGATION_CONTRACTIONS = frozenset({"n't"})

# SO-CAL shift value (adapted from shift=4 on [-5,5] to shift=0.4 on [-1,1])
DEFAULT_NEGATION_SHIFT = 0.4

# Near-negators have stronger effect (from SO-CAL int_dictionary)
NEAR_NEGATOR_SHIFTS = {
    "hardly": 0.6,
    "barely": 0.6,
    "scarcely": 0.6,
    "never": 0.5,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NegationInfo:
    """
    Information about detected negation.

    Implements SO-CAL shift semantics where negation shifts
    the value toward the opposite pole by a fixed amount.

    Attributes:
        negated: Whether the adjective is negated
        negation_type: Type of negation detected
        negation_word: The negation word (e.g., "not", "never")
        shift_value: The shift magnitude to apply
    """

    negated: bool = False
    negation_type: NegationType = NegationType.NONE
    negation_word: str | None = None
    shift_value: float = DEFAULT_NEGATION_SHIFT

    def apply_shift(self, value: float, clamp: bool = True) -> float:
        """
        Apply SO-CAL negation shift.

        - Positive values shift toward negative (subtract shift)
        - Negative values shift toward positive (add shift)

        Args:
            value: Original value in [-1, 1]
            clamp: Whether to clamp result to [-1, 1]

        Returns:
            Shifted value
        """
        if not self.negated:
            return value

        if value >= 0:
            result = value - self.shift_value
        else:
            result = value + self.shift_value

        if clamp:
            result = max(-1.0, min(1.0, result))

        return result


@dataclass
class ParsedPhrase:
    """
    Fully parsed adjective phrase with modifier and negation.

    This is the primary output class, containing all information
    needed to compute final personality intensity.

    Attributes:
        adjective: The base personality adjective
        modifier: Optional intensity modifier
        modifier_value: SO-CAL modifier value (0.0 if none)
        modifier_entry: Full ModifierEntry if found
        negation: Negation information
        source_phrase: Original input phrase
        pos_tag: Part-of-speech tag from SpaCy
        confidence: Parsing confidence [0, 1]
    """

    adjective: str
    modifier: str | None = None
    modifier_value: float = 0.0
    modifier_entry: ModifierEntry | None = field(default=None, repr=False)
    negation: NegationInfo = field(default_factory=NegationInfo)
    source_phrase: str = ""
    pos_tag: str | None = None
    confidence: float = 1.0

    def __post_init__(self):
        if not self.source_phrase:
            parts = []
            if self.negation.negated and self.negation.negation_word:
                parts.append(self.negation.negation_word)
            if self.modifier:
                parts.append(self.modifier)
            parts.append(self.adjective)
            self.source_phrase = " ".join(parts)

    @property
    def is_modified(self) -> bool:
        return self.modifier is not None

    @property
    def is_negated(self) -> bool:
        return self.negation.negated

    @property
    def multiplier(self) -> float:
        return 1.0 + self.modifier_value

    def compute_intensity(
        self, base_intensity: float, polarity: int = 1, clamp: bool = True
    ) -> float:
        """
        Compute final intensity with modifier and negation.

        Order of operations (following SO-CAL):
        1. Apply modifier (multiplicative)
        2. Apply polarity
        3. Apply negation shift (additive)

        Args:
            base_intensity: Base intensity from adjective weight [0, 1]
            polarity: +1 for positive pole, -1 for negative pole
            clamp: Whether to clamp result to [-1, 1]

        Returns:
            Final intensity value
        """
        modified = base_intensity * self.multiplier
        polarized = modified * polarity
        result = self.negation.apply_shift(polarized, clamp=False)

        if clamp:
            result = max(-1.0, min(1.0, result))

        return result


# =============================================================================
# Phrase Parser
# =============================================================================


class PhraseParser:
    """
    Parser for modified personality adjective phrases.

    Supports:
    - Intensity modifiers: "very lazy", "extremely creative"
    - Negation: "not lazy", "never happy"
    - Combined: "not very lazy", "hardly ever confident"

    Uses SpaCy for accurate POS tagging and dependency parsing.

    Example:
        parser = PhraseParser.from_default_resources()

        result = parser.parse("not very lazy")
        assert result.adjective == "lazy"
        assert result.modifier == "very"
        assert result.is_negated
    """

    def __init__(
        self,
        lexicon: ModifierLexicon,
        spacy_model: str = "en_core_web_sm",
        negation_shift: float = DEFAULT_NEGATION_SHIFT,
    ):
        """
        Initialize the parser.

        Args:
            lexicon: ModifierLexicon for intensity modifiers
            spacy_model: SpaCy model name
            negation_shift: Default negation shift value
        """
        self.lexicon = lexicon
        self._spacy_model = spacy_model
        self.negation_shift = negation_shift
        self._nlp: spacy.Language | None = None

    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load SpaCy model."""
        if self._nlp is None:
            self._nlp = get_spacy_nlp(self._spacy_model)
        return self._nlp

    @classmethod
    def from_default_resources(
        cls, negation_shift: float = DEFAULT_NEGATION_SHIFT
    ) -> "PhraseParser":
        """Create parser with default lexicon."""
        lexicon = ModifierLexicon.from_default_resources()
        return cls(lexicon, negation_shift=negation_shift)

    @classmethod
    def from_lexicon_path(cls, path: Path) -> "PhraseParser":
        """Create parser with lexicon from specified path."""
        lexicon = ModifierLexicon.from_yaml(path)
        return cls(lexicon)

    def parse(self, phrase: str) -> ParsedPhrase:
        """
        Parse an adjective phrase.

        Args:
            phrase: Input phrase (e.g., "not very lazy")

        Returns:
            ParsedPhrase with all components
        """
        phrase = phrase.strip()
        original = phrase

        doc = self.nlp(phrase.lower())

        negation_info = self._extract_negation(doc)
        adjective, pos_tag, adj_token = self._find_adjective(doc)
        modifier_entry = self._extract_modifier(doc, adj_token, negation_info)
        confidence = self._calculate_confidence(adjective, pos_tag, modifier_entry)

        return ParsedPhrase(
            adjective=adjective or self._fallback_adjective(doc),
            modifier=modifier_entry.phrase if modifier_entry else None,
            modifier_value=modifier_entry.value if modifier_entry else 0.0,
            modifier_entry=modifier_entry,
            negation=negation_info,
            source_phrase=original,
            pos_tag=pos_tag,
            confidence=confidence,
        )

    def parse_list(self, phrases: list[str]) -> list[ParsedPhrase]:
        """Parse multiple phrases."""
        return [self.parse(p) for p in phrases]

    def _extract_negation(self, doc: spacy.tokens.Doc) -> NegationInfo:
        """Extract negation using SpaCy."""
        for token in doc:
            # SpaCy's dependency label for negation
            if token.dep_ == "neg":
                shift = NEAR_NEGATOR_SHIFTS.get(token.text.lower(), self.negation_shift)
                return NegationInfo(
                    negated=True,
                    negation_type=NegationType.STANDARD,
                    negation_word=token.text.lower(),
                    shift_value=shift,
                )

            # Contracted negation
            if token.text.lower() in NEGATION_CONTRACTIONS:
                return NegationInfo(
                    negated=True,
                    negation_type=NegationType.CONTRACTED,
                    negation_word=token.text.lower(),
                    shift_value=self.negation_shift,
                )

            # Known negation words
            if token.text.lower() in NEGATION_WORDS:
                shift = NEAR_NEGATOR_SHIFTS.get(token.text.lower(), self.negation_shift)
                neg_type = (
                    NegationType.LEXICAL
                    if token.text.lower() in {"never", "without", "lack", "lacking"}
                    else NegationType.STANDARD
                )
                return NegationInfo(
                    negated=True,
                    negation_type=neg_type,
                    negation_word=token.text.lower(),
                    shift_value=shift,
                )

        return NegationInfo()

    def _find_adjective(
        self, doc: spacy.tokens.Doc
    ) -> tuple[str | None, str | None, spacy.tokens.Token | None]:
        """
        Find the target adjective using SpaCy POS tagging.

        Returns the last ADJ token that isn't part of a known modifier,
        to correctly handle phrases like "a little bit nervous" where
        "little" is tagged as ADJ but is part of the modifier phrase.
        """
        # Collect all ADJ tokens
        adj_candidates = [token for token in doc if token.pos_ == "ADJ"]

        # Return the last ADJ that isn't in the modifier lexicon
        for token in reversed(adj_candidates):
            if token.text.lower() not in self.lexicon:
                return (token.text.lower(), token.pos_, token)

        # If all ADJs are modifiers, return the last one anyway
        if adj_candidates:
            token = adj_candidates[-1]
            return (token.text.lower(), token.pos_, token)

        # Fallback: last non-function word that isn't a negation or modifier
        for token in reversed(list(doc)):
            if token.pos_ not in {"DET", "ADP", "PART", "PUNCT", "SPACE"}:
                if not self._is_negation_token(token):
                    if token.text.lower() not in self.lexicon:
                        return (token.text.lower(), token.pos_, token)

        return (None, None, None)

    def _extract_modifier(
        self,
        doc: spacy.tokens.Doc,
        adj_token: spacy.tokens.Token | None,
        negation_info: NegationInfo,
    ) -> ModifierEntry | None:
        """
        Extract intensity modifier from tokens.

        Collects all tokens BEFORE the target adjective (excluding negation)
        and attempts to match them against the modifier lexicon.

        This handles multi-word modifiers like "a little bit" where individual
        tokens may have various POS tags (DET, ADJ, NOUN).
        """
        if adj_token is None:
            return None

        # Collect all tokens before the adjective, excluding negation
        modifier_tokens = []
        for token in doc:
            # Stop when we reach the adjective
            if token.i >= adj_token.i:
                break

            # Skip negation tokens
            if self._is_negation_token(token):
                continue

            # Skip punctuation
            if token.pos_ == "PUNCT":
                continue

            modifier_tokens.append(token.text.lower())

        if not modifier_tokens:
            return None

        # Build phrase and try to match against lexicon
        # Try progressively shorter prefixes (greedy matching)
        for end_idx in range(len(modifier_tokens), 0, -1):
            candidate_phrase = " ".join(modifier_tokens[:end_idx])
            entry, remainder = self.lexicon.extract_modifier(
                candidate_phrase, category_filter=ModifierCategory.ADVERB
            )
            if entry is not None:
                return entry

        return None

    def _is_negation_token(self, token: spacy.tokens.Token) -> bool:
        """Check if token is negation."""
        if token.dep_ == "neg":
            return True
        text = token.text.lower()
        return text in NEGATION_WORDS or text in NEGATION_CONTRACTIONS

    def _fallback_adjective(self, doc: spacy.tokens.Doc) -> str:
        """Fallback adjective extraction."""
        tokens = [t for t in doc if t.pos_ != "PUNCT"]
        if tokens:
            return tokens[-1].text.lower()
        return ""

    def _calculate_confidence(
        self, adjective: str | None, pos_tag: str | None, modifier: ModifierEntry | None
    ) -> float:
        """Calculate parsing confidence."""
        confidence = 1.0
        if adjective is None:
            confidence -= 0.4
        if pos_tag and pos_tag != "ADJ":
            confidence -= 0.2
        if modifier:
            confidence = min(1.0, confidence + 0.05)
        return max(0.0, confidence)


# =============================================================================
# Convenience Functions
# =============================================================================


def parse_adjective_phrases(phrases: list[str]) -> list[ParsedPhrase]:
    """
    Convenience function to parse a list of adjective phrases.

    Args:
        phrases: List of adjective phrases

    Returns:
        List of ParsedPhrase objects
    """
    parser = PhraseParser.from_default_resources()
    return parser.parse_list(phrases)
