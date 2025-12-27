"""
Modifier Lexicon - SO-CAL Intensifier Support for RnB Framework

This module provides access to the SO-CAL intensifier lexicon for
modifying personality adjective intensity.

Based on: Taboada et al. (2011) "Lexicon-Based Methods for Sentiment Analysis"

The lexicon contains ~200 modifiers (amplifiers and downtoners) with
empirically-validated intensity values from Mechanical Turk studies.

Example:
    lexicon = ModifierLexicon.from_default_resources()

    # Look up a modifier
    entry = lexicon.get("very")
    # ModifierEntry(modifier="very", value=0.2, category="adverb_intensifiers")

    # Apply to a base intensity
    base = 0.7  # From adjective weight
    modified = base * (1 + entry.value)  # 0.7 * 1.2 = 0.84
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ModifierCategory(str, Enum):
    """Categories of modifiers in the SO-CAL lexicon."""

    ADVERB = "adverb_intensifiers"
    ADJECTIVE = "adjective_intensifiers"
    QUANTIFIER = "quantifier_intensifiers"
    SPECIAL = "special_constructions"


class ModifierType(str, Enum):
    """Classification based on effect direction."""

    AMPLIFIER = "amplifier"  # value > 0, increases intensity
    DOWNTONER = "downtoner"  # value < 0, decreases intensity
    NEUTRAL = "neutral"  # value == 0, no effect
    NEGATOR = "near_negator"  # value <= -1.0, inverts/nullifies


@dataclass(frozen=True)
class ModifierEntry:
    """
    A single modifier from the SO-CAL lexicon.

    Attributes:
        modifier: The modifier word/phrase (normalized, underscores for spaces)
        phrase: Human-readable form (spaces instead of underscores)
        value: SO-CAL intensity modifier (-3.0 to +1.0)
        category: Which lexicon category this belongs to
    """

    modifier: str
    phrase: str
    value: float
    category: ModifierCategory

    @property
    def modifier_type(self) -> ModifierType:
        """Classify this modifier by its effect."""
        if self.value <= -1.0:
            return ModifierType.NEGATOR
        elif self.value < 0:
            return ModifierType.DOWNTONER
        elif self.value > 0:
            return ModifierType.AMPLIFIER
        else:
            return ModifierType.NEUTRAL

    @property
    def multiplier(self) -> float:
        """
        The multiplicative factor to apply.

        Formula: modified = base × multiplier
        """
        return 1.0 + self.value

    def apply(self, base_value: float, clamp: bool = True) -> float:
        """
        Apply this modifier to a base value.

        Uses SO-CAL multiplicative formula:
            modified = base × (1 + modifier_value)

        Args:
            base_value: Original intensity value
            clamp: If True, clamp result to [-1.0, 1.0]

        Returns:
            Modified intensity value

        Example:
            >>> entry = ModifierEntry("very", "very", 0.2, ModifierCategory.ADVERB)
            >>> entry.apply(0.7)
            0.84
        """
        result = base_value * self.multiplier
        if clamp:
            result = max(-1.0, min(1.0, result))
        return result

    def __repr__(self) -> str:
        sign = "+" if self.value >= 0 else ""
        return f"ModifierEntry('{self.phrase}', {sign}{self.value})"


class ModifierLexicon:
    """
    SO-CAL intensifier lexicon for personality adjective modification.

    Provides O(1) lookup of modifier values and application of the
    SO-CAL multiplicative intensity formula.

    The lexicon distinguishes between:
    - Adverb intensifiers: "very", "extremely", "slightly" (modify adjectives)
    - Adjective intensifiers: "total", "complete", "minor" (modify nouns)
    - Quantifiers: "a lot of", "plenty of" (modify quantities)
    - Special constructions: "difficult to", "hard to"

    For personality adjective parsing, adverb intensifiers are primary.

    Attributes:
        entries: Dict mapping normalized modifier → ModifierEntry

    Example:
        lexicon = ModifierLexicon.from_default_resources()

        # Look up modifier
        if entry := lexicon.get("extremely"):
            modified = entry.apply(base_intensity)

        # Check if phrase starts with a modifier
        modifier, remainder = lexicon.extract_modifier("very lazy")
        # modifier = ModifierEntry("very", 0.2, ...)
        # remainder = "lazy"
    """

    def __init__(self, entries: dict[str, ModifierEntry]):
        """
        Initialize with pre-built entry dictionary.

        Args:
            entries: Dict mapping normalized modifier string to ModifierEntry
        """
        self._entries = entries

        # Build phrase-to-modifier index for extraction
        # Sort by length (longest first) for greedy matching
        self._phrases_by_length: list[tuple[str, str]] = sorted(
            [(e.phrase, e.modifier) for e in entries.values()],
            key=lambda x: len(x[0]),
            reverse=True,
        )

        logger.debug(f"ModifierLexicon initialized with {len(entries)} entries")

    @classmethod
    def from_yaml(cls, path: Path) -> "ModifierLexicon":
        """
        Load lexicon from YAML file.

        Args:
            path: Path to modifiers.yaml

        Returns:
            Configured ModifierLexicon
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        entries: dict[str, ModifierEntry] = {}

        # Process each category
        category_keys = [
            ("adverb_intensifiers", ModifierCategory.ADVERB),
            ("adjective_intensifiers", ModifierCategory.ADJECTIVE),
            ("quantifier_intensifiers", ModifierCategory.QUANTIFIER),
            ("special_constructions", ModifierCategory.SPECIAL),
        ]

        for yaml_key, category in category_keys:
            if yaml_key not in data:
                continue

            for modifier, value in data[yaml_key].items():
                # Normalize: lowercase, strip
                normalized = modifier.lower().strip()
                # Convert underscores to spaces for phrase form
                phrase = normalized.replace("_", " ")

                entry = ModifierEntry(
                    modifier=normalized,
                    phrase=phrase,
                    value=float(value),
                    category=category,
                )
                entries[normalized] = entry

                # Also index by phrase form if different
                if phrase != normalized:
                    entries[phrase] = entry

        logger.info(f"Loaded {len(entries)} modifier entries from {path}")
        return cls(entries)

    @classmethod
    def from_default_resources(cls) -> "ModifierLexicon":
        """
        Load lexicon from default resource location.

        Returns:
            Configured ModifierLexicon
        """
        resources_dir = Path(__file__).parent / "data"
        return cls.from_yaml(resources_dir / "modifiers.yaml")

    def get(self, modifier: str) -> ModifierEntry | None:
        """
        Look up a modifier by name.

        Args:
            modifier: Modifier word/phrase (case-insensitive)

        Returns:
            ModifierEntry if found, None otherwise
        """
        normalized = modifier.lower().strip()
        return self._entries.get(normalized)

    def __getitem__(self, modifier: str) -> ModifierEntry:
        """
        Look up modifier, raise KeyError if not found.

        Args:
            modifier: Modifier word/phrase

        Returns:
            ModifierEntry

        Raises:
            KeyError: If modifier not in lexicon
        """
        entry = self.get(modifier)
        if entry is None:
            raise KeyError(f"Modifier '{modifier}' not found in lexicon")
        return entry

    def __contains__(self, modifier: str) -> bool:
        """Check if modifier is in lexicon."""
        return self.get(modifier) is not None

    def __len__(self) -> int:
        """Number of unique modifiers (excluding duplicate phrase forms)."""
        # Count unique by modifier field
        return len(set(e.modifier for e in self._entries.values()))

    def extract_modifier(
        self, phrase: str, category_filter: ModifierCategory | None = None
    ) -> tuple[ModifierEntry | None, str]:
        """
        Extract leading modifier from a phrase.

        Uses greedy matching (longest modifier first) to handle
        multi-word modifiers like "a little bit".

        Args:
            phrase: Input phrase like "very lazy" or "a little bit tired"
            category_filter: If set, only match modifiers from this category

        Returns:
            Tuple of (ModifierEntry or None, remaining phrase)

        Example:
            >>> lexicon.extract_modifier("very lazy")
            (ModifierEntry("very", 0.2, ...), "lazy")

            >>> lexicon.extract_modifier("a little bit tired")
            (ModifierEntry("a_little_bit", -0.5, ...), "tired")

            >>> lexicon.extract_modifier("lazy")
            (None, "lazy")
        """
        phrase_lower = phrase.lower().strip()

        # Try each known modifier phrase (longest first)
        for mod_phrase, mod_key in self._phrases_by_length:
            if phrase_lower.startswith(mod_phrase):
                entry = self._entries[mod_key]

                # Apply category filter
                if category_filter and entry.category != category_filter:
                    continue

                # Extract remainder
                remainder = phrase_lower[len(mod_phrase) :].lstrip()
                return (entry, remainder)

        # No modifier found
        return (None, phrase)

    def list_modifiers(
        self,
        category: ModifierCategory | None = None,
        modifier_type: ModifierType | None = None,
    ) -> list[ModifierEntry]:
        """
        List modifiers matching filters.

        Args:
            category: Filter by category (ADVERB, ADJECTIVE, etc.)
            modifier_type: Filter by type (AMPLIFIER, DOWNTONER, etc.)

        Returns:
            List of matching ModifierEntry objects
        """
        # Get unique entries
        seen = set()
        results = []

        for entry in self._entries.values():
            if entry.modifier in seen:
                continue
            seen.add(entry.modifier)

            if category and entry.category != category:
                continue
            if modifier_type and entry.modifier_type != modifier_type:
                continue

            results.append(entry)

        return sorted(results, key=lambda e: (e.category.value, -e.value))

    def apply_modifier(
        self, modifier: str, base_value: float, clamp: bool = True
    ) -> float:
        """
        Apply a modifier to a base value.

        Convenience method that looks up and applies in one step.
        Returns base_value unchanged if modifier not found.

        Args:
            modifier: Modifier word/phrase
            base_value: Original intensity value
            clamp: If True, clamp result to [-1.0, 1.0]

        Returns:
            Modified value (or base_value if modifier not found)
        """
        entry = self.get(modifier)
        if entry is None:
            return base_value
        return entry.apply(base_value, clamp=clamp)

    def get_adverb_modifiers(self) -> list[ModifierEntry]:
        """Get all adverb intensifiers (primary for adjective modification)."""
        return self.list_modifiers(category=ModifierCategory.ADVERB)

    def get_amplifiers(self) -> list[ModifierEntry]:
        """Get all amplifiers (increase intensity)."""
        return self.list_modifiers(modifier_type=ModifierType.AMPLIFIER)

    def get_downtoners(self) -> list[ModifierEntry]:
        """Get all downtoners (decrease intensity)."""
        return self.list_modifiers(modifier_type=ModifierType.DOWNTONER)
