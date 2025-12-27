"""
Unit Tests for SO-CAL Modifier System

Tests for:
- ModifierLexicon loading and lookup
- ModifierEntry application
- PhraseParser parsing
- Integration with personality intensity calculation
"""

from pathlib import Path

import pytest

from rnb.resources.modifier_lexicon import (
    ModifierCategory,
    ModifierEntry,
    ModifierLexicon,
    ModifierType,
)
from rnb.resources.phrase_parser import ModifiedAdjective, PhraseParser

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def modifiers_yaml(tmp_path) -> Path:
    """Create a minimal test modifiers.yaml file."""
    content = """
metadata:
  source: "test"

adverb_intensifiers:
  very: 0.2
  extremely: 0.4
  slightly: -0.5
  somewhat: -0.3
  a_little_bit: -0.5
  the_most: 1.0

adjective_intensifiers:
  total: 0.5
  minor: -0.3
"""
    path = tmp_path / "modifiers.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def lexicon(modifiers_yaml) -> ModifierLexicon:
    """Create lexicon from test YAML."""
    return ModifierLexicon.from_yaml(modifiers_yaml)


@pytest.fixture
def parser(lexicon) -> PhraseParser:
    """Create parser with test lexicon."""
    return PhraseParser(lexicon)


# =============================================================================
# ModifierEntry Tests
# =============================================================================


class TestModifierEntry:
    """Tests for ModifierEntry dataclass."""

    def test_amplifier_type(self):
        """Test amplifier classification."""
        entry = ModifierEntry("very", "very", 0.2, ModifierCategory.ADVERB)
        assert entry.modifier_type == ModifierType.AMPLIFIER

    def test_downtoner_type(self):
        """Test downtoner classification."""
        entry = ModifierEntry("slightly", "slightly", -0.5, ModifierCategory.ADVERB)
        assert entry.modifier_type == ModifierType.DOWNTONER

    def test_negator_type(self):
        """Test near-negator classification."""
        entry = ModifierEntry("barely", "barely", -1.5, ModifierCategory.ADVERB)
        assert entry.modifier_type == ModifierType.NEGATOR

    def test_multiplier_positive(self):
        """Test multiplier calculation for amplifiers."""
        entry = ModifierEntry("very", "very", 0.2, ModifierCategory.ADVERB)
        assert entry.multiplier == 1.2

    def test_multiplier_negative(self):
        """Test multiplier calculation for downtoners."""
        entry = ModifierEntry("slightly", "slightly", -0.5, ModifierCategory.ADVERB)
        assert entry.multiplier == 0.5

    def test_apply_amplifier(self):
        """Test applying amplifier to base value."""
        entry = ModifierEntry("very", "very", 0.2, ModifierCategory.ADVERB)
        result = entry.apply(0.7)
        assert result == pytest.approx(0.84)

    def test_apply_downtoner(self):
        """Test applying downtoner to base value."""
        entry = ModifierEntry("slightly", "slightly", -0.5, ModifierCategory.ADVERB)
        result = entry.apply(0.8)
        assert result == pytest.approx(0.4)

    def test_apply_clamping(self):
        """Test that values are clamped to [-1, 1]."""
        entry = ModifierEntry("the_most", "the most", 1.0, ModifierCategory.ADVERB)
        result = entry.apply(0.8)  # 0.8 * 2.0 = 1.6 → clamped to 1.0
        assert result == 1.0

    def test_apply_negative_base(self):
        """Test applying modifier to negative base value."""
        entry = ModifierEntry("very", "very", 0.2, ModifierCategory.ADVERB)
        result = entry.apply(-0.7)
        assert result == pytest.approx(-0.84)

    def test_apply_downtoner_to_negative(self):
        """Test downtoner makes negative less negative."""
        entry = ModifierEntry("slightly", "slightly", -0.5, ModifierCategory.ADVERB)
        result = entry.apply(-0.8)
        assert result == pytest.approx(-0.4)


# =============================================================================
# ModifierLexicon Tests
# =============================================================================


class TestModifierLexicon:
    """Tests for ModifierLexicon class."""

    def test_load_from_yaml(self, modifiers_yaml):
        """Test loading lexicon from YAML file."""
        lexicon = ModifierLexicon.from_yaml(modifiers_yaml)
        assert len(lexicon) > 0

    def test_get_existing(self, lexicon):
        """Test getting existing modifier."""
        entry = lexicon.get("very")
        assert entry is not None
        assert entry.value == 0.2

    def test_get_nonexistent(self, lexicon):
        """Test getting non-existent modifier returns None."""
        entry = lexicon.get("nonexistent")
        assert entry is None

    def test_get_case_insensitive(self, lexicon):
        """Test lookup is case-insensitive."""
        entry = lexicon.get("VERY")
        assert entry is not None
        assert entry.value == 0.2

    def test_getitem(self, lexicon):
        """Test dictionary-style access."""
        entry = lexicon["very"]
        assert entry.value == 0.2

    def test_getitem_raises_keyerror(self, lexicon):
        """Test KeyError for missing modifier."""
        with pytest.raises(KeyError):
            _ = lexicon["nonexistent"]

    def test_contains(self, lexicon):
        """Test 'in' operator."""
        assert "very" in lexicon
        assert "nonexistent" not in lexicon

    def test_extract_single_word_modifier(self, lexicon):
        """Test extracting single-word modifier."""
        entry, remainder = lexicon.extract_modifier("very lazy")
        assert entry is not None
        assert entry.phrase == "very"
        assert remainder == "lazy"

    def test_extract_multi_word_modifier(self, lexicon):
        """Test extracting multi-word modifier."""
        entry, remainder = lexicon.extract_modifier("a little bit lazy")
        assert entry is not None
        assert entry.phrase == "a little bit"
        assert remainder == "lazy"

    def test_extract_no_modifier(self, lexicon):
        """Test when no modifier present."""
        entry, remainder = lexicon.extract_modifier("lazy")
        assert entry is None
        assert remainder == "lazy"

    def test_extract_with_category_filter(self, lexicon):
        """Test extraction with category filter."""
        # "total" is adjective intensifier, shouldn't match with adverb filter
        entry, remainder = lexicon.extract_modifier(
            "total disaster", category_filter=ModifierCategory.ADVERB
        )
        assert entry is None
        assert remainder == "total disaster"

    def test_list_modifiers(self, lexicon):
        """Test listing all modifiers."""
        modifiers = lexicon.list_modifiers()
        assert len(modifiers) > 0

    def test_list_modifiers_by_category(self, lexicon):
        """Test listing modifiers by category."""
        adverbs = lexicon.list_modifiers(category=ModifierCategory.ADVERB)
        adjectives = lexicon.list_modifiers(category=ModifierCategory.ADJECTIVE)

        assert all(m.category == ModifierCategory.ADVERB for m in adverbs)
        assert all(m.category == ModifierCategory.ADJECTIVE for m in adjectives)

    def test_list_modifiers_by_type(self, lexicon):
        """Test listing modifiers by type."""
        amplifiers = lexicon.list_modifiers(modifier_type=ModifierType.AMPLIFIER)
        downtoners = lexicon.list_modifiers(modifier_type=ModifierType.DOWNTONER)

        assert all(m.value > 0 for m in amplifiers)
        assert all(m.value < 0 for m in downtoners)

    def test_apply_modifier(self, lexicon):
        """Test convenience apply method."""
        result = lexicon.apply_modifier("very", 0.7)
        assert result == pytest.approx(0.84)

    def test_apply_modifier_not_found(self, lexicon):
        """Test apply returns unchanged value for unknown modifier."""
        result = lexicon.apply_modifier("unknown", 0.7)
        assert result == 0.7


# =============================================================================
# ModifiedAdjective Tests
# =============================================================================


class TestModifiedAdjective:
    """Tests for ModifiedAdjective dataclass."""

    def test_unmodified_adjective(self):
        """Test creating unmodified adjective."""
        ma = ModifiedAdjective(adjective="lazy")
        assert ma.adjective == "lazy"
        assert ma.modifier is None
        assert ma.modifier_value == 0.0
        assert not ma.is_modified

    def test_modified_adjective(self):
        """Test creating modified adjective."""
        ma = ModifiedAdjective(adjective="lazy", modifier="very", modifier_value=0.2)
        assert ma.adjective == "lazy"
        assert ma.modifier == "very"
        assert ma.modifier_value == 0.2
        assert ma.is_modified

    def test_source_phrase_auto(self):
        """Test source_phrase is auto-generated."""
        ma = ModifiedAdjective(adjective="lazy", modifier="very", modifier_value=0.2)
        assert ma.source_phrase == "very lazy"

    def test_source_phrase_unmodified(self):
        """Test source_phrase for unmodified adjective."""
        ma = ModifiedAdjective(adjective="lazy")
        assert ma.source_phrase == "lazy"

    def test_multiplier_unmodified(self):
        """Test multiplier is 1.0 for unmodified."""
        ma = ModifiedAdjective(adjective="lazy")
        assert ma.multiplier == 1.0

    def test_multiplier_amplified(self):
        """Test multiplier for amplified adjective."""
        ma = ModifiedAdjective(adjective="lazy", modifier="very", modifier_value=0.2)
        assert ma.multiplier == 1.2

    def test_apply_to_intensity(self):
        """Test applying to base intensity."""
        ma = ModifiedAdjective(adjective="lazy", modifier="very", modifier_value=0.2)
        result = ma.apply_to_intensity(0.7)
        assert result == pytest.approx(0.84)

    def test_apply_to_intensity_unmodified(self):
        """Test applying unmodified returns same value."""
        ma = ModifiedAdjective(adjective="lazy")
        result = ma.apply_to_intensity(0.7)
        assert result == 0.7


# =============================================================================
# PhraseParser Tests
# =============================================================================


class TestPhraseParser:
    """Tests for PhraseParser class."""

    def test_parse_unmodified(self, parser):
        """Test parsing unmodified adjective."""
        result = parser.parse("lazy")
        assert result.adjective == "lazy"
        assert result.modifier is None
        assert result.modifier_value == 0.0

    def test_parse_with_modifier(self, parser):
        """Test parsing modified adjective."""
        result = parser.parse("very lazy")
        assert result.adjective == "lazy"
        assert result.modifier == "very"
        assert result.modifier_value == 0.2

    def test_parse_case_insensitive(self, parser):
        """Test parsing is case-insensitive."""
        result = parser.parse("VERY LAZY")
        assert result.adjective == "lazy"
        assert result.modifier == "very"

    def test_parse_with_whitespace(self, parser):
        """Test parsing handles extra whitespace."""
        result = parser.parse("  very   lazy  ")
        assert result.adjective == "lazy"
        assert result.modifier == "very"

    def test_parse_multi_word_modifier(self, parser):
        """Test parsing multi-word modifier."""
        result = parser.parse("a little bit nervous")
        assert result.adjective == "nervous"
        assert result.modifier == "a little bit"
        assert result.modifier_value == -0.5

    def test_parse_preserves_source(self, parser):
        """Test original phrase is preserved."""
        result = parser.parse("very lazy")
        assert result.source_phrase == "very lazy"

    def test_parse_list(self, parser):
        """Test batch parsing."""
        results = parser.parse_list(["lazy", "very organized", "slightly ambitious"])

        assert len(results) == 3
        assert results[0].adjective == "lazy"
        assert results[0].modifier is None

        assert results[1].adjective == "organized"
        assert results[1].modifier == "very"

        assert results[2].adjective == "ambitious"
        assert results[2].modifier == "slightly"

    def test_parse_mixed_input(self, parser):
        """Test parsing mixed string/ModifiedAdjective input."""
        existing = ModifiedAdjective(
            adjective="creative", modifier="highly", modifier_value=0.2
        )
        results = parser.parse_mixed_input(["lazy", "very organized", existing])

        assert len(results) == 3
        assert results[0].adjective == "lazy"
        assert results[1].adjective == "organized"
        assert results[2] is existing


# =============================================================================
# Integration Tests
# =============================================================================


class TestModifierIntegration:
    """Integration tests for the modifier system."""

    def test_full_workflow(self, lexicon, parser):
        """Test complete parsing and application workflow."""
        # Parse phrase
        result = parser.parse("extremely lazy")
        assert result.adjective == "lazy"
        assert result.modifier == "extremely"

        # Apply to base intensity (simulating weight-based calculation)
        base_intensity = 0.7  # Would come from adjective weight
        modified = result.apply_to_intensity(base_intensity)

        # extremely has value 0.4, so 0.7 * 1.4 = 0.98
        assert modified == pytest.approx(0.98)

    def test_downtoner_workflow(self, lexicon, parser):
        """Test downtoner reduces intensity."""
        result = parser.parse("slightly ambitious")

        base_intensity = 0.8
        modified = result.apply_to_intensity(base_intensity)

        # slightly has value -0.5, so 0.8 * 0.5 = 0.4
        assert modified == pytest.approx(0.4)

    def test_unmodified_unchanged(self, parser):
        """Test unmodified adjective leaves intensity unchanged."""
        result = parser.parse("lazy")

        base_intensity = 0.7
        modified = result.apply_to_intensity(base_intensity)

        assert modified == base_intensity

    def test_clamping_at_max(self, lexicon, parser):
        """Test intensity is clamped at 1.0."""
        result = parser.parse("the most excellent")

        # the_most has value 1.0 (doubles)
        base_intensity = 0.8
        modified = result.apply_to_intensity(base_intensity)

        # 0.8 * 2.0 = 1.6 → clamped to 1.0
        assert modified == 1.0


# =============================================================================
# SO-CAL Paper Examples
# =============================================================================


class TestSOCALPaperExamples:
    """
    Tests based on examples from the SO-CAL paper.

    Taboada et al. (2011), Section 2.3:
    "if sleazy has an SO value of −3, somewhat sleazy would have an SO value of:
     −3 × (100% − 30%) = −2.1"
    """

    @pytest.fixture
    def paper_lexicon(self, tmp_path):
        """Create lexicon matching paper examples."""
        content = """
adverb_intensifiers:
  slightly: -0.5
  somewhat: -0.3
  very: 0.25
  most: 1.0
  really: 0.15
"""
        path = tmp_path / "paper_modifiers.yaml"
        path.write_text(content)
        return ModifierLexicon.from_yaml(path)

    def test_somewhat_sleazy_example(self, paper_lexicon):
        """Test 'somewhat sleazy' example from paper."""
        # Paper: sleazy = -3, somewhat = -30%, result = -2.1
        # Our scale: -3 normalized to ~-0.6 (if scale is -5 to 5)
        # Let's use raw values to match paper

        entry = paper_lexicon["somewhat"]
        base = -3.0  # sleazy in paper's scale

        # -3 * (1 + (-0.3)) = -3 * 0.7 = -2.1
        result = entry.apply(base, clamp=False)
        assert result == pytest.approx(-2.1)

    def test_most_excellent_example(self, paper_lexicon):
        """Test 'most excellent' example from paper."""
        # Paper: excellent = 5, most = +100%, result = 10
        entry = paper_lexicon["most"]
        base = 5.0  # excellent in paper's scale

        # 5 * (1 + 1.0) = 5 * 2.0 = 10
        result = entry.apply(base, clamp=False)
        assert result == pytest.approx(10.0)

    def test_really_very_good_recursive(self, paper_lexicon):
        """Test recursive intensification from paper."""
        # Paper: good = 3, "really very good"
        # = (3 × 1.25) × 1.15 = 4.3125

        really = paper_lexicon["really"]
        very = paper_lexicon["very"]
        base = 3.0  # good

        # Apply inner first: very good = 3 * 1.25 = 3.75
        after_very = very.apply(base, clamp=False)
        assert after_very == pytest.approx(3.75)

        # Then outer: really (very good) = 3.75 * 1.15 = 4.3125
        result = really.apply(after_very, clamp=False)
        assert result == pytest.approx(4.3125)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
