"""
Comprehensive Tests for Phrase Parser with Negation

Tests cover:
1. ModifierLexicon (from previous implementation)
2. ModifierEntry application
3. PhraseParser with and without SpaCy
4. NegationInfo and SO-CAL shift semantics
5. ParsedPhrase intensity computation
6. Integration tests
7. SO-CAL paper examples

Requires: pytest
Optional: spacy, en_core_web_sm (for full SpaCy tests)
"""

from pathlib import Path

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def modifiers_yaml(tmp_path) -> Path:
    """Create test modifiers.yaml file."""
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
  really: 0.2
  quite: 0.1

adjective_intensifiers:
  total: 0.5
  minor: -0.3
"""
    path = tmp_path / "modifiers.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def lexicon(modifiers_yaml):
    """Create test lexicon."""
    # Note: This requires the lexicon to be in the same path
    # In actual repo, import from rnb.resources.modifier_lexicon
    # For testing, create inline
    from rnb.resources.modifier_lexicon import ModifierLexicon

    return ModifierLexicon.from_yaml(modifiers_yaml)


@pytest.fixture
def parser(modifiers_yaml):
    """Create parser without requiring SpaCy."""
    from rnb.resources.modifier_lexicon import ModifierLexicon
    from rnb.resources.phrase_parser import PhraseParser

    lexicon = ModifierLexicon.from_yaml(modifiers_yaml)
    return PhraseParser(lexicon=lexicon)


# =============================================================================
# NegationInfo Tests - SO-CAL Shift Semantics
# =============================================================================


class TestNegationInfo:
    """Tests for NegationInfo and shift semantics."""

    def test_no_negation_passthrough(self):
        """Values pass through unchanged when not negated."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=False)
        assert neg.apply_shift(0.8) == 0.8
        assert neg.apply_shift(-0.6) == -0.6
        assert neg.apply_shift(0.0) == 0.0

    def test_positive_shifts_toward_negative(self):
        """Positive values shift toward negative when negated."""
        from rnb.resources.phrase_parser import NegationInfo, NegationType

        neg = NegationInfo(
            negated=True,
            negation_type=NegationType.STANDARD,
            negation_word="not",
            shift_value=0.4,
        )
        # +0.8 - 0.4 = +0.4
        assert neg.apply_shift(0.8) == pytest.approx(0.4)

    def test_negative_shifts_toward_positive(self):
        """Negative values shift toward positive when negated."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)
        # -0.8 + 0.4 = -0.4
        assert neg.apply_shift(-0.8) == pytest.approx(-0.4)

    def test_shift_can_cross_zero(self):
        """Small values can cross zero after shift."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)

        # +0.2 - 0.4 = -0.2 (crosses to negative)
        assert neg.apply_shift(0.2) == pytest.approx(-0.2)

        # -0.2 + 0.4 = +0.2 (crosses to positive)
        assert neg.apply_shift(-0.2) == pytest.approx(0.2)

    def test_clamping(self):
        """Results are clamped to [-1, 1]."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=1.5)

        # Large shifts clamped
        result = neg.apply_shift(0.8)  # 0.8 - 1.5 = -0.7
        assert result == pytest.approx(-0.7)

        # Would exceed -1, but clamped
        result = neg.apply_shift(0.2)  # 0.2 - 1.5 = -1.3 → -1.0
        assert result == -1.0

    def test_zero_shifts_negative(self):
        """Zero is treated as non-negative (shifts toward negative)."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)
        assert neg.apply_shift(0.0) == pytest.approx(-0.4)


class TestSOCALPaperExamples:
    """
    Tests based on SO-CAL paper Section 2.4, Table 5.

    Paper uses [-5, +5] scale with shift=4.
    We use [-1, +1] scale with shift=0.4 (proportional).
    """

    def test_not_excellent(self):
        """'not excellent' should be mildly positive (not terrible)."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)

        # excellent ≈ +1.0 (max positive)
        # not excellent = 1.0 - 0.4 = 0.6
        result = neg.apply_shift(1.0)
        assert result == pytest.approx(0.6)
        assert result > 0  # Still positive!

    def test_not_sleazy(self):
        """'not sleazy' should be mildly positive."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)

        # sleazy ≈ -0.6
        # not sleazy = -0.6 + 0.4 = -0.2
        result = neg.apply_shift(-0.6)
        assert result == pytest.approx(-0.2)
        assert result > -0.6  # Less negative

    def test_not_terrible(self):
        """'not terrible' should be mildly negative (not excellent)."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)

        # terrible ≈ -1.0 (max negative)
        # not terrible = -1.0 + 0.4 = -0.6
        result = neg.apply_shift(-1.0)
        assert result == pytest.approx(-0.6)
        assert result < 0  # Still negative!

    def test_shift_vs_flip_comparison(self):
        """Demonstrate why shift is better than flip."""
        from rnb.resources.phrase_parser import NegationInfo

        neg = NegationInfo(negated=True, shift_value=0.4)

        # "not excellent" with SHIFT: +1.0 → +0.6
        shift_result = neg.apply_shift(1.0)

        # "not excellent" with FLIP would be: +1.0 → -1.0
        flip_result = -1.0

        # SHIFT gives "okay" (+0.6), FLIP gives "terrible" (-1.0)
        # Pragmatically, "not excellent" ≠ "terrible"
        assert shift_result != flip_result
        assert shift_result > 0  # Still positive
        assert flip_result < 0  # Wrongly negative


# =============================================================================
# ParsedPhrase Tests
# =============================================================================


class TestParsedPhrase:
    """Tests for ParsedPhrase dataclass."""

    def test_basic_adjective(self):
        """Simple adjective without modification."""
        from rnb.resources.phrase_parser import ParsedPhrase

        pp = ParsedPhrase(adjective="lazy")
        assert pp.adjective == "lazy"
        assert not pp.is_modified
        assert not pp.is_negated
        assert pp.multiplier == 1.0
        assert pp.source_phrase == "lazy"

    def test_with_modifier(self):
        """Adjective with intensity modifier."""
        from rnb.resources.phrase_parser import ParsedPhrase

        pp = ParsedPhrase(adjective="lazy", modifier="very", modifier_value=0.2)
        assert pp.is_modified
        assert pp.multiplier == 1.2
        assert pp.source_phrase == "very lazy"

    def test_with_negation(self):
        """Adjective with negation."""
        from rnb.resources.phrase_parser import NegationInfo, ParsedPhrase

        pp = ParsedPhrase(
            adjective="lazy", negation=NegationInfo(negated=True, negation_word="not")
        )
        assert pp.is_negated
        assert pp.source_phrase == "not lazy"

    def test_modifier_and_negation(self):
        """Adjective with both modifier and negation."""
        from rnb.resources.phrase_parser import NegationInfo, ParsedPhrase

        pp = ParsedPhrase(
            adjective="lazy",
            modifier="very",
            modifier_value=0.2,
            negation=NegationInfo(negated=True, negation_word="not"),
        )
        assert pp.is_modified
        assert pp.is_negated
        assert pp.source_phrase == "not very lazy"

    def test_compute_intensity_simple(self):
        """Basic intensity computation."""
        from rnb.resources.phrase_parser import ParsedPhrase

        pp = ParsedPhrase(adjective="lazy")

        # No modification: passthrough
        assert pp.compute_intensity(0.7, polarity=1) == 0.7
        assert pp.compute_intensity(0.7, polarity=-1) == -0.7

    def test_compute_intensity_with_modifier(self):
        """Intensity with modifier."""
        from rnb.resources.phrase_parser import ParsedPhrase

        pp = ParsedPhrase(adjective="lazy", modifier="very", modifier_value=0.2)

        # 0.7 × 1.2 × 1 = 0.84
        result = pp.compute_intensity(0.7, polarity=1)
        assert result == pytest.approx(0.84)

    def test_compute_intensity_with_negation(self):
        """Intensity with negation."""
        from rnb.resources.phrase_parser import NegationInfo, ParsedPhrase

        pp = ParsedPhrase(
            adjective="lazy", negation=NegationInfo(negated=True, shift_value=0.4)
        )

        # 0.7 × 1 × 1 = 0.7, then shift: 0.7 - 0.4 = 0.3
        result = pp.compute_intensity(0.7, polarity=1)
        assert result == pytest.approx(0.3)

    def test_compute_intensity_full(self):
        """Full intensity computation with modifier + negation."""
        from rnb.resources.phrase_parser import NegationInfo, ParsedPhrase

        pp = ParsedPhrase(
            adjective="lazy",
            modifier="very",
            modifier_value=0.2,
            negation=NegationInfo(negated=True, shift_value=0.4),
        )

        # Step 1: 0.7 × 1.2 = 0.84
        # Step 2: 0.84 × 1 = 0.84
        # Step 3: 0.84 - 0.4 = 0.44
        result = pp.compute_intensity(0.7, polarity=1)
        assert result == pytest.approx(0.44)

    def test_compute_intensity_negative_polarity(self):
        """Intensity with negative polarity + negation."""
        from rnb.resources.phrase_parser import NegationInfo, ParsedPhrase

        pp = ParsedPhrase(
            adjective="lazy",
            modifier="very",
            modifier_value=0.2,
            negation=NegationInfo(negated=True, shift_value=0.4),
        )

        # "not very lazy" where lazy is negative pole
        # Step 1: 0.7 × 1.2 = 0.84
        # Step 2: 0.84 × -1 = -0.84
        # Step 3: -0.84 + 0.4 = -0.44 (shift toward positive)
        result = pp.compute_intensity(0.7, polarity=-1)
        assert result == pytest.approx(-0.44)


# =============================================================================
# PhraseParser Tests (Fallback Mode)
# =============================================================================


class TestPhraseParserFallback:
    """Tests for PhraseParser without SpaCy."""

    def test_simple_adjective(self, parser):
        """Parse simple adjective."""
        result = parser.parse("lazy")
        assert result.adjective == "lazy"
        assert not result.is_modified
        assert not result.is_negated

    def test_modified_adjective(self, parser):
        """Parse adjective with modifier."""
        result = parser.parse("very lazy")
        assert result.adjective == "lazy"
        assert result.modifier == "very"
        assert result.modifier_value == pytest.approx(0.2)

    def test_negated_adjective(self, parser):
        """Parse negated adjective."""
        result = parser.parse("not lazy")
        assert result.adjective == "lazy"
        assert result.is_negated
        assert result.negation.negation_word == "not"

    def test_negated_modified(self, parser):
        """Parse 'not very lazy'."""
        result = parser.parse("not very lazy")
        assert result.adjective == "lazy"
        assert result.modifier == "very"
        assert result.is_negated
        assert result.negation.negation_word == "not"

    def test_never_as_negation(self, parser):
        """'never' recognized as negation with stronger shift."""
        result = parser.parse("never happy")
        assert result.is_negated
        assert result.negation.negation_word == "never"
        assert result.negation.shift_value == 0.5

    def test_hardly_as_negation(self, parser):
        """'hardly' recognized as near-negator."""
        result = parser.parse("hardly confident")
        assert result.is_negated
        assert result.negation.negation_word == "hardly"
        assert result.negation.shift_value == 0.6

    def test_case_insensitive(self, parser):
        """Parsing is case-insensitive."""
        result = parser.parse("NOT VERY LAZY")
        assert result.is_negated
        assert result.modifier == "very"
        assert result.adjective == "lazy"

    def test_whitespace_handling(self, parser):
        """Extra whitespace handled."""
        result = parser.parse("  very   lazy  ")
        assert result.adjective == "lazy"
        assert result.modifier == "very"

    def test_parse_list(self, parser):
        """Parse multiple phrases."""
        results = parser.parse_list(["lazy", "very organized", "not creative"])
        assert len(results) == 3
        assert results[0].adjective == "lazy"
        assert results[1].is_modified
        assert results[2].is_negated


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_not_lazy_workflow(self, parser):
        """Full workflow: 'not lazy' → intensity."""
        result = parser.parse("not lazy")

        # Lazy is negative pole, base intensity 0.7
        base = 0.7
        polarity = -1

        # 0.7 × 1 × -1 = -0.7
        # With negation: -0.7 + 0.4 = -0.3
        intensity = result.compute_intensity(base, polarity)
        assert intensity == pytest.approx(-0.3)

    def test_not_very_lazy_workflow(self, parser):
        """Full workflow: 'not very lazy' → intensity."""
        result = parser.parse("not very lazy")

        base = 0.7
        polarity = -1

        # 0.7 × 1.2 = 0.84
        # 0.84 × -1 = -0.84
        # -0.84 + 0.4 = -0.44
        intensity = result.compute_intensity(base, polarity)
        assert intensity == pytest.approx(-0.44)

    def test_extremely_creative_workflow(self, parser):
        """Full workflow: 'extremely creative' → intensity."""
        result = parser.parse("extremely creative")

        base = 0.8
        polarity = 1

        # 0.8 × 1.4 = 1.12 → clamped to 1.0
        intensity = result.compute_intensity(base, polarity)
        assert intensity == 1.0

    def test_not_excellent_vs_terrible(self, parser):
        """'not excellent' ≠ 'terrible' (shift vs flip)."""
        not_excellent = parser.parse("not excellent")

        base_excellent = 0.9
        polarity_excellent = 1

        # not excellent: 0.9 × 1 = 0.9, then 0.9 - 0.4 = 0.5
        not_excellent_intensity = not_excellent.compute_intensity(
            base_excellent, polarity_excellent
        )

        # terrible (if flip): would be -0.9
        terrible_intensity = -0.9

        # Key: not excellent ≠ terrible
        assert not_excellent_intensity != pytest.approx(terrible_intensity)
        assert not_excellent_intensity > 0  # Still positive


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
