"""
Conformity tests for RnB personality validation.

Tests whether RnB agents produce personality assessment results
that match their designed personality profiles.

Run with:
    pytest tests/validation/test_conformity.py -v
    pytest tests/validation/test_conformity.py -v --archetype=resilient
    pytest tests/validation/test_conformity.py -v --instrument=bfi2s
    pytest tests/validation/test_conformity.py -v --tolerance=0.3
"""

import json

import pytest

from rnb.validation import (
    AssessmentResult,
    calculate_conformity_score,
    check_range_conformity,
    convert_to_rnb_scale,
    parse_batch_response,
)

# =============================================================================
# Instrument Loading Tests
# =============================================================================


@pytest.mark.validation
class TestInstrumentLoading:
    """Tests for instrument loading and structure."""

    def test_tipi_loads_correctly(self, tipi_instrument):
        """Test TIPI instrument loads with correct structure."""
        assert tipi_instrument.acronym == "TIPI"
        assert tipi_instrument.items_count == 10
        assert len(tipi_instrument.items) == 10
        assert tipi_instrument.scale.min == 1
        assert tipi_instrument.scale.max == 7

    def test_bfi2s_loads_correctly(self, bfi2s_instrument):
        """Test BFI-2-S instrument loads with correct structure."""
        assert bfi2s_instrument.acronym == "BFI-2-S"
        assert bfi2s_instrument.items_count == 30
        assert len(bfi2s_instrument.items) == 30
        assert bfi2s_instrument.scale.min == 1
        assert bfi2s_instrument.scale.max == 5

    def test_tipi_has_all_domains(self, tipi_instrument):
        """Test TIPI covers all Big Five domains."""
        domains = {item.domain for item in tipi_instrument.items}
        expected = {
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "emotional_stability",
            "openness",
        }
        assert domains == expected

    def test_bfi2s_has_all_domains(self, bfi2s_instrument):
        """Test BFI-2-S covers all Big Five domains."""
        domains = {item.domain for item in bfi2s_instrument.items}
        expected = {
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        }
        assert domains == expected

    def test_bfi2s_has_all_facets(self, bfi2s_instrument):
        """Test BFI-2-S has 15 facets (3 per domain)."""
        facets = {item.facet for item in bfi2s_instrument.items if item.facet}
        assert len(facets) == 15


# =============================================================================
# Archetype Loading Tests
# =============================================================================


@pytest.mark.validation
class TestArchetypeLoading:
    """Tests for archetype loading and structure."""

    def test_archetypes_load(self, archetypes):
        """Test archetypes load correctly."""
        assert len(archetypes) >= 4
        assert "resilient" in archetypes
        assert "overcontrolled" in archetypes
        assert "undercontrolled" in archetypes

    def test_archetype_has_required_fields(self, archetypes):
        """Test each archetype has required fields."""
        for _name, archetype in archetypes.items():
            assert archetype.traits is not None
            assert len(archetype.traits) == 5
            assert archetype.behavioral_markers is not None
            assert archetype.references is not None

    def test_archetype_traits_in_valid_range(self, archetypes):
        """Test archetype trait values are in [-1, 1] range."""
        for name, archetype in archetypes.items():
            for trait, value in archetype.traits.items():
                assert -1.0 <= value <= 1.0, f"{name}.{trait} = {value} out of range"

    def test_ruo_archetypes_have_literature_references(self, archetypes):
        """Test RUO archetypes have proper references."""
        ruo_names = ["resilient", "overcontrolled", "undercontrolled"]
        for name in ruo_names:
            if name in archetypes:
                refs = archetypes[name].references
                assert len(refs) > 0, f"{name} missing references"
                # Should cite key papers
                refs_text = " ".join(refs).lower()
                assert any(author in refs_text for author in ["asendorpf", "robins"])


# =============================================================================
# Response Parsing Tests
# =============================================================================


@pytest.mark.validation
class TestResponseParsing:
    """Tests for parsing LLM assessment responses."""

    def test_parse_json_response(self, tipi_instrument):
        """Test parsing JSON format response."""
        response = '{"1": 5, "2": 3, "3": 6, "4": 2, "5": 5, "6": 3, "7": 6, "8": 2, "9": 6, "10": 3}'
        parsed = parse_batch_response(response, tipi_instrument)

        assert len(parsed) == 10
        assert parsed[1] == 5
        assert parsed[10] == 3

    def test_parse_json_with_code_block(self, tipi_instrument):
        """Test parsing JSON wrapped in markdown code block."""
        response = '```json\n{"1": 5, "2": 3, "3": 6, "4": 2, "5": 5, "6": 3, "7": 6, "8": 2, "9": 6, "10": 3}\n```'
        parsed = parse_batch_response(response, tipi_instrument)

        assert len(parsed) == 10

    def test_parse_numbered_list(self, tipi_instrument):
        """Test parsing numbered list format."""
        response = "1. 5\n2. 3\n3. 6\n4. 2\n5. 5\n6. 3\n7. 6\n8. 2\n9. 6\n10. 3"
        parsed = parse_batch_response(response, tipi_instrument)

        assert len(parsed) == 10
        assert parsed[1] == 5

    def test_parse_invalid_response_raises(self, tipi_instrument):
        """Test that invalid response raises ValueError."""
        response = "I don't understand the question."

        with pytest.raises(ValueError):
            parse_batch_response(response, tipi_instrument)


# =============================================================================
# Scale Conversion Tests
# =============================================================================


@pytest.mark.validation
class TestScaleConversion:
    """Tests for scale conversion functions."""

    def test_tipi_to_rnb_conversion(self):
        """Test TIPI (1-7) to RnB (-1 to 1) conversion."""
        # TIPI midpoint = 4, half_range = 3
        assert convert_to_rnb_scale(7, 4.0, 3.0) == 1.0
        assert convert_to_rnb_scale(1, 4.0, 3.0) == -1.0
        assert convert_to_rnb_scale(4, 4.0, 3.0) == 0.0
        assert abs(convert_to_rnb_scale(5.5, 4.0, 3.0) - 0.5) < 0.01

    def test_bfi2s_to_rnb_conversion(self):
        """Test BFI-2-S (1-5) to RnB (-1 to 1) conversion."""
        # BFI-2-S midpoint = 3, half_range = 2
        assert convert_to_rnb_scale(5, 3.0, 2.0) == 1.0
        assert convert_to_rnb_scale(1, 3.0, 2.0) == -1.0
        assert convert_to_rnb_scale(3, 3.0, 2.0) == 0.0

    def test_neuroticism_inversion(self):
        """Test neuroticism score inversion for TIPI."""
        # TIPI emotional_stability -> neuroticism (inverted)
        es_score = 6.0  # High emotional stability
        rnb_neuroticism = convert_to_rnb_scale(es_score, 4.0, 3.0, invert=True)

        # High ES should give LOW neuroticism (negative on RnB scale)
        assert rnb_neuroticism < 0


# =============================================================================
# Scoring Tests
# =============================================================================


@pytest.mark.validation
class TestScoring:
    """Tests for domain and facet scoring."""

    def test_tipi_domain_scoring(self, tipi_assessor):
        """Test TIPI domain score calculation."""
        # Create mock responses for a resilient-like profile
        responses = {
            1: 6,  # Extraverted (E+)
            2: 2,  # Critical (A-, reversed)
            3: 6,  # Dependable (C+)
            4: 2,  # Anxious (N-, reversed for ES)
            5: 6,  # Open (O+)
            6: 2,  # Reserved (E-, reversed)
            7: 6,  # Sympathetic (A+)
            8: 2,  # Disorganized (C-, reversed)
            9: 6,  # Calm (ES+)
            10: 2,  # Conventional (O-, reversed)
        }

        result = tipi_assessor._calculate_scores(responses)

        # All domains should be high (above 5 on 1-7 scale)
        for domain, score in result.domain_scores.items():
            assert score >= 5.0, f"{domain} = {score} is too low"

    def test_bfi2s_facet_scoring(self, bfi2s_assessor):
        """Test BFI-2-S facet score calculation."""
        # Create neutral responses (all 3s on 1-5 scale)
        responses = {i: 3 for i in range(1, 31)}

        result = bfi2s_assessor._calculate_scores(responses)

        # Should have 15 facet scores
        assert result.facet_scores is not None
        assert len(result.facet_scores) == 15

        # All facets should be near midpoint
        for facet, score in result.facet_scores.items():
            assert 2.5 <= score <= 3.5, f"{facet} = {score}"


# =============================================================================
# Conformity Check Tests
# =============================================================================


@pytest.mark.validation
@pytest.mark.conformity
class TestConformityChecks:
    """Tests for conformity checking against archetypes."""

    def test_conformity_score_calculation(self):
        """Test conformity score calculation using pingouin."""
        designed = {
            "openness": 0.5,
            "conscientiousness": 0.6,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": -0.7,
        }

        # Perfect match
        measured_perfect = designed.copy()
        corr, devs, _ = calculate_conformity_score(designed, measured_perfect)
        assert corr > 0.99
        assert all(abs(d) < 0.01 for d in devs.values())

        # Complete opposite
        measured_opposite = {k: -v for k, v in designed.items()}
        corr, devs, _ = calculate_conformity_score(designed, measured_opposite)
        assert corr < -0.99

    def test_range_conformity_with_tolerance(self, archetypes, tolerance):
        """Test range conformity checking with tolerance parameter."""
        resilient = archetypes["resilient"]

        # Scores within tolerance of designed traits
        good_scores = {
            "openness": 0.4,  # Designed: 0.5, within 0.25 tolerance
            "conscientiousness": 0.7,  # Designed: 0.6, within 0.25 tolerance
            "extraversion": 0.6,  # Designed: 0.5, within 0.25 tolerance
            "agreeableness": 0.4,  # Designed: 0.5, within 0.25 tolerance
            "neuroticism": -0.6,  # Designed: -0.7, within 0.25 tolerance
        }

        all_in, per_domain, ranges = check_range_conformity(
            good_scores, resilient.traits, tolerance=tolerance
        )
        assert all_in

        # Scores outside tolerance
        bad_scores = {
            "openness": 0.0,  # Designed: 0.5, outside 0.25 tolerance
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.4,
            "neuroticism": -0.6,
        }

        all_in, per_domain, ranges = check_range_conformity(
            bad_scores, resilient.traits, tolerance=tolerance
        )
        assert not all_in
        assert not per_domain["openness"]

    def test_tolerance_affects_range(self):
        """Test that tolerance parameter correctly affects expected ranges."""
        designed = {"openness": 0.5}
        measured = {"openness": 0.6}

        # With small tolerance, should fail
        all_in, _, _ = check_range_conformity(measured, designed, tolerance=0.05)
        assert not all_in

        # With larger tolerance, should pass
        all_in, _, _ = check_range_conformity(measured, designed, tolerance=0.15)
        assert all_in

    def test_assessor_conformity_check(self, tipi_assessor, archetypes, tolerance):
        """Test assessor's conformity check method."""
        resilient = archetypes["resilient"]

        # Create a mock result matching resilient profile
        result = AssessmentResult(
            instrument="TIPI",
            raw_responses={i: 6 for i in range(1, 11)},
            domain_scores={
                "extraversion": 6.0,
                "agreeableness": 6.0,
                "conscientiousness": 6.0,
                "emotional_stability": 6.0,
                "openness": 6.0,
            },
            rnb_scores={
                "extraversion": 0.67,
                "agreeableness": 0.67,
                "conscientiousness": 0.67,
                "neuroticism": -0.67,
                "openness": 0.67,
            },
        )

        conformity = tipi_assessor.check_conformity(
            result, resilient, tolerance=tolerance
        )

        # Should have reasonable correlation with resilient profile
        assert conformity.correlation > 0.5


# =============================================================================
# Mock Assessment Tests
# =============================================================================


@pytest.mark.validation
@pytest.mark.conformity
class TestMockAssessment:
    """Tests using mock LLM client."""

    def test_batch_assessment_with_mock(self, tipi_assessor, mock_llm_client):
        """Test batch assessment with mock client."""
        # Configure mock to return neutral JSON
        mock_llm_client.fixed_response = json.dumps({str(i): 4 for i in range(1, 11)})

        result = tipi_assessor.assess(mock_llm_client, method="batch")

        assert result.instrument == "TIPI"
        assert len(result.raw_responses) == 10
        assert result.rnb_scores is not None

        # Neutral responses should give near-zero RnB scores
        for _domain, score in result.rnb_scores.items():
            assert abs(score) < 0.2

    def test_archetype_mock_produces_expected_profile(
        self, bfi2s_assessor, ruo_archetype, mock_llm_for_archetype
    ):
        """Test mock client produces profile matching archetype."""
        result = bfi2s_assessor.assess(mock_llm_for_archetype, method="batch")

        # Check that at least the key marker traits show expected direction
        if ruo_archetype.name == "resilient":
            # Resilient: low neuroticism is key marker
            assert result.rnb_scores["neuroticism"] < 0.3
        elif ruo_archetype.name == "overcontrolled":
            # Overcontrolled: low extraversion, high neuroticism
            assert result.rnb_scores["extraversion"] < 0.3
        elif ruo_archetype.name == "undercontrolled":
            # Undercontrolled: low conscientiousness
            assert result.rnb_scores["conscientiousness"] < 0.3


# =============================================================================
# Parametrized Archetype Tests
# =============================================================================


@pytest.mark.validation
@pytest.mark.conformity
class TestArchetypeConformity:
    """Parametrized conformity tests for all archetypes."""

    @pytest.mark.parametrize(
        "archetype_name",
        [
            "resilient",
            "overcontrolled",
            "undercontrolled",
            "average",
        ],
    )
    def test_archetype_traits_are_valid(self, archetype_name, archetypes):
        """Test that archetype traits are in valid RnB range."""
        archetype = archetypes[archetype_name]

        # Check all traits are in RnB range [-1, 1]
        for trait, value in archetype.traits.items():
            assert -1.0 <= value <= 1.0, f"{archetype_name}.{trait}: {value}"

        # Check all Big Five traits are present
        expected_traits = {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }
        assert set(archetype.traits.keys()) == expected_traits

    @pytest.mark.parametrize(
        "archetype_name",
        [
            "resilient",
            "overcontrolled",
            "undercontrolled",
        ],
    )
    def test_archetype_key_markers(self, archetype_name, archetypes):
        """Test that archetypes have expected key trait markers."""
        archetype = archetypes[archetype_name]

        if archetype_name == "resilient":
            # Key marker: low neuroticism
            assert archetype.traits["neuroticism"] < -0.5
            # Above average on other traits
            for trait in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
            ]:
                assert archetype.traits[trait] > 0

        elif archetype_name == "overcontrolled":
            # Key markers: high N, low E
            assert archetype.traits["neuroticism"] > 0.4
            assert archetype.traits["extraversion"] < -0.4

        elif archetype_name == "undercontrolled":
            # Key marker: low conscientiousness
            assert archetype.traits["conscientiousness"] < -0.4
