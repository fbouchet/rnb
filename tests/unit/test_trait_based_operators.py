"""Unit tests for trait-based influence operators"""

import pytest

from rnb.influence.context import InfluenceContext
from rnb.influence.trait_based import (
    DetailOrientedInfluence,
    EnthusiasmInfluence,
    ExpressionInfluence,
    PrecisionInfluence,
    SocialEnergyInfluence,
    StructureInfluence,
)
from rnb.personality.state import FFMTrait, PersonalityState

# ===== Fixtures =====


@pytest.fixture
def high_conscientiousness_context():
    """Agent with high conscientiousness (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_conscientiousness_context():
    """Agent with low conscientiousness (-0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.CONSCIENTIOUSNESS] = -0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def neutral_conscientiousness_context():
    """Agent with neutral conscientiousness (0.0)"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.0
    return InfluenceContext.from_personality(state)


@pytest.fixture
def high_extraversion_context():
    """Agent with high extraversion (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_extraversion_context():
    """Agent with low extraversion (-0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = -0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def neutral_extraversion_context():
    """Agent with neutral extraversion (0.0)"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.0
    return InfluenceContext.from_personality(state)


# ===== Conscientiousness Operator Tests =====


class TestStructureInfluence:
    """Tests for StructureInfluence operator (C2: Order)"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = StructureInfluence()

        assert op.name == "conscientiousness_structure"
        assert op.category == "trait_based"
        assert "C2: Order" in op.description

    def test_applies_high_conscientiousness(self, high_conscientiousness_context):
        """Test operator activates for high conscientiousness"""
        op = StructureInfluence()
        assert op.applies(high_conscientiousness_context) is True

    def test_applies_low_conscientiousness(self, low_conscientiousness_context):
        """Test operator activates for low conscientiousness"""
        op = StructureInfluence()
        assert op.applies(low_conscientiousness_context) is True

    def test_not_applies_neutral(self, neutral_conscientiousness_context):
        """Test operator doesn't activate for neutral conscientiousness"""
        op = StructureInfluence()
        assert op.applies(neutral_conscientiousness_context) is False

    def test_apply_high_conscientiousness(self, high_conscientiousness_context):
        """Test operator adds structure guidance for high conscientiousness"""
        op = StructureInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, high_conscientiousness_context)

        assert base in result
        assert "Structure:" in result
        assert "clear sections" in result.lower() or "logical flow" in result.lower()

    def test_apply_low_conscientiousness(self, low_conscientiousness_context):
        """Test operator adds flexibility guidance for low conscientiousness"""
        op = StructureInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, low_conscientiousness_context)

        assert base in result
        assert "Structure:" in result
        assert "flexible" in result.lower() or "freeform" in result.lower()

    def test_priority(self):
        """Test operator priority is appropriate for traits"""
        op = StructureInfluence()
        assert op.get_activation_priority() == 70


class TestDetailOrientedInfluence:
    """Tests for DetailOrientedInfluence operator (C1: Competence, C6: Deliberation)"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = DetailOrientedInfluence()

        assert op.name == "conscientiousness_detail"
        assert op.category == "trait_based"

    def test_applies_very_high_conscientiousness(self, high_conscientiousness_context):
        """Test operator activates for very high conscientiousness (>0.6)"""
        op = DetailOrientedInfluence()
        assert op.applies(high_conscientiousness_context) is True

    def test_not_applies_moderate_conscientiousness(self):
        """Test operator doesn't activate for moderate conscientiousness (0.5)"""
        state = PersonalityState(agent_id="test")
        state.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.5
        context = InfluenceContext.from_personality(state)

        op = DetailOrientedInfluence()
        assert op.applies(context) is False

    def test_apply_high_conscientiousness(self, high_conscientiousness_context):
        """Test operator adds thoroughness guidance"""
        op = DetailOrientedInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, high_conscientiousness_context)

        assert base in result
        assert "thorough" in result.lower() or "comprehensive" in result.lower()
        assert "caveats" in result.lower() or "limitations" in result.lower()


class TestPrecisionInfluence:
    """Tests for PrecisionInfluence operator (C6: Deliberation)"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = PrecisionInfluence()

        assert op.name == "conscientiousness_precision"
        assert op.category == "trait_based"

    def test_applies_only_at_extremes(self):
        """Test operator only activates at very high extremes (>0.7)"""
        op = PrecisionInfluence()

        # Should activate at 0.8
        state_high = PersonalityState(agent_id="test")
        state_high.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.8
        context_high = InfluenceContext.from_personality(state_high)
        assert op.applies(context_high) is True

        # Should NOT activate at 0.6
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.6
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_conscientiousness(self, high_conscientiousness_context):
        """Test operator adds precision guidance"""
        op = PrecisionInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, high_conscientiousness_context)

        assert base in result
        assert "precise" in result.lower() or "careful" in result.lower()

    def test_priority_higher_than_others(self):
        """Test precision has higher priority (applied earlier)"""
        precision = PrecisionInfluence()
        structure = StructureInfluence()

        assert precision.get_activation_priority() < structure.get_activation_priority()


# ===== Extraversion Operator Tests =====


class TestEnthusiasmInfluence:
    """Tests for EnthusiasmInfluence operator (E6: Positive Emotions)"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = EnthusiasmInfluence()

        assert op.name == "extraversion_enthusiasm"
        assert op.category == "trait_based"
        assert "E6: Positive Emotions" in op.description

    def test_applies_high_extraversion(self, high_extraversion_context):
        """Test operator activates for high extraversion"""
        op = EnthusiasmInfluence()
        assert op.applies(high_extraversion_context) is True

    def test_applies_low_extraversion(self, low_extraversion_context):
        """Test operator activates for low extraversion"""
        op = EnthusiasmInfluence()
        assert op.applies(low_extraversion_context) is True

    def test_not_applies_neutral(self, neutral_extraversion_context):
        """Test operator doesn't activate for neutral extraversion"""
        op = EnthusiasmInfluence()
        assert op.applies(neutral_extraversion_context) is False

    def test_apply_high_extraversion(self, high_extraversion_context):
        """Test operator adds enthusiasm for high extraversion"""
        op = EnthusiasmInfluence()
        base = "Explain recursion."

        result = op.apply(base, high_extraversion_context)

        assert base in result
        assert "enthusiastic" in result.lower() or "energetic" in result.lower()

    def test_apply_low_extraversion(self, low_extraversion_context):
        """Test operator adds reserve for low extraversion"""
        op = EnthusiasmInfluence()
        base = "Explain recursion."

        result = op.apply(base, low_extraversion_context)

        assert base in result
        assert "reserved" in result.lower() or "measured" in result.lower()


class TestExpressionInfluence:
    """Tests for ExpressionInfluence operator (E1: Warmth, E6: Positive Emotions)"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = ExpressionInfluence()

        assert op.name == "extraversion_expression"
        assert op.category == "trait_based"

    def test_applies_only_at_higher_extremes(self, high_extraversion_context):
        """Test operator activates at |value| > 0.6"""
        op = ExpressionInfluence()

        # Should activate at 0.8
        assert op.applies(high_extraversion_context) is True

        # Should NOT activate at 0.5
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.traits[FFMTrait.EXTRAVERSION] = 0.5
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_extraversion(self, high_extraversion_context):
        """Test operator adds expressiveness guidance"""
        op = ExpressionInfluence()
        base = "Explain recursion."

        result = op.apply(base, high_extraversion_context)

        assert base in result
        assert "expressive" in result.lower() or "engaging" in result.lower()


class TestSocialEnergyInfluence:
    """Tests for SocialEnergyInfluence operator (E2: Gregariousness, E4: Activity)"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = SocialEnergyInfluence()

        assert op.name == "extraversion_social_energy"
        assert op.category == "trait_based"

    def test_applies_only_at_very_high_extremes(self, high_extraversion_context):
        """Test operator activates only at |value| > 0.7"""
        op = SocialEnergyInfluence()

        # Should activate at 0.8
        assert op.applies(high_extraversion_context) is True

        # Should NOT activate at 0.6
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.traits[FFMTrait.EXTRAVERSION] = 0.6
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_extraversion(self, high_extraversion_context):
        """Test operator adds interactive engagement"""
        op = SocialEnergyInfluence()
        base = "Explain recursion."

        result = op.apply(base, high_extraversion_context)

        assert base in result
        assert "interactive" in result.lower() or "conversational" in result.lower()

    def test_priority_lower_than_specific_traits(self):
        """Test social energy has lower priority than specific traits"""
        social = SocialEnergyInfluence()
        enthusiasm = EnthusiasmInfluence()

        assert social.get_activation_priority() > enthusiasm.get_activation_priority()


# ===== Integration Tests =====


class TestOperatorActivationThresholds:
    """Test that operators activate at correct thresholds"""

    def test_activation_threshold_progression(self):
        """Test operators activate at progressively higher thresholds"""
        # StructureInfluence: > 0.5
        structure = StructureInfluence()

        # DetailOrientedInfluence: > 0.6
        detail = DetailOrientedInfluence()

        # PrecisionInfluence: > 0.7
        precision = PrecisionInfluence()

        # Test at 0.5: only structure activates
        state_05 = PersonalityState(agent_id="test")
        state_05.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.55
        context_05 = InfluenceContext.from_personality(state_05)

        assert structure.applies(context_05) is True
        assert detail.applies(context_05) is False
        assert precision.applies(context_05) is False

        # Test at 0.65: structure and detail activate
        state_065 = PersonalityState(agent_id="test")
        state_065.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.65
        context_065 = InfluenceContext.from_personality(state_065)

        assert structure.applies(context_065) is True
        assert detail.applies(context_065) is True
        assert precision.applies(context_065) is False

        # Test at 0.75: all activate
        state_075 = PersonalityState(agent_id="test")
        state_075.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.75
        context_075 = InfluenceContext.from_personality(state_075)

        assert structure.applies(context_075) is True
        assert detail.applies(context_075) is True
        assert precision.applies(context_075) is True


class TestOperatorChaining:
    """Test that multiple operators can be applied in sequence"""

    def test_multiple_conscientiousness_operators(self, high_conscientiousness_context):
        """Test applying multiple conscientiousness operators"""
        base = "Explain photosynthesis."

        # Apply in priority order (precision -> structure -> detail)
        precision = PrecisionInfluence()
        structure = StructureInfluence()
        detail = DetailOrientedInfluence()

        result = base
        result = precision.apply(result, high_conscientiousness_context)
        result = structure.apply(result, high_conscientiousness_context)
        result = detail.apply(result, high_conscientiousness_context)

        # All modifications should be present
        assert "precise" in result.lower() or "careful" in result.lower()
        assert "structure" in result.lower() or "organized" in result.lower()
        assert "thorough" in result.lower() or "comprehensive" in result.lower()

    def test_multiple_extraversion_operators(self, high_extraversion_context):
        """Test applying multiple extraversion operators"""
        base = "Explain recursion."

        enthusiasm = EnthusiasmInfluence()
        expression = ExpressionInfluence()
        social = SocialEnergyInfluence()

        result = base
        result = enthusiasm.apply(result, high_extraversion_context)
        result = expression.apply(result, high_extraversion_context)
        result = social.apply(result, high_extraversion_context)

        # All modifications should be present
        assert "enthusiastic" in result.lower() or "energetic" in result.lower()
        assert "expressive" in result.lower() or "engaging" in result.lower()
        assert "interactive" in result.lower() or "conversational" in result.lower()
