"""Unit tests for affect-based and mood-based influence operators"""

import pytest

from rnb.influence.affect_based import (
    CooperationEngagementInfluence,
    CooperationHelpfulnessInfluence,
    CooperationVerbosityInfluence,
    DominanceAssertivenessInfluence,
    DominanceDirectivenessInfluence,
    TrustOpennessInfluence,
    TrustVulnerabilityInfluence,
)
from rnb.influence.context import InfluenceContext
from rnb.influence.mood_based import (
    EnergyInitiativeInfluence,
    EnergyLengthInfluence,
    HappinessPositivityInfluence,
    HappinessToneInfluence,
    SatisfactionPatienceInfluence,
)
from rnb.personality.state import AffectDimension, MoodDimension, PersonalityState

# ===== Affect-based Fixtures =====


@pytest.fixture
def high_cooperation_context():
    """Agent with high cooperation (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.COOPERATION] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_cooperation_context():
    """Agent with low cooperation (-0.5)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.COOPERATION] = -0.5
    return InfluenceContext.from_personality(state)


@pytest.fixture
def neutral_cooperation_context():
    """Agent with neutral cooperation (0.0)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.COOPERATION] = 0.0
    return InfluenceContext.from_personality(state)


@pytest.fixture
def high_trust_context():
    """Agent with high trust (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.TRUST] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_trust_context():
    """Agent with low trust (-0.5)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.TRUST] = -0.5
    return InfluenceContext.from_personality(state)


@pytest.fixture
def high_dominance_context():
    """Agent with high dominance (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.DOMINANCE] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_dominance_context():
    """Agent with low dominance (-0.5)"""
    state = PersonalityState(agent_id="test_agent")
    state.affects[AffectDimension.DOMINANCE] = -0.5
    return InfluenceContext.from_personality(state)


# ===== Mood-based Fixtures =====


@pytest.fixture
def high_energy_context():
    """Agent with high energy (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.moods[MoodDimension.ENERGY] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_energy_context():
    """Agent with low energy (-0.5)"""
    state = PersonalityState(agent_id="test_agent")
    state.moods[MoodDimension.ENERGY] = -0.5
    return InfluenceContext.from_personality(state)


@pytest.fixture
def high_happiness_context():
    """Agent with high happiness (0.9)"""
    state = PersonalityState(agent_id="test_agent")
    state.moods[MoodDimension.HAPPINESS] = 0.9
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_happiness_context():
    """Agent with low happiness (-0.5)"""
    state = PersonalityState(agent_id="test_agent")
    state.moods[MoodDimension.HAPPINESS] = -0.5
    return InfluenceContext.from_personality(state)


@pytest.fixture
def high_satisfaction_context():
    """Agent with high satisfaction (0.8)"""
    state = PersonalityState(agent_id="test_agent")
    state.moods[MoodDimension.SATISFACTION] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_satisfaction_context():
    """Agent with low satisfaction (-0.6)"""
    state = PersonalityState(agent_id="test_agent")
    state.moods[MoodDimension.SATISFACTION] = -0.6
    return InfluenceContext.from_personality(state)


# ===== Cooperation Operator Tests =====


class TestCooperationVerbosityInfluence:
    """Tests for CooperationVerbosityInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = CooperationVerbosityInfluence()

        assert op.name == "cooperation_verbosity"
        assert op.category == "affect_based"
        assert "detail" in op.description.lower()

    def test_applies_always(
        self,
        high_cooperation_context,
        low_cooperation_context,
        neutral_cooperation_context,
    ):
        """Test operator always applies (cooperation is fundamental)"""
        op = CooperationVerbosityInfluence()

        assert op.applies(high_cooperation_context) is True
        assert op.applies(low_cooperation_context) is True
        assert op.applies(neutral_cooperation_context) is True

    def test_apply_high_cooperation(self, high_cooperation_context):
        """Test operator adds thorough/detailed guidance for high cooperation"""
        op = CooperationVerbosityInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, high_cooperation_context)

        assert base in result
        assert "thorough" in result.lower() or "detailed" in result.lower()

    def test_apply_low_cooperation(self, low_cooperation_context):
        """Test operator adds brief/minimal guidance for low cooperation"""
        op = CooperationVerbosityInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, low_cooperation_context)

        assert base in result
        assert "brief" in result.lower() or "minimal" in result.lower()

    def test_priority(self):
        """Test operator has high priority (sets basic style)"""
        op = CooperationVerbosityInfluence()
        assert op.get_activation_priority() == 150


class TestCooperationHelpfulnessInfluence:
    """Tests for CooperationHelpfulnessInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = CooperationHelpfulnessInfluence()

        assert op.name == "cooperation_helpfulness"
        assert op.category == "affect_based"

    def test_applies_high_cooperation(self, high_cooperation_context):
        """Test operator activates for high cooperation (>0.6)"""
        op = CooperationHelpfulnessInfluence()
        assert op.applies(high_cooperation_context) is True

    def test_applies_low_cooperation(self, low_cooperation_context):
        """Test operator activates for low cooperation (<-0.3)"""
        op = CooperationHelpfulnessInfluence()
        assert op.applies(low_cooperation_context) is True

    def test_not_applies_neutral(self, neutral_cooperation_context):
        """Test operator doesn't activate for neutral cooperation"""
        op = CooperationHelpfulnessInfluence()
        assert op.applies(neutral_cooperation_context) is False

    def test_apply_high_cooperation(self, high_cooperation_context):
        """Test operator adds proactive helpfulness"""
        op = CooperationHelpfulnessInfluence()
        base = "Explain recursion."

        result = op.apply(base, high_cooperation_context)

        assert base in result
        assert "proactive" in result.lower() or "additional" in result.lower()


class TestCooperationEngagementInfluence:
    """Tests for CooperationEngagementInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = CooperationEngagementInfluence()

        assert op.name == "cooperation_engagement"
        assert op.category == "affect_based"

    def test_applies_only_at_extremes(self):
        """Test operator activates only at extremes"""
        op = CooperationEngagementInfluence()

        # Should activate at 0.8
        state_high = PersonalityState(agent_id="test")
        state_high.affects[AffectDimension.COOPERATION] = 0.8
        context_high = InfluenceContext.from_personality(state_high)
        assert op.applies(context_high) is True

        # Should NOT activate at 0.5
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.affects[AffectDimension.COOPERATION] = 0.5
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_cooperation(self, high_cooperation_context):
        """Test operator adds engagement depth"""
        op = CooperationEngagementInfluence()
        base = "Explain recursion."

        result = op.apply(base, high_cooperation_context)

        assert base in result
        assert "engagement" in result.lower() or "interest" in result.lower()


# ===== Trust Operator Tests =====


class TestTrustOpennessInfluence:
    """Tests for TrustOpennessInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = TrustOpennessInfluence()

        assert op.name == "trust_openness"
        assert op.category == "affect_based"
        assert "transparency" in op.description.lower()

    def test_applies_high_trust(self, high_trust_context):
        """Test operator activates for high trust (>0.6)"""
        op = TrustOpennessInfluence()
        assert op.applies(high_trust_context) is True

    def test_applies_low_trust(self, low_trust_context):
        """Test operator activates for low trust (<-0.3)"""
        op = TrustOpennessInfluence()
        assert op.applies(low_trust_context) is True

    def test_apply_high_trust(self, high_trust_context):
        """Test operator adds openness about limitations"""
        op = TrustOpennessInfluence()
        base = "Explain quantum computing."

        result = op.apply(base, high_trust_context)

        assert base in result
        assert "limitations" in result.lower() or "uncertainties" in result.lower()

    def test_apply_low_trust(self, low_trust_context):
        """Test operator maintains confident tone for low trust"""
        op = TrustOpennessInfluence()
        base = "Explain quantum computing."

        result = op.apply(base, low_trust_context)

        assert base in result
        assert "confident" in result.lower() or "avoid" in result.lower()


class TestTrustVulnerabilityInfluence:
    """Tests for TrustVulnerabilityInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = TrustVulnerabilityInfluence()

        assert op.name == "trust_vulnerability"
        assert op.category == "affect_based"

    def test_applies_only_at_extremes(self):
        """Test operator activates only at extremes (>0.7 or <-0.5)"""
        op = TrustVulnerabilityInfluence()

        # Should activate at 0.8
        state_high = PersonalityState(agent_id="test")
        state_high.affects[AffectDimension.TRUST] = 0.8
        context_high = InfluenceContext.from_personality(state_high)
        assert op.applies(context_high) is True

        # Should NOT activate at 0.5
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.affects[AffectDimension.TRUST] = 0.5
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_trust(self, high_trust_context):
        """Test operator adds nuanced expression"""
        op = TrustVulnerabilityInfluence()
        base = "Explain AI ethics."

        result = op.apply(base, high_trust_context)

        assert base in result
        assert "nuanced" in result.lower() or "complexity" in result.lower()


# ===== Dominance Operator Tests =====


class TestDominanceAssertivenessInfluence:
    """Tests for DominanceAssertivenessInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = DominanceAssertivenessInfluence()

        assert op.name == "dominance_assertiveness"
        assert op.category == "affect_based"
        assert "assertiveness" in op.description.lower()

    def test_applies_high_dominance(self, high_dominance_context):
        """Test operator activates for high dominance"""
        op = DominanceAssertivenessInfluence()
        assert op.applies(high_dominance_context) is True

    def test_applies_low_dominance(self, low_dominance_context):
        """Test operator activates for low dominance"""
        op = DominanceAssertivenessInfluence()
        assert op.applies(low_dominance_context) is True

    def test_apply_high_dominance(self, high_dominance_context):
        """Test operator adds assertive, direct language"""
        op = DominanceAssertivenessInfluence()
        base = "Should I learn Python?"

        result = op.apply(base, high_dominance_context)

        assert base in result
        assert "direct" in result.lower() or "assertive" in result.lower()

    def test_apply_low_dominance(self, low_dominance_context):
        """Test operator adds deferential language"""
        op = DominanceAssertivenessInfluence()
        base = "Should I learn Python?"

        result = op.apply(base, low_dominance_context)

        assert base in result
        assert "suggestions" in result.lower() or "defer" in result.lower()


class TestDominanceDirectivenessInfluence:
    """Tests for DominanceDirectivenessInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = DominanceDirectivenessInfluence()

        assert op.name == "dominance_directiveness"
        assert op.category == "affect_based"

    def test_applies_only_at_extremes(self):
        """Test operator activates only at extremes"""
        op = DominanceDirectivenessInfluence()

        # Should activate at 0.8
        state_high = PersonalityState(agent_id="test")
        state_high.affects[AffectDimension.DOMINANCE] = 0.8
        context_high = InfluenceContext.from_personality(state_high)
        assert op.applies(context_high) is True

        # Should NOT activate at 0.5
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.affects[AffectDimension.DOMINANCE] = 0.5
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_dominance(self, high_dominance_context):
        """Test operator adds instructive approach"""
        op = DominanceDirectivenessInfluence()
        base = "Help me with this task."

        result = op.apply(base, high_dominance_context)

        assert base in result
        assert "instructive" in result.lower() or "guiding" in result.lower()


# ===== Energy Mood Operator Tests =====


class TestEnergyLengthInfluence:
    """Tests for EnergyLengthInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = EnergyLengthInfluence()

        assert op.name == "energy_length"
        assert op.category == "mood_based"
        assert "length" in op.description.lower()

    def test_applies_high_energy(self, high_energy_context):
        """Test operator activates for high energy (>0.6)"""
        op = EnergyLengthInfluence()
        assert op.applies(high_energy_context) is True

    def test_applies_low_energy(self, low_energy_context):
        """Test operator activates for low energy (<-0.3)"""
        op = EnergyLengthInfluence()
        assert op.applies(low_energy_context) is True

    def test_not_applies_neutral(self):
        """Test operator doesn't activate for neutral energy"""
        state = PersonalityState(agent_id="test")
        state.moods[MoodDimension.ENERGY] = 0.0
        context = InfluenceContext.from_personality(state)

        op = EnergyLengthInfluence()
        assert op.applies(context) is False

    def test_apply_low_energy(self, low_energy_context):
        """Test operator adds conciseness for low energy"""
        op = EnergyLengthInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, low_energy_context)

        assert base in result
        assert "concise" in result.lower() or "essential" in result.lower()

    def test_apply_high_energy(self, high_energy_context):
        """Test operator allows elaboration for high energy"""
        op = EnergyLengthInfluence()
        base = "Explain photosynthesis."

        result = op.apply(base, high_energy_context)

        assert base in result
        assert "elaborate" in result.lower() or "extended" in result.lower()

    def test_priority(self):
        """Test operator priority is appropriate for moods"""
        op = EnergyLengthInfluence()
        assert op.get_activation_priority() == 120


class TestEnergyInitiativeInfluence:
    """Tests for EnergyInitiativeInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = EnergyInitiativeInfluence()

        assert op.name == "energy_initiative"
        assert op.category == "mood_based"

    def test_applies_only_at_extremes(self):
        """Test operator activates only at extremes (>0.7 or <-0.5)"""
        op = EnergyInitiativeInfluence()

        # Should activate at 0.8
        state_high = PersonalityState(agent_id="test")
        state_high.moods[MoodDimension.ENERGY] = 0.8
        context_high = InfluenceContext.from_personality(state_high)
        assert op.applies(context_high) is True

        # Should NOT activate at 0.5
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.moods[MoodDimension.ENERGY] = 0.5
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_energy(self, high_energy_context):
        """Test operator adds initiative for high energy"""
        op = EnergyInitiativeInfluence()
        base = "Explain recursion."

        result = op.apply(base, high_energy_context)

        assert base in result
        assert "initiative" in result.lower() or "follow-up" in result.lower()


# ===== Happiness Mood Operator Tests =====


class TestHappinessToneInfluence:
    """Tests for HappinessToneInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = HappinessToneInfluence()

        assert op.name == "happiness_tone"
        assert op.category == "mood_based"
        assert "tone" in op.description.lower()

    def test_applies_high_happiness(self, high_happiness_context):
        """Test operator activates for high happiness (>0.7)"""
        op = HappinessToneInfluence()
        assert op.applies(high_happiness_context) is True

    def test_applies_low_happiness(self, low_happiness_context):
        """Test operator activates for low happiness (<-0.3)"""
        op = HappinessToneInfluence()
        assert op.applies(low_happiness_context) is True

    def test_apply_high_happiness(self, high_happiness_context):
        """Test operator adds warmth for high happiness"""
        op = HappinessToneInfluence()
        base = "Tell me about your day."

        result = op.apply(base, high_happiness_context)

        assert base in result
        assert "warmth" in result.lower() or "positivity" in result.lower()

    def test_apply_low_happiness(self, low_happiness_context):
        """Test operator maintains neutral tone for low happiness"""
        op = HappinessToneInfluence()
        base = "Tell me about your day."

        result = op.apply(base, low_happiness_context)

        assert base in result
        assert "neutral" in result.lower() or "professional" in result.lower()


class TestHappinessPositivityInfluence:
    """Tests for HappinessPositivityInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = HappinessPositivityInfluence()

        assert op.name == "happiness_positivity"
        assert op.category == "mood_based"

    def test_applies_only_at_high_extremes(self):
        """Test operator activates only at high extremes (>0.8 or <-0.5)"""
        op = HappinessPositivityInfluence()

        # Should activate at 0.9
        state_high = PersonalityState(agent_id="test")
        state_high.moods[MoodDimension.HAPPINESS] = 0.9
        context_high = InfluenceContext.from_personality(state_high)
        assert op.applies(context_high) is True

        # Should NOT activate at 0.7
        state_moderate = PersonalityState(agent_id="test")
        state_moderate.moods[MoodDimension.HAPPINESS] = 0.7
        context_moderate = InfluenceContext.from_personality(state_moderate)
        assert op.applies(context_moderate) is False

    def test_apply_high_happiness(self, high_happiness_context):
        """Test operator adds positive framing"""
        op = HappinessPositivityInfluence()
        base = "Discuss challenges in AI."

        result = op.apply(base, high_happiness_context)

        assert base in result
        assert "positive" in result.lower() or "encouraging" in result.lower()


# ===== Satisfaction Mood Operator Tests =====


class TestSatisfactionPatienceInfluence:
    """Tests for SatisfactionPatienceInfluence operator"""

    def test_initialization(self):
        """Test operator is properly initialized"""
        op = SatisfactionPatienceInfluence()

        assert op.name == "satisfaction_patience"
        assert op.category == "mood_based"
        assert "patience" in op.description.lower()

    def test_applies_high_satisfaction(self, high_satisfaction_context):
        """Test operator activates for high satisfaction (>0.7)"""
        op = SatisfactionPatienceInfluence()
        assert op.applies(high_satisfaction_context) is True

    def test_applies_low_satisfaction(self, low_satisfaction_context):
        """Test operator activates for low satisfaction (<-0.4)"""
        op = SatisfactionPatienceInfluence()
        assert op.applies(low_satisfaction_context) is True

    def test_apply_high_satisfaction(self, high_satisfaction_context):
        """Test operator adds patience and thoroughness"""
        op = SatisfactionPatienceInfluence()
        base = "Explain a complex topic."

        result = op.apply(base, high_satisfaction_context)

        assert base in result
        assert "thorough" in result.lower() or "patient" in result.lower()

    def test_apply_low_satisfaction(self, low_satisfaction_context):
        """Test operator focuses on efficiency for low satisfaction"""
        op = SatisfactionPatienceInfluence()
        base = "Explain a complex topic."

        result = op.apply(base, low_satisfaction_context)

        assert base in result
        assert "efficiency" in result.lower() or "streamlined" in result.lower()


# ===== Integration Tests =====


class TestAffectMoodOperatorChaining:
    """Test that affect and mood operators can be applied together"""

    def test_cooperation_and_energy_operators(self):
        """Test combining cooperation and energy operators"""
        state = PersonalityState(agent_id="test")
        state.affects[AffectDimension.COOPERATION] = 0.8
        state.moods[MoodDimension.ENERGY] = -0.5
        context = InfluenceContext.from_personality(state)

        coop_op = CooperationVerbosityInfluence()
        energy_op = EnergyLengthInfluence()

        base = "Explain photosynthesis."
        result = base
        result = coop_op.apply(result, context)
        result = energy_op.apply(result, context)

        # Should have both influences
        # High cooperation wants thorough, but low energy wants concise
        # Energy should override (applied later) or create tension
        assert "Communication style:" in result
        assert "Length:" in result

    def test_trust_and_happiness_operators(self):
        """Test combining trust and happiness operators"""
        state = PersonalityState(agent_id="test")
        state.affects[AffectDimension.TRUST] = 0.8
        state.moods[MoodDimension.HAPPINESS] = 0.9
        context = InfluenceContext.from_personality(state)

        trust_op = TrustOpennessInfluence()
        happiness_op = HappinessToneInfluence()

        base = "Discuss AI limitations."
        result = base
        result = trust_op.apply(result, context)
        result = happiness_op.apply(result, context)

        # Both should be present
        assert "Transparency:" in result or "limitations" in result.lower()
        assert "Emotional tone:" in result or "warmth" in result.lower()


class TestOperatorPriorityOrdering:
    """Test that operators have appropriate priority ordering"""

    def test_affect_operators_higher_priority_than_moods(self):
        """Test affect operators have higher priority than mood operators"""
        # Affects are more fundamental to interaction style
        coop_op = CooperationVerbosityInfluence()
        energy_op = EnergyLengthInfluence()

        assert coop_op.get_activation_priority() > energy_op.get_activation_priority()

    def test_mood_operators_have_consistent_range(self):
        """Test mood operators are in 120-130 range"""
        energy_len = EnergyLengthInfluence()
        energy_init = EnergyInitiativeInfluence()
        happiness_tone = HappinessToneInfluence()
        happiness_pos = HappinessPositivityInfluence()
        satisfaction = SatisfactionPatienceInfluence()

        priorities = [
            energy_len.get_activation_priority(),
            energy_init.get_activation_priority(),
            happiness_tone.get_activation_priority(),
            happiness_pos.get_activation_priority(),
            satisfaction.get_activation_priority(),
        ]

        assert all(120 <= p <= 130 for p in priorities)

    def test_affect_operators_have_consistent_range(self):
        """Test affect operators are in 150-170 range"""
        coop_verb = CooperationVerbosityInfluence()
        coop_help = CooperationHelpfulnessInfluence()
        trust_open = TrustOpennessInfluence()
        dom_assert = DominanceAssertivenessInfluence()

        priorities = [
            coop_verb.get_activation_priority(),
            coop_help.get_activation_priority(),
            trust_open.get_activation_priority(),
            dom_assert.get_activation_priority(),
        ]

        assert all(150 <= p <= 170 for p in priorities)
