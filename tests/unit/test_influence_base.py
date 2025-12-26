"""Unit tests for InfluenceOperator base classes"""

import pytest
from rnb.personality.state import (
    PersonalityState,
    FFMTrait,
    AffectDimension
)
from rnb.influence.context import InfluenceContext
from rnb.influence.base import (
    InfluenceOperator,
    CompositeInfluenceOperator
)


# ===== Test Implementations =====

class AlwaysActiveOperator(InfluenceOperator):
    """Test operator that always activates"""
    
    def __init__(self):
        super().__init__(
            name="always_active",
            description="Always active test operator",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return True
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [MODIFIED]"


class NeverActiveOperator(InfluenceOperator):
    """Test operator that never activates"""
    
    def __init__(self):
        super().__init__(
            name="never_active",
            description="Never active test operator",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return False
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [SHOULD_NOT_APPEAR]"


class ConditionalOperator(InfluenceOperator):
    """Test operator with conditional activation based on extraversion"""
    
    def __init__(self):
        super().__init__(
            name="conditional",
            description="Activates when extraversion > 0.5",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return context.personality.traits[FFMTrait.EXTRAVERSION] > 0.5
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " Be enthusiastic!"


class HighPriorityOperator(InfluenceOperator):
    """Test operator with high priority (low number)"""
    
    def __init__(self):
        super().__init__(
            name="high_priority",
            description="High priority operator",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return True
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [HIGH]"
    
    def get_activation_priority(self) -> int:
        return 10


class LowPriorityOperator(InfluenceOperator):
    """Test operator with low priority (high number)"""
    
    def __init__(self):
        super().__init__(
            name="low_priority",
            description="Low priority operator",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return True
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [LOW]"
    
    def get_activation_priority(self) -> int:
        return 200


# ===== Fixtures =====

@pytest.fixture
def sample_context():
    """Create sample influence context for testing"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.6
    state.affects[AffectDimension.COOPERATION] = 0.7
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_extraversion_context():
    """Create context with low extraversion"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.3
    return InfluenceContext.from_personality(state)


# ===== InfluenceOperator Tests =====

def test_operator_initialization():
    """Test basic operator initialization"""
    op = AlwaysActiveOperator()
    
    assert op.name == "always_active"
    assert op.description == "Always active test operator"
    assert op.category == "test"


def test_operator_default_priority():
    """Test default activation priority is 100"""
    op = AlwaysActiveOperator()
    assert op.get_activation_priority() == 100


def test_operator_custom_priority():
    """Test custom activation priority"""
    op = HighPriorityOperator()
    assert op.get_activation_priority() == 10


def test_operator_applies_always_active(sample_context):
    """Test operator that always activates"""
    op = AlwaysActiveOperator()
    assert op.applies(sample_context) is True


def test_operator_applies_never_active(sample_context):
    """Test operator that never activates"""
    op = NeverActiveOperator()
    assert op.applies(sample_context) is False


def test_operator_applies_conditional_true(sample_context):
    """Test conditional operator activates when condition met"""
    op = ConditionalOperator()
    # sample_context has extraversion = 0.6 > 0.5
    assert op.applies(sample_context) is True


def test_operator_applies_conditional_false(low_extraversion_context):
    """Test conditional operator doesn't activate when condition not met"""
    op = ConditionalOperator()
    # low_extraversion_context has extraversion = 0.3 < 0.5
    assert op.applies(low_extraversion_context) is False


def test_operator_apply_modifies_prompt(sample_context):
    """Test operator modifies prompt correctly"""
    op = AlwaysActiveOperator()
    base = "Explain recursion."
    
    result = op.apply(base, sample_context)
    
    assert result == "Explain recursion. [MODIFIED]"
    assert base in result


def test_operator_apply_conditional(sample_context):
    """Test conditional operator applies modification"""
    op = ConditionalOperator()
    base = "Explain recursion."
    
    result = op.apply(base, sample_context)
    
    assert "Be enthusiastic!" in result


def test_operator_repr():
    """Test operator string representation"""
    op = AlwaysActiveOperator()
    repr_str = repr(op)
    
    assert "AlwaysActiveOperator" in repr_str
    assert "always_active" in repr_str
    assert "test" in repr_str


def test_operator_str():
    """Test operator human-readable string"""
    op = AlwaysActiveOperator()
    str_repr = str(op)
    
    assert op.name in str_repr
    assert op.description in str_repr


# ===== CompositeInfluenceOperator Tests =====

def test_composite_initialization():
    """Test composite operator initialization"""
    op1 = AlwaysActiveOperator()
    op2 = ConditionalOperator()
    
    composite = CompositeInfluenceOperator(
        name="test_composite",
        description="Test composite operator",
        operators=[op1, op2]
    )
    
    assert composite.name == "test_composite"
    assert composite.category == "composite"
    assert len(composite.operators) == 2


def test_composite_applies_any_active(sample_context):
    """Test composite activates if any component activates"""
    always_active = AlwaysActiveOperator()
    never_active = NeverActiveOperator()
    
    composite = CompositeInfluenceOperator(
        name="test_composite",
        description="Test composite",
        operators=[always_active, never_active]
    )
    
    # Should activate because always_active applies
    assert composite.applies(sample_context) is True


def test_composite_applies_none_active(low_extraversion_context):
    """Test composite doesn't activate if no components activate"""
    never_active = NeverActiveOperator()
    conditional = ConditionalOperator()  # Won't activate with low extraversion
    
    composite = CompositeInfluenceOperator(
        name="test_composite",
        description="Test composite",
        operators=[never_active, conditional]
    )
    
    assert composite.applies(low_extraversion_context) is False


def test_composite_apply_chains_operators(sample_context):
    """Test composite applies all active operators in sequence"""
    op1 = AlwaysActiveOperator()  # Adds " [MODIFIED]"
    op2 = ConditionalOperator()    # Adds " Be enthusiastic!"
    
    composite = CompositeInfluenceOperator(
        name="test_composite",
        description="Test composite",
        operators=[op1, op2]
    )
    
    base = "Explain recursion."
    result = composite.apply(base, sample_context)
    
    # Both modifications should be present
    assert "[MODIFIED]" in result
    assert "Be enthusiastic!" in result
    assert result.startswith(base)


def test_composite_apply_respects_priority(sample_context):
    """Test composite applies operators in priority order"""
    low_priority = LowPriorityOperator()   # Priority 200, adds " [LOW]"
    high_priority = HighPriorityOperator() # Priority 10, adds " [HIGH]"
    
    composite = CompositeInfluenceOperator(
        name="test_composite",
        description="Test composite",
        operators=[low_priority, high_priority]  # Added in wrong order
    )
    
    base = "Test."
    result = composite.apply(base, sample_context)
    
    # High priority should be applied first (appears earlier in string)
    high_index = result.index("[HIGH]")
    low_index = result.index("[LOW]")
    assert high_index < low_index


def test_composite_apply_skips_inactive(sample_context):
    """Test composite skips operators that don't apply"""
    always_active = AlwaysActiveOperator()
    never_active = NeverActiveOperator()
    
    composite = CompositeInfluenceOperator(
        name="test_composite",
        description="Test composite",
        operators=[always_active, never_active]
    )
    
    base = "Test."
    result = composite.apply(base, sample_context)
    
    # Only always_active should modify
    assert "[MODIFIED]" in result
    assert "[SHOULD_NOT_APPEAR]" not in result


def test_composite_empty_operators_list(sample_context):
    """Test composite with no operators"""
    composite = CompositeInfluenceOperator(
        name="empty_composite",
        description="Empty composite",
        operators=[]
    )
    
    # Should not apply (no operators)
    assert composite.applies(sample_context) is False
    
    # Should return prompt unchanged
    base = "Test."
    result = composite.apply(base, sample_context)
    assert result == base