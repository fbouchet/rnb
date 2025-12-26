"""Unit tests for OperatorRegistry (RnB activation matrix)"""

import pytest
from rnb.personality.state import (
    PersonalityState,
    FFMTrait,
    MoodDimension,
    AffectDimension
)
from rnb.influence.context import InfluenceContext
from rnb.influence.base import InfluenceOperator
from rnb.influence.registry import OperatorRegistry


# ===== Test Operator Implementations =====

class MockOperatorAlpha(InfluenceOperator):
    """Mock operator that always activates"""
    
    def __init__(self):
        super().__init__(
            name="alpha",
            description="Always active test operator",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return True
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [ALPHA]"
    
    def get_activation_priority(self) -> int:
        return 50


class MockOperatorBeta(InfluenceOperator):
    """Mock operator with conditional activation"""
    
    def __init__(self):
        super().__init__(
            name="beta",
            description="Activates when extraversion > 0.5",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return context.personality.traits[FFMTrait.EXTRAVERSION] > 0.5
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [BETA]"
    
    def get_activation_priority(self) -> int:
        return 100


class MockOperatorGamma(InfluenceOperator):
    """Mock operator in different category"""
    
    def __init__(self):
        super().__init__(
            name="gamma",
            description="Different category operator",
            category="other"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return True
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [GAMMA]"
    
    def get_activation_priority(self) -> int:
        return 150


class MockOperatorNever(InfluenceOperator):
    """Mock operator that never activates"""
    
    def __init__(self):
        super().__init__(
            name="never",
            description="Never active",
            category="test"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        return False
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        return base_prompt + " [NEVER]"


# ===== Fixtures =====

@pytest.fixture
def registry():
    """Create empty registry for testing"""
    return OperatorRegistry()


@pytest.fixture
def high_extraversion_context():
    """Create context with high extraversion"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.8
    return InfluenceContext.from_personality(state)


@pytest.fixture
def low_extraversion_context():
    """Create context with low extraversion"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.3
    return InfluenceContext.from_personality(state)


# ===== Registration Tests =====

def test_register_operator(registry):
    """Test registering an operator"""
    op = MockOperatorAlpha()
    registry.register(op)
    
    assert len(registry) == 1
    assert "alpha" in registry
    assert registry.get_operator("alpha") == op


def test_register_duplicate_name_raises_error(registry):
    """Test registering operator with duplicate name raises error"""
    op1 = MockOperatorAlpha()
    registry.register(op1)
    
    op2 = MockOperatorAlpha()  # Same name
    with pytest.raises(ValueError) as exc_info:
        registry.register(op2)
    
    assert "already registered" in str(exc_info.value)


def test_register_multiple_operators(registry):
    """Test registering multiple operators"""
    registry.register(MockOperatorAlpha())
    registry.register(MockOperatorBeta())
    registry.register(MockOperatorGamma())
    
    assert len(registry) == 3


def test_register_with_custom_activation(registry, high_extraversion_context):
    """Test registering operator with custom activation rule"""
    op = MockOperatorNever()  # Normally never activates
    
    # Custom rule: always activate
    registry.register(op, custom_activation=lambda ctx: True)
    
    active = registry.get_active_operators(high_extraversion_context)
    assert op in active


# ===== Unregistration Tests =====

def test_unregister_existing_operator(registry):
    """Test unregistering an operator"""
    registry.register(MockOperatorAlpha())
    
    result = registry.unregister("alpha")
    
    assert result is True
    assert len(registry) == 0
    assert "alpha" not in registry


def test_unregister_nonexistent_operator(registry):
    """Test unregistering non-existent operator returns False"""
    result = registry.unregister("nonexistent")
    assert result is False


# ===== Query Tests =====

def test_get_operator_existing(registry):
    """Test retrieving existing operator"""
    op = MockOperatorAlpha()
    registry.register(op)
    
    retrieved = registry.get_operator("alpha")
    assert retrieved == op


def test_get_operator_nonexistent(registry):
    """Test retrieving non-existent operator returns None"""
    retrieved = registry.get_operator("nonexistent")
    assert retrieved is None


def test_list_operators_all(registry):
    """Test listing all operators"""
    registry.register(MockOperatorAlpha())
    registry.register(MockOperatorBeta())
    registry.register(MockOperatorGamma())
    
    operators = registry.list_operators()
    
    assert len(operators) == 3


def test_list_operators_by_category(registry):
    """Test listing operators by category"""
    registry.register(MockOperatorAlpha())   # category: test
    registry.register(MockOperatorBeta())    # category: test
    registry.register(MockOperatorGamma())   # category: other
    
    test_ops = registry.list_operators(category="test")
    other_ops = registry.list_operators(category="other")
    
    assert len(test_ops) == 2
    assert len(other_ops) == 1


def test_list_categories(registry):
    """Test listing all categories"""
    registry.register(MockOperatorAlpha())   # category: test
    registry.register(MockOperatorGamma())   # category: other
    
    categories = registry.list_categories()
    
    assert len(categories) == 2
    assert "test" in categories
    assert "other" in categories


# ===== Activation Tests =====

def test_get_active_operators_all_active(registry, high_extraversion_context):
    """Test getting active operators when all apply"""
    registry.register(MockOperatorAlpha())  # Always active
    registry.register(MockOperatorBeta())   # Active when extraversion > 0.5
    
    active = registry.get_active_operators(high_extraversion_context)
    
    assert len(active) == 2


def test_get_active_operators_some_active(registry, low_extraversion_context):
    """Test getting active operators when only some apply"""
    registry.register(MockOperatorAlpha())  # Always active
    registry.register(MockOperatorBeta())   # NOT active (extraversion too low)
    
    active = registry.get_active_operators(low_extraversion_context)
    
    assert len(active) == 1
    assert active[0].name == "alpha"


def test_get_active_operators_none_active(registry, high_extraversion_context):
    """Test getting active operators when none apply"""
    registry.register(MockOperatorNever())
    
    active = registry.get_active_operators(high_extraversion_context)
    
    assert len(active) == 0


def test_get_active_operators_sorted_by_priority(registry, high_extraversion_context):
    """Test active operators are sorted by priority"""
    alpha = MockOperatorAlpha()   # Priority 50
    beta = MockOperatorBeta()     # Priority 100
    gamma = MockOperatorGamma()   # Priority 150
    
    # Register in wrong order
    registry.register(gamma)
    registry.register(alpha)
    registry.register(beta)
    
    active = registry.get_active_operators(high_extraversion_context)
    
    # Should be sorted by priority (low to high)
    assert active[0].name == "alpha"    # 50
    assert active[1].name == "beta"     # 100
    assert active[2].name == "gamma"    # 150


def test_get_active_operators_by_category(registry, high_extraversion_context):
    """Test getting active operators filtered by category"""
    registry.register(MockOperatorAlpha())   # test category
    registry.register(MockOperatorBeta())    # test category
    registry.register(MockOperatorGamma())   # other category
    
    active_test = registry.get_active_operators(
        high_extraversion_context,
        category="test"
    )
    
    assert len(active_test) == 2
    assert all(op.category == "test" for op in active_test)


# ===== Application Tests =====

def test_apply_all_basic(registry, high_extraversion_context):
    """Test applying all active operators"""
    registry.register(MockOperatorAlpha())
    registry.register(MockOperatorBeta())
    
    base = "Explain recursion."
    result = registry.apply_all(base, high_extraversion_context)
    
    # Both operators should have modified
    assert "[ALPHA]" in result
    assert "[BETA]" in result
    assert result.startswith(base)


def test_apply_all_respects_priority(registry, high_extraversion_context):
    """Test apply_all applies operators in priority order"""
    registry.register(MockOperatorGamma())  # Priority 150
    registry.register(MockOperatorAlpha())  # Priority 50
    
    base = "Test."
    result = registry.apply_all(base, high_extraversion_context)
    
    # Alpha (priority 50) should appear before Gamma (priority 150)
    alpha_index = result.index("[ALPHA]")
    gamma_index = result.index("[GAMMA]")
    assert alpha_index < gamma_index


def test_apply_all_skips_inactive(registry, low_extraversion_context):
    """Test apply_all skips inactive operators"""
    registry.register(MockOperatorAlpha())  # Active
    registry.register(MockOperatorBeta())   # Not active (low extraversion)
    
    base = "Test."
    result = registry.apply_all(base, low_extraversion_context)
    
    assert "[ALPHA]" in result
    assert "[BETA]" not in result


def test_apply_all_empty_registry(registry, high_extraversion_context):
    """Test apply_all with no operators returns unchanged prompt"""
    base = "Test."
    result = registry.apply_all(base, high_extraversion_context)
    
    assert result == base


def test_apply_all_by_category(registry, high_extraversion_context):
    """Test applying only operators from specific category"""
    registry.register(MockOperatorAlpha())   # test category
    registry.register(MockOperatorGamma())   # other category
    
    base = "Test."
    result = registry.apply_all(base, high_extraversion_context, category="test")
    
    assert "[ALPHA]" in result
    assert "[GAMMA]" not in result


def test_apply_all_dry_run(registry, high_extraversion_context, capsys):
    """Test dry_run mode doesn't modify prompt"""
    registry.register(MockOperatorAlpha())
    
    base = "Test."
    result = registry.apply_all(base, high_extraversion_context, dry_run=True)
    
    # Should return unchanged
    assert result == base
    
    # Should print what would be applied
    captured = capsys.readouterr()
    assert "Would apply" in captured.out
    assert "alpha" in captured.out


# ===== Summary Tests =====

def test_get_activation_summary(registry, high_extraversion_context):
    """Test getting activation summary"""
    registry.register(MockOperatorAlpha())   # Active
    registry.register(MockOperatorBeta())    # Active (high extraversion)
    registry.register(MockOperatorGamma())   # Active
    registry.register(MockOperatorNever())   # Not active
    
    summary = registry.get_activation_summary(high_extraversion_context)
    
    assert summary["total_operators"] == 4
    assert summary["active_operators"] == 3
    assert summary["activation_rate"] == 0.75
    assert "test" in summary["active_by_category"]
    assert "other" in summary["active_by_category"]
    assert "alpha" in summary["active_operator_names"]
    assert "never" not in summary["active_operator_names"]


# ===== Utility Tests =====

def test_clear_registry(registry):
    """Test clearing all operators from registry"""
    registry.register(MockOperatorAlpha())
    registry.register(MockOperatorBeta())
    
    registry.clear()
    
    assert len(registry) == 0
    assert len(registry.list_categories()) == 0


def test_len_operator(registry):
    """Test __len__ returns correct count"""
    assert len(registry) == 0
    
    registry.register(MockOperatorAlpha())
    assert len(registry) == 1
    
    registry.register(MockOperatorBeta())
    assert len(registry) == 2


def test_contains_operator(registry):
    """Test __contains__ checks operator presence"""
    assert "alpha" not in registry
    
    registry.register(MockOperatorAlpha())
    assert "alpha" in registry


def test_repr(registry):
    """Test string representation"""
    registry.register(MockOperatorAlpha())
    registry.register(MockOperatorGamma())
    
    repr_str = repr(registry)
    
    assert "OperatorRegistry" in repr_str
    assert "operators=2" in repr_str
    assert "categories=2" in repr_str