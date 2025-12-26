"""Unit tests for InfluenceContext (RnB Model M)"""

import pytest
from rnb.personality.state import (
    PersonalityState,
    FFMTrait,
    MoodDimension,
    AffectDimension
)
from rnb.influence.context import InfluenceContext


@pytest.fixture
def sample_personality():
    """Create sample personality state for testing"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.6
    state.moods[MoodDimension.HAPPINESS] = 0.5
    state.affects[AffectDimension.COOPERATION] = 0.7
    return state


def test_context_creation_from_personality(sample_personality):
    """Test creating context from personality state (M.A only)"""
    context = InfluenceContext.from_personality(sample_personality)
    
    assert context.personality == sample_personality
    assert context.user_context is None
    assert context.session_context is None
    assert context.task_context is None


def test_context_creation_with_all_components(sample_personality):
    """Test creating context with all Model M components"""
    user_ctx = {"preferences": {"verbosity": "high"}}
    session_ctx = {"turn": 5, "topic": "programming"}
    task_ctx = {"domain": "education", "difficulty": "intermediate"}
    
    context = InfluenceContext(
        personality=sample_personality,
        user_context=user_ctx,
        session_context=session_ctx,
        task_context=task_ctx
    )
    
    assert context.personality == sample_personality
    assert context.user_context == user_ctx
    assert context.session_context == session_ctx
    assert context.task_context == task_ctx


def test_has_user_context_true():
    """Test has_user_context returns True when M.U is present"""
    state = PersonalityState(agent_id="test")
    context = InfluenceContext(
        personality=state,
        user_context={"data": "value"}
    )
    
    assert context.has_user_context() is True


def test_has_user_context_false():
    """Test has_user_context returns False when M.U is absent"""
    state = PersonalityState(agent_id="test")
    context = InfluenceContext.from_personality(state)
    
    assert context.has_user_context() is False


def test_has_session_context_true():
    """Test has_session_context returns True when M.S is present"""
    state = PersonalityState(agent_id="test")
    context = InfluenceContext(
        personality=state,
        session_context={"turn": 1}
    )
    
    assert context.has_session_context() is True


def test_has_session_context_false():
    """Test has_session_context returns False when M.S is absent"""
    state = PersonalityState(agent_id="test")
    context = InfluenceContext.from_personality(state)
    
    assert context.has_session_context() is False


def test_has_task_context_true():
    """Test has_task_context returns True when M.T is present"""
    state = PersonalityState(agent_id="test")
    context = InfluenceContext(
        personality=state,
        task_context={"domain": "math"}
    )
    
    assert context.has_task_context() is True


def test_has_task_context_false():
    """Test has_task_context returns False when M.T is absent"""
    state = PersonalityState(agent_id="test")
    context = InfluenceContext.from_personality(state)
    
    assert context.has_task_context() is False


def test_get_agent_id(sample_personality):
    """Test retrieving agent ID from context"""
    context = InfluenceContext.from_personality(sample_personality)
    
    assert context.get_agent_id() == "test_agent"


def test_context_repr_personality_only(sample_personality):
    """Test string representation with only M.A"""
    context = InfluenceContext.from_personality(sample_personality)
    repr_str = repr(context)
    
    assert "M.A(personality)" in repr_str
    assert "M.U" not in repr_str
    assert "M.S" not in repr_str
    assert "M.T" not in repr_str


def test_context_repr_all_components(sample_personality):
    """Test string representation with all Model M components"""
    context = InfluenceContext(
        personality=sample_personality,
        user_context={"key": "value"},
        session_context={"key": "value"},
        task_context={"key": "value"}
    )
    repr_str = repr(context)
    
    assert "M.A(personality)" in repr_str
    assert "M.U(user)" in repr_str
    assert "M.S(session)" in repr_str
    assert "M.T(task)" in repr_str


def test_context_personality_access(sample_personality):
    """Test accessing personality state attributes through context"""
    context = InfluenceContext.from_personality(sample_personality)
    
    # Should be able to access FFM traits
    assert context.personality.traits[FFMTrait.EXTRAVERSION] == 0.6
    
    # Should be able to access moods
    assert context.personality.moods[MoodDimension.HAPPINESS] == 0.5
    
    # Should be able to access affects
    assert context.personality.affects[AffectDimension.COOPERATION] == 0.7


def test_context_immutability_concern():
    """
    Test that modifying personality through context affects the original.
    
    Note: This is expected behavior - context holds reference, not copy.
    If immutability needed, should be handled at PersonalityState level.
    """
    state = PersonalityState(agent_id="test")
    state.traits[FFMTrait.OPENNESS] = 0.5
    
    context = InfluenceContext.from_personality(state)
    
    # Modify through context
    context.personality.traits[FFMTrait.OPENNESS] = 0.8
    
    # Original state is also modified (same reference)
    assert state.traits[FFMTrait.OPENNESS] == 0.8