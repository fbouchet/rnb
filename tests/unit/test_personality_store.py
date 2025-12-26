"""Unit tests for PersonalityStateStore"""

import pytest
from types import MappingProxyType
from rnb.personality.backend import RedisBackend
from rnb.personality.store import PersonalityStateStore
from rnb.personality.state import (
    PersonalityState,
    FFMTrait,
    MoodDimension,
    AffectDimension
)
from rnb.personality.exceptions import AgentNotFoundError, InvalidValueError


@pytest.fixture
def backend():
    """Create RedisBackend for testing"""
    backend = RedisBackend()
    yield backend
    backend.flush_db()
    backend.close()


@pytest.fixture
def store(backend):
    """Create PersonalityStateStore for testing"""
    return PersonalityStateStore(backend)


@pytest.fixture
def test_agent(store):
    """Create a test agent"""
    state = PersonalityState(agent_id="test_agent")
    state.traits[FFMTrait.EXTRAVERSION] = 0.6
    state.moods[MoodDimension.HAPPINESS] = 0.5
    state.affects[AffectDimension.COOPERATION] = 0.7
    store.set_state(state)
    return "test_agent"


def test_get_state_existing(store, test_agent):
    """Test retrieving an existing agent's state"""
    state = store.get_state(test_agent)
    
    assert state is not None
    assert state.agent_id == test_agent
    assert state.traits[FFMTrait.EXTRAVERSION] == 0.6


def test_get_state_nonexistent(store):
    """Test retrieving a non-existent agent"""
    state = store.get_state("nonexistent")
    assert state is None


def test_set_state_creates_new(store):
    """Test setting state creates new agent"""
    new_state = PersonalityState(agent_id="new_agent")
    new_state.traits[FFMTrait.OPENNESS] = 0.8
    
    store.set_state(new_state)
    
    retrieved = store.get_state("new_agent")
    assert retrieved is not None
    assert retrieved.traits[FFMTrait.OPENNESS] == 0.8


def test_set_state_updates_existing(store, test_agent):
    """Test setting state updates existing agent"""
    state = store.get_state(test_agent)
    state.traits[FFMTrait.CONSCIENTIOUSNESS] = 0.9
    
    store.set_state(state)
    
    retrieved = store.get_state(test_agent)
    assert retrieved.traits[FFMTrait.CONSCIENTIOUSNESS] == 0.9


# ===== Trait Tests =====

def test_get_trait(store, test_agent):
    """Test getting single trait"""
    extraversion = store.get_trait(test_agent, FFMTrait.EXTRAVERSION)
    assert extraversion == 0.6


def test_get_trait_nonexistent_agent(store):
    """Test getting trait for non-existent agent raises error"""
    with pytest.raises(AgentNotFoundError) as exc_info:
        store.get_trait("nonexistent", FFMTrait.EXTRAVERSION)
    assert "nonexistent" in str(exc_info.value)


def test_get_traits_returns_immutable(store, test_agent):
    """Test get_traits returns immutable view"""
    traits = store.get_traits(test_agent)
    
    assert isinstance(traits, MappingProxyType)
    
    # Verify we can't modify it
    with pytest.raises(TypeError):
        traits[FFMTrait.OPENNESS] = 0.9


def test_get_traits_nonexistent_agent(store):
    """Test getting traits for non-existent agent raises error"""
    with pytest.raises(AgentNotFoundError):
        store.get_traits("nonexistent")


def test_set_trait(store, test_agent):
    """Test setting single trait"""
    store.set_trait(test_agent, FFMTrait.OPENNESS, 0.8)
    
    value = store.get_trait(test_agent, FFMTrait.OPENNESS)
    assert value == 0.8


def test_set_trait_invalid_range(store, test_agent):
    """Test setting trait with invalid value raises error"""
    with pytest.raises(InvalidValueError) as exc_info:
        store.set_trait(test_agent, FFMTrait.OPENNESS, 1.5)
    
    assert "1.5" in str(exc_info.value)


def test_set_traits_multiple(store, test_agent):
    """Test setting multiple traits"""
    new_traits = {
        FFMTrait.OPENNESS: 0.8,
        FFMTrait.CONSCIENTIOUSNESS: 0.9
    }
    
    store.set_traits(test_agent, new_traits)
    
    assert store.get_trait(test_agent, FFMTrait.OPENNESS) == 0.8
    assert store.get_trait(test_agent, FFMTrait.CONSCIENTIOUSNESS) == 0.9


# ===== Mood Tests =====

def test_get_mood(store, test_agent):
    """Test getting single mood dimension"""
    happiness = store.get_mood(test_agent, MoodDimension.HAPPINESS)
    assert happiness == 0.5


def test_get_moods_returns_immutable(store, test_agent):
    """Test get_moods returns immutable view"""
    moods = store.get_moods(test_agent)
    
    assert isinstance(moods, MappingProxyType)
    
    with pytest.raises(TypeError):
        moods[MoodDimension.ENERGY] = 0.9


def test_set_mood(store, test_agent):
    """Test setting single mood dimension"""
    store.set_mood(test_agent, MoodDimension.ENERGY, 0.7)
    
    value = store.get_mood(test_agent, MoodDimension.ENERGY)
    assert value == 0.7


def test_set_moods_multiple(store, test_agent):
    """Test setting multiple mood dimensions"""
    new_moods = {
        MoodDimension.HAPPINESS: 0.8,
        MoodDimension.ENERGY: 0.6
    }
    
    store.set_moods(test_agent, new_moods)
    
    assert store.get_mood(test_agent, MoodDimension.HAPPINESS) == 0.8
    assert store.get_mood(test_agent, MoodDimension.ENERGY) == 0.6


def test_update_mood_additive(store, test_agent):
    """Test update_mood adds to current values"""
    initial = store.get_mood(test_agent, MoodDimension.HAPPINESS)
    
    store.update_mood(test_agent, {MoodDimension.HAPPINESS: 0.2})
    
    final = store.get_mood(test_agent, MoodDimension.HAPPINESS)
    assert final == initial + 0.2


def test_update_mood_clamping_upper(store, test_agent):
    """Test update_mood clamps to upper bound"""
    store.set_mood(test_agent, MoodDimension.HAPPINESS, 0.9)
    
    store.update_mood(test_agent, {MoodDimension.HAPPINESS: 0.5})
    
    final = store.get_mood(test_agent, MoodDimension.HAPPINESS)
    assert final == 1.0


def test_update_mood_clamping_lower(store, test_agent):
    """Test update_mood clamps to lower bound"""
    store.set_mood(test_agent, MoodDimension.HAPPINESS, -0.9)
    
    store.update_mood(test_agent, {MoodDimension.HAPPINESS: -0.5})
    
    final = store.get_mood(test_agent, MoodDimension.HAPPINESS)
    assert final == -1.0


# ===== Affect Tests =====

def test_get_affect(store, test_agent):
    """Test getting single affect dimension"""
    cooperation = store.get_affect(test_agent, AffectDimension.COOPERATION)
    assert cooperation == 0.7


def test_get_affects_returns_immutable(store, test_agent):
    """Test get_affects returns immutable view"""
    affects = store.get_affects(test_agent)
    
    assert isinstance(affects, MappingProxyType)
    
    with pytest.raises(TypeError):
        affects[AffectDimension.TRUST] = 0.9


def test_set_affect(store, test_agent):
    """Test setting single affect dimension"""
    store.set_affect(test_agent, AffectDimension.TRUST, 0.8)
    
    value = store.get_affect(test_agent, AffectDimension.TRUST)
    assert value == 0.8


def test_set_affects_multiple(store, test_agent):
    """Test setting multiple affect dimensions"""
    new_affects = {
        AffectDimension.TRUST: 0.8,
        AffectDimension.DOMINANCE: 0.3
    }
    
    store.set_affects(test_agent, new_affects)
    
    assert store.get_affect(test_agent, AffectDimension.TRUST) == 0.8
    assert store.get_affect(test_agent, AffectDimension.DOMINANCE) == 0.3


def test_update_affect_additive(store, test_agent):
    """Test update_affect adds to current values"""
    initial = store.get_affect(test_agent, AffectDimension.COOPERATION)
    
    store.update_affect(test_agent, {AffectDimension.COOPERATION: -0.2})
    
    final = store.get_affect(test_agent, AffectDimension.COOPERATION)
    assert final == initial - 0.2


# ===== Metadata Tests =====

def test_increment_interaction(store, test_agent):
    """Test incrementing interaction counter"""
    initial = store.get_interaction_count(test_agent)
    
    count = store.increment_interaction(test_agent)
    
    assert count == initial + 1
    assert store.get_interaction_count(test_agent) == count


def test_get_interaction_count(store, test_agent):
    """Test getting interaction count"""
    count = store.get_interaction_count(test_agent)
    assert count == 0  # New agent starts at 0


def test_get_last_updated(store, test_agent):
    """Test getting last updated timestamp"""
    timestamp = store.get_last_updated(test_agent)
    assert timestamp is not None