"""Unit tests for AgentManager"""

import pytest
from rnb.personality.backend import RedisBackend
from rnb.personality.store import PersonalityStateStore
from rnb.personality.manager import AgentManager
from rnb.personality.state import (
    PersonalityState,
    FFMTrait,
    MoodDimension,
    AffectDimension
)
from rnb.personality.exceptions import (
    AgentNotFoundError,
    AgentAlreadyExistsError,
    InvalidValueError
)


@pytest.fixture
def backend():
    """Create RedisBackend for testing"""
    backend = RedisBackend()
    yield backend
    backend.flush_db()
    backend.close()


@pytest.fixture
def manager(backend):
    """Create AgentManager for testing"""
    store = PersonalityStateStore(backend)
    return AgentManager(store)


def test_create_agent_defaults(manager):
    """Test creating agent with default values"""
    state = manager.create_agent("test_agent")
    
    assert state.agent_id == "test_agent"
    assert all(v == 0.0 for v in state.traits.values())
    assert all(v == 0.0 for v in state.moods.values())
    assert all(v == 0.0 for v in state.affects.values())


def test_create_agent_with_traits(manager):
    """Test creating agent with custom traits"""
    state = manager.create_agent(
        "test_agent",
        traits={
            FFMTrait.EXTRAVERSION: 0.6,
            FFMTrait.CONSCIENTIOUSNESS: 0.8
        }
    )
    
    assert state.traits[FFMTrait.EXTRAVERSION] == 0.6
    assert state.traits[FFMTrait.CONSCIENTIOUSNESS] == 0.8


def test_create_agent_with_moods_and_affects(manager):
    """Test creating agent with moods and affects"""
    state = manager.create_agent(
        "test_agent",
        moods={MoodDimension.HAPPINESS: 0.5},
        affects={AffectDimension.COOPERATION: 0.7}
    )
    
    assert state.moods[MoodDimension.HAPPINESS] == 0.5
    assert state.affects[AffectDimension.COOPERATION] == 0.7


def test_create_agent_already_exists(manager):
    """Test creating agent that already exists raises error"""
    manager.create_agent("test_agent")
    
    with pytest.raises(AgentAlreadyExistsError) as exc_info:
        manager.create_agent("test_agent")
    
    assert "test_agent" in str(exc_info.value)


def test_create_agent_invalid_value(manager):
    """Test creating agent with invalid value raises error"""
    with pytest.raises(InvalidValueError):
        manager.create_agent(
            "test_agent",
            traits={FFMTrait.EXTRAVERSION: 1.5}
        )


def test_delete_agent_existing(manager):
    """Test deleting an existing agent"""
    manager.create_agent("test_agent")
    
    deleted = manager.delete_agent("test_agent")
    
    assert deleted is True
    assert not manager.agent_exists("test_agent")


def test_delete_agent_nonexistent(manager):
    """Test deleting non-existent agent returns False"""
    deleted = manager.delete_agent("nonexistent")
    assert deleted is False


def test_agent_exists_true(manager):
    """Test agent_exists returns True for existing agent"""
    manager.create_agent("test_agent")
    assert manager.agent_exists("test_agent") is True


def test_agent_exists_false(manager):
    """Test agent_exists returns False for non-existent agent"""
    assert manager.agent_exists("nonexistent") is False


def test_list_agents_empty(manager):
    """Test listing agents when none exist"""
    agents = manager.list_agents()
    assert agents == []


def test_list_agents_multiple(manager):
    """Test listing multiple agents"""
    manager.create_agent("agent1")
    manager.create_agent("agent2")
    manager.create_agent("agent3")
    
    agents = manager.list_agents()
    
    assert len(agents) == 3
    assert "agent1" in agents
    assert "agent2" in agents
    assert "agent3" in agents


def test_get_agent_summary(manager):
    """Test getting agent summary"""
    manager.create_agent(
        "test_agent",
        traits={
            FFMTrait.EXTRAVERSION: 0.7,
            FFMTrait.CONSCIENTIOUSNESS: 0.8
        }
    )
    
    summary = manager.get_agent_summary("test_agent")
    
    assert summary["agent_id"] == "test_agent"
    assert summary["interaction_count"] == 0
    assert "extraversion" in summary["dominant_traits"]
    assert "conscientiousness" in summary["dominant_traits"]
    assert "neutral" in summary["current_mood"]


def test_get_agent_summary_nonexistent(manager):
    """Test getting summary for non-existent agent raises error"""
    with pytest.raises(AgentNotFoundError):
        manager.get_agent_summary("nonexistent")


def test_clone_agent(manager):
    """Test cloning an agent"""
    manager.create_agent(
        "source",
        traits={FFMTrait.OPENNESS: 0.9},
        moods={MoodDimension.HAPPINESS: 0.6}
    )
    
    cloned_state = manager.clone_agent("source", "clone")
    
    assert cloned_state.agent_id == "clone"
    assert cloned_state.traits[FFMTrait.OPENNESS] == 0.9
    assert cloned_state.moods[MoodDimension.HAPPINESS] == 0.6
    # Metadata should be reset by default
    assert cloned_state.interaction_count == 0


def test_clone_agent_preserve_metadata(manager):
    """Test cloning agent with metadata preservation"""
    manager.create_agent("source")
    manager.store.increment_interaction("source")
    manager.store.increment_interaction("source")
    
    cloned_state = manager.clone_agent("source", "clone", reset_metadata=False)
    
    assert cloned_state.interaction_count == 2


def test_clone_agent_source_not_found(manager):
    """Test cloning non-existent agent raises error"""
    with pytest.raises(AgentNotFoundError):
        manager.clone_agent("nonexistent", "clone")


def test_clone_agent_target_exists(manager):
    """Test cloning to existing agent raises error"""
    manager.create_agent("source")
    manager.create_agent("target")
    
    with pytest.raises(AgentAlreadyExistsError):
        manager.clone_agent("source", "target")


def test_apply_update_rule(manager):
    """Test applying update rule to agent"""
    manager.create_agent(
        "test_agent",
        moods={MoodDimension.HAPPINESS: 0.5}
    )
    
    def increase_happiness(state: PersonalityState) -> PersonalityState:
        state.moods[MoodDimension.HAPPINESS] += 0.2
        return state
    
    manager.apply_update_rule("test_agent", increase_happiness)
    
    happiness = manager.store.get_mood("test_agent", MoodDimension.HAPPINESS)
    assert happiness == 0.7


def test_apply_update_rule_nonexistent(manager):
    """Test applying update rule to non-existent agent raises error"""
    def dummy_rule(state: PersonalityState) -> PersonalityState:
        return state
    
    with pytest.raises(AgentNotFoundError):
        manager.apply_update_rule("nonexistent", dummy_rule)


def test_reset_agent_moods(manager):
    """Test resetting agent moods to neutral"""
    manager.create_agent(
        "test_agent",
        moods={
            MoodDimension.HAPPINESS: 0.8,
            MoodDimension.ENERGY: -0.5
        }
    )
    
    manager.reset_agent_moods("test_agent")
    
    moods = manager.store.get_moods("test_agent")
    assert all(v == 0.0 for v in moods.values())


def test_reset_agent_affects(manager):
    """Test resetting agent affects to neutral"""
    manager.create_agent(
        "test_agent",
        affects={
            AffectDimension.COOPERATION: 0.7,
            AffectDimension.TRUST: -0.3
        }
    )
    
    manager.reset_agent_affects("test_agent")
    
    affects = manager.store.get_affects("test_agent")
    assert all(v == 0.0 for v in affects.values())