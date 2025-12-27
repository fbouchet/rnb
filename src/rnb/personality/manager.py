"""High-level agent lifecycle management"""

from collections.abc import Callable

from .exceptions import AgentAlreadyExistsError, AgentNotFoundError, InvalidValueError
from .state import AffectDimension, MoodDimension, PersonalityState
from .store import PersonalityStateStore
from .taxonomy import Trait


class AgentManager:
    """
    Manages agent lifecycle: creation, deletion, listing.
    Provides high-level operations on agent personalities.
    """

    def __init__(self, store: PersonalityStateStore):
        """
        Initialize manager with personality store.

        Args:
            store: PersonalityStateStore instance
        """
        self.store = store

    def create_agent(
        self,
        agent_id: str,
        traits: dict[Trait, float] | None = None,
        moods: dict[MoodDimension, float] | None = None,
        affects: dict[AffectDimension, float] | None = None,
    ) -> PersonalityState:
        """
        Create new agent with initial personality state.

        Args:
            agent_id: Unique agent identifier
            traits: Optional initial trait values (default: all 0.0)
            moods: Optional initial mood values (default: all 0.0)
            affects: Optional initial affect values (default: all 0.0)

        Returns:
            Created PersonalityState

        Raises:
            AgentAlreadyExistsError: If agent already exists
            InvalidValueError: If any value out of range

        Example:
            manager.create_agent(
                "tutor_001",
                traits={Trait.EXTRAVERSION: 0.6, Trait.CONSCIENTIOUSNESS: 0.8},
                affects={AffectDimension.COOPERATION: 0.7}
            )
        """
        if self.store._exists(agent_id):
            raise AgentAlreadyExistsError(agent_id)

        # Create default state
        state = PersonalityState(agent_id=agent_id)

        # Validate and apply custom values if provided
        if traits:
            for trait, value in traits.items():
                if not -1.0 <= value <= 1.0:
                    raise InvalidValueError(trait.value, value)
                # Set trait value (will distribute to all schemes under this trait)
                self._set_trait_uniformly(state, trait, value)

        if moods:
            for mood, value in moods.items():
                if not -1.0 <= value <= 1.0:
                    raise InvalidValueError(mood.value, value)
            state.moods.update(moods)

        if affects:
            for affect, value in affects.items():
                if not -1.0 <= value <= 1.0:
                    raise InvalidValueError(affect.value, value)
            state.affects.update(affects)

        # Store the state
        self.store.set_state(state)
        return state

    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete agent and all associated state.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if agent was deleted, False if it didn't exist
        """
        if not self.store._exists(agent_id):
            return False

        key = self.store._key(agent_id)
        return self.store.backend.delete(key)

    def agent_exists(self, agent_id: str) -> bool:
        """
        Check if agent exists.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if agent exists, False otherwise
        """
        return self.store._exists(agent_id)

    def list_agents(self) -> list[str]:
        """
        List all agent IDs in the system.

        Returns:
            List of agent ID strings
        """
        keys = self.store.backend.list_keys("rnb:personality:*")
        # Extract agent_id from keys like "rnb:personality:agent_001"
        return [key.split(":")[-1] for key in keys]

    def get_agent_summary(self, agent_id: str) -> dict:
        """
        Get summary information about an agent.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Dictionary with agent summary

        Raises:
            AgentNotFoundError: If agent doesn't exist

        Example return:
            {
                "agent_id": "tutor_001",
                "interaction_count": 42,
                "last_updated": "2024-01-15T10:30:00",
                "dominant_traits": ["conscientiousness", "agreeableness"],
                "current_mood": "positive_energetic"
            }
        """
        state = self.store.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        # Find dominant traits (|value| > 0.5)
        dominant_traits = [
            trait.value for trait, value in state.traits.items() if abs(value) > 0.5
        ]

        # Determine mood category
        avg_mood = sum(state.moods.values()) / len(state.moods)
        energy = state.moods[MoodDimension.ENERGY]

        if avg_mood > 0.3:
            mood_valence = "positive"
        elif avg_mood < -0.3:
            mood_valence = "negative"
        else:
            mood_valence = "neutral"

        if energy > 0.3:
            mood_energy = "energetic"
        elif energy < -0.3:
            mood_energy = "low_energy"
        else:
            mood_energy = "moderate_energy"

        current_mood = f"{mood_valence}_{mood_energy}"

        return {
            "agent_id": agent_id,
            "interaction_count": state.interaction_count,
            "last_updated": state.last_updated.isoformat(),
            "dominant_traits": dominant_traits,
            "current_mood": current_mood,
        }

    def clone_agent(
        self, source_agent_id: str, new_agent_id: str, reset_metadata: bool = True
    ) -> PersonalityState:
        """
        Create a copy of an existing agent.

        Args:
            source_agent_id: Agent to clone from
            new_agent_id: ID for new agent
            reset_metadata: If True, reset interaction_count and last_updated

        Returns:
            Newly created PersonalityState

        Raises:
            AgentNotFoundError: If source doesn't exist
            AgentAlreadyExistsError: If new_agent_id already exists
        """
        source_state = self.store.get_state(source_agent_id)
        if source_state is None:
            raise AgentNotFoundError(source_agent_id)

        if self.store._exists(new_agent_id):
            raise AgentAlreadyExistsError(new_agent_id)

        # Create new state with same personality parameters
        new_state = PersonalityState(
            agent_id=new_agent_id,
            traits=source_state.traits.copy(),
            moods=source_state.moods.copy(),
            affects=source_state.affects.copy(),
        )

        if not reset_metadata:
            new_state.interaction_count = source_state.interaction_count
            new_state.last_updated = source_state.last_updated

        self.store.set_state(new_state)
        return new_state

    def apply_update_rule(
        self, agent_id: str, rule: Callable[[PersonalityState], PersonalityState]
    ) -> None:
        """
        Apply arbitrary update function to agent state.

        This is the extensibility point for implementing custom update rules
        from RnB's behavioral dynamics (mood evolution, affect development, etc.)

        Args:
            agent_id: Unique agent identifier
            rule: Function that takes PersonalityState and returns modified state

        Raises:
            AgentNotFoundError: If agent doesn't exist

        Example:
            def frustration_rule(state: PersonalityState) -> PersonalityState:
                # After user criticism, increase neuroticism-based mood change
                neuroticism = state.traits[Trait.NEUROTICISM]
                if neuroticism > 0.5:
                    state.moods[MoodDimension.HAPPINESS] -= 0.3
                    state.moods[MoodDimension.CALMNESS] -= 0.2
                return state

            manager.apply_update_rule("agent_001", frustration_rule)
        """
        state = self.store.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        updated_state = rule(state)
        self.store.set_state(updated_state)

    def reset_agent_moods(self, agent_id: str) -> None:
        """
        Reset all mood dimensions to neutral (0.0).

        Args:
            agent_id: Unique agent identifier

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        self.store.set_moods(agent_id, {mood: 0.0 for mood in MoodDimension})

    def reset_agent_affects(self, agent_id: str) -> None:
        """
        Reset all affect dimensions to neutral (0.0).

        Args:
            agent_id: Unique agent identifier

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        self.store.set_affects(agent_id, {affect: 0.0 for affect in AffectDimension})

    def _set_trait_uniformly(
        self, state: PersonalityState, trait: Trait, value: float
    ) -> None:
        """
        Set all schemes under a trait to a uniform value.

        For testing/initialization purposes, creates one representative
        scheme per facet and sets them all to the trait value.

        This gives us ~6 schemes per trait (one per facet), which is
        sufficient for the trait value to aggregate correctly.

        Args:
            state: PersonalityState to modify
            trait: Trait to set
            value: Value in [-1, 1] to apply to all schemes
        """
        from .taxonomy import FACETS_BY_TRAIT

        # Get all facets for this trait
        facets = FACETS_BY_TRAIT.get(trait, [])

        # Create one representative scheme per facet
        # Format: "Trait_Facet_VALUE"
        for facet in facets:
            scheme_key = f"{trait.value}_{facet}_VALUE"
            state.set_scheme(scheme_key, value)
