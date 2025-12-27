"""High-level personality state management"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType

from .backend import RedisBackend
from .exceptions import AgentNotFoundError, InvalidValueError
from .state import AffectDimension, MoodDimension, PersonalityState
from .taxonomy import Trait


class PersonalityStateStore:
    """
    Manages personality states with complete get/set operations.
    Separates read/write for traits, moods, and affects.
    """

    def __init__(self, backend: RedisBackend | None = None):
        """
        Initialize store with Redis backend.

        Args:
            backend: RedisBackend instance. If None, creates default connection.
        """
        self.backend = backend or RedisBackend()

    def _key(self, agent_id: str) -> str:
        """Generate Redis key for agent"""
        return f"rnb:personality:{agent_id}"

    def _exists(self, agent_id: str) -> bool:
        """
        Internal check if agent exists.
        Use AgentManager.agent_exists() for public API.
        """
        return self.backend.exists(self._key(agent_id))

    def _validate_range(self, dimension: str, value: float) -> None:
        """Validate that value is in [-1, 1] range"""
        if not -1.0 <= value <= 1.0:
            raise InvalidValueError(dimension, value)

    # ===== Complete State Operations =====

    def get_state(self, agent_id: str) -> PersonalityState | None:
        """
        Retrieve complete personality state.

        Args:
            agent_id: Unique agent identifier

        Returns:
            PersonalityState object, or None if agent doesn't exist
        """
        data = self.backend.get(self._key(agent_id))
        if data is None:
            return None
        return PersonalityState.from_dict(data)

    def set_state(self, state: PersonalityState) -> None:
        """
        Store complete personality state.

        Args:
            state: PersonalityState object to store
        """
        state.last_updated = datetime.now()
        self.backend.set(self._key(state.agent_id), state.to_dict())

    # ===== Trait Operations =====

    def get_trait(self, agent_id: str, trait: Trait) -> float:
        """
        Get single trait value.

        Args:
            agent_id: Unique agent identifier
            trait: FFM trait to retrieve

        Returns:
            Trait value in range [-1, 1]

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return state.traits[trait]

    def get_traits(self, agent_id: str) -> Mapping[Trait, float]:
        """
        Get all trait values as immutable view.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Immutable mapping of all traits

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return MappingProxyType(state.traits)

    def set_trait(self, agent_id: str, trait: Trait, value: float) -> None:
        """
        Set single trait value.
        Note: Traits are supposed to be stable. Use sparingly.

        Args:
            agent_id: Unique agent identifier
            trait: FFM trait to set
            value: New value in range [-1, 1]

        Raises:
            AgentNotFoundError: If agent doesn't exist
            InvalidValueError: If value out of range
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        self._validate_range(trait.value, value)
        state.traits[trait] = value
        self.set_state(state)

    def set_traits(self, agent_id: str, traits: dict[Trait, float]) -> None:
        """
        Set multiple trait values.

        Args:
            agent_id: Unique agent identifier
            traits: Dictionary of traits to set

        Raises:
            AgentNotFoundError: If agent doesn't exist
            InvalidValueError: If any value out of range
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        for trait, value in traits.items():
            self._validate_range(trait.value, value)

        state.traits.update(traits)
        self.set_state(state)

    # ===== Mood Operations =====

    def get_mood(self, agent_id: str, mood: MoodDimension) -> float:
        """
        Get single mood dimension value.

        Args:
            agent_id: Unique agent identifier
            mood: Mood dimension to retrieve

        Returns:
            Mood value in range [-1, 1]

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return state.moods[mood]

    def get_moods(self, agent_id: str) -> Mapping[MoodDimension, float]:
        """
        Get all mood dimension values as immutable view.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Immutable mapping of all moods

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return MappingProxyType(state.moods)

    def set_mood(self, agent_id: str, mood: MoodDimension, value: float) -> None:
        """
        Set single mood dimension value (direct assignment).

        Args:
            agent_id: Unique agent identifier
            mood: Mood dimension to set
            value: New value in range [-1, 1]

        Raises:
            AgentNotFoundError: If agent doesn't exist
            InvalidValueError: If value out of range
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        self._validate_range(mood.value, value)
        state.moods[mood] = value
        self.set_state(state)

    def set_moods(self, agent_id: str, moods: dict[MoodDimension, float]) -> None:
        """
        Set multiple mood dimension values (direct assignment).

        Args:
            agent_id: Unique agent identifier
            moods: Dictionary of moods to set

        Raises:
            AgentNotFoundError: If agent doesn't exist
            InvalidValueError: If any value out of range
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        for mood, value in moods.items():
            self._validate_range(mood.value, value)

        state.moods.update(moods)
        self.set_state(state)

    def update_mood(self, agent_id: str, changes: dict[MoodDimension, float]) -> None:
        """
        Update mood dimensions by adding deltas (for use by update rules).
        Values are clamped to [-1, 1] range.

        Args:
            agent_id: Unique agent identifier
            changes: Dictionary of delta values to add to current moods

        Raises:
            AgentNotFoundError: If agent doesn't exist

        Example:
            # If current happiness is 0.3 and you update with +0.5,
            # resulting happiness will be 0.8
            store.update_mood(agent_id, {MoodDimension.HAPPINESS: 0.5})
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        for mood, delta in changes.items():
            current = state.moods[mood]
            new_value = current + delta
            # Clamp to valid range
            state.moods[mood] = max(-1.0, min(1.0, new_value))

        self.set_state(state)

    # ===== Affect Operations =====

    def get_affect(self, agent_id: str, affect: AffectDimension) -> float:
        """
        Get single affect dimension value.

        Args:
            agent_id: Unique agent identifier
            affect: Affect dimension to retrieve

        Returns:
            Affect value in range [-1, 1]

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return state.affects[affect]

    def get_affects(self, agent_id: str) -> Mapping[AffectDimension, float]:
        """
        Get all affect dimension values as immutable view.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Immutable mapping of all affects

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return MappingProxyType(state.affects)

    def set_affect(self, agent_id: str, affect: AffectDimension, value: float) -> None:
        """
        Set single affect dimension value (direct assignment).

        Args:
            agent_id: Unique agent identifier
            affect: Affect dimension to set
            value: New value in range [-1, 1]

        Raises:
            AgentNotFoundError: If agent doesn't exist
            InvalidValueError: If value out of range
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        self._validate_range(affect.value, value)
        state.affects[affect] = value
        self.set_state(state)

    def set_affects(self, agent_id: str, affects: dict[AffectDimension, float]) -> None:
        """
        Set multiple affect dimension values (direct assignment).

        Args:
            agent_id: Unique agent identifier
            affects: Dictionary of affects to set

        Raises:
            AgentNotFoundError: If agent doesn't exist
            InvalidValueError: If any value out of range
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        for affect, value in affects.items():
            self._validate_range(affect.value, value)

        state.affects.update(affects)
        self.set_state(state)

    def update_affect(
        self, agent_id: str, changes: dict[AffectDimension, float]
    ) -> None:
        """
        Update affect dimensions by adding deltas (for use by update rules).
        Values are clamped to [-1, 1] range.

        Args:
            agent_id: Unique agent identifier
            changes: Dictionary of delta values to add to current affects

        Raises:
            AgentNotFoundError: If agent doesn't exist

        Example:
            # If current cooperation is 0.5 and you update with -0.2,
            # resulting cooperation will be 0.3
            store.update_affect(agent_id, {AffectDimension.COOPERATION: -0.2})
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        for affect, delta in changes.items():
            current = state.affects[affect]
            new_value = current + delta
            # Clamp to valid range
            state.affects[affect] = max(-1.0, min(1.0, new_value))

        self.set_state(state)

    # ===== Metadata Operations =====

    def increment_interaction(self, agent_id: str) -> int:
        """
        Increment interaction counter.

        Args:
            agent_id: Unique agent identifier

        Returns:
            New interaction count

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)

        state.interaction_count += 1
        self.set_state(state)
        return state.interaction_count

    def get_interaction_count(self, agent_id: str) -> int:
        """
        Get current interaction count.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Interaction count

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return state.interaction_count

    def get_last_updated(self, agent_id: str) -> datetime:
        """
        Get timestamp of last state update.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Datetime of last update

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        state = self.get_state(agent_id)
        if state is None:
            raise AgentNotFoundError(agent_id)
        return state.last_updated
