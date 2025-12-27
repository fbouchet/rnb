"""Custom exceptions for RnB personality system"""


class RnBException(Exception):
    """Base exception for RnB framework"""

    pass


class AgentNotFoundError(RnBException):
    """Raised when attempting to access a non-existent agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent '{agent_id}' does not exist")


class AgentAlreadyExistsError(RnBException):
    """Raised when attempting to create an agent that already exists"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent '{agent_id}' already exists")


class InvalidValueError(RnBException):
    """Raised when a personality value is outside valid range"""

    def __init__(self, dimension: str, value: float, valid_range: str = "[-1, 1]"):
        self.dimension = dimension
        self.value = value
        self.valid_range = valid_range
        super().__init__(f"{dimension} value must be in {valid_range}, got {value}")


class StorageConnectionError(RnBException):
    """Raised when unable to connect to storage backend"""

    def __init__(self, backend: str, details: str):
        self.backend = backend
        self.details = details
        super().__init__(f"Failed to connect to {backend}: {details}")
