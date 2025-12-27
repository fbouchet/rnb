"""Unit tests for memory types (M.U, M.S, M.T models)"""

from datetime import datetime

from rnb.memory.types import Memory, MemoryType, SessionMemory, TaskMemory, UserMemory


class TestMemoryBase:
    """Tests for base Memory class"""

    def test_memory_creation(self):
        """Test basic memory creation"""
        memory = Memory(
            id="test_001",
            type=MemoryType.USER,
            content="Test memory content",
            metadata={"key": "value"},
        )

        assert memory.id == "test_001"
        assert memory.type == MemoryType.USER
        assert memory.content == "Test memory content"
        assert memory.metadata == {"key": "value"}
        assert isinstance(memory.timestamp, datetime)

    def test_memory_default_metadata(self):
        """Test memory with default empty metadata"""
        memory = Memory(id="test_002", type=MemoryType.SESSION, content="Content")

        assert memory.metadata == {}

    def test_memory_serialization(self):
        """Test memory can be serialized to dict"""
        memory = Memory(
            id="test_003",
            type=MemoryType.TASK,
            content="Task content",
            metadata={"status": "active"},
        )

        data = memory.model_dump()

        assert data["id"] == "test_003"
        assert data["type"] == "task"
        assert data["content"] == "Task content"
        assert data["metadata"]["status"] == "active"


class TestUserMemory:
    """Tests for UserMemory (M.U)"""

    def test_user_memory_creation(self):
        """Test UserMemory creation with explicit type"""
        memory = UserMemory(
            id="user_001",
            content="User prefers Python over Java",
            metadata={"category": "preference"},
        )

        assert memory.id == "user_001"
        assert memory.type == MemoryType.USER
        assert memory.content == "User prefers Python over Java"
        assert memory.metadata["category"] == "preference"

    def test_user_memory_default_type(self):
        """Test UserMemory has USER type by default"""
        memory = UserMemory(id="user_002", content="User expertise in algorithms")

        assert memory.type == MemoryType.USER

    def test_user_memory_preference_example(self):
        """Test realistic user preference memory"""
        memory = UserMemory(
            id="user_pref_001",
            content="User prefers concise explanations with practical examples",
            metadata={
                "category": "preference",
                "confidence": 0.8,
                "observed_interactions": 5,
            },
        )

        assert "concise" in memory.content
        assert memory.metadata["confidence"] == 0.8
        assert memory.metadata["observed_interactions"] == 5

    def test_user_memory_expertise_example(self):
        """Test realistic user expertise memory"""
        memory = UserMemory(
            id="user_exp_001",
            content="User has intermediate Python knowledge, beginner in algorithms",
            metadata={
                "category": "expertise",
                "domain": "programming",
                "python_level": "intermediate",
                "algorithms_level": "beginner",
            },
        )

        assert "Python" in memory.content
        assert memory.metadata["domain"] == "programming"


class TestSessionMemory:
    """Tests for SessionMemory (M.S)"""

    def test_session_memory_creation(self):
        """Test SessionMemory creation"""
        memory = SessionMemory(
            id="session_001",
            content="User asked about recursion; agent explained with examples",
            metadata={"session_id": "conv_001", "turn": 1},
        )

        assert memory.id == "session_001"
        assert memory.type == MemoryType.SESSION
        assert memory.metadata["session_id"] == "conv_001"
        assert memory.metadata["turn"] == 1

    def test_session_memory_default_type(self):
        """Test SessionMemory has SESSION type by default"""
        memory = SessionMemory(id="session_002", content="Dialogue turn")

        assert memory.type == MemoryType.SESSION

    def test_session_memory_dialogue_turn(self):
        """Test realistic dialogue turn memory"""
        memory = SessionMemory(
            id="session_turn_003",
            content="User asked about recursion; agent provided definition and factorial example",
            metadata={
                "session_id": "conv_20250101_001",
                "turn": 3,
                "user_message": "Can you explain recursion?",
                "agent_response": "Recursion is when a function calls itself...",
                "topics": ["recursion", "programming", "functions"],
            },
        )

        assert memory.metadata["turn"] == 3
        assert "recursion" in memory.metadata["topics"]
        assert "user_message" in memory.metadata

    def test_session_memory_cross_reference(self):
        """Test session memory with cross-references"""
        memory = SessionMemory(
            id="session_ref_001",
            content="Discussion referenced previous binary search explanation from turn 5",
            metadata={
                "session_id": "conv_001",
                "turn": 8,
                "reference_turn": 5,
                "type": "cross_reference",
            },
        )

        assert memory.metadata["reference_turn"] == 5
        assert memory.metadata["type"] == "cross_reference"


class TestTaskMemory:
    """Tests for TaskMemory (M.T)"""

    def test_task_memory_creation(self):
        """Test TaskMemory creation"""
        memory = TaskMemory(
            id="task_001",
            content="Help user implement quicksort in Python",
            metadata={"task_id": "task_001", "status": "in_progress"},
        )

        assert memory.id == "task_001"
        assert memory.type == MemoryType.TASK
        assert memory.metadata["status"] == "in_progress"

    def test_task_memory_default_type(self):
        """Test TaskMemory has TASK type by default"""
        memory = TaskMemory(id="task_002", content="Task description")

        assert memory.type == MemoryType.TASK

    def test_task_memory_definition(self):
        """Test realistic task definition memory"""
        memory = TaskMemory(
            id="task_def_001",
            content="Help user implement quicksort algorithm in Python with test cases",
            metadata={
                "task_id": "task_001",
                "status": "in_progress",
                "created": "2025-01-01T10:00:00",
                "priority": "high",
                "estimated_turns": 10,
            },
        )

        assert "quicksort" in memory.content
        assert memory.metadata["priority"] == "high"
        assert memory.metadata["estimated_turns"] == 10

    def test_task_memory_progress(self):
        """Test task progress tracking memory"""
        memory = TaskMemory(
            id="task_progress_001",
            content="Task progress: explained concept, showed pseudocode. Remaining: write code, create tests",
            metadata={
                "task_id": "task_001",
                "status": "in_progress",
                "steps_completed": ["explain_concept", "provide_pseudocode"],
                "steps_remaining": ["write_code", "create_tests"],
                "completion_percentage": 50,
            },
        )

        assert len(memory.metadata["steps_completed"]) == 2
        assert len(memory.metadata["steps_remaining"]) == 2
        assert memory.metadata["completion_percentage"] == 50


class TestMemoryTypeEnum:
    """Tests for MemoryType enum"""

    def test_memory_type_values(self):
        """Test MemoryType enum values"""
        assert MemoryType.USER.value == "user"
        assert MemoryType.SESSION.value == "session"
        assert MemoryType.TASK.value == "task"

    def test_memory_type_comparison(self):
        """Test MemoryType enum comparison"""
        assert MemoryType.USER == MemoryType.USER
        assert MemoryType.USER != MemoryType.SESSION
        assert MemoryType.SESSION != MemoryType.TASK
