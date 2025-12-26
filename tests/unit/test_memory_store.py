"""Unit tests for MemoryStore high-level interface"""

import pytest
import tempfile
import shutil
from rnb.memory.store import MemoryStore
from rnb.memory.types import MemoryType
from rnb.memory.backend import ChromaBackend


@pytest.fixture
def temp_chroma_dir():
    """Create temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_store(temp_chroma_dir):
    """Create MemoryStore with temporary storage"""
    store = MemoryStore(persist_directory=temp_chroma_dir)
    yield store
    store.close()


class TestMemoryStoreInitialization:
    """Tests for MemoryStore initialization"""
    
    def test_store_creation(self, temp_chroma_dir):
        """Test store can be created"""
        store = MemoryStore(persist_directory=temp_chroma_dir)
        
        assert store.backend is not None
        store.close()
    
    def test_store_with_custom_backend(self, temp_chroma_dir):
        """Test store with custom backend"""
        backend = ChromaBackend(persist_directory=temp_chroma_dir)
        store = MemoryStore(backend=backend)
        
        assert store.backend is backend
        store.close()


class TestUserMemoryOperations:
    """Tests for M.U (User Model) operations"""
    
    def test_add_user_memory(self, memory_store):
        """Test adding user memory"""
        memory = memory_store.add_user_memory(
            agent_id="agent_001",
            content="User prefers Python",
            metadata={"category": "preference"}
        )
        
        assert memory.id.startswith("user_")
        assert memory.type == MemoryType.USER
        assert memory.content == "User prefers Python"
        assert memory.metadata["category"] == "preference"
    
    def test_get_user_memories(self, memory_store):
        """Test retrieving user memories"""
        # Add some user memories
        memory_store.add_user_memory(
            "agent_001",
            "User prefers concise explanations"
        )
        memory_store.add_user_memory(
            "agent_001",
            "User has expertise in Python"
        )
        
        # Retrieve user memories
        memories = memory_store.get_user_memories("agent_001", n_results=10)
        
        assert len(memories) == 2
        assert all(m['metadata']['type'] == 'user' for m in memories)
    
    def test_search_user_memories(self, memory_store):
        """Test searching user memories semantically"""
        # Add user preferences
        memory_store.add_user_memory(
            "agent_001",
            "User prefers Python for data science"
        )
        memory_store.add_user_memory(
            "agent_001",
            "User likes detailed code examples"
        )
        
        # Search for programming preferences
        results = memory_store.get_user_memories(
            "agent_001",
            query="programming language preferences",
            n_results=5
        )
        
        assert len(results) > 0
        assert any("Python" in r['content'] for r in results)


class TestSessionMemoryOperations:
    """Tests for M.S (Session Model) operations"""
    
    def test_add_session_memory(self, memory_store):
        """Test adding session memory"""
        memory = memory_store.add_session_memory(
            agent_id="agent_001",
            content="User asked about recursion",
            metadata={
                "session_id": "conv_001",
                "turn": 1
            }
        )
        
        assert memory.id.startswith("session_")
        assert memory.type == MemoryType.SESSION
        assert memory.metadata["session_id"] == "conv_001"
    
    def test_get_session_memories(self, memory_store):
        """Test retrieving session memories"""
        # Add session memories
        memory_store.add_session_memory(
            "agent_001",
            "Turn 1: discussed recursion",
            metadata={"session_id": "conv_001", "turn": 1}
        )
        memory_store.add_session_memory(
            "agent_001",
            "Turn 2: provided examples",
            metadata={"session_id": "conv_001", "turn": 2}
        )
        
        # Retrieve session memories
        memories = memory_store.get_session_memories(
            "agent_001",
            session_id="conv_001",
            n_results=10
        )
        
        assert len(memories) >= 2
        assert all(m['metadata']['type'] == 'session' for m in memories)


class TestTaskMemoryOperations:
    """Tests for M.T (Task Model) operations"""
    
    def test_add_task_memory(self, memory_store):
        """Test adding task memory"""
        memory = memory_store.add_task_memory(
            agent_id="agent_001",
            content="Help user implement quicksort",
            metadata={
                "task_id": "task_001",
                "status": "in_progress"
            }
        )
        
        assert memory.id.startswith("task_")
        assert memory.type == MemoryType.TASK
        assert memory.metadata["status"] == "in_progress"
    
    def test_get_task_memories(self, memory_store):
        """Test retrieving task memories"""
        # Add task memories
        memory_store.add_task_memory(
            "agent_001",
            "Task: Implement sorting algorithm",
            metadata={"status": "in_progress"}
        )
        memory_store.add_task_memory(
            "agent_001",
            "Task: Write unit tests",
            metadata={"status": "completed"}
        )
        
        # Get all tasks
        all_tasks = memory_store.get_task_memories("agent_001", n_results=10)
        assert len(all_tasks) >= 2
        
        # Get only in-progress tasks
        active_tasks = memory_store.get_task_memories(
            "agent_001",
            status="in_progress",
            n_results=10
        )
        assert len(active_tasks) >= 1
        assert all(m['metadata'].get('status') == 'in_progress' for m in active_tasks)


class TestMemorySearch:
    """Tests for cross-type memory search"""
    
    def test_search_all_memory_types(self, memory_store):
        """Test searching across all memory types"""
        # Add different types of memories
        memory_store.add_user_memory(
            "agent_001",
            "User prefers Python programming"
        )
        memory_store.add_session_memory(
            "agent_001",
            "Discussed Python best practices"
        )
        memory_store.add_task_memory(
            "agent_001",
            "Task: Write Python script"
        )
        
        # Search across all types
        results = memory_store.search_memories(
            agent_id="agent_001",
            query="Python programming",
            n_results=5
        )
        
        # Should find memories from multiple types
        assert len(results) >= 2
        types_found = {m['metadata']['type'] for m in results}
        assert len(types_found) > 1
    
    def test_search_specific_memory_type(self, memory_store):
        """Test searching specific memory type"""
        # Add memories
        memory_store.add_user_memory("agent_001", "User likes Python")
        memory_store.add_session_memory("agent_001", "Discussed Python")
        
        # Search only user memories
        results = memory_store.search_memories(
            agent_id="agent_001",
            query="Python",
            memory_type=MemoryType.USER,
            n_results=5
        )
        
        # Should only get user memories
        assert all(m['metadata']['type'] == 'user' for m in results)


class TestMemoryManagement:
    """Tests for memory lifecycle management"""
    
    def test_count_memories(self, memory_store):
        """Test counting memories"""
        # Add memories
        memory_store.add_user_memory("agent_001", "User memory 1")
        memory_store.add_user_memory("agent_001", "User memory 2")
        memory_store.add_session_memory("agent_001", "Session memory 1")
        
        # Count all
        total = memory_store.count_memories("agent_001")
        assert total == 3
        
        # Count by type
        user_count = memory_store.count_memories("agent_001", MemoryType.USER)
        assert user_count == 2
        
        session_count = memory_store.count_memories("agent_001", MemoryType.SESSION)
        assert session_count == 1
    
    def test_clear_memories_by_type(self, memory_store):
        """Test clearing memories by type"""
        # Add memories
        memory_store.add_user_memory("agent_001", "User memory")
        memory_store.add_session_memory("agent_001", "Session memory")
        
        # Clear session memories
        memory_store.clear_memories("agent_001", MemoryType.SESSION)
        
        # User memories should remain
        assert memory_store.count_memories("agent_001", MemoryType.USER) == 1
        assert memory_store.count_memories("agent_001", MemoryType.SESSION) == 0
    
    def test_clear_all_memories(self, memory_store):
        """Test clearing all memories for agent"""
        # Add memories
        memory_store.add_user_memory("agent_001", "Memory 1")
        memory_store.add_session_memory("agent_001", "Memory 2")
        memory_store.add_task_memory("agent_001", "Memory 3")
        
        # Clear all
        memory_store.clear_memories("agent_001")
        
        # Should be empty
        assert memory_store.count_memories("agent_001") == 0