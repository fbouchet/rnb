"""Unit tests for ChromaDB backend"""

import shutil
import tempfile

import pytest

from rnb.memory.backend import ChromaBackend


@pytest.fixture
def temp_chroma_dir():
    """Create temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def chroma_backend(temp_chroma_dir):
    """Create ChromaDB backend with temporary storage"""
    backend = ChromaBackend(persist_directory=temp_chroma_dir)
    yield backend
    backend.close()


class TestChromaBackendInitialization:
    """Tests for ChromaDB backend initialization"""

    def test_backend_creation(self, temp_chroma_dir):
        """Test backend can be created"""
        backend = ChromaBackend(persist_directory=temp_chroma_dir)

        assert backend.persist_directory == temp_chroma_dir
        assert backend.collection_prefix == "memories"

        backend.close()

    def test_backend_custom_prefix(self, temp_chroma_dir):
        """Test backend with custom collection prefix"""
        backend = ChromaBackend(
            persist_directory=temp_chroma_dir, collection_prefix="test_memories"
        )

        assert backend.collection_prefix == "test_memories"
        backend.close()


class TestChromaBackendCollections:
    """Tests for collection management"""

    def test_get_collection(self, chroma_backend):
        """Test getting/creating collection for agent"""
        collection = chroma_backend.get_collection("agent_001")

        assert collection is not None
        assert collection.name == "memories_agent_001"

    def test_collection_per_agent(self, chroma_backend):
        """Test separate collections for different agents"""
        collection1 = chroma_backend.get_collection("agent_001")
        collection2 = chroma_backend.get_collection("agent_002")

        assert collection1.name == "memories_agent_001"
        assert collection2.name == "memories_agent_002"
        assert collection1.name != collection2.name


class TestChromaBackendMemoryOperations:
    """Tests for memory CRUD operations"""

    def test_add_memory(self, chroma_backend):
        """Test adding a memory"""
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_001",
            content="User prefers Python",
            metadata={"type": "user", "category": "preference"},
        )

        # Verify memory was added
        memory = chroma_backend.get_memory("agent_001", "mem_001")
        assert memory is not None
        assert memory["content"] == "User prefers Python"
        assert memory["metadata"]["type"] == "user"

    def test_get_memory(self, chroma_backend):
        """Test retrieving specific memory"""
        # Add memory
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_002",
            content="Test content",
            metadata={"key": "value"},
        )

        # Retrieve it
        memory = chroma_backend.get_memory("agent_001", "mem_002")

        assert memory["id"] == "mem_002"
        assert memory["content"] == "Test content"
        assert memory["metadata"]["key"] == "value"

    def test_get_nonexistent_memory(self, chroma_backend):
        """Test retrieving non-existent memory returns None"""
        memory = chroma_backend.get_memory("agent_001", "nonexistent")
        assert memory is None

    def test_delete_memory(self, chroma_backend):
        """Test deleting a memory"""
        # Add memory (ChromaDB requires non-empty metadata)
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_003",
            content="To be deleted",
            metadata={"test": "value"},
        )

        # Verify it exists
        assert chroma_backend.get_memory("agent_001", "mem_003") is not None

        # Delete it
        chroma_backend.delete_memory("agent_001", "mem_003")

        # Verify it's gone
        assert chroma_backend.get_memory("agent_001", "mem_003") is None


class TestChromaBackendSearch:
    """Tests for semantic search functionality"""

    def test_search_memories(self, chroma_backend):
        """Test semantic search"""
        # Add some memories
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_001",
            content="User prefers Python for data science",
            metadata={"type": "user"},
        )
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_002",
            content="User asked about machine learning algorithms",
            metadata={"type": "session"},
        )

        # Search for related content
        results = chroma_backend.search_memories(
            agent_id="agent_001", query="programming language preferences", n_results=2
        )

        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)

    def test_search_with_metadata_filter(self, chroma_backend):
        """Test search with metadata filtering"""
        # Add memories of different types
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="user_001",
            content="User prefers concise explanations",
            metadata={"type": "user", "category": "preference"},
        )
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="session_001",
            content="Discussed recursion in session",
            metadata={"type": "session"},
        )

        # Search only user memories
        results = chroma_backend.search_memories(
            agent_id="agent_001",
            query="explanations",
            n_results=5,
            where={"type": "user"},
        )

        # Should only return user-type memories
        for result in results:
            assert result["metadata"]["type"] == "user"

    def test_search_no_results(self, chroma_backend):
        """Test search with no matching results"""
        results = chroma_backend.search_memories(
            agent_id="agent_empty", query="nonexistent content", n_results=5
        )

        assert len(results) == 0


class TestChromaBackendBulkOperations:
    """Tests for bulk operations"""

    def test_count_memories(self, chroma_backend):
        """Test counting memories"""
        # Add multiple memories
        for i in range(5):
            chroma_backend.add_memory(
                agent_id="agent_001",
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                metadata={"type": "user"},
            )

        count = chroma_backend.count_memories("agent_001")
        assert count == 5

    def test_count_with_filter(self, chroma_backend):
        """Test counting with metadata filter"""
        # Add memories of different types
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="user_001",
            content="User memory",
            metadata={"type": "user"},
        )
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="session_001",
            content="Session memory",
            metadata={"type": "session"},
        )

        user_count = chroma_backend.count_memories("agent_001", where={"type": "user"})

        assert user_count == 1

    def test_clear_memories_with_filter(self, chroma_backend):
        """Test clearing memories with metadata filter"""
        # Add memories of different types
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="user_001",
            content="User memory",
            metadata={"type": "user"},
        )
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="session_001",
            content="Session memory",
            metadata={"type": "session"},
        )

        # Clear only session memories
        chroma_backend.clear_memories("agent_001", where={"type": "session"})

        # User memory should still exist
        assert chroma_backend.get_memory("agent_001", "user_001") is not None
        # Session memory should be gone
        assert chroma_backend.get_memory("agent_001", "session_001") is None

    def test_clear_all_memories(self, chroma_backend):
        """Test clearing all memories for agent"""
        # Add memories
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_001",
            content="Memory 1",
            metadata={"index": 1},
        )
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_002",
            content="Memory 2",
            metadata={"index": 2},
        )

        # Clear all
        chroma_backend.clear_memories("agent_001")

        # Count should be 0
        count = chroma_backend.count_memories("agent_001")
        assert count == 0


class TestChromaBackendIsolation:
    """Tests for agent isolation"""

    def test_agent_memory_isolation(self, chroma_backend):
        """Test memories are isolated per agent"""
        # Add memory for agent 1
        chroma_backend.add_memory(
            agent_id="agent_001",
            memory_id="mem_001",
            content="Agent 1 memory",
            metadata={"agent": "1"},
        )

        # Add memory for agent 2
        chroma_backend.add_memory(
            agent_id="agent_002",
            memory_id="mem_001",  # Same ID but different agent
            content="Agent 2 memory",
            metadata={"agent": "2"},
        )

        # Retrieve from each agent
        mem1 = chroma_backend.get_memory("agent_001", "mem_001")
        mem2 = chroma_backend.get_memory("agent_002", "mem_001")

        assert mem1["content"] == "Agent 1 memory"
        assert mem2["content"] == "Agent 2 memory"
