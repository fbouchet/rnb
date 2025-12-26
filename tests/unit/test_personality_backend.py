"""Unit tests for RedisBackend"""

import pytest
import json
from rnb.personality.backend import RedisBackend
from rnb.personality.exceptions import StorageConnectionError


@pytest.fixture
def backend():
    """Create RedisBackend instance for testing"""
    backend = RedisBackend()
    yield backend
    # Cleanup: flush test data
    backend.flush_db()
    backend.close()


def test_backend_connection():
    """Test successful Redis connection"""
    backend = RedisBackend()
    assert backend.redis is not None
    backend.close()


def test_backend_connection_failure():
    """Test connection failure handling"""
    with pytest.raises(StorageConnectionError) as exc_info:
        RedisBackend(host='nonexistent', port=9999)
    
    assert "Redis at nonexistent:9999" in str(exc_info.value)


def test_set_and_get(backend):
    """Test basic set/get operations"""
    test_data = {"name": "test", "value": 42}
    
    backend.set("test:key", test_data)
    retrieved = backend.get("test:key")
    
    assert retrieved == test_data


def test_get_nonexistent_key(backend):
    """Test getting a key that doesn't exist"""
    result = backend.get("nonexistent:key")
    assert result is None


def test_delete_existing_key(backend):
    """Test deleting an existing key"""
    backend.set("test:key", {"data": "value"})
    
    deleted = backend.delete("test:key")
    assert deleted is True
    
    result = backend.get("test:key")
    assert result is None


def test_delete_nonexistent_key(backend):
    """Test deleting a key that doesn't exist"""
    deleted = backend.delete("nonexistent:key")
    assert deleted is False


def test_exists_true(backend):
    """Test exists() returns True for existing key"""
    backend.set("test:key", {"data": "value"})
    assert backend.exists("test:key") is True


def test_exists_false(backend):
    """Test exists() returns False for non-existing key"""
    assert backend.exists("nonexistent:key") is False


def test_list_keys_with_pattern(backend):
    """Test listing keys with pattern matching"""
    backend.set("rnb:test:1", {"id": 1})
    backend.set("rnb:test:2", {"id": 2})
    backend.set("other:key", {"id": 3})
    
    keys = backend.list_keys("rnb:test:*")
    
    assert len(keys) == 2
    assert "rnb:test:1" in keys
    assert "rnb:test:2" in keys
    assert "other:key" not in keys


def test_list_keys_no_matches(backend):
    """Test listing keys with no matches"""
    keys = backend.list_keys("nonexistent:*")
    assert len(keys) == 0


def test_flush_db(backend):
    """Test flushing database"""
    backend.set("key1", {"data": "value1"})
    backend.set("key2", {"data": "value2"})
    
    backend.flush_db()
    
    assert backend.get("key1") is None
    assert backend.get("key2") is None