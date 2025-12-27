"""Pure Redis storage backend - no business logic"""

import json

import redis

from .exceptions import StorageConnectionError


class RedisBackend:
    """Low-level Redis operations for personality state storage"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        decode_responses: bool = True,
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number (0-15)
            decode_responses: If True, responses are decoded to strings

        Raises:
            StorageConnectionError: If unable to connect to Redis
        """
        try:
            self.redis = redis.Redis(
                host=host, port=port, db=db, decode_responses=decode_responses
            )
            # Test connection
            self.redis.ping()
        except redis.ConnectionError as e:
            raise StorageConnectionError(
                backend=f"Redis at {host}:{port}", details=str(e)
            ) from e

    def get(self, key: str) -> dict | None:
        """
        Retrieve data from Redis.

        Args:
            key: Redis key to retrieve

        Returns:
            Dictionary of stored data, or None if key doesn't exist
        """
        data = self.redis.get(key)
        if data is None:
            return None
        return json.loads(data)

    def set(self, key: str, data: dict) -> None:
        """
        Store data in Redis.

        Args:
            key: Redis key to store under
            data: Dictionary to store (will be JSON-serialized)
        """
        self.redis.set(key, json.dumps(data))

    def delete(self, key: str) -> bool:
        """
        Delete key from Redis.

        Args:
            key: Redis key to delete

        Returns:
            True if key was deleted, False if it didn't exist
        """
        return bool(self.redis.delete(key))

    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.

        Args:
            key: Redis key to check

        Returns:
            True if key exists, False otherwise
        """
        return bool(self.redis.exists(key))

    def list_keys(self, pattern: str) -> list[str]:
        """
        List all keys matching pattern.

        Args:
            pattern: Redis pattern (e.g., "rnb:personality:*")

        Returns:
            List of matching keys
        """
        return self.redis.keys(pattern)

    def flush_db(self) -> None:
        """
        WARNING: Delete ALL keys in current database.
        Use only for testing/development.
        """
        self.redis.flushdb()

    def close(self) -> None:
        """Close Redis connection"""
        self.redis.close()
