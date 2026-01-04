"""Data Caching Layer for StockAI.

Provides caching for API responses to reduce API calls and improve performance.
"""

import functools
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, TypeVar

from stockai.config import get_settings
from stockai.data.database import get_db, session_scope
from stockai.data.models import CacheEntry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheManager:
    """Manages caching of data with TTL support."""

    def __init__(self, ttl: int | None = None):
        """Initialize cache manager.

        Args:
            ttl: Time to live in seconds (default from settings)
        """
        settings = get_settings()
        self.ttl = ttl or settings.cache_ttl

    def _make_key(self, prefix: str, *args: Any) -> str:
        """Generate a cache key from prefix and arguments."""
        key_parts = [prefix] + [str(a) for a in args]
        return ":".join(key_parts)

    def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        try:
            with session_scope() as session:
                entry = (
                    session.query(CacheEntry)
                    .filter(CacheEntry.cache_key == key)
                    .filter(CacheEntry.expires_at > datetime.utcnow())
                    .first()
                )

                if entry:
                    try:
                        return json.loads(entry.cache_value)
                    except json.JSONDecodeError:
                        return entry.cache_value
                return None
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Optional TTL override in seconds

        Returns:
            True if successful
        """
        ttl = ttl or self.ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        try:
            # Serialize value
            if isinstance(value, str):
                cache_value = value
            else:
                cache_value = json.dumps(value, default=str)

            with session_scope() as session:
                # Upsert
                entry = (
                    session.query(CacheEntry)
                    .filter(CacheEntry.cache_key == key)
                    .first()
                )

                if entry:
                    entry.cache_value = cache_value
                    entry.expires_at = expires_at
                else:
                    entry = CacheEntry(
                        cache_key=key,
                        cache_value=cache_value,
                        expires_at=expires_at,
                    )
                    session.add(entry)

            return True
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            with session_scope() as session:
                deleted = (
                    session.query(CacheEntry)
                    .filter(CacheEntry.cache_key == key)
                    .delete()
                )
                return deleted > 0
        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")
            return False

    def clear_expired(self) -> int:
        """Clear all expired cache entries.

        Returns:
            Number of entries cleared
        """
        try:
            with session_scope() as session:
                deleted = (
                    session.query(CacheEntry)
                    .filter(CacheEntry.expires_at < datetime.utcnow())
                    .delete()
                )
                return deleted
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        try:
            with session_scope() as session:
                deleted = session.query(CacheEntry).delete()
                return deleted
        except Exception as e:
            logger.warning(f"Cache clear all error: {e}")
            return 0

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: int | None = None,
    ) -> T | None:
        """Get from cache or compute and set.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Optional TTL override

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            logger.debug(f"Cache hit: {key}")
            return value

        logger.debug(f"Cache miss: {key}")
        value = factory()
        if value is not None:
            self.set(key, value, ttl)
        return value


# In-memory cache for session-level caching
_memory_cache: dict[str, tuple[Any, datetime]] = {}


def memory_cache_get(key: str) -> Any | None:
    """Get from in-memory cache."""
    if key in _memory_cache:
        value, expires = _memory_cache[key]
        if expires > datetime.utcnow():
            return value
        del _memory_cache[key]
    return None


def memory_cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Set in in-memory cache."""
    expires = datetime.utcnow() + timedelta(seconds=ttl)
    _memory_cache[key] = (value, expires)


def memory_cache_clear() -> None:
    """Clear in-memory cache."""
    _memory_cache.clear()


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(prefix: str, ttl: int | None = None):
    """Decorator to cache function results.

    Args:
        prefix: Cache key prefix
        ttl: Optional TTL in seconds

    Example:
        @cached("stock_info")
        def get_stock_info(symbol: str) -> dict:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = get_cache()
            key = cache._make_key(prefix, *args, *kwargs.values())
            return cache.get_or_set(key, lambda: func(*args, **kwargs), ttl)
        return wrapper
    return decorator


def async_cached(prefix: str, ttl: int | None = None):
    """Decorator to cache async function results.

    Args:
        prefix: Cache key prefix
        ttl: Optional TTL in seconds

    Example:
        @async_cached("sentiment")
        async def get_sentiment(symbol: str) -> dict:
            ...
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = get_cache()
            key = cache._make_key(prefix, *args, *kwargs.values())

            # Check cache first
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {key}")
                return cached_value

            # Cache miss - await the async function
            logger.debug(f"Cache miss: {key}")
            value = await func(*args, **kwargs)

            # Store result in cache
            if value is not None:
                cache.set(key, value, ttl)

            return value
        return wrapper
    return decorator
