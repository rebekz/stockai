"""Unit Tests for Async Cache Decorator.

Tests the async_cached decorator in isolation:
- Cache hit returns cached value without calling function
- Cache miss calls underlying function
- Key generation with different arguments
- Custom TTL is respected
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta


class TestAsyncCachedDecorator:
    """Test async_cached decorator functionality."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        mock = MagicMock()
        mock._make_key = MagicMock(side_effect=lambda prefix, *args: f"{prefix}:{':'.join(str(a) for a in args)}")
        return mock

    @pytest.mark.asyncio
    async def test_cache_miss_calls_underlying_function(self, mock_cache_manager):
        """Test that cache miss calls the underlying async function."""
        # Setup: mock cache.get returns None (cache miss)
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            # Track function calls
            call_count = 0

            @async_cached("test_prefix")
            async def test_function(symbol: str) -> dict:
                nonlocal call_count
                call_count += 1
                return {"symbol": symbol, "data": "from_function"}

            # Call the decorated function
            result = await test_function("BBCA")

            # Verify: function was called once
            assert call_count == 1
            assert result == {"symbol": "BBCA", "data": "from_function"}

            # Verify: cache.get was called to check cache
            mock_cache_manager.get.assert_called_once()

            # Verify: cache.set was called to store result
            mock_cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value_without_calling_function(self, mock_cache_manager):
        """Test that cache hit returns cached value without calling underlying function."""
        # Setup: mock cache.get returns cached value (cache hit)
        cached_data = {"symbol": "BBCA", "data": "from_cache"}
        mock_cache_manager.get.return_value = cached_data

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            # Track function calls
            call_count = 0

            @async_cached("test_prefix")
            async def test_function(symbol: str) -> dict:
                nonlocal call_count
                call_count += 1
                return {"symbol": symbol, "data": "from_function"}

            # Call the decorated function
            result = await test_function("BBCA")

            # Verify: function was NOT called (cache hit)
            assert call_count == 0
            assert result == {"symbol": "BBCA", "data": "from_cache"}

            # Verify: cache.get was called to check cache
            mock_cache_manager.get.assert_called_once()

            # Verify: cache.set was NOT called (no need to store)
            mock_cache_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_key_generation_with_positional_args(self, mock_cache_manager):
        """Test cache key is generated correctly with positional arguments."""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            @async_cached("sentiment")
            async def get_sentiment(symbol: str, days: int) -> dict:
                return {"symbol": symbol, "days": days}

            await get_sentiment("BBCA", 7)

            # Verify key was made with prefix and both args
            mock_cache_manager._make_key.assert_called_with("sentiment", "BBCA", 7)

    @pytest.mark.asyncio
    async def test_key_generation_with_kwargs(self, mock_cache_manager):
        """Test cache key is generated correctly with keyword arguments."""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            @async_cached("prediction")
            async def get_prediction(symbol: str, model: str = "default") -> dict:
                return {"symbol": symbol, "model": model}

            await get_prediction("TLKM", model="advanced")

            # Verify key was made with prefix and both args (positional + kwarg value)
            mock_cache_manager._make_key.assert_called_with("prediction", "TLKM", "advanced")

    @pytest.mark.asyncio
    async def test_different_args_produce_different_cache_keys(self, mock_cache_manager):
        """Test that different arguments produce different cache keys."""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True
        captured_keys = []

        def track_key(prefix, *args):
            key = f"{prefix}:{':'.join(str(a) for a in args)}"
            captured_keys.append(key)
            return key

        mock_cache_manager._make_key = MagicMock(side_effect=track_key)

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            @async_cached("stock")
            async def get_stock_data(symbol: str) -> dict:
                return {"symbol": symbol}

            await get_stock_data("BBCA")
            await get_stock_data("TLKM")
            await get_stock_data("ASII")

            # Verify different symbols produce different keys
            assert len(captured_keys) == 3
            assert captured_keys[0] == "stock:BBCA"
            assert captured_keys[1] == "stock:TLKM"
            assert captured_keys[2] == "stock:ASII"

    @pytest.mark.asyncio
    async def test_custom_ttl_is_passed_to_cache_set(self, mock_cache_manager):
        """Test that custom TTL is passed to cache.set()."""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            @async_cached("data", ttl=300)  # 5 minutes TTL
            async def get_data(item: str) -> dict:
                return {"item": item}

            await get_data("test_item")

            # Verify cache.set was called with custom TTL
            mock_cache_manager.set.assert_called_once()
            call_args = mock_cache_manager.set.call_args
            # set(key, value, ttl)
            assert call_args[0][2] == 300  # Third positional arg is TTL

    @pytest.mark.asyncio
    async def test_default_ttl_when_not_specified(self, mock_cache_manager):
        """Test that None is passed for TTL when not specified (uses default)."""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            @async_cached("data")  # No TTL specified
            async def get_data(item: str) -> dict:
                return {"item": item}

            await get_data("test_item")

            # Verify cache.set was called with None TTL (uses default)
            mock_cache_manager.set.assert_called_once()
            call_args = mock_cache_manager.set.call_args
            assert call_args[0][2] is None  # Third positional arg is TTL

    @pytest.mark.asyncio
    async def test_none_result_is_not_cached(self, mock_cache_manager):
        """Test that None results are not stored in cache."""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import async_cached

            @async_cached("data")
            async def get_data(item: str) -> dict | None:
                return None  # Returns None

            result = await get_data("test_item")

            # Verify result is None
            assert result is None

            # Verify cache.set was NOT called (don't cache None)
            mock_cache_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that @functools.wraps preserves function name and docstring."""
        with patch("stockai.data.cache.get_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.get.return_value = None
            mock_cache.set.return_value = True
            mock_cache._make_key = MagicMock(return_value="key")
            mock_get_cache.return_value = mock_cache

            from stockai.data.cache import async_cached

            @async_cached("test")
            async def my_documented_function(x: int) -> int:
                """This is the docstring."""
                return x * 2

            # Check function metadata is preserved
            assert my_documented_function.__name__ == "my_documented_function"
            assert my_documented_function.__doc__ == "This is the docstring."


class TestCacheKeyGeneration:
    """Test cache key generation logic."""

    def test_make_key_with_single_arg(self):
        """Test key generation with single argument."""
        from stockai.data.cache import CacheManager

        manager = CacheManager(ttl=60)
        key = manager._make_key("prefix", "arg1")

        assert key == "prefix:arg1"

    def test_make_key_with_multiple_args(self):
        """Test key generation with multiple arguments."""
        from stockai.data.cache import CacheManager

        manager = CacheManager(ttl=60)
        key = manager._make_key("sentiment", "BBCA", 7, "daily")

        assert key == "sentiment:BBCA:7:daily"

    def test_make_key_with_no_args(self):
        """Test key generation with no additional arguments."""
        from stockai.data.cache import CacheManager

        manager = CacheManager(ttl=60)
        key = manager._make_key("global")

        assert key == "global"

    def test_make_key_handles_numeric_args(self):
        """Test key generation handles numeric arguments properly."""
        from stockai.data.cache import CacheManager

        manager = CacheManager(ttl=60)
        key = manager._make_key("data", 123, 45.67)

        assert key == "data:123:45.67"


class TestSyncCachedDecorator:
    """Test the sync cached decorator for comparison."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        mock = MagicMock()
        mock._make_key = MagicMock(
            side_effect=lambda prefix, *args: f"{prefix}:{':'.join(str(a) for a in args)}"
        )
        return mock

    def test_sync_cached_cache_miss(self, mock_cache_manager):
        """Test sync cached decorator handles cache miss."""
        mock_cache_manager.get_or_set = MagicMock(return_value={"data": "computed"})

        with patch("stockai.data.cache.get_cache", return_value=mock_cache_manager):
            from stockai.data.cache import cached

            @cached("sync_test")
            def sync_function(x: int) -> dict:
                return {"x": x}

            result = sync_function(42)

            # The cached decorator uses get_or_set which handles everything
            mock_cache_manager.get_or_set.assert_called_once()
            assert result == {"data": "computed"}

    def test_sync_cached_preserves_function_metadata(self):
        """Test that sync cached decorator preserves function metadata."""
        with patch("stockai.data.cache.get_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.get_or_set = MagicMock(return_value={"result": 1})
            mock_cache._make_key = MagicMock(return_value="key")
            mock_get_cache.return_value = mock_cache

            from stockai.data.cache import cached

            @cached("test")
            def my_sync_function(x: int) -> int:
                """Sync function docstring."""
                return x + 1

            assert my_sync_function.__name__ == "my_sync_function"
            assert my_sync_function.__doc__ == "Sync function docstring."
