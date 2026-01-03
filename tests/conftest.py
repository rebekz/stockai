"""Pytest configuration and fixtures for StockAI tests."""

import os
import pytest


@pytest.fixture(autouse=True)
def isolate_database():
    """Ensure database isolation for all tests.

    This fixture runs before every test to ensure the DatabaseManager
    singleton is properly reset, preventing test pollution.
    """
    # Store original env var if exists
    original_db_path = os.environ.get("STOCKAI_DB_PATH")

    # Import here to avoid circular imports at module load
    from stockai.data import database as db_module
    from stockai.data.database import DatabaseManager
    from stockai import config as config_module

    # Clear the settings cache so new env vars are picked up
    config_module.get_settings.cache_clear()

    # Clear database singleton before test
    if DatabaseManager._engine is not None:
        try:
            DatabaseManager._engine.dispose()
        except Exception:
            pass
    DatabaseManager._instance = None
    DatabaseManager._engine = None
    DatabaseManager._SessionLocal = None

    # Also reset the global db_manager
    db_module._db_manager = None

    yield

    # Clear everything after test
    if DatabaseManager._engine is not None:
        try:
            DatabaseManager._engine.dispose()
        except Exception:
            pass
    DatabaseManager._instance = None
    DatabaseManager._engine = None
    DatabaseManager._SessionLocal = None
    db_module._db_manager = None

    # Clear settings cache again for next test
    config_module.get_settings.cache_clear()

    # Restore original env var
    if original_db_path is not None:
        os.environ["STOCKAI_DB_PATH"] = original_db_path
    elif "STOCKAI_DB_PATH" in os.environ:
        del os.environ["STOCKAI_DB_PATH"]
