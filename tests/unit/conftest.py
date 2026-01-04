"""Unit test configuration.

Overrides the autouse isolate_database fixture from root conftest.py
for unit tests that don't need database isolation.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_database():
    """Override the global fixture - unit tests mock everything and don't need DB isolation."""
    yield
