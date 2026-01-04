"""E2E Tests for Watchlist API Endpoints.

Tests for all watchlist CRUD operations using FastAPI TestClient.
Includes success cases, error cases (404, 409), and full CRUD workflow.
"""

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from stockai.web.app import create_app
from stockai.data.database import DatabaseManager, init_database, session_scope
from stockai.data.models import Stock


# ============ Fixtures ============


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.

    Uses a unique database file per test to ensure proper isolation.
    """
    from stockai.data import database as db_module
    from stockai import config as config_module

    with tempfile.TemporaryDirectory() as tmpdir:
        unique_id = uuid.uuid4().hex[:8]
        db_path = os.path.join(tmpdir, f"test_{unique_id}.db")
        os.environ["STOCKAI_DB_PATH"] = db_path

        config_module.get_settings.cache_clear()

        if DatabaseManager._engine is not None:
            try:
                DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._SessionLocal = None
        db_module._db_manager = None

        init_database()

        yield tmpdir

        if DatabaseManager._engine is not None:
            try:
                DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._SessionLocal = None
        db_module._db_manager = None

        config_module.get_settings.cache_clear()

        if "STOCKAI_DB_PATH" in os.environ:
            del os.environ["STOCKAI_DB_PATH"]


@pytest.fixture
def client(temp_db):
    """Create test client for FastAPI app with isolated database."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_stock(temp_db):
    """Create a sample stock for testing."""
    with session_scope() as session:
        stock = Stock(
            symbol="BBCA",
            name="Bank Central Asia Tbk",
            sector="Financial",
            is_active=True,
        )
        session.add(stock)
        session.flush()
        stock_id = stock.id

    with session_scope() as session:
        stock = session.query(Stock).filter(Stock.id == stock_id).first()
        session.expunge(stock)
        return stock


@pytest.fixture
def sample_watchlist_item(client, sample_stock):
    """Create a sample watchlist item via API for testing."""
    response = client.post(
        "/api/watchlist",
        json={
            "symbol": "BBCA",
            "alert_price_above": 10000.0,
            "alert_price_below": 8000.0,
            "notes": "Test watchlist item",
        },
    )
    assert response.status_code == 201
    return response.json()


# ============ GET /api/watchlist Tests ============


class TestListWatchlist:
    """Tests for GET /api/watchlist endpoint."""

    def test_list_empty_watchlist(self, client):
        """Should return empty list when watchlist is empty."""
        response = client.get("/api/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 0
        assert data["items"] == []

    def test_list_watchlist_with_items(self, sample_watchlist_item, client):
        """Should return list of watchlist items."""
        response = client.get("/api/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1
        assert len(data["items"]) == 1

        item = data["items"][0]
        assert item["stock"]["symbol"] == "BBCA"
        assert item["alert_price_above"] == 10000.0
        assert item["alert_price_below"] == 8000.0
        assert item["notes"] == "Test watchlist item"

    def test_list_watchlist_includes_stock_info(self, sample_watchlist_item, client):
        """Should include stock information in each item."""
        response = client.get("/api/watchlist")
        data = response.json()

        item = data["items"][0]
        assert "stock" in item
        assert "id" in item["stock"]
        assert "symbol" in item["stock"]
        assert "name" in item["stock"]
        assert item["stock"]["symbol"] == "BBCA"
        assert item["stock"]["name"] == "Bank Central Asia Tbk"

    def test_list_watchlist_multiple_items(self, client, temp_db):
        """Should return all watchlist items."""
        symbols = ["BBCA", "BBRI", "TLKM"]

        for symbol in symbols:
            response = client.post("/api/watchlist", json={"symbol": symbol})
            assert response.status_code == 201

        response = client.get("/api/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 3
        assert len(data["items"]) == 3

        item_symbols = {item["stock"]["symbol"] for item in data["items"]}
        assert item_symbols == {"BBCA", "BBRI", "TLKM"}


# ============ POST /api/watchlist Tests ============


class TestCreateWatchlistItem:
    """Tests for POST /api/watchlist endpoint."""

    def test_create_watchlist_item_by_symbol(self, client, temp_db):
        """Should create watchlist item by symbol."""
        response = client.post(
            "/api/watchlist",
            json={"symbol": "BBCA"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["stock"]["symbol"] == "BBCA"
        assert "id" in data
        assert "created_at" in data

    def test_create_watchlist_item_with_alerts(self, client, temp_db):
        """Should create watchlist item with alert prices."""
        response = client.post(
            "/api/watchlist",
            json={
                "symbol": "BBCA",
                "alert_price_above": 10000.0,
                "alert_price_below": 8000.0,
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["alert_price_above"] == 10000.0
        assert data["alert_price_below"] == 8000.0

    def test_create_watchlist_item_with_notes(self, client, temp_db):
        """Should create watchlist item with notes."""
        response = client.post(
            "/api/watchlist",
            json={
                "symbol": "BBCA",
                "notes": "Bank stock to watch",
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["notes"] == "Bank stock to watch"

    def test_create_watchlist_item_creates_stock(self, client, temp_db):
        """Should create stock if it doesn't exist."""
        response = client.post(
            "/api/watchlist",
            json={"symbol": "NEWSTOCK"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["stock"]["symbol"] == "NEWSTOCK"
        assert data["stock"]["id"] is not None

    def test_create_watchlist_item_symbol_case_insensitive(self, client, temp_db):
        """Should normalize symbol to uppercase."""
        response = client.post(
            "/api/watchlist",
            json={"symbol": "bbca"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["stock"]["symbol"] == "BBCA"

    def test_create_watchlist_item_duplicate_returns_409(
        self, sample_watchlist_item, client
    ):
        """Should return 409 Conflict for duplicate stock."""
        response = client.post(
            "/api/watchlist",
            json={"symbol": "BBCA"},
            headers={"accept": "application/json"},
        )
        assert response.status_code == 409
        assert "already in the watchlist" in response.json()["error"]

    def test_create_watchlist_item_requires_symbol_or_stock_id(self, client, temp_db):
        """Should return 422 if neither symbol nor stock_id provided."""
        response = client.post(
            "/api/watchlist",
            json={},
        )
        assert response.status_code == 422

    def test_create_watchlist_item_by_stock_id(self, client, sample_stock):
        """Should create watchlist item by stock_id."""
        response = client.post(
            "/api/watchlist",
            json={"stock_id": sample_stock.id},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["stock"]["symbol"] == "BBCA"
        assert data["stock"]["id"] == sample_stock.id

    def test_create_watchlist_item_validates_alert_prices(self, client, temp_db):
        """Should validate alert_price_below < alert_price_above."""
        response = client.post(
            "/api/watchlist",
            json={
                "symbol": "BBCA",
                "alert_price_above": 8000.0,
                "alert_price_below": 10000.0,
            },
        )
        assert response.status_code == 422


# ============ GET /api/watchlist/{item_id} Tests ============


class TestGetWatchlistItem:
    """Tests for GET /api/watchlist/{item_id} endpoint."""

    def test_get_watchlist_item_by_id(self, sample_watchlist_item, client):
        """Should return watchlist item by ID."""
        item_id = sample_watchlist_item["id"]

        response = client.get(f"/api/watchlist/{item_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == item_id
        assert data["stock"]["symbol"] == "BBCA"
        assert data["notes"] == "Test watchlist item"

    def test_get_watchlist_item_includes_stock_info(self, sample_watchlist_item, client):
        """Should include stock information."""
        item_id = sample_watchlist_item["id"]

        response = client.get(f"/api/watchlist/{item_id}")
        data = response.json()

        assert "stock" in data
        assert data["stock"]["symbol"] == "BBCA"
        assert data["stock"]["name"] == "Bank Central Asia Tbk"
        assert data["stock"]["sector"] == "Financial"

    def test_get_watchlist_item_not_found_returns_404(self, client, temp_db):
        """Should return 404 for non-existent ID."""
        response = client.get(
            "/api/watchlist/99999",
            headers={"accept": "application/json"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["error"]


# ============ PUT /api/watchlist/{item_id} Tests ============


class TestUpdateWatchlistItem:
    """Tests for PUT /api/watchlist/{item_id} endpoint."""

    def test_update_alert_price_above(self, sample_watchlist_item, client):
        """Should update alert_price_above."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={"alert_price_above": 12000.0},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["alert_price_above"] == 12000.0
        # Other fields unchanged
        assert data["alert_price_below"] == 8000.0
        assert data["notes"] == "Test watchlist item"

    def test_update_alert_price_below(self, sample_watchlist_item, client):
        """Should update alert_price_below."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={"alert_price_below": 7000.0},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["alert_price_below"] == 7000.0

    def test_update_notes(self, sample_watchlist_item, client):
        """Should update notes."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={"notes": "Updated notes"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["notes"] == "Updated notes"

    def test_update_multiple_fields(self, sample_watchlist_item, client):
        """Should update multiple fields at once."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={
                "alert_price_above": 15000.0,
                "alert_price_below": 6000.0,
                "notes": "All fields updated",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["alert_price_above"] == 15000.0
        assert data["alert_price_below"] == 6000.0
        assert data["notes"] == "All fields updated"

    def test_update_clear_alert_above(self, sample_watchlist_item, client):
        """Should clear alert_price_above when set to 0."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={"alert_price_above": 0},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["alert_price_above"] is None

    def test_update_clear_notes(self, sample_watchlist_item, client):
        """Should clear notes when set to empty string."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={"notes": ""},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["notes"] is None

    def test_update_not_found_returns_404(self, client, temp_db):
        """Should return 404 for non-existent ID."""
        response = client.put(
            "/api/watchlist/99999",
            json={"notes": "test"},
            headers={"accept": "application/json"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["error"]

    def test_update_returns_stock_info(self, sample_watchlist_item, client):
        """Should include stock info in response."""
        item_id = sample_watchlist_item["id"]

        response = client.put(
            f"/api/watchlist/{item_id}",
            json={"notes": "Updated"},
        )
        data = response.json()

        assert "stock" in data
        assert data["stock"]["symbol"] == "BBCA"


# ============ DELETE /api/watchlist/{item_id} Tests ============


class TestDeleteWatchlistItem:
    """Tests for DELETE /api/watchlist/{item_id} endpoint."""

    def test_delete_watchlist_item(self, sample_watchlist_item, client):
        """Should delete watchlist item by ID."""
        item_id = sample_watchlist_item["id"]

        response = client.delete(f"/api/watchlist/{item_id}")
        assert response.status_code == 200

        data = response.json()
        assert "Successfully removed" in data["message"]
        assert data["deleted_item"]["id"] == item_id

    def test_delete_returns_deleted_item_info(self, sample_watchlist_item, client):
        """Should return deleted item information."""
        item_id = sample_watchlist_item["id"]

        response = client.delete(f"/api/watchlist/{item_id}")
        data = response.json()

        assert "deleted_item" in data
        assert data["deleted_item"]["stock"]["symbol"] == "BBCA"
        assert data["deleted_item"]["notes"] == "Test watchlist item"

    def test_delete_item_is_removed(self, sample_watchlist_item, client):
        """Should actually remove item from database."""
        item_id = sample_watchlist_item["id"]

        # Delete the item
        delete_response = client.delete(f"/api/watchlist/{item_id}")
        assert delete_response.status_code == 200

        # Verify it's gone
        get_response = client.get(f"/api/watchlist/{item_id}")
        assert get_response.status_code == 404

    def test_delete_not_found_returns_404(self, client, temp_db):
        """Should return 404 for non-existent ID."""
        response = client.delete(
            "/api/watchlist/99999",
            headers={"accept": "application/json"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["error"]


# ============ DELETE /api/watchlist/symbol/{symbol} Tests ============


class TestDeleteWatchlistItemBySymbol:
    """Tests for DELETE /api/watchlist/symbol/{symbol} endpoint."""

    def test_delete_by_symbol(self, sample_watchlist_item, client):
        """Should delete watchlist item by symbol."""
        response = client.delete("/api/watchlist/symbol/BBCA")
        assert response.status_code == 200

        data = response.json()
        assert "Successfully removed" in data["message"]
        assert data["deleted_item"]["stock"]["symbol"] == "BBCA"

    def test_delete_by_symbol_case_insensitive(self, sample_watchlist_item, client):
        """Should handle lowercase symbols."""
        response = client.delete("/api/watchlist/symbol/bbca")
        assert response.status_code == 200

        data = response.json()
        assert data["deleted_item"]["stock"]["symbol"] == "BBCA"

    def test_delete_by_symbol_item_is_removed(self, sample_watchlist_item, client):
        """Should actually remove item from database."""
        # Delete by symbol
        delete_response = client.delete("/api/watchlist/symbol/BBCA")
        assert delete_response.status_code == 200

        # Verify watchlist is empty
        list_response = client.get("/api/watchlist")
        assert list_response.json()["count"] == 0

    def test_delete_by_symbol_not_found_returns_404(self, client, temp_db):
        """Should return 404 for symbol not in watchlist."""
        response = client.delete(
            "/api/watchlist/symbol/NOTFOUND",
            headers={"accept": "application/json"},
        )
        assert response.status_code == 404
        assert "not in the watchlist" in response.json()["error"]


# ============ Full CRUD Workflow Tests ============


class TestWatchlistCRUDWorkflow:
    """Integration tests for complete CRUD workflow."""

    def test_full_crud_workflow(self, client, temp_db):
        """Test complete create-read-update-delete workflow."""
        # 1. Create a watchlist item
        create_response = client.post(
            "/api/watchlist",
            json={
                "symbol": "BBRI",
                "alert_price_above": 5000.0,
                "alert_price_below": 4000.0,
                "notes": "Initial notes",
            },
        )
        assert create_response.status_code == 201
        item_id = create_response.json()["id"]

        # 2. Read the item
        get_response = client.get(f"/api/watchlist/{item_id}")
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["stock"]["symbol"] == "BBRI"
        assert data["notes"] == "Initial notes"

        # 3. Update the item
        update_response = client.put(
            f"/api/watchlist/{item_id}",
            json={
                "alert_price_above": 5500.0,
                "notes": "Updated notes",
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["alert_price_above"] == 5500.0
        assert update_response.json()["notes"] == "Updated notes"

        # 4. Verify update persisted
        verify_response = client.get(f"/api/watchlist/{item_id}")
        assert verify_response.json()["alert_price_above"] == 5500.0

        # 5. Delete the item
        delete_response = client.delete(f"/api/watchlist/{item_id}")
        assert delete_response.status_code == 200

        # 6. Verify deletion
        final_response = client.get(f"/api/watchlist/{item_id}")
        assert final_response.status_code == 404

    def test_multiple_stocks_workflow(self, client, temp_db):
        """Test managing multiple stocks in watchlist."""
        symbols = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII"]

        # Add all stocks
        created_ids = []
        for symbol in symbols:
            response = client.post("/api/watchlist", json={"symbol": symbol})
            assert response.status_code == 201
            created_ids.append(response.json()["id"])

        # Verify all added
        list_response = client.get("/api/watchlist")
        assert list_response.json()["count"] == 5

        # Update one
        update_response = client.put(
            f"/api/watchlist/{created_ids[0]}",
            json={"notes": "Updated BBCA"},
        )
        assert update_response.status_code == 200

        # Delete by symbol
        delete_response = client.delete("/api/watchlist/symbol/TLKM")
        assert delete_response.status_code == 200

        # Verify removal
        final_list = client.get("/api/watchlist")
        assert final_list.json()["count"] == 4
        remaining_symbols = {
            item["stock"]["symbol"] for item in final_list.json()["items"]
        }
        assert "TLKM" not in remaining_symbols

    def test_add_same_stock_twice_fails(self, client, temp_db):
        """Test that adding the same stock twice returns 409."""
        # Add first time - should succeed
        first_response = client.post("/api/watchlist", json={"symbol": "BBCA"})
        assert first_response.status_code == 201

        # Add second time - should fail with 409
        second_response = client.post("/api/watchlist", json={"symbol": "BBCA"})
        assert second_response.status_code == 409

    def test_delete_then_add_again(self, client, temp_db):
        """Test that deleting and re-adding a stock works."""
        # Add stock
        create_response = client.post("/api/watchlist", json={"symbol": "BBCA"})
        assert create_response.status_code == 201
        item_id = create_response.json()["id"]

        # Delete stock
        delete_response = client.delete(f"/api/watchlist/{item_id}")
        assert delete_response.status_code == 200

        # Add again - should succeed
        re_add_response = client.post("/api/watchlist", json={"symbol": "BBCA"})
        assert re_add_response.status_code == 201
        assert re_add_response.json()["stock"]["symbol"] == "BBCA"


# ============ Response Format Tests ============


class TestResponseFormat:
    """Tests for API response format and structure."""

    def test_list_response_format(self, sample_watchlist_item, client):
        """Test that list response has correct format."""
        response = client.get("/api/watchlist")
        data = response.json()

        assert "count" in data
        assert "items" in data
        assert isinstance(data["count"], int)
        assert isinstance(data["items"], list)

    def test_item_response_format(self, sample_watchlist_item, client):
        """Test that item response has all required fields."""
        item_id = sample_watchlist_item["id"]
        response = client.get(f"/api/watchlist/{item_id}")
        data = response.json()

        # Check required fields
        assert "id" in data
        assert "stock" in data
        assert "alert_price_above" in data
        assert "alert_price_below" in data
        assert "notes" in data
        assert "created_at" in data

        # Check stock nested object
        stock = data["stock"]
        assert "id" in stock
        assert "symbol" in stock
        assert "name" in stock

    def test_delete_response_format(self, sample_watchlist_item, client):
        """Test that delete response has correct format."""
        item_id = sample_watchlist_item["id"]
        response = client.delete(f"/api/watchlist/{item_id}")
        data = response.json()

        assert "message" in data
        assert "deleted_item" in data
        assert isinstance(data["message"], str)

    def test_error_response_format(self, client, temp_db):
        """Test that error responses have correct format."""
        response = client.get(
            "/api/watchlist/99999",
            headers={"accept": "application/json"},
        )
        assert response.status_code == 404

        data = response.json()
        assert "error" in data
        assert isinstance(data["error"], str)
