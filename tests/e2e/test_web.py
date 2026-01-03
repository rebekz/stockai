"""E2E Tests for StockAI Web Dashboard.

Tests FastAPI routes, API endpoints, and page rendering.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from stockai.web.app import create_app
from stockai import __version__


# ============ Fixtures ============

@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_idx_source():
    """Mock IDX data source."""
    with patch("stockai.web.routes.IDXIndexSource") as mock:
        instance = MagicMock()
        instance.get_idx30_stocks.return_value = [
            {"symbol": "BBCA", "name": "Bank Central Asia", "price": 9500, "change_pct": 1.5, "volume": 10000000},
            {"symbol": "BBRI", "name": "Bank Rakyat Indonesia", "price": 5200, "change_pct": -0.5, "volume": 15000000},
        ]
        instance.get_lq45_stocks.return_value = [
            {"symbol": "TLKM", "name": "Telkom Indonesia", "price": 3800, "change_pct": 0.2, "volume": 8000000},
        ]
        instance.get_stock_details.return_value = {
            "symbol": "BBCA",
            "name": "Bank Central Asia",
            "sector": "Finance",
            "industry": "Banking",
            "price": 9500,
            "change_pct": 1.5,
            "volume": 10000000,
            "is_idx30": True,
            "is_lq45": True,
        }
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_yahoo_source():
    """Mock Yahoo Finance data source."""
    import pandas as pd
    from datetime import datetime, timedelta

    with patch("stockai.web.routes.YahooFinanceSource") as mock:
        instance = MagicMock()

        # Generate sample price history with native Python types
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        df = pd.DataFrame({
            "date": dates,
            "open": [float(9400 + i * 10) for i in range(30)],
            "high": [float(9500 + i * 10) for i in range(30)],
            "low": [float(9300 + i * 10) for i in range(30)],
            "close": [float(9450 + i * 10) for i in range(30)],
            "volume": [int(10000000 + i * 100000) for i in range(30)],
        })
        instance.get_price_history.return_value = df
        mock.return_value = instance
        yield instance


# ============ Health Check Tests ============

class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_ok(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__

    def test_health_check_json_format(self, client):
        """Health check should return valid JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


# ============ API Status Tests ============

class TestAPIStatus:
    """Tests for API status endpoint."""

    def test_api_status_returns_ok(self, client):
        """API status should return ok."""
        response = client.get("/api/status")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__
        assert "timestamp" in data

    def test_api_status_timestamp_format(self, client):
        """API status timestamp should be ISO format."""
        response = client.get("/api/status")
        data = response.json()

        # Should be valid ISO format
        from datetime import datetime
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert timestamp is not None


# ============ Stock List API Tests ============

class TestStockListAPI:
    """Tests for stock list API endpoint."""

    def test_list_idx30_stocks(self, client, mock_idx_source):
        """Should return IDX30 stocks."""
        response = client.get("/api/stocks?index=IDX30")
        assert response.status_code == 200

        data = response.json()
        assert data["index"] == "IDX30"
        assert data["count"] == 2
        assert len(data["stocks"]) == 2

    def test_list_lq45_stocks(self, client, mock_idx_source):
        """Should return LQ45 stocks."""
        response = client.get("/api/stocks?index=LQ45")
        assert response.status_code == 200

        data = response.json()
        assert data["index"] == "LQ45"
        assert data["count"] == 1

    def test_list_stocks_with_prices(self, client, mock_idx_source):
        """Should include prices when requested."""
        response = client.get("/api/stocks?index=IDX30&include_prices=true")
        assert response.status_code == 200

        data = response.json()
        assert len(data["stocks"]) > 0
        mock_idx_source.get_idx30_stocks.assert_called_with(include_prices=True)

    def test_list_stocks_invalid_index(self, client, mock_idx_source):
        """Should return 400 for invalid index."""
        response = client.get("/api/stocks?index=INVALID")
        assert response.status_code == 400


# ============ Stock Info API Tests ============

class TestStockInfoAPI:
    """Tests for stock info API endpoint."""

    def test_get_stock_info(self, client, mock_idx_source):
        """Should return stock details."""
        response = client.get("/api/stocks/BBCA")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA"
        assert data["name"] == "Bank Central Asia"
        assert data["sector"] == "Finance"

    def test_get_stock_info_case_insensitive(self, client, mock_idx_source):
        """Should handle lowercase symbols."""
        response = client.get("/api/stocks/bbca")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA"

    def test_get_stock_info_not_found(self, client, mock_idx_source):
        """Should return 404 for unknown stock."""
        mock_idx_source.get_stock_details.return_value = None

        response = client.get("/api/stocks/UNKNOWN")
        assert response.status_code == 404


# ============ Stock History API Tests ============

class TestStockHistoryAPI:
    """Tests for stock price history API endpoint."""

    def test_get_stock_history(self, client, mock_yahoo_source):
        """Should return price history."""
        response = client.get("/api/stocks/BBCA/history")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA"
        assert data["period"] == "1mo"
        assert "history" in data
        assert len(data["history"]) > 0

    def test_get_stock_history_custom_period(self, client, mock_yahoo_source):
        """Should accept custom period."""
        response = client.get("/api/stocks/BBCA/history?period=3mo")
        assert response.status_code == 200

        data = response.json()
        assert data["period"] == "3mo"
        mock_yahoo_source.get_price_history.assert_called_with("BBCA", period="3mo")

    def test_get_stock_history_fields(self, client, mock_yahoo_source):
        """History should include OHLCV fields."""
        response = client.get("/api/stocks/BBCA/history")
        data = response.json()

        if data["history"]:
            entry = data["history"][0]
            assert "date" in entry
            assert "open" in entry
            assert "high" in entry
            assert "low" in entry
            assert "close" in entry
            assert "volume" in entry


# ============ Stock Chart API Tests ============

class TestStockChartAPI:
    """Tests for stock chart data API endpoint."""

    def test_get_chart_data(self, client, mock_yahoo_source):
        """Should return Plotly-formatted chart data."""
        response = client.get("/api/stocks/BBCA/chart")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA"
        assert "dates" in data
        assert "open" in data
        assert "high" in data
        assert "low" in data
        assert "close" in data
        assert "volume" in data

    def test_get_chart_data_arrays(self, client, mock_yahoo_source):
        """Chart data should have matching array lengths."""
        response = client.get("/api/stocks/BBCA/chart")
        data = response.json()

        length = len(data["dates"])
        assert len(data["open"]) == length
        assert len(data["high"]) == length
        assert len(data["low"]) == length
        assert len(data["close"]) == length
        assert len(data["volume"]) == length


# ============ Portfolio API Tests ============

class TestPortfolioAPI:
    """Tests for portfolio API endpoint."""

    def test_get_portfolio(self, client):
        """Should return portfolio data."""
        with patch("stockai.web.routes.init_database"):
            with patch("stockai.core.portfolio.PnLCalculator") as mock_calc:
                mock_instance = MagicMock()
                mock_instance.get_portfolio_summary.return_value = {
                    "positions": [],
                    "summary": {"total_value": 0, "total_cost": 0},
                }
                mock_calc.return_value = mock_instance

                response = client.get("/api/portfolio")

                # Should return 200 even with empty portfolio
                assert response.status_code == 200


# ============ Export API Tests ============

@pytest.fixture
def mock_sentiment_modules():
    """Mock sentiment modules for export tests."""
    with patch.dict("sys.modules", {
        "stockai.core.sentiment": MagicMock(),
    }):
        yield


class TestExportAPI:
    """Tests for export API endpoint."""

    def test_export_report_structure(self, client, mock_idx_source, mock_yahoo_source):
        """Export should return report data structure."""
        # The export endpoint handles sentiment errors gracefully,
        # so we can test without mocking sentiment
        response = client.get("/api/export/BBCA")
        assert response.status_code == 200

        data = response.json()
        assert "symbol" in data
        assert "generated_at" in data
        assert "version" in data
        assert "stock_info" in data

    def test_export_includes_stock_info(self, client, mock_idx_source, mock_yahoo_source):
        """Export should include stock information."""
        response = client.get("/api/export/BBCA")
        data = response.json()

        assert data["symbol"] == "BBCA"
        assert data["stock_info"]["name"] == "Bank Central Asia"

    def test_export_includes_price_history(self, client, mock_idx_source, mock_yahoo_source):
        """Export should include price history."""
        response = client.get("/api/export/BBCA")
        data = response.json()

        assert "price_history" in data
        assert "price_stats" in data

    def test_export_not_found(self, client, mock_idx_source, mock_yahoo_source):
        """Export should return 404 for unknown stock."""
        mock_idx_source.get_stock_details.return_value = None

        response = client.get("/api/export/UNKNOWN")
        assert response.status_code == 404


# ============ Page Rendering Tests ============

class TestPageRendering:
    """Tests for HTML page rendering."""

    def test_home_page_renders(self, client):
        """Home page should render HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_home_page_contains_title(self, client):
        """Home page should contain title."""
        response = client.get("/")
        assert b"StockAI" in response.content

    def test_stocks_page_renders(self, client, mock_idx_source):
        """Stocks page should render."""
        response = client.get("/stocks")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_analyze_page_renders(self, client, mock_idx_source):
        """Analyze page should render."""
        response = client.get("/analyze/BBCA")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_analyze_page_contains_symbol(self, client, mock_idx_source):
        """Analyze page should show stock symbol."""
        response = client.get("/analyze/BBCA")
        assert b"BBCA" in response.content

    def test_analyze_page_not_found(self, client, mock_idx_source):
        """Analyze page should return 404 for unknown stock."""
        mock_idx_source.get_stock_details.return_value = None

        response = client.get("/analyze/UNKNOWN")
        assert response.status_code == 404

    def test_portfolio_page_renders(self, client):
        """Portfolio page should render."""
        with patch("stockai.web.routes.init_database"):
            with patch("stockai.core.portfolio.PnLCalculator") as mock_calc:
                mock_instance = MagicMock()
                mock_instance.get_portfolio_summary.return_value = {
                    "positions": [],
                    "summary": {},
                }
                mock_calc.return_value = mock_instance

                response = client.get("/portfolio")
                assert response.status_code == 200

    def test_sentiment_page_renders(self, client):
        """Sentiment page should render."""
        response = client.get("/sentiment")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# ============ Error Handling Tests ============

class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_unknown_route(self, client):
        """Should return 404 for unknown routes."""
        response = client.get("/unknown/route")
        assert response.status_code == 404

    def test_api_error_returns_json(self, client, mock_idx_source):
        """API errors should return JSON."""
        mock_idx_source.get_stock_details.return_value = None

        response = client.get(
            "/api/stocks/UNKNOWN",
            headers={"accept": "application/json"}
        )
        assert response.status_code == 404
        assert "application/json" in response.headers["content-type"]


# ============ CLI Web Command Tests ============

class TestCLIWebCommand:
    """Tests for CLI web command."""

    def test_web_command_exists(self):
        """Web command should be registered."""
        from stockai.cli.main import app
        from typer.testing import CliRunner

        runner = CliRunner()

        # Check help shows web command
        result = runner.invoke(app, ["--help"])
        assert "web" in result.output

    def test_web_command_help(self):
        """Web command should have help text."""
        from stockai.cli.main import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["web", "--help"])

        assert result.exit_code == 0
        assert "Start the StockAI web dashboard" in result.output
        assert "--host" in result.output
        assert "--port" in result.output


# ============ App Factory Tests ============

class TestAppFactory:
    """Tests for app factory."""

    def test_create_app_returns_fastapi(self):
        """create_app should return FastAPI instance."""
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_has_routes(self):
        """App should have routes configured."""
        app = create_app()

        # Check some expected routes exist
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/api/status" in routes

    def test_create_app_has_templates(self):
        """App should have templates configured."""
        app = create_app()
        assert hasattr(app.state, "templates")

    def test_create_app_docs_url(self):
        """App should have API docs configured."""
        app = create_app()
        assert app.docs_url == "/api/docs"


# ============ Integration Tests ============

class TestWebIntegration:
    """Integration tests for web module."""

    def test_full_stock_workflow(self, client, mock_idx_source, mock_yahoo_source):
        """Test complete stock analysis workflow."""
        # 1. List stocks
        list_response = client.get("/api/stocks")
        assert list_response.status_code == 200

        stocks = list_response.json()["stocks"]
        symbol = stocks[0]["symbol"]

        # 2. Get stock details
        info_response = client.get(f"/api/stocks/{symbol}")
        assert info_response.status_code == 200

        # 3. Get price history
        history_response = client.get(f"/api/stocks/{symbol}/history")
        assert history_response.status_code == 200

        # 4. Get chart data
        chart_response = client.get(f"/api/stocks/{symbol}/chart")
        assert chart_response.status_code == 200

        # 5. Export report
        with patch("stockai.core.sentiment.SentimentAnalyzer"):
            with patch("stockai.core.sentiment.NewsAggregator") as mock_news:
                mock_news.return_value.fetch_all.return_value = []
                export_response = client.get(f"/api/export/{symbol}")
                assert export_response.status_code == 200

    def test_navigation_between_pages(self, client, mock_idx_source):
        """Test navigation between pages."""
        # Home page
        home = client.get("/")
        assert home.status_code == 200

        # Navigate to stocks
        stocks = client.get("/stocks")
        assert stocks.status_code == 200

        # Navigate to analyze
        analyze = client.get("/analyze/BBCA")
        assert analyze.status_code == 200

        # Navigate to sentiment
        sentiment = client.get("/sentiment")
        assert sentiment.status_code == 200
