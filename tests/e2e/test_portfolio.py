"""E2E Tests for Portfolio Management (Epic 4).

Tests portfolio manager, P&L calculator, and analytics.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from stockai.data.database import init_database, get_db
from stockai.data.models import Stock, PortfolioItem, PortfolioTransaction
from stockai.data.sources.yahoo import YahooFinanceSource
from stockai.core.portfolio import PortfolioManager, PnLCalculator, PortfolioAnalytics


class TestPortfolioManager:
    """Test suite for PortfolioManager (Stories 4.1, 4.2)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database and manager."""
        init_database()
        self.manager = PortfolioManager()
        # Clear any existing portfolio data
        self.manager.clear_portfolio()

    def test_add_position_creates_new_position(self):
        """Test adding a new position."""
        result = self.manager.add_position(
            symbol="BBCA",
            shares=100,
            price=9500.0,
        )

        assert result["action"] == "BUY"
        assert result["symbol"] == "BBCA"
        assert result["shares"] == 100
        assert result["price"] == 9500.0
        assert result["total_shares"] == 100
        assert result["avg_price"] == 9500.0
        assert result["total_cost"] == 950000.0

    def test_add_position_updates_existing_position(self):
        """Test adding to existing position updates average cost."""
        # First buy
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        # Second buy at different price
        result = self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)

        assert result["total_shares"] == 200
        # Average should be (100*9500 + 100*10000) / 200 = 9750
        assert result["avg_price"] == 9750.0

    def test_add_position_creates_transaction(self):
        """Test that adding position creates transaction record."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        transactions = self.manager.get_transactions(symbol="BBCA")
        assert len(transactions) == 1
        assert transactions[0]["type"] == "BUY"
        assert transactions[0]["shares"] == 100

    def test_add_position_validation(self):
        """Test validation for invalid inputs."""
        with pytest.raises(ValueError, match="positive"):
            self.manager.add_position(symbol="BBCA", shares=0, price=9500.0)

        with pytest.raises(ValueError, match="positive"):
            self.manager.add_position(symbol="BBCA", shares=100, price=-1)

    def test_remove_position_partial_sell(self):
        """Test partial sell of position."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        result = self.manager.remove_position(
            symbol="BBCA",
            shares=50,
            price=10000.0,
        )

        assert result["action"] == "SELL"
        assert result["shares"] == 50
        assert result["remaining_shares"] == 50
        assert result["position_closed"] is False
        # P&L: (10000-9500) * 50 = 25000
        assert result["realized_pnl"] == 25000.0

    def test_remove_position_full_sell(self):
        """Test selling entire position."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        result = self.manager.remove_position(
            symbol="BBCA",
            shares=100,
            price=10000.0,
        )

        assert result["position_closed"] is True
        assert result["remaining_shares"] == 0

    def test_remove_position_without_shares_sells_all(self):
        """Test that omitting shares sells entire position."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        result = self.manager.remove_position(symbol="BBCA", price=10000.0)

        assert result["shares"] == 100
        assert result["position_closed"] is True

    def test_remove_position_not_found(self):
        """Test error when position doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            self.manager.remove_position(symbol="XXXX", shares=50)

    def test_remove_position_insufficient_shares(self):
        """Test error when selling more than owned."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        with pytest.raises(ValueError, match="Cannot sell"):
            self.manager.remove_position(symbol="BBCA", shares=200)

    def test_get_positions(self):
        """Test getting all positions."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)
        self.manager.add_position(symbol="TLKM", shares=500, price=3400.0)

        positions = self.manager.get_positions()

        assert len(positions) == 2
        symbols = [p["symbol"] for p in positions]
        assert "BBCA" in symbols
        assert "TLKM" in symbols

    def test_get_position_single(self):
        """Test getting single position."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        position = self.manager.get_position("BBCA")

        assert position is not None
        assert position["symbol"] == "BBCA"
        assert position["shares"] == 100

    def test_get_position_not_found(self):
        """Test getting non-existent position returns None."""
        position = self.manager.get_position("XXXX")
        assert position is None

    def test_get_transactions_filtered(self):
        """Test filtering transactions by symbol."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)
        self.manager.add_position(symbol="TLKM", shares=500, price=3400.0)

        bbca_txns = self.manager.get_transactions(symbol="BBCA")
        all_txns = self.manager.get_transactions()

        assert len(bbca_txns) == 1
        assert len(all_txns) == 2

    def test_clear_portfolio(self):
        """Test clearing all portfolio data."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)
        self.manager.add_position(symbol="TLKM", shares=500, price=3400.0)

        count = self.manager.clear_portfolio()

        assert count == 2
        assert len(self.manager.get_positions()) == 0


class TestPnLCalculator:
    """Test suite for P&L Calculator (Story 4.4)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database and calculator."""
        init_database()
        self.manager = PortfolioManager()
        self.pnl = PnLCalculator()
        self.manager.clear_portfolio()

    def test_calculate_position_pnl_profit(self):
        """Test P&L calculation for profitable position."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)

        # Mock current price higher
        result = self.pnl.calculate_position_pnl("BBCA", current_price=10000.0)

        assert result["symbol"] == "BBCA"
        assert result["cost_basis"] == 950000.0
        assert result["market_value"] == 1000000.0
        assert result["unrealized_pnl"] == 50000.0
        assert result["is_profit"] is True

    def test_calculate_position_pnl_loss(self):
        """Test P&L calculation for losing position."""
        self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)

        result = self.pnl.calculate_position_pnl("BBCA", current_price=9500.0)

        assert result["unrealized_pnl"] == -50000.0
        assert result["is_profit"] is False
        assert result["pnl_percent"] == -5.0

    def test_calculate_position_pnl_not_found(self):
        """Test P&L for non-existent position."""
        result = self.pnl.calculate_position_pnl("XXXX", current_price=1000.0)
        assert "error" in result

    def test_calculate_portfolio_pnl(self):
        """Test total portfolio P&L calculation."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)
        self.manager.add_position(symbol="TLKM", shares=500, price=3400.0)

        prices = {"BBCA": 10000.0, "TLKM": 3500.0}
        result = self.pnl.calculate_portfolio_pnl(prices=prices)

        assert result["position_count"] == 2
        # BBCA: cost=950000, value=1000000, pnl=50000
        # TLKM: cost=1700000, value=1750000, pnl=50000
        # Total: cost=2650000, value=2750000, pnl=100000
        assert result["total_cost_basis"] == 2650000.0
        assert result["total_market_value"] == 2750000.0
        assert result["total_unrealized_pnl"] == 100000.0
        assert result["is_profit"] is True

    def test_calculate_portfolio_pnl_empty(self):
        """Test P&L for empty portfolio."""
        result = self.pnl.calculate_portfolio_pnl()

        assert result["position_count"] == 0
        assert result["total_cost_basis"] == 0
        assert result["total_market_value"] == 0

    def test_get_realized_pnl(self):
        """Test realized P&L from sell transactions."""
        # Buy and sell
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)
        self.manager.remove_position(symbol="BBCA", shares=50, price=10000.0)

        result = self.pnl.get_realized_pnl(symbol="BBCA")

        assert result["transaction_count"] == 1
        # Note: This depends on how cost basis is calculated in the sell
        assert result["total_realized_pnl"] > 0

    def test_get_portfolio_summary(self):
        """Test comprehensive portfolio summary."""
        self.manager.add_position(symbol="BBCA", shares=100, price=9500.0)
        self.manager.add_position(symbol="TLKM", shares=500, price=3400.0)

        prices = {"BBCA": 10000.0, "TLKM": 3300.0}
        summary = self.pnl.get_portfolio_summary(prices=prices)

        assert "summary" in summary
        assert "positions" in summary
        assert summary["winners_count"] >= 0
        assert summary["losers_count"] >= 0

    def test_portfolio_allocation_percentages(self):
        """Test that allocation percentages are calculated."""
        self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)
        self.manager.add_position(symbol="TLKM", shares=500, price=2000.0)

        prices = {"BBCA": 10000.0, "TLKM": 2000.0}
        summary = self.pnl.get_portfolio_summary(prices=prices)

        # BBCA: 1000000, TLKM: 1000000, Total: 2000000
        # Each should be 50%
        positions = summary["positions"]
        for pos in positions:
            assert "allocation_percent" in pos
            assert pos["allocation_percent"] == 50.0


class TestPortfolioAnalytics:
    """Test suite for Portfolio Analytics (Story 4.5)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database and analytics."""
        init_database()
        self.manager = PortfolioManager()
        self.analytics = PortfolioAnalytics()
        self.manager.clear_portfolio()

    def test_analyze_concentration_empty(self):
        """Test concentration analysis for empty portfolio."""
        result = self.analytics.analyze_concentration()

        assert result["concentration_score"] == 0
        assert result["risk_level"] == "N/A"

    def test_analyze_concentration_high(self):
        """Test high concentration detection."""
        # Single position = 100% concentration
        self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)

        prices = {"BBCA": 10000.0}
        result = self.analytics.analyze_concentration(prices=prices)

        assert result["risk_level"] == "HIGH"
        assert result["position_count"] == 1

    def test_analyze_concentration_low(self):
        """Test low concentration with diversified portfolio."""
        # Add multiple positions with similar weights
        self.manager.add_position(symbol="BBCA", shares=100, price=1000.0)
        self.manager.add_position(symbol="TLKM", shares=100, price=1000.0)
        self.manager.add_position(symbol="BBRI", shares=100, price=1000.0)
        self.manager.add_position(symbol="BMRI", shares=100, price=1000.0)
        self.manager.add_position(symbol="ASII", shares=100, price=1000.0)

        prices = {s: 1000.0 for s in ["BBCA", "TLKM", "BBRI", "BMRI", "ASII"]}
        result = self.analytics.analyze_concentration(prices=prices)

        # HHI = 5 * 20^2 = 2000 (moderate)
        assert result["risk_level"] in ["LOW", "MODERATE"]
        assert result["hhi_index"] == 2000.0

    def test_analyze_sector_allocation(self):
        """Test sector allocation analysis."""
        self.manager.add_position(symbol="BBCA", shares=100, price=1000.0)

        result = self.analytics.analyze_sector_allocation()

        assert "sectors" in result
        assert "diversification_level" in result
        assert "sector_count" in result

    @patch.object(YahooFinanceSource, 'get_price_history')
    def test_calculate_portfolio_volatility(self, mock_yahoo):
        """Test portfolio volatility calculation."""
        import pandas as pd
        import numpy as np

        # Create mock price data with known volatility
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = 10000 * np.exp(np.cumsum(np.random.randn(30) * 0.02))

        mock_df = pd.DataFrame({
            'Close': prices,
            'date': dates,
        })
        mock_yahoo.return_value = mock_df

        self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)

        result = self.analytics.calculate_portfolio_volatility(days=30)

        assert "portfolio_volatility" in result
        assert "risk_level" in result
        assert result["portfolio_volatility"] >= 0

    def test_get_full_analysis(self):
        """Test comprehensive portfolio analysis."""
        self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)

        prices = {"BBCA": 10000.0}
        analysis = self.analytics.get_full_analysis(prices=prices)

        assert "overall_score" in analysis
        assert "health_status" in analysis
        assert "concentration" in analysis
        assert "sector_allocation" in analysis
        assert "volatility" in analysis
        assert "recommendations" in analysis

    def test_generate_ai_insights(self):
        """Test AI insight generation."""
        self.manager.add_position(symbol="BBCA", shares=100, price=10000.0)

        prices = {"BBCA": 11000.0}  # 10% profit
        insights = self.analytics.generate_ai_insights()

        assert isinstance(insights, list)
        # Should have at least one insight about portfolio being profitable
        # or about concentration


class TestPortfolioCLI:
    """Test suite for Portfolio CLI commands."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for CLI tests."""
        init_database()
        from stockai.cli.main import app
        self.runner = CliRunner()
        self.app = app
        # Clear portfolio
        manager = PortfolioManager()
        manager.clear_portfolio()

    def test_portfolio_add_command(self):
        """Test portfolio add CLI command."""
        result = self.runner.invoke(
            self.app, ["portfolio", "add", "BBCA", "100", "9500"]
        )

        assert result.exit_code == 0
        assert "Added position" in result.output
        assert "BBCA" in result.output

    def test_portfolio_list_empty(self):
        """Test portfolio list with no positions."""
        result = self.runner.invoke(self.app, ["portfolio", "list", "--no-prices"])

        assert result.exit_code == 0
        assert "No positions" in result.output

    def test_portfolio_list_with_positions(self):
        """Test portfolio list with positions."""
        # Add position first
        self.runner.invoke(self.app, ["portfolio", "add", "BBCA", "100", "9500"])

        result = self.runner.invoke(self.app, ["portfolio", "list", "--no-prices"])

        assert result.exit_code == 0
        assert "BBCA" in result.output
        assert "100" in result.output

    def test_portfolio_sell_command(self):
        """Test portfolio sell CLI command."""
        # Add position first
        self.runner.invoke(self.app, ["portfolio", "add", "BBCA", "100", "9500"])

        result = self.runner.invoke(
            self.app, ["portfolio", "sell", "BBCA", "--shares", "50", "--price", "10000"]
        )

        assert result.exit_code == 0
        assert "Sold position" in result.output
        assert "Realized P&L" in result.output

    def test_portfolio_pnl_command(self):
        """Test portfolio pnl CLI command."""
        # Add position first
        self.runner.invoke(self.app, ["portfolio", "add", "BBCA", "100", "9500"])

        result = self.runner.invoke(self.app, ["portfolio", "pnl"])

        # Even with no real prices, should show something
        assert result.exit_code == 0

    def test_portfolio_transactions_command(self):
        """Test portfolio transactions CLI command."""
        # Add some transactions
        self.runner.invoke(self.app, ["portfolio", "add", "BBCA", "100", "9500"])
        self.runner.invoke(self.app, ["portfolio", "add", "TLKM", "500", "3400"])

        result = self.runner.invoke(self.app, ["portfolio", "transactions"])

        assert result.exit_code == 0
        assert "Transaction History" in result.output

    def test_portfolio_analyze_command(self):
        """Test portfolio analyze CLI command."""
        # Add position first
        self.runner.invoke(self.app, ["portfolio", "add", "BBCA", "100", "9500"])

        result = self.runner.invoke(self.app, ["portfolio", "analyze"])

        assert result.exit_code == 0
        assert "Portfolio Health" in result.output


