"""E2E Tests for Yahoo Finance Data Connector (Story 1.3).

Note: These tests make real API calls to Yahoo Finance.
They may be slow and subject to rate limiting.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from stockai.data.sources.yahoo import YahooFinanceSource, fetch_stock_data


class TestYahooFinanceSource:
    """Test Yahoo Finance data source."""

    @pytest.fixture
    def source(self):
        """Create a Yahoo Finance source instance."""
        return YahooFinanceSource()

    def test_ticker_symbol_conversion(self, source):
        """Test conversion of symbol to Yahoo Finance format."""
        assert source._get_ticker_symbol("BBCA") == "BBCA.JK"
        assert source._get_ticker_symbol("bbca") == "BBCA.JK"
        assert source._get_ticker_symbol("BBCA.JK") == "BBCA.JK"
        assert source._get_ticker_symbol(" tlkm ") == "TLKM.JK"

    def test_clean_symbol(self, source):
        """Test cleaning Yahoo Finance symbol."""
        assert source._clean_symbol("BBCA.JK") == "BBCA"
        assert source._clean_symbol("bbca.jk") == "BBCA"
        assert source._clean_symbol("TLKM") == "TLKM"


class TestStockInfo:
    """Test stock info retrieval."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_get_stock_info_bbca(self, source):
        """AC1.3.1: Fetch BBCA stock info from Yahoo Finance."""
        info = source.get_stock_info("BBCA")

        assert info is not None
        assert info["symbol"] == "BBCA"
        assert info["name"] is not None
        assert len(info["name"]) > 0
        assert info["current_price"] is not None
        assert info["current_price"] > 0

    @pytest.mark.network
    def test_get_stock_info_tlkm(self, source):
        """Test fetching TLKM stock info."""
        info = source.get_stock_info("TLKM")

        assert info is not None
        assert info["symbol"] == "TLKM"
        assert info["current_price"] is not None

    @pytest.mark.network
    def test_get_stock_info_invalid(self, source):
        """Test handling invalid symbol."""
        info = source.get_stock_info("INVALIDXYZ123")
        assert info is None

    @pytest.mark.network
    def test_stock_info_has_required_fields(self, source):
        """Test that stock info contains required fields."""
        info = source.get_stock_info("BBRI")

        assert info is not None
        required_fields = [
            "symbol",
            "name",
            "current_price",
            "previous_close",
            "volume",
        ]
        for field in required_fields:
            assert field in info, f"Missing field: {field}"


class TestPriceHistory:
    """Test price history retrieval."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_get_price_history_1mo(self, source):
        """AC1.3.2: Fetch 1 month price history."""
        df = source.get_price_history("BBCA", period="1mo")

        assert not df.empty
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Should have ~20 trading days in a month
        assert len(df) >= 10

    @pytest.mark.network
    def test_get_price_history_3mo(self, source):
        """Test 3 month price history."""
        df = source.get_price_history("TLKM", period="3mo")

        assert not df.empty
        assert len(df) >= 40  # ~60 trading days in 3 months

    @pytest.mark.network
    def test_get_price_history_with_dates(self, source):
        """Test price history with specific date range."""
        end = datetime.now()
        start = end - timedelta(days=30)

        df = source.get_price_history("ASII", start=start, end=end)

        assert not df.empty
        assert df["date"].min() >= start - timedelta(days=5)  # Allow some tolerance

    @pytest.mark.network
    def test_price_history_data_types(self, source):
        """Test that price data has correct types."""
        df = source.get_price_history("BBCA", period="5d")

        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert pd.api.types.is_numeric_dtype(df["open"])
            assert pd.api.types.is_numeric_dtype(df["close"])
            assert pd.api.types.is_numeric_dtype(df["volume"])


class TestCurrentPrice:
    """Test current price retrieval."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_get_current_price(self, source):
        """Test getting current price."""
        price = source.get_current_price("BBCA")

        assert price is not None
        assert price["symbol"] == "BBCA"
        assert price["price"] is not None
        assert price["price"] > 0

    @pytest.mark.network
    def test_get_multiple_prices(self, source):
        """Test getting prices for multiple stocks."""
        symbols = ["BBCA", "TLKM", "BBRI"]
        prices = source.get_multiple_prices(symbols)

        assert len(prices) >= 1  # At least some should succeed
        for symbol, data in prices.items():
            assert data["price"] is not None


class TestDividends:
    """Test dividend data retrieval."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_get_dividends(self, source):
        """Test getting dividend history."""
        df = source.get_dividends("BBCA")

        # BBCA typically pays dividends
        if not df.empty:
            assert "symbol" in df.columns
            assert "date" in df.columns
            assert "dividend" in df.columns


class TestFinancials:
    """Test financial statement retrieval."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_get_financials(self, source):
        """Test getting financial statements."""
        financials = source.get_financials("BBCA")

        assert "income_statement" in financials
        assert "balance_sheet" in financials
        assert "cash_flow" in financials


class TestValidation:
    """Test symbol validation."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_validate_valid_symbol(self, source):
        """Test validating a valid symbol."""
        assert source.validate_symbol("BBCA") is True
        assert source.validate_symbol("TLKM") is True

    @pytest.mark.network
    def test_validate_invalid_symbol(self, source):
        """Test validating an invalid symbol."""
        assert source.validate_symbol("INVALIDXYZ") is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.network
    def test_fetch_stock_data(self):
        """Test the quick fetch function."""
        df = fetch_stock_data("BBCA", period="5d")

        assert not df.empty
        assert "close" in df.columns


class TestIDXSpecificBehavior:
    """Test IDX-specific behavior."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    @pytest.mark.network
    def test_idx_suffix_handling(self, source):
        """AC1.3.3: Properly handle .JK suffix for IDX stocks."""
        # Both should return same data
        info1 = source.get_stock_info("BBCA")
        info2 = source.get_stock_info("BBCA.JK")

        if info1 and info2:
            assert info1["symbol"] == info2["symbol"]
            assert info1["name"] == info2["name"]

    @pytest.mark.network
    def test_currency_is_idr(self, source):
        """Test that IDX stocks are in IDR currency."""
        info = source.get_stock_info("BBCA")

        if info:
            assert info.get("currency") == "IDR"
