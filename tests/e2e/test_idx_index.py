"""E2E Tests for IDX Index Data (Story 1.4)."""

import pytest

from stockai.data.sources.idx import (
    IDXIndexSource,
    IDX30_SYMBOLS,
    LQ45_SYMBOLS,
    IDX_SECTORS,
    get_idx30,
    get_lq45,
)


class TestIDXIndexSymbols:
    """Test IDX index symbol lists."""

    def test_idx30_has_30_stocks(self):
        """AC1.4.1: IDX30 contains 30 stocks."""
        assert len(IDX30_SYMBOLS) == 30

    def test_lq45_has_45_stocks(self):
        """AC1.4.2: LQ45 contains 45 stocks."""
        assert len(LQ45_SYMBOLS) == 45

    def test_idx30_symbols_unique(self):
        """Test IDX30 symbols are unique."""
        assert len(IDX30_SYMBOLS) == len(set(IDX30_SYMBOLS))

    def test_lq45_symbols_unique(self):
        """Test LQ45 symbols are unique."""
        assert len(LQ45_SYMBOLS) == len(set(LQ45_SYMBOLS))

    def test_idx30_all_uppercase(self):
        """Test all IDX30 symbols are uppercase."""
        for symbol in IDX30_SYMBOLS:
            assert symbol == symbol.upper()

    def test_major_stocks_in_idx30(self):
        """Test major Indonesian stocks are in IDX30."""
        major_stocks = ["BBCA", "BBRI", "TLKM", "ASII", "BMRI"]
        for symbol in major_stocks:
            assert symbol in IDX30_SYMBOLS, f"{symbol} should be in IDX30"

    def test_idx30_is_subset_of_lq45(self):
        """Test most IDX30 stocks are in LQ45."""
        # Most IDX30 should be in LQ45 (allow some variance)
        overlap = set(IDX30_SYMBOLS) & set(LQ45_SYMBOLS)
        assert len(overlap) >= 25, "IDX30 should significantly overlap with LQ45"


class TestIDXIndexSource:
    """Test IDX Index data source."""

    @pytest.fixture
    def source(self):
        return IDXIndexSource()

    def test_get_idx30_symbols(self, source):
        """Test getting IDX30 symbols."""
        symbols = source.get_idx30_symbols()
        assert len(symbols) == 30
        assert "BBCA" in symbols

    def test_get_lq45_symbols(self, source):
        """Test getting LQ45 symbols."""
        symbols = source.get_lq45_symbols()
        assert len(symbols) == 45
        assert "BBCA" in symbols

    def test_get_index_symbols_idx30(self, source):
        """Test getting symbols by index name - IDX30."""
        symbols = source.get_index_symbols("IDX30")
        assert len(symbols) == 30

    def test_get_index_symbols_lq45(self, source):
        """Test getting symbols by index name - LQ45."""
        symbols = source.get_index_symbols("LQ45")
        assert len(symbols) == 45

    def test_get_index_symbols_case_insensitive(self, source):
        """Test index name is case insensitive."""
        symbols1 = source.get_index_symbols("idx30")
        symbols2 = source.get_index_symbols("IDX30")
        assert symbols1 == symbols2

    def test_get_index_symbols_unknown(self, source):
        """Test unknown index returns empty list."""
        symbols = source.get_index_symbols("UNKNOWN")
        assert symbols == []


class TestIDXIndexStocks:
    """Test IDX index stock retrieval."""

    @pytest.fixture
    def source(self):
        return IDXIndexSource()

    def test_get_idx30_stocks_basic(self, source):
        """Test getting IDX30 stocks without prices."""
        stocks = source.get_idx30_stocks(include_prices=False)
        assert len(stocks) == 30
        for stock in stocks:
            assert "symbol" in stock
            assert "index" in stock
            assert stock["index"] == "IDX30"

    def test_get_lq45_stocks_basic(self, source):
        """Test getting LQ45 stocks without prices."""
        stocks = source.get_lq45_stocks(include_prices=False)
        assert len(stocks) == 45


class TestIDXStockDetails:
    """Test stock details with index membership."""

    @pytest.fixture
    def source(self):
        return IDXIndexSource()

    @pytest.mark.network
    def test_get_stock_details_bbca(self, source):
        """AC1.4.3: Get stock details with index membership."""
        details = source.get_stock_details("BBCA")

        assert details is not None
        assert details["symbol"] == "BBCA"
        assert details["is_idx30"] is True
        assert details["is_lq45"] is True

    @pytest.mark.network
    def test_get_stock_details_non_index(self, source):
        """Test stock not in major indices."""
        # BRIS might not be in IDX30/LQ45
        details = source.get_stock_details("BRIS")

        if details:
            assert "is_idx30" in details
            assert "is_lq45" in details


class TestIDXIndexPerformance:
    """Test index performance features."""

    @pytest.fixture
    def source(self):
        return IDXIndexSource()

    @pytest.mark.network
    def test_get_top_gainers(self, source):
        """Test getting top gainers."""
        gainers = source.get_top_gainers("IDX30", limit=5)

        # May be empty if market is closed
        if gainers:
            assert len(gainers) <= 5
            for stock in gainers:
                assert "symbol" in stock

    @pytest.mark.network
    def test_get_top_losers(self, source):
        """Test getting top losers."""
        losers = source.get_top_losers("IDX30", limit=5)

        if losers:
            assert len(losers) <= 5

    @pytest.mark.network
    def test_get_most_active(self, source):
        """Test getting most active stocks."""
        active = source.get_most_active("IDX30", limit=5)

        if active:
            assert len(active) <= 5


class TestIDXSectors:
    """Test IDX sector data."""

    @pytest.fixture
    def source(self):
        return IDXIndexSource()

    def test_get_all_sectors(self, source):
        """Test getting all IDX sectors."""
        sectors = source.get_all_sectors()

        assert len(sectors) > 0
        assert "H" in sectors  # Finance
        assert sectors["H"] == "Finance"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_idx30(self):
        """Test get_idx30 function."""
        symbols = get_idx30()
        assert len(symbols) == 30
        assert "BBCA" in symbols

    def test_get_lq45(self):
        """Test get_lq45 function."""
        symbols = get_lq45()
        assert len(symbols) == 45


class TestIDXValidation:
    """Test IDX symbol validation."""

    @pytest.fixture
    def source(self):
        return IDXIndexSource()

    @pytest.mark.network
    def test_is_valid_idx_symbol(self, source):
        """Test validating IDX symbols."""
        assert source.is_valid_idx_symbol("BBCA") is True
        assert source.is_valid_idx_symbol("INVALIDXYZ") is False
