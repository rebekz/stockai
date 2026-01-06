"""IDX Index Data - Indonesian Stock Exchange Index Components.

Provides lists of stocks in various IDX indices (IDX30, LQ45, etc.)
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from stockai.data.sources.yahoo import YahooFinanceSource

logger = logging.getLogger(__name__)


# IDX30 components as of 2024 (updated periodically by IDX)
# These are the 30 most liquid stocks on IDX
IDX30_SYMBOLS = [
    "ACES", "ADRO", "AMRT", "ANTM", "ASII",
    "BBCA", "BBNI", "BBRI", "BBTN", "BMRI",
    "BRPT", "BUKA", "CPIN", "EMTK", "EXCL",
    "GGRM", "GOTO", "HMSP", "ICBP", "INCO",
    "INDF", "INKP", "KLBF", "MAPI", "MDKA",
    "MEDC", "PGAS", "SMGR", "TLKM", "UNVR",
]

# LQ45 components (45 most liquid stocks)
LQ45_SYMBOLS = [
    "ACES", "ADRO", "AKRA", "AMMN", "AMRT",
    "ANTM", "ARTO", "ASII", "BBCA", "BBNI",
    "BBRI", "BBTN", "BMRI", "BRPT", "BUKA",
    "CPIN", "EMTK", "ERAA", "ESSA", "EXCL",
    "GGRM", "GOTO", "HMSP", "HRUM", "ICBP",
    "INCO", "INDF", "INKP", "ITMG", "JPFA",
    "KLBF", "MAPI", "MBMA", "MDKA", "MEDC",
    "MIKA", "PGAS", "PGEO", "PTBA", "SIDO",
    "SMGR", "TBIG", "TKIM", "TLKM", "UNVR",
]

# JII70 components (Jakarta Islamic Index 70 - 70 most liquid sharia-compliant stocks)
# Note: Excludes conventional banks and non-halal businesses
JII70_SYMBOLS = [
    # Top tier by market cap
    "AMMN", "TLKM", "BYAN", "TPIA", "ASII",
    "GOTO", "DSSA", "UNTR", "INDF", "PANI",
    # High market cap sharia stocks
    "BRPT", "BRMS", "BUMI", "UNVR", "BRIS",
    "ICBP", "ANTM", "ISAT", "CPIN", "ADRO",
    # Mid-large cap sharia stocks
    "INCO", "MDKA", "KLBF", "PTBA", "SMGR",
    "INTP", "MYOR", "JPFA", "EXCL", "JSMR",
    # Infrastructure & Property sharia stocks
    "TBIG", "TOWR", "WIKA", "WSKT", "CTRA",
    "SMRA", "BSDE", "PWON",
    # Consumer & Healthcare sharia stocks
    "SIDO", "MIKA", "ERAA", "MAPI", "ACES",
    "AMRT",
    # Energy & Mining sharia stocks
    "MEDC", "HRUM", "ITMG", "AKRA", "ESSA",
    "PGAS", "TINS",
    # Technology & Media sharia stocks
    "EMTK", "BUKA", "MNCN", "SCMA",
    # Paper & Materials sharia stocks
    "INKP", "TKIM",
    # Additional liquid sharia stocks
    "BTPS", "SRTG", "MLPT", "KPIG", "BMTR",
    "LSIP", "AALI", "SSMS", "SILO", "AUTO",
    "SMDR",  # Note: INKA removed (delisted from Yahoo Finance)
]

# Major sectors in IDX
IDX_SECTORS = {
    "A": "Agriculture",
    "B": "Mining",
    "C": "Basic Industry and Chemicals",
    "D": "Miscellaneous Industry",
    "E": "Consumer Goods Industry",
    "F": "Property, Real Estate and Building Construction",
    "G": "Infrastructure, Utilities and Transportation",
    "H": "Finance",
    "I": "Trade, Services and Investment",
    "J": "Manufacturing",
}


class IDXIndexSource:
    """Data source for IDX index components and stock listings."""

    def __init__(self):
        """Initialize IDX Index source."""
        self.yahoo = YahooFinanceSource()
        self._cache: dict[str, Any] = {}
        self._cache_expiry: dict[str, datetime] = {}

    def get_idx30_symbols(self) -> list[str]:
        """Get list of IDX30 component symbols.

        Returns:
            List of stock symbols in IDX30
        """
        return IDX30_SYMBOLS.copy()

    def get_lq45_symbols(self) -> list[str]:
        """Get list of LQ45 component symbols.

        Returns:
            List of stock symbols in LQ45
        """
        return LQ45_SYMBOLS.copy()

    def get_jii70_symbols(self) -> list[str]:
        """Get list of JII70 component symbols (Jakarta Islamic Index 70).

        Returns:
            List of sharia-compliant stock symbols in JII70
        """
        return JII70_SYMBOLS.copy()

    def get_index_symbols(self, index_name: str) -> list[str]:
        """Get component symbols for a given index.

        Args:
            index_name: Index name (IDX30, LQ45, JII70)

        Returns:
            List of stock symbols
        """
        index_name = index_name.upper()
        if index_name == "IDX30":
            return self.get_idx30_symbols()
        elif index_name == "LQ45":
            return self.get_lq45_symbols()
        elif index_name == "JII70":
            return self.get_jii70_symbols()
        else:
            logger.warning(f"Unknown index: {index_name}")
            return []

    def get_idx30_stocks(self, include_prices: bool = False) -> list[dict[str, Any]]:
        """Get IDX30 stocks with basic info.

        Args:
            include_prices: Whether to fetch current prices

        Returns:
            List of stock dictionaries
        """
        return self._get_index_stocks(IDX30_SYMBOLS, "IDX30", include_prices)

    def get_lq45_stocks(self, include_prices: bool = False) -> list[dict[str, Any]]:
        """Get LQ45 stocks with basic info.

        Args:
            include_prices: Whether to fetch current prices

        Returns:
            List of stock dictionaries
        """
        return self._get_index_stocks(LQ45_SYMBOLS, "LQ45", include_prices)

    def get_jii70_stocks(self, include_prices: bool = False) -> list[dict[str, Any]]:
        """Get JII70 stocks with basic info (Jakarta Islamic Index 70).

        Args:
            include_prices: Whether to fetch current prices

        Returns:
            List of sharia-compliant stock dictionaries
        """
        return self._get_index_stocks(JII70_SYMBOLS, "JII70", include_prices)

    def _get_index_stocks(
        self, symbols: list[str], index_name: str, include_prices: bool
    ) -> list[dict[str, Any]]:
        """Get stocks for an index with optional prices.

        Args:
            symbols: List of stock symbols
            index_name: Name of the index
            include_prices: Whether to fetch prices

        Returns:
            List of stock dictionaries
        """
        stocks = []
        for symbol in symbols:
            stock_data = {
                "symbol": symbol,
                "index": index_name,
            }

            if include_prices:
                try:
                    price_info = self.yahoo.get_current_price(symbol)
                    if price_info:
                        stock_data.update({
                            "price": price_info.get("price"),
                            "change": price_info.get("change"),
                            "change_percent": price_info.get("change_percent"),
                            "volume": price_info.get("volume"),
                        })
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")

            stocks.append(stock_data)

        return stocks

    def get_stock_details(self, symbol: str) -> dict[str, Any] | None:
        """Get detailed stock information with index membership.

        Args:
            symbol: Stock symbol

        Returns:
            Stock details dictionary
        """
        symbol = symbol.upper()
        info = self.yahoo.get_stock_info(symbol)

        if info:
            info["is_idx30"] = symbol in IDX30_SYMBOLS
            info["is_lq45"] = symbol in LQ45_SYMBOLS
            info["is_jii70"] = symbol in JII70_SYMBOLS

        return info

    def get_index_performance(
        self, index_name: str, period: str = "1mo"
    ) -> pd.DataFrame:
        """Get price history for all stocks in an index.

        Args:
            index_name: Index name (IDX30, LQ45)
            period: Time period

        Returns:
            DataFrame with combined price data
        """
        symbols = self.get_index_symbols(index_name)
        all_data = []

        for symbol in symbols:
            try:
                df = self.yahoo.get_price_history(symbol, period=period)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to get history for {symbol}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def get_top_gainers(
        self, index_name: str = "IDX30", limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get top gaining stocks in an index.

        Args:
            index_name: Index name
            limit: Number of stocks to return

        Returns:
            List of top gainers with price data
        """
        stocks = self._get_index_stocks(
            self.get_index_symbols(index_name), index_name, include_prices=True
        )

        # Filter stocks with valid change data and sort
        valid_stocks = [s for s in stocks if s.get("change_percent") is not None]
        valid_stocks.sort(key=lambda x: x.get("change_percent", 0), reverse=True)

        return valid_stocks[:limit]

    def get_top_losers(
        self, index_name: str = "IDX30", limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get top losing stocks in an index.

        Args:
            index_name: Index name
            limit: Number of stocks to return

        Returns:
            List of top losers with price data
        """
        stocks = self._get_index_stocks(
            self.get_index_symbols(index_name), index_name, include_prices=True
        )

        # Filter stocks with valid change data and sort
        valid_stocks = [s for s in stocks if s.get("change_percent") is not None]
        valid_stocks.sort(key=lambda x: x.get("change_percent", 0))

        return valid_stocks[:limit]

    def get_most_active(
        self, index_name: str = "IDX30", limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get most actively traded stocks by volume.

        Args:
            index_name: Index name
            limit: Number of stocks to return

        Returns:
            List of most active stocks with volume data
        """
        stocks = self._get_index_stocks(
            self.get_index_symbols(index_name), index_name, include_prices=True
        )

        # Filter stocks with valid volume and sort
        valid_stocks = [s for s in stocks if s.get("volume") is not None]
        valid_stocks.sort(key=lambda x: x.get("volume", 0), reverse=True)

        return valid_stocks[:limit]

    def is_valid_idx_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a valid IDX stock.

        Args:
            symbol: Stock symbol to check

        Returns:
            True if valid IDX stock
        """
        return self.yahoo.validate_symbol(symbol)

    def get_all_sectors(self) -> dict[str, str]:
        """Get all IDX sector codes and names.

        Returns:
            Dictionary of sector codes to names
        """
        return IDX_SECTORS.copy()


# Convenience functions
def get_idx_source() -> IDXIndexSource:
    """Get an IDX Index data source instance."""
    return IDXIndexSource()


def get_idx30() -> list[str]:
    """Quick function to get IDX30 symbols."""
    return IDX30_SYMBOLS.copy()


def get_lq45() -> list[str]:
    """Quick function to get LQ45 symbols."""
    return LQ45_SYMBOLS.copy()


def get_jii70() -> list[str]:
    """Quick function to get JII70 symbols (Jakarta Islamic Index 70)."""
    return JII70_SYMBOLS.copy()
