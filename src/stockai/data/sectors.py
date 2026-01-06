"""Indonesian Stock Exchange (IDX) Sector Data.

Provides sector classification and sector index symbols for IDX stocks.
Based on IDX sector classification system.
"""

import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# IDX Sector Classifications with representative sector index/ETF
IDX_SECTORS = {
    "Finance": {
        "description": "Banks, Insurance, Multi-Finance, Securities",
        "index": "^JKFINA",  # IDX Finance Sector Index
        "fallback_stocks": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"],
    },
    "Consumer Cyclicals": {
        "description": "Retail, Media, Automotive, Hotels, Restaurants",
        "index": "^JKCONS",
        "fallback_stocks": ["ASII.JK", "MAPI.JK", "ACES.JK"],
    },
    "Consumer Non-Cyclicals": {
        "description": "Food & Beverage, Tobacco, Household Goods, Healthcare",
        "index": "^JKBIND",
        "fallback_stocks": ["UNVR.JK", "ICBP.JK", "INDF.JK", "KLBF.JK"],
    },
    "Basic Materials": {
        "description": "Mining, Chemicals, Building Materials, Forestry",
        "index": "^JKMING",
        "fallback_stocks": ["ANTM.JK", "INCO.JK", "SMGR.JK", "TPIA.JK"],
    },
    "Energy": {
        "description": "Oil & Gas, Coal Mining",
        "index": "^JKAGRI",
        "fallback_stocks": ["ADRO.JK", "PTBA.JK", "PGAS.JK", "MEDC.JK"],
    },
    "Infrastructures": {
        "description": "Telco, Utilities, Transportation, Construction",
        "index": "^JKINFR",
        "fallback_stocks": ["TLKM.JK", "JSMR.JK", "PGAS.JK", "WIKA.JK"],
    },
    "Industrials": {
        "description": "Heavy Equipment, Industrial Machinery",
        "index": "^JKMNFG",
        "fallback_stocks": ["UNTR.JK", "INTP.JK"],
    },
    "Technology": {
        "description": "IT, Tech Services, Digital",
        "index": "^JKTECH",
        "fallback_stocks": ["BUKA.JK", "GOTO.JK"],
    },
    "Property & Real Estate": {
        "description": "Property, Real Estate",
        "index": "^JKPROP",
        "fallback_stocks": ["BSDE.JK", "CTRA.JK", "SMRA.JK"],
    },
    "Healthcare": {
        "description": "Hospitals, Pharma, Medical Devices",
        "index": "^JKHLTH",
        "fallback_stocks": ["KLBF.JK", "SIDO.JK"],
    },
    "Transportation & Logistics": {
        "description": "Airlines, Shipping, Logistics",
        "index": "^JKTRNS",
        "fallback_stocks": ["GIAA.JK", "SMDR.JK"],
    },
}

# Stock to Sector mapping for major IDX stocks
STOCK_SECTOR_MAP = {
    # Finance
    "BBCA": "Finance",
    "BBRI": "Finance",
    "BMRI": "Finance",
    "BBNI": "Finance",
    "BRIS": "Finance",
    "BTPS": "Finance",
    "MEGA": "Finance",
    "PNBN": "Finance",
    "BNGA": "Finance",
    "NISP": "Finance",
    "ADMF": "Finance",
    "BDMN": "Finance",
    # Consumer Non-Cyclicals
    "UNVR": "Consumer Non-Cyclicals",
    "ICBP": "Consumer Non-Cyclicals",
    "INDF": "Consumer Non-Cyclicals",
    "GGRM": "Consumer Non-Cyclicals",
    "HMSP": "Consumer Non-Cyclicals",
    "KLBF": "Consumer Non-Cyclicals",
    "MYOR": "Consumer Non-Cyclicals",
    "SIDO": "Consumer Non-Cyclicals",
    "CPIN": "Consumer Non-Cyclicals",
    "JPFA": "Consumer Non-Cyclicals",
    # Consumer Cyclicals
    "ASII": "Consumer Cyclicals",
    "MAPI": "Consumer Cyclicals",
    "ACES": "Consumer Cyclicals",
    "LPPF": "Consumer Cyclicals",
    "ERAA": "Consumer Cyclicals",
    "MSIN": "Consumer Cyclicals",
    # Basic Materials
    "ANTM": "Basic Materials",
    "INCO": "Basic Materials",
    "SMGR": "Basic Materials",
    "INTP": "Basic Materials",
    "TPIA": "Basic Materials",
    "BRPT": "Basic Materials",
    "MDKA": "Basic Materials",
    # Energy
    "ADRO": "Energy",
    "PTBA": "Energy",
    "MEDC": "Energy",
    "ITMG": "Energy",
    "PGAS": "Energy",
    "AKRA": "Energy",
    # Infrastructures
    "TLKM": "Infrastructures",
    "JSMR": "Infrastructures",
    "TOWR": "Infrastructures",
    "TBIG": "Infrastructures",
    "WIKA": "Infrastructures",
    "WSKT": "Infrastructures",
    "PTPP": "Infrastructures",
    "EXCL": "Infrastructures",
    "ISAT": "Infrastructures",
    # Industrials
    "UNTR": "Industrials",
    "SRIL": "Industrials",
    # Technology
    "BUKA": "Technology",
    "GOTO": "Technology",
    "EMTK": "Technology",
    # Property & Real Estate
    "BSDE": "Property & Real Estate",
    "CTRA": "Property & Real Estate",
    "SMRA": "Property & Real Estate",
    "PWON": "Property & Real Estate",
    "LPKR": "Property & Real Estate",
    # Transportation
    "GIAA": "Transportation & Logistics",
    "SMDR": "Transportation & Logistics",
}


class SectorDataProvider:
    """Provides sector data for IDX stocks."""

    def __init__(self):
        """Initialize sector data provider."""
        self._cache: dict[str, pd.DataFrame] = {}

    def get_stock_sector(self, symbol: str) -> str | None:
        """Get sector for a stock symbol.

        Args:
            symbol: Stock symbol (e.g., BBCA)

        Returns:
            Sector name or None if unknown
        """
        clean_symbol = symbol.upper().replace(".JK", "")
        return STOCK_SECTOR_MAP.get(clean_symbol)

    def get_sector_info(self, sector: str) -> dict[str, Any] | None:
        """Get sector information.

        Args:
            sector: Sector name

        Returns:
            Sector info dict or None
        """
        return IDX_SECTORS.get(sector)

    def get_sector_history(
        self,
        sector: str,
        period: str = "3mo",
    ) -> pd.DataFrame:
        """Get historical data for a sector index.

        Uses sector index if available, otherwise creates a composite
        from major sector stocks.

        Args:
            sector: Sector name
            period: Time period

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{sector}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        sector_info = IDX_SECTORS.get(sector)
        if not sector_info:
            logger.warning(f"Unknown sector: {sector}")
            return pd.DataFrame()

        # Try sector index first
        index_symbol = sector_info.get("index")
        try:
            ticker = yf.Ticker(index_symbol)
            df = ticker.history(period=period)
            if not df.empty and len(df) > 10:
                df.columns = [c.lower() for c in df.columns]
                self._cache[cache_key] = df
                return df
        except Exception as e:
            logger.debug(f"Sector index {index_symbol} not available: {e}")

        # Fallback: Create composite from major stocks
        fallback_stocks = sector_info.get("fallback_stocks", [])
        if not fallback_stocks:
            return pd.DataFrame()

        try:
            dfs = []
            for stock in fallback_stocks[:3]:  # Use top 3 stocks
                ticker = yf.Ticker(stock)
                df = ticker.history(period=period)
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    dfs.append(df["close"])

            if not dfs:
                return pd.DataFrame()

            # Create equal-weighted composite
            composite = pd.concat(dfs, axis=1).mean(axis=1)
            result = pd.DataFrame({"close": composite})

            # Calculate pseudo OHLV from close
            result["open"] = result["close"].shift(1).fillna(result["close"])
            result["high"] = result["close"] * 1.005  # Approximate
            result["low"] = result["close"] * 0.995
            result["volume"] = 0

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"Error creating sector composite for {sector}: {e}")
            return pd.DataFrame()

    def calculate_sector_relative_strength(
        self,
        stock_df: pd.DataFrame,
        symbol: str,
        period: int = 20,
    ) -> pd.Series:
        """Calculate sector relative strength for a stock.

        Sector Relative Strength = (Stock Return / Sector Return) - 1
        Positive value means stock outperforms sector.

        Args:
            stock_df: Stock OHLCV DataFrame
            symbol: Stock symbol
            period: Rolling period for calculation

        Returns:
            Series with sector relative strength values
        """
        if stock_df.empty:
            return pd.Series(dtype=float)

        # Get stock's sector
        sector = self.get_stock_sector(symbol)
        if not sector:
            logger.debug(f"No sector found for {symbol}, using market-wide")
            return pd.Series(0.0, index=stock_df.index)

        # Get sector data
        sector_df = self.get_sector_history(sector, period="6mo")
        if sector_df.empty:
            logger.debug(f"No sector data for {sector}")
            return pd.Series(0.0, index=stock_df.index)

        try:
            # Prepare stock data - handle both indexed and column-based date formats
            if "date" in stock_df.columns and not isinstance(stock_df.index, pd.DatetimeIndex):
                # Data has 'date' column but integer index (from YahooFinanceSource)
                stock_work = stock_df.set_index("date")
            else:
                stock_work = stock_df

            # Align data by date
            stock_close = stock_work["close"] if "close" in stock_work.columns else stock_work.iloc[:, 0]
            sector_close = sector_df["close"]

            # Normalize indices to timezone-naive for comparison
            if hasattr(stock_close.index, 'tz') and stock_close.index.tz is not None:
                stock_close = stock_close.copy()
                stock_close.index = stock_close.index.tz_localize(None)
            if hasattr(sector_close.index, 'tz') and sector_close.index.tz is not None:
                sector_close = sector_close.copy()
                sector_close.index = sector_close.index.tz_localize(None)

            # Reindex sector to match stock dates
            sector_close = sector_close.reindex(stock_close.index, method="ffill")

            # Calculate rolling returns
            stock_return = stock_close.pct_change(period)
            sector_return = sector_close.pct_change(period)

            # Calculate relative strength
            # RS > 0 means outperforming sector
            rs = (1 + stock_return) / (1 + sector_return.replace(0, 0.0001)) - 1

            # Handle edge cases
            rs = rs.replace([float("inf"), float("-inf")], 0)
            rs = rs.fillna(0)

            # Clip extreme values
            rs = rs.clip(-2, 2)

            return rs

        except Exception as e:
            logger.warning(f"Error calculating sector RS for {symbol}: {e}")
            return pd.Series(0.0, index=stock_df.index)


# Convenience functions
_sector_provider: SectorDataProvider | None = None


def get_sector_provider() -> SectorDataProvider:
    """Get singleton sector data provider."""
    global _sector_provider
    if _sector_provider is None:
        _sector_provider = SectorDataProvider()
    return _sector_provider


def get_stock_sector(symbol: str) -> str | None:
    """Get sector for a stock symbol."""
    return get_sector_provider().get_stock_sector(symbol)


def get_sector_relative_strength(
    stock_df: pd.DataFrame,
    symbol: str,
    period: int = 20,
) -> pd.Series:
    """Calculate sector relative strength for a stock."""
    return get_sector_provider().calculate_sector_relative_strength(
        stock_df, symbol, period
    )
