"""Yahoo Finance Data Source for Indonesian Stocks.

Fetches stock data from Yahoo Finance with .JK suffix for IDX stocks.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceSource:
    """Data source for Indonesian stocks via Yahoo Finance.

    Indonesian stocks use the .JK suffix (Jakarta Stock Exchange).
    Example: BBCA.JK, TLKM.JK, BBRI.JK
    """

    IDX_SUFFIX = ".JK"

    def __init__(self):
        """Initialize Yahoo Finance source."""
        self._cache: dict[str, Any] = {}

    def _get_ticker_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format.

        Args:
            symbol: Stock symbol (e.g., BBCA)

        Returns:
            Yahoo Finance symbol (e.g., BBCA.JK)
        """
        symbol = symbol.upper().strip()
        if not symbol.endswith(self.IDX_SUFFIX):
            return f"{symbol}{self.IDX_SUFFIX}"
        return symbol

    def _clean_symbol(self, symbol: str) -> str:
        """Remove .JK suffix from symbol.

        Args:
            symbol: Yahoo Finance symbol (e.g., BBCA.JK)

        Returns:
            Clean symbol (e.g., BBCA)
        """
        return symbol.upper().replace(self.IDX_SUFFIX, "").strip()

    def get_stock_info(self, symbol: str) -> dict[str, Any] | None:
        """Get detailed stock information.

        Args:
            symbol: Stock symbol (e.g., BBCA)

        Returns:
            Dictionary with stock info or None if not found
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            if not info or info.get("regularMarketPrice") is None:
                logger.warning(f"No data found for {ticker_symbol}")
                return None

            return {
                "symbol": self._clean_symbol(symbol),
                "name": info.get("longName") or info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "IDR"),
                "exchange": info.get("exchange", "JKT"),
                "current_price": info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("trailingEps"),
                "beta": info.get("beta"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary", ""),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker_symbol}: {e}")
            return None

    def get_price_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Get historical price data (OHLCV).

        Args:
            symbol: Stock symbol (e.g., BBCA)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Start date (overrides period if provided)
            end: End date (defaults to today)

        Returns:
            DataFrame with OHLCV data
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)

            if start is not None:
                df = ticker.history(start=start, end=end or datetime.now(), interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No price history found for {ticker_symbol}")
                return pd.DataFrame()

            # Standardize column names
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Ensure datetime format
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

            # Add symbol column
            df["symbol"] = self._clean_symbol(symbol)

            # Select and order columns
            columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
            if "adj_close" in df.columns:
                columns.append("adj_close")
            elif "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adj_close"})
                columns.append("adj_close")

            available_cols = [c for c in columns if c in df.columns]
            return df[available_cols]

        except Exception as e:
            logger.error(f"Error fetching history for {ticker_symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> dict[str, Any] | None:
        """Get current/latest price data.

        Args:
            symbol: Stock symbol (e.g., BBCA)

        Returns:
            Dictionary with current price data or None
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            if not info or info.get("regularMarketPrice") is None:
                return None

            return {
                "symbol": self._clean_symbol(symbol),
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "change_percent": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume"),
                "market_time": datetime.fromtimestamp(info.get("regularMarketTime", 0))
                if info.get("regularMarketTime")
                else None,
            }
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker_symbol}: {e}")
            return None

    def get_multiple_prices(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get current prices for multiple stocks.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to price data
        """
        results = {}
        for symbol in symbols:
            price_data = self.get_current_price(symbol)
            if price_data:
                results[self._clean_symbol(symbol)] = price_data
        return results

    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """Get dividend history.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with dividend history
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            dividends = ticker.dividends

            if dividends.empty:
                return pd.DataFrame()

            df = dividends.reset_index()
            df.columns = ["date", "dividend"]
            df["symbol"] = self._clean_symbol(symbol)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

            return df[["symbol", "date", "dividend"]]
        except Exception as e:
            logger.error(f"Error fetching dividends for {ticker_symbol}: {e}")
            return pd.DataFrame()

    def get_financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Get financial statements.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with income_statement, balance_sheet, cash_flow DataFrames
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)

            return {
                "income_statement": ticker.income_stmt,
                "balance_sheet": ticker.balance_sheet,
                "cash_flow": ticker.cashflow,
            }
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker_symbol}: {e}")
            return {
                "income_statement": pd.DataFrame(),
                "balance_sheet": pd.DataFrame(),
                "cash_flow": pd.DataFrame(),
            }

    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol exists and has data.

        Args:
            symbol: Stock symbol to validate

        Returns:
            True if symbol exists and has data
        """
        info = self.get_stock_info(symbol)
        return info is not None and info.get("current_price") is not None

    def search_stocks(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """Search for stocks by name or symbol.

        Uses local IDX stock database for fast fuzzy search,
        with fallback to Yahoo Finance for unlisted stocks.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching stocks
        """
        from stockai.data.listings import search_stocks as local_search

        # First: Try local database (fast, fuzzy search)
        local_results = local_search(query, limit=limit)
        if local_results:
            return [
                {
                    "symbol": r["symbol"],
                    "name": r["name"],
                    "sector": r.get("sector", ""),
                    "score": r.get("score", 1.0),
                }
                for r in local_results
            ]

        # Fallback: Try Yahoo Finance direct lookup
        try:
            info = self.get_stock_info(query)
            if info:
                return [
                    {
                        "symbol": info["symbol"],
                        "name": info["name"],
                        "sector": info.get("sector", ""),
                        "score": 1.0,
                    }
                ]
        except Exception as e:
            logger.debug(f"Yahoo search fallback failed: {e}")

        return []


# Convenience functions
def get_yahoo_source() -> YahooFinanceSource:
    """Get a Yahoo Finance data source instance."""
    return YahooFinanceSource()


def fetch_stock_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """Quick function to fetch stock price history.

    Args:
        symbol: Stock symbol (e.g., BBCA)
        period: Time period

    Returns:
        DataFrame with OHLCV data
    """
    source = YahooFinanceSource()
    return source.get_price_history(symbol, period=period)
