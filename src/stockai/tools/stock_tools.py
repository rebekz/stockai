"""Stock Analysis Tools for StockAI Agent.

Provides tools for fetching and analyzing Indonesian stock data.
These tools wrap the data sources and add agent-friendly interfaces.
"""

import logging
from datetime import datetime
from typing import Any

from stockai.data.sources.yahoo import YahooFinanceSource
from stockai.data.sources.idx import IDXIndexSource, get_idx30, get_lq45
from stockai.tools.registry import stockai_tool, get_registry

logger = logging.getLogger(__name__)


# Initialize data sources
_yahoo = YahooFinanceSource()
_idx = IDXIndexSource()


@stockai_tool(name="get_stock_info", category="data")
def get_stock_info(symbol: str) -> dict[str, Any]:
    """Get basic information about a stock.

    Fetches company name, sector, market cap, and other metadata.

    Args:
        symbol: Stock ticker symbol (e.g., BBCA, TLKM)

    Returns:
        Dictionary with stock information
    """
    logger.info(f"Fetching stock info for {symbol}")

    info = _yahoo.get_stock_info(symbol)
    if not info:
        return {"error": f"Stock {symbol} not found"}

    # Add index membership info
    info["is_idx30"] = symbol.upper() in get_idx30()
    info["is_lq45"] = symbol.upper() in get_lq45()

    return info


@stockai_tool(name="get_current_price", category="data")
def get_current_price(symbol: str) -> dict[str, Any]:
    """Get the current/latest price of a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with price, change, change_percent, volume
    """
    logger.info(f"Fetching current price for {symbol}")

    price_info = _yahoo.get_current_price(symbol)
    if not price_info:
        return {"error": f"Price data for {symbol} not available"}

    return price_info


@stockai_tool(name="get_price_history", category="data")
def get_price_history(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """Get historical price data for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)

    Returns:
        Dictionary with price history data
    """
    logger.info(f"Fetching {period} price history for {symbol}")

    df = _yahoo.get_price_history(symbol, period=period)

    if df.empty:
        return {"error": f"No price history for {symbol}"}

    # Convert to list of dictionaries
    records = df.to_dict("records")

    # Calculate summary statistics
    if len(records) > 1:
        first_close = records[0]["close"]
        last_close = records[-1]["close"]
        change = last_close - first_close
        change_pct = (change / first_close) * 100

        summary = {
            "period": period,
            "start_date": str(records[0]["date"]),
            "end_date": str(records[-1]["date"]),
            "start_price": first_close,
            "end_price": last_close,
            "change": change,
            "change_percent": change_pct,
            "high": df["high"].max(),
            "low": df["low"].min(),
            "avg_volume": df["volume"].mean(),
            "total_records": len(records),
        }
    else:
        summary = {"total_records": len(records)}

    return {
        "symbol": symbol.upper(),
        "summary": summary,
        "data": records[-10:],  # Last 10 records for context
    }


@stockai_tool(name="get_technical_indicators", category="analysis")
def get_technical_indicators(symbol: str, period: str = "3mo") -> dict[str, Any]:
    """Calculate technical indicators for a stock.

    Computes RSI, MACD, Bollinger Bands, and moving averages.

    Args:
        symbol: Stock ticker symbol
        period: Period for calculation (default 3mo for sufficient data)

    Returns:
        Dictionary with technical indicator values
    """
    logger.info(f"Calculating technical indicators for {symbol}")

    df = _yahoo.get_price_history(symbol, period=period)

    if df.empty or len(df) < 20:
        return {"error": f"Insufficient data for technical analysis of {symbol}"}

    try:
        import ta

        # RSI
        rsi = ta.momentum.RSIIndicator(df["close"], window=14)
        current_rsi = rsi.rsi().iloc[-1]

        # MACD
        macd = ta.trend.MACD(df["close"])
        current_macd = macd.macd().iloc[-1]
        current_signal = macd.macd_signal().iloc[-1]
        current_histogram = macd.macd_diff().iloc[-1]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]

        # Moving Averages
        sma_20 = df["close"].rolling(window=20).mean().iloc[-1]
        sma_50 = df["close"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        ema_12 = df["close"].ewm(span=12).mean().iloc[-1]
        ema_26 = df["close"].ewm(span=26).mean().iloc[-1]

        current_price = df["close"].iloc[-1]

        # Generate signals
        signals = []
        if current_rsi < 30:
            signals.append("🟢 RSI oversold (potential buy)")
        elif current_rsi > 70:
            signals.append("🔴 RSI overbought (potential sell)")

        if current_macd > current_signal:
            signals.append("🟢 MACD bullish crossover")
        else:
            signals.append("🔴 MACD bearish crossover")

        if current_price > bb_upper:
            signals.append("🔴 Price above upper Bollinger Band")
        elif current_price < bb_lower:
            signals.append("🟢 Price below lower Bollinger Band")

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "indicators": {
                "rsi": {
                    "value": round(current_rsi, 2),
                    "interpretation": "oversold" if current_rsi < 30 else "overbought" if current_rsi > 70 else "neutral",
                },
                "macd": {
                    "macd": round(current_macd, 4),
                    "signal": round(current_signal, 4),
                    "histogram": round(current_histogram, 4),
                    "interpretation": "bullish" if current_macd > current_signal else "bearish",
                },
                "bollinger_bands": {
                    "upper": round(bb_upper, 2),
                    "middle": round(bb_middle, 2),
                    "lower": round(bb_lower, 2),
                },
                "moving_averages": {
                    "sma_20": round(sma_20, 2),
                    "sma_50": round(sma_50, 2) if sma_50 else None,
                    "ema_12": round(ema_12, 2),
                    "ema_26": round(ema_26, 2),
                },
            },
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except ImportError:
        return {"error": "Technical analysis library not available"}
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        return {"error": str(e)}


@stockai_tool(name="get_idx30_stocks", category="index")
def get_idx30_stocks() -> dict[str, Any]:
    """Get list of IDX30 index components.

    Returns:
        Dictionary with IDX30 stock symbols
    """
    symbols = get_idx30()
    return {
        "index": "IDX30",
        "count": len(symbols),
        "symbols": symbols,
        "description": "30 most liquid stocks on IDX",
    }


@stockai_tool(name="get_lq45_stocks", category="index")
def get_lq45_stocks() -> dict[str, Any]:
    """Get list of LQ45 index components.

    Returns:
        Dictionary with LQ45 stock symbols
    """
    symbols = get_lq45()
    return {
        "index": "LQ45",
        "count": len(symbols),
        "symbols": symbols,
        "description": "45 most liquid stocks on IDX",
    }


@stockai_tool(name="compare_stocks", category="analysis")
def compare_stocks(symbols: list[str] | str) -> dict[str, Any]:
    """Compare multiple stocks side by side.

    Args:
        symbols: List of stock symbols or comma-separated string

    Returns:
        Dictionary with comparison data
    """
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbols = [s.upper() for s in symbols]

    logger.info(f"Comparing stocks: {symbols}")

    comparison = []
    for symbol in symbols[:5]:  # Limit to 5 stocks
        info = _yahoo.get_stock_info(symbol)
        price = _yahoo.get_current_price(symbol)

        if info and price:
            comparison.append({
                "symbol": symbol,
                "name": info.get("name", "N/A"),
                "sector": info.get("sector", "N/A"),
                "price": price.get("price"),
                "change_percent": price.get("change_percent"),
                "market_cap": info.get("market_cap"),
                "pe_ratio": info.get("pe_ratio"),
                "pb_ratio": info.get("pb_ratio"),
                "dividend_yield": info.get("dividend_yield"),
            })

    return {
        "stocks": comparison,
        "count": len(comparison),
        "timestamp": datetime.utcnow().isoformat(),
    }


@stockai_tool(name="get_sector_info", category="index")
def get_sector_info() -> dict[str, Any]:
    """Get IDX sector classifications.

    Returns:
        Dictionary with sector codes and names
    """
    sectors = _idx.get_all_sectors()
    return {
        "sectors": sectors,
        "count": len(sectors),
    }


def register_stock_tools() -> None:
    """Ensure all stock tools are registered.

    Call this function to force registration of all tools.
    Tools are normally auto-registered via the decorator.
    """
    # Tools are registered via decorator on import
    # This function exists for explicit registration if needed
    registry = get_registry()
    tool_count = len(registry.list_tools())
    logger.info(f"Stock tools registered: {tool_count} tools available")
