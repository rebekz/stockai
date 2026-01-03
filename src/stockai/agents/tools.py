"""Agent Tools.

Wraps StockAI functionality as LangChain tools for agent use.
"""

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Data Tools (from stockai.data)
# =============================================================================

@tool
def get_idx30_list() -> list[str]:
    """Get list of IDX30 stock symbols.

    Returns the 30 most liquid stocks on the Indonesia Stock Exchange.
    """
    from stockai.data import get_idx30_list as _get_idx30_list
    return _get_idx30_list()


@tool
def get_lq45_list() -> list[str]:
    """Get list of LQ45 stock symbols.

    Returns the 45 most liquid stocks on the Indonesia Stock Exchange.
    """
    from stockai.data import get_lq45_list as _get_lq45_list
    return _get_lq45_list()


@tool
def search_stocks(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search for stocks by symbol or name.

    Args:
        query: Search query (symbol or company name)
        limit: Maximum number of results

    Returns:
        List of matching stocks with symbol, name, sector, and match score
    """
    from stockai.data import search_stocks as _search_stocks
    return _search_stocks(query, limit=limit)


@tool
def get_stock_info(symbol: str) -> dict[str, Any] | None:
    """Get detailed stock information.

    Args:
        symbol: Stock symbol (e.g., BBCA)

    Returns:
        Dictionary with stock info including name, sector, price, metrics
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    return source.get_stock_info(symbol)


@tool
def get_stock_sector(symbol: str) -> str | None:
    """Get the sector for a stock.

    Args:
        symbol: Stock symbol (e.g., BBCA)

    Returns:
        Sector name or None if not found
    """
    from stockai.data import get_stock_sector as _get_stock_sector
    return _get_stock_sector(symbol)


@tool
def get_sector_relative_strength(symbol: str, period: int = 20) -> float:
    """Calculate sector relative strength for a stock.

    Args:
        symbol: Stock symbol
        period: Lookback period in days

    Returns:
        Relative strength score (positive = outperforming sector)
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    from stockai.data import get_sector_relative_strength as _get_sector_rs

    source = YahooFinanceSource()
    df = source.get_price_history(symbol, period="3mo")

    if df.empty:
        return 0.0

    return _get_sector_rs(df, symbol, period=period)


# =============================================================================
# Price History Tools
# =============================================================================

@tool
def fetch_stock_data(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """Fetch historical stock price data (OHLCV).

    Args:
        symbol: Stock symbol (e.g., BBCA)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)

    Returns:
        Dictionary with price data and basic statistics
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    df = source.get_price_history(symbol, period=period)

    if df.empty:
        return {"error": f"No data found for {symbol}"}

    return {
        "symbol": symbol,
        "period": period,
        "data_points": len(df),
        "latest_close": float(df["close"].iloc[-1]) if "close" in df.columns else None,
        "high": float(df["high"].max()) if "high" in df.columns else None,
        "low": float(df["low"].min()) if "low" in df.columns else None,
        "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else None,
        "price_change": float(df["close"].iloc[-1] - df["close"].iloc[0]) if len(df) > 1 else 0,
        "price_change_pct": float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100) if len(df) > 1 else 0,
    }


@tool
def get_price_history(symbol: str, period: str = "3mo") -> list[dict[str, Any]]:
    """Get detailed price history as a list of records.

    Args:
        symbol: Stock symbol (e.g., BBCA)
        period: Time period

    Returns:
        List of daily price records with date, open, high, low, close, volume
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    df = source.get_price_history(symbol, period=period)

    if df.empty:
        return []

    # Convert to records, limiting to last 60 days for context size
    df_recent = df.tail(60)
    records = df_recent.to_dict(orient="records")

    # Clean up datetime serialization
    for record in records:
        if "date" in record:
            record["date"] = str(record["date"])[:10]

    return records


@tool
def get_multiple_prices(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Get current prices for multiple stocks.

    Args:
        symbols: List of stock symbols

    Returns:
        Dictionary mapping symbols to their current price data
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    return source.get_multiple_prices(symbols)


@tool
def get_current_price(symbol: str) -> dict[str, Any] | None:
    """Get current/latest price data for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Current price, change, change percent, volume
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    return source.get_current_price(symbol)


# =============================================================================
# Financial Data Tools
# =============================================================================

@tool
def get_financials(symbol: str) -> dict[str, Any]:
    """Get financial statements for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with income_statement, balance_sheet, cash_flow DataFrames
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    financials = source.get_financials(symbol)

    # Convert DataFrames to summary dicts for context size
    result = {}
    for key, df in financials.items():
        if df is not None and not df.empty:
            # Get latest year's data
            result[key] = {
                "available": True,
                "columns": list(df.columns)[:5],  # First 5 periods
                "rows": list(df.index)[:10],  # First 10 items
            }
        else:
            result[key] = {"available": False}

    return result


@tool
def get_dividends(symbol: str) -> list[dict[str, Any]]:
    """Get dividend history for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        List of dividend payments with date and amount
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    source = YahooFinanceSource()
    df = source.get_dividends(symbol)

    if df.empty:
        return []

    # Convert to records
    records = df.tail(20).to_dict(orient="records")
    for record in records:
        if "date" in record:
            record["date"] = str(record["date"])[:10]
    return records


# =============================================================================
# Technical Analysis Tools
# =============================================================================

@tool
def generate_features(symbol: str, period: str = "3mo") -> dict[str, Any]:
    """Generate technical features for a stock.

    Calculates ~45 technical indicators including RSI, MACD, Bollinger Bands,
    volume ratios, and more.

    Args:
        symbol: Stock symbol
        period: Time period for historical data

    Returns:
        Dictionary with latest feature values and interpretations
    """
    from stockai.data.sources.yahoo import YahooFinanceSource
    from stockai.core.predictor.features import FeatureEngineer

    source = YahooFinanceSource()
    df = source.get_price_history(symbol, period=period)

    if df.empty:
        return {"error": f"No data found for {symbol}"}

    # Generate features
    engineer = FeatureEngineer(include_market_features=False, normalize=False)
    features_df = engineer.generate_features(df, symbol=symbol)

    if features_df.empty:
        return {"error": "Could not generate features"}

    # Get latest values
    latest = features_df.iloc[-1].to_dict()

    # Key indicators summary
    summary = {
        "symbol": symbol,
        "rsi_14": latest.get("rsi_14"),
        "macd": latest.get("macd"),
        "macd_signal": latest.get("macd_signal"),
        "bb_position": latest.get("bb_position"),  # 0-1 range within bands
        "stoch_k": latest.get("stoch_k"),
        "atr_14": latest.get("atr_14"),
        "adx_14": latest.get("adx_14"),
        "volume_sma20_ratio": latest.get("volume_sma20_ratio"),
        "price_sma20_ratio": latest.get("price_sma20_ratio"),
        "volatility_20d": latest.get("volatility_20d"),
    }

    # Interpretations
    interpretations = []
    if summary.get("rsi_14"):
        rsi = summary["rsi_14"]
        if rsi > 70:
            interpretations.append("RSI overbought (>70)")
        elif rsi < 30:
            interpretations.append("RSI oversold (<30)")
        else:
            interpretations.append("RSI neutral")

    if summary.get("macd") and summary.get("macd_signal"):
        if summary["macd"] > summary["macd_signal"]:
            interpretations.append("MACD bullish crossover")
        else:
            interpretations.append("MACD bearish crossover")

    if summary.get("bb_position"):
        bb = summary["bb_position"]
        if bb > 0.8:
            interpretations.append("Near upper Bollinger Band")
        elif bb < 0.2:
            interpretations.append("Near lower Bollinger Band")

    summary["interpretations"] = interpretations
    return summary


# =============================================================================
# Sentiment Tools
# =============================================================================

@tool
def fetch_stock_news(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent news articles about a stock.

    Args:
        symbol: Stock symbol
        limit: Maximum number of articles

    Returns:
        List of news articles with title, source, date, sentiment
    """
    from stockai.core.sentiment.news import NewsAggregator

    aggregator = NewsAggregator()
    articles = aggregator.fetch_all(symbol, limit=limit)

    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", ""),
            "published": str(a.get("published", ""))[:10],
            "url": a.get("url", ""),
        }
        for a in articles[:limit]
    ]


@tool
def analyze_sentiment(symbol: str) -> dict[str, Any]:
    """Analyze news sentiment for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Sentiment analysis with scores and summary
    """
    from stockai.core.sentiment.analyzer import SentimentAnalyzer
    from stockai.core.sentiment.news import NewsAggregator

    aggregator = NewsAggregator()
    analyzer = SentimentAnalyzer()

    articles = aggregator.fetch_all(symbol, limit=20)

    if not articles:
        return {
            "symbol": symbol,
            "articles_analyzed": 0,
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "message": "No news articles found",
        }

    # Analyze each article
    sentiments = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        result = analyzer.analyze(text)
        sentiments.append(result)

    # Aggregate
    positive = sum(1 for s in sentiments if s.get("label") == "positive")
    negative = sum(1 for s in sentiments if s.get("label") == "negative")
    neutral = sum(1 for s in sentiments if s.get("label") == "neutral")
    total = len(sentiments)

    avg_score = sum(s.get("score", 0) for s in sentiments) / total if total > 0 else 0

    if positive > negative * 1.5:
        overall = "bullish"
    elif negative > positive * 1.5:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "symbol": symbol,
        "articles_analyzed": total,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "average_score": round(avg_score, 3),
        "overall_sentiment": overall,
    }


# =============================================================================
# Tool Registry
# =============================================================================

# All available tools
ALL_TOOLS = [
    # Data tools
    get_idx30_list,
    get_lq45_list,
    search_stocks,
    get_stock_info,
    get_stock_sector,
    get_sector_relative_strength,
    # Price tools
    fetch_stock_data,
    get_price_history,
    get_multiple_prices,
    get_current_price,
    # Financial tools
    get_financials,
    get_dividends,
    # Technical tools
    generate_features,
    # Sentiment tools
    fetch_stock_news,
    analyze_sentiment,
]


def get_agent_tools(agent_type: str | None = None) -> list:
    """Get tools for a specific agent type.

    Args:
        agent_type: Type of agent (market_scanner, research, technical, etc.)
                   None returns all tools

    Returns:
        List of LangChain tools for the agent
    """
    if agent_type is None:
        return ALL_TOOLS

    tool_mapping = {
        "market_scanner": [
            get_lq45_list,
            get_idx30_list,
            fetch_stock_data,
            get_multiple_prices,
            get_sector_relative_strength,
            get_stock_sector,
        ],
        "research": [
            get_stock_info,
            get_financials,
            get_dividends,
            get_price_history,
            search_stocks,
            get_stock_sector,
        ],
        "technical": [
            get_price_history,
            fetch_stock_data,
            generate_features,
            get_stock_info,
        ],
        "sentiment": [
            fetch_stock_news,
            analyze_sentiment,
            get_stock_info,
        ],
        "portfolio": [
            get_stock_info,
            get_price_history,
            get_stock_sector,
            get_sector_relative_strength,
        ],
        "risk": [
            get_price_history,
            fetch_stock_data,
            generate_features,
            get_stock_info,
            get_stock_sector,
        ],
        "execution": [
            get_stock_info,
            get_current_price,
            get_price_history,
        ],
    }

    return tool_mapping.get(agent_type, ALL_TOOLS)
