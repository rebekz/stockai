"""Web Routes for StockAI Dashboard.

API and page routes for the web interface.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from stockai import __version__
from stockai.data.cache import async_cached
from stockai.data.database import init_database
from stockai.data.sources.yahoo import YahooFinanceSource
from stockai.data.sources.idx import IDXIndexSource

logger = logging.getLogger(__name__)

# API Router
api_router = APIRouter(tags=["api"])

# Pages Router
pages_router = APIRouter(tags=["pages"])


# ============ API ROUTES ============

@api_router.get("/status")
async def api_status() -> dict:
    """Get API status and version."""
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat(),
    }


@api_router.get("/stocks")
async def list_stocks(
    index: str = Query("IDX30", description="Index to list (IDX30, LQ45)"),
    include_prices: bool = Query(False, description="Include current prices"),
) -> dict:
    """List stocks in an index."""
    idx_source = IDXIndexSource()

    if index.upper() == "IDX30":
        stocks = idx_source.get_idx30_stocks(include_prices=include_prices)
    elif index.upper() == "LQ45":
        stocks = idx_source.get_lq45_stocks(include_prices=include_prices)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown index: {index}")

    return {
        "index": index.upper(),
        "count": len(stocks),
        "stocks": stocks,
    }


@api_router.get("/stocks/{symbol}")
async def get_stock_info(symbol: str) -> dict:
    """Get detailed stock information."""
    idx_source = IDXIndexSource()
    info = idx_source.get_stock_details(symbol.upper())

    if not info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    return info


@api_router.get("/stocks/{symbol}/history")
async def get_stock_history(
    symbol: str,
    period: str = Query("1mo", description="Time period (1d,5d,1mo,3mo,6mo,1y,2y)"),
) -> dict:
    """Get stock price history."""
    yahoo = YahooFinanceSource()
    df = yahoo.get_price_history(symbol.upper(), period=period)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No history for {symbol}")

    # Convert to dict
    history = []
    for _, row in df.iterrows():
        history.append({
            "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
            "open": round(row["open"], 2),
            "high": round(row["high"], 2),
            "low": round(row["low"], 2),
            "close": round(row["close"], 2),
            "volume": int(row["volume"]),
        })

    return {
        "symbol": symbol.upper(),
        "period": period,
        "count": len(history),
        "history": history,
    }


@api_router.get("/stocks/{symbol}/chart")
async def get_stock_chart_data(
    symbol: str,
    period: str = Query("3mo", description="Time period"),
) -> dict:
    """Get stock chart data formatted for Plotly."""
    yahoo = YahooFinanceSource()
    df = yahoo.get_price_history(symbol.upper(), period=period)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    # Format for candlestick chart
    return {
        "symbol": symbol.upper(),
        "dates": [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in df["date"]],
        "open": df["open"].round(2).tolist(),
        "high": df["high"].round(2).tolist(),
        "low": df["low"].round(2).tolist(),
        "close": df["close"].round(2).tolist(),
        "volume": df["volume"].astype(int).tolist(),
    }


@api_router.get("/portfolio")
async def get_portfolio() -> dict:
    """Get portfolio positions with P&L."""
    init_database()

    from stockai.core.portfolio import PnLCalculator

    pnl_calc = PnLCalculator()
    summary = pnl_calc.get_portfolio_summary()

    return summary


@api_router.get("/portfolio/analytics")
async def get_portfolio_analytics() -> dict:
    """Get portfolio analytics."""
    init_database()

    from stockai.core.portfolio import PortfolioAnalytics

    analytics = PortfolioAnalytics()
    analysis = analytics.get_full_analysis()
    insights = analytics.generate_ai_insights(analysis)

    analysis["insights"] = insights
    return analysis


@api_router.get("/sentiment/{symbol}")
@async_cached("sentiment")
async def get_sentiment(
    symbol: str,
    days: int = Query(7, description="Days of news to analyze"),
) -> dict:
    """Get sentiment analysis for a stock."""
    # Normalize symbol for consistent cache keys
    symbol = symbol.upper()

    from stockai.core.sentiment import SentimentAnalyzer, NewsAggregator

    news_agg = NewsAggregator()
    articles = news_agg.fetch_all(symbol, max_articles=15, days_back=days)

    if not articles:
        return {
            "symbol": symbol,
            "article_count": 0,
            "sentiment": None,
            "message": "No recent news found",
        }

    analyzer = SentimentAnalyzer()
    aggregated = analyzer.aggregate_sentiment(articles, symbol)

    return aggregated.to_dict()


@api_router.get("/predict/{symbol}")
@async_cached("prediction")
async def get_prediction(symbol: str) -> dict:
    """Get stock prediction."""
    # Normalize symbol for consistent cache keys
    symbol = symbol.upper()

    from stockai.config import get_settings
    from stockai.core.predictor import EnsemblePredictor

    settings = get_settings()
    yahoo = YahooFinanceSource()

    df = yahoo.get_price_history(symbol, period="6mo")
    if df.empty or len(df) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {symbol}",
        )

    model_dir = settings.project_root / "data" / "models"
    ensemble = EnsemblePredictor(
        xgboost_path=model_dir / "xgboost_v1.json",
        lstm_path=model_dir / "lstm_v1.pt",
    )

    loaded = ensemble.load_models()
    if not any(loaded.values()):
        return {
            "symbol": symbol,
            "prediction": None,
            "message": "No trained models available",
        }

    # Get prediction with sentiment
    result = ensemble.predict_with_sentiment(df, symbol)

    return {
        "symbol": symbol,
        "prediction": result,
    }


# ============ PAGE ROUTES ============

@pages_router.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Home page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "StockAI Dashboard",
            "version": __version__,
        },
    )


@pages_router.get("/stocks", response_class=HTMLResponse)
async def stocks_page(request: Request):
    """Stocks listing page."""
    templates = request.app.state.templates

    # Get stock list
    idx_source = IDXIndexSource()
    idx30 = idx_source.get_idx30_stocks(include_prices=True)

    return templates.TemplateResponse(
        "stocks.html",
        {
            "request": request,
            "title": "Stock List",
            "stocks": idx30,
        },
    )


@pages_router.get("/analyze/{symbol}", response_class=HTMLResponse)
async def analyze_page(request: Request, symbol: str):
    """Stock analysis page."""
    templates = request.app.state.templates

    # Get stock info
    idx_source = IDXIndexSource()
    info = idx_source.get_stock_details(symbol.upper())

    if not info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    return templates.TemplateResponse(
        "analyze.html",
        {
            "request": request,
            "title": f"Analyze {symbol.upper()}",
            "symbol": symbol.upper(),
            "stock_info": info,
        },
    )


@pages_router.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    """Portfolio page."""
    templates = request.app.state.templates

    # Get portfolio data
    init_database()
    from stockai.core.portfolio import PnLCalculator

    pnl_calc = PnLCalculator()
    summary = pnl_calc.get_portfolio_summary()

    return templates.TemplateResponse(
        "portfolio.html",
        {
            "request": request,
            "title": "Portfolio",
            "portfolio": summary,
        },
    )


@pages_router.get("/sentiment", response_class=HTMLResponse)
async def sentiment_page(request: Request):
    """Sentiment analysis page."""
    templates = request.app.state.templates

    return templates.TemplateResponse(
        "sentiment.html",
        {
            "request": request,
            "title": "Sentiment Analysis",
        },
    )


@api_router.get("/export/{symbol}")
async def export_stock_report(symbol: str) -> dict:
    """Generate stock analysis report data for PDF export.

    Returns comprehensive analysis data that can be used
    to generate a PDF report client-side or server-side.
    """
    from datetime import datetime
    from stockai.core.sentiment import SentimentAnalyzer, NewsAggregator

    symbol = symbol.upper()
    report_data: dict[str, Any] = {
        "symbol": symbol,
        "generated_at": datetime.utcnow().isoformat(),
        "version": __version__,
    }

    # Get stock info
    idx_source = IDXIndexSource()
    stock_info = idx_source.get_stock_details(symbol)

    if not stock_info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    report_data["stock_info"] = stock_info

    # Get price history
    yahoo = YahooFinanceSource()
    df = yahoo.get_price_history(symbol, period="3mo")

    if not df.empty:
        history = []
        for _, row in df.tail(30).iterrows():  # Last 30 days
            history.append({
                "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
                "close": round(row["close"], 2),
                "volume": int(row["volume"]),
            })
        report_data["price_history"] = history

        # Calculate basic stats
        if len(df) > 1:
            first_close = df.iloc[0]["close"]
            last_close = df.iloc[-1]["close"]
            change_pct = ((last_close - first_close) / first_close) * 100

            report_data["price_stats"] = {
                "current_price": round(last_close, 2),
                "period_change_pct": round(change_pct, 2),
                "high": round(df["high"].max(), 2),
                "low": round(df["low"].min(), 2),
                "avg_volume": int(df["volume"].mean()),
            }

    # Get sentiment
    try:
        news_agg = NewsAggregator()
        articles = news_agg.fetch_all(symbol, max_articles=10, days_back=7)

        if articles:
            analyzer = SentimentAnalyzer()
            aggregated = analyzer.aggregate_sentiment(articles, symbol)
            report_data["sentiment"] = {
                "overall": aggregated.dominant_label.value,
                "score": round(aggregated.avg_sentiment_score, 2),
                "confidence": round(aggregated.confidence, 2),
                "article_count": aggregated.article_count,
                "signal_strength": aggregated.signal_strength,
            }
    except Exception:
        report_data["sentiment"] = None

    # Get prediction (if models available)
    try:
        from stockai.config import get_settings
        from stockai.core.predictor import EnsemblePredictor

        settings = get_settings()
        model_dir = settings.project_root / "data" / "models"

        ensemble = EnsemblePredictor(
            xgboost_path=model_dir / "xgboost_v1.json",
            lstm_path=model_dir / "lstm_v1.pt",
        )

        if not df.empty and len(df) >= 50:
            loaded = ensemble.load_models()
            if any(loaded.values()):
                result = ensemble.predict(df)
                report_data["prediction"] = {
                    "direction": result.get("direction"),
                    "confidence": round(result.get("confidence", 0), 2),
                    "confidence_level": result.get("confidence_level"),
                }
    except Exception:
        report_data["prediction"] = None

    return report_data
