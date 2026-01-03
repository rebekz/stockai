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


@stockai_tool(name="get_volume_analysis", category="analysis")
def get_volume_analysis(symbol: str, period: str = "3mo") -> dict[str, Any]:
    """Analyze volume-based indicators for a stock.

    Computes On-Balance Volume (OBV), VWAP, Accumulation/Distribution Line,
    Money Flow Index (MFI), and volume ratios relative to moving averages.

    Args:
        symbol: Stock ticker symbol
        period: Period for calculation (default 3mo for sufficient data)

    Returns:
        Dictionary with volume indicator values and signals
    """
    logger.info(f"Calculating volume analysis for {symbol}")

    df = _yahoo.get_price_history(symbol, period=period)

    if df.empty or len(df) < 20:
        return {"error": f"Insufficient data for volume analysis of {symbol}"}

    try:
        import ta

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # On-Balance Volume (OBV)
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
        obv_values = obv.on_balance_volume()
        current_obv = obv_values.iloc[-1]
        obv_sma5 = obv_values.rolling(5).mean().iloc[-1]

        # OBV trend interpretation
        obv_prev = obv_values.iloc[-5] if len(obv_values) >= 5 else obv_values.iloc[0]
        obv_change_pct = ((current_obv - obv_prev) / abs(obv_prev) * 100) if obv_prev != 0 else 0
        obv_trend = "rising" if current_obv > obv_sma5 else "falling"

        # Volume Weighted Average Price (VWAP)
        vwap = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume)
        current_vwap = vwap.volume_weighted_average_price().iloc[-1]

        # Accumulation/Distribution Line
        ad = ta.volume.AccDistIndexIndicator(high, low, close, volume)
        ad_values = ad.acc_dist_index()
        current_ad = ad_values.iloc[-1]
        ad_sma5 = ad_values.rolling(5).mean().iloc[-1]
        ad_trend = "accumulation" if current_ad > ad_sma5 else "distribution"

        # Money Flow Index (MFI)
        mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14)
        current_mfi = mfi.money_flow_index().iloc[-1]

        # MFI interpretation
        if current_mfi < 20:
            mfi_interpretation = "oversold"
        elif current_mfi > 80:
            mfi_interpretation = "overbought"
        else:
            mfi_interpretation = "neutral"

        # Volume Ratios
        vol_sma5 = volume.rolling(5).mean().iloc[-1]
        vol_sma10 = volume.rolling(10).mean().iloc[-1]
        vol_sma20 = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]

        volume_ratio_5d = current_volume / vol_sma5 if vol_sma5 > 0 else 0
        volume_ratio_10d = current_volume / vol_sma10 if vol_sma10 > 0 else 0
        volume_ratio_20d = current_volume / vol_sma20 if vol_sma20 > 0 else 0

        current_price = close.iloc[-1]

        # Generate signals
        signals = []

        # MFI signals
        if current_mfi < 20:
            signals.append("🟢 MFI oversold (potential buy)")
        elif current_mfi > 80:
            signals.append("🔴 MFI overbought (potential sell)")

        # OBV trend signals
        if obv_trend == "rising" and close.iloc[-1] > close.iloc[-5]:
            signals.append("🟢 OBV confirming price uptrend")
        elif obv_trend == "falling" and close.iloc[-1] < close.iloc[-5]:
            signals.append("🔴 OBV confirming price downtrend")
        elif obv_trend == "rising" and close.iloc[-1] < close.iloc[-5]:
            signals.append("🟢 OBV divergence (bullish)")
        elif obv_trend == "falling" and close.iloc[-1] > close.iloc[-5]:
            signals.append("🔴 OBV divergence (bearish)")

        # A/D line signals
        if ad_trend == "accumulation":
            signals.append("🟢 Accumulation detected (buying pressure)")
        else:
            signals.append("🔴 Distribution detected (selling pressure)")

        # VWAP signals
        if current_price > current_vwap:
            signals.append("🟢 Price above VWAP (bullish)")
        else:
            signals.append("🔴 Price below VWAP (bearish)")

        # Volume spike signals
        if volume_ratio_20d > 2.0:
            signals.append("⚠️ Volume spike (>2x average)")
        elif volume_ratio_20d < 0.5:
            signals.append("⚠️ Low volume (<0.5x average)")

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "current_volume": int(current_volume),
            "indicators": {
                "obv": {
                    "value": round(current_obv, 2),
                    "sma5": round(obv_sma5, 2),
                    "change_pct_5d": round(obv_change_pct, 2),
                    "trend": obv_trend,
                    "interpretation": "bullish" if obv_trend == "rising" else "bearish",
                },
                "vwap": {
                    "value": round(current_vwap, 2),
                    "price_position": "above" if current_price > current_vwap else "below",
                    "interpretation": "bullish" if current_price > current_vwap else "bearish",
                },
                "accumulation_distribution": {
                    "value": round(current_ad, 2),
                    "sma5": round(ad_sma5, 2),
                    "trend": ad_trend,
                    "interpretation": "bullish" if ad_trend == "accumulation" else "bearish",
                },
                "mfi": {
                    "value": round(current_mfi, 2),
                    "interpretation": mfi_interpretation,
                },
                "volume_ratios": {
                    "vs_5d_avg": round(volume_ratio_5d, 2),
                    "vs_10d_avg": round(volume_ratio_10d, 2),
                    "vs_20d_avg": round(volume_ratio_20d, 2),
                    "avg_volume_5d": int(vol_sma5),
                    "avg_volume_10d": int(vol_sma10),
                    "avg_volume_20d": int(vol_sma20),
                },
            },
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except ImportError:
        return {"error": "Technical analysis library not available"}
    except Exception as e:
        logger.error(f"Volume analysis failed: {e}")
        return {"error": str(e)}


@stockai_tool(name="get_volume_profile", category="analysis")
def get_volume_profile(symbol: str, period: str = "1mo", num_levels: int = 20) -> dict[str, Any]:
    """Calculate volume profile by price level for a stock.

    Shows where most trading volume occurred across price levels.
    Useful for identifying support/resistance zones based on volume.

    Args:
        symbol: Stock ticker symbol
        period: Period for calculation (default 1mo)
        num_levels: Number of price levels to divide into (default 20)

    Returns:
        Dictionary with volume profile data including POC, VAH, VAL
    """
    logger.info(f"Calculating volume profile for {symbol}")

    df = _yahoo.get_price_history(symbol, period=period)

    if df.empty or len(df) < 5:
        return {"error": f"Insufficient data for volume profile of {symbol}"}

    try:
        import numpy as np

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Determine price range
        price_min = low.min()
        price_max = high.max()
        price_range = price_max - price_min

        if price_range <= 0:
            return {"error": f"Invalid price range for {symbol}"}

        # Create price bins
        bin_size = price_range / num_levels
        bins = np.linspace(price_min, price_max, num_levels + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate volume at each price level
        # For each bar, distribute volume across price levels based on price range
        volume_by_level = np.zeros(num_levels)

        for i in range(len(df)):
            bar_low = low.iloc[i]
            bar_high = high.iloc[i]
            bar_volume = volume.iloc[i]
            bar_close = close.iloc[i]

            # Find which bins this bar's price range covers
            for j in range(num_levels):
                level_low = bins[j]
                level_high = bins[j + 1]

                # Calculate overlap between bar's price range and this level
                overlap_low = max(bar_low, level_low)
                overlap_high = min(bar_high, level_high)

                if overlap_high > overlap_low:
                    # Distribute volume proportionally based on overlap
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_ratio = (overlap_high - overlap_low) / bar_range
                        volume_by_level[j] += bar_volume * overlap_ratio
                    else:
                        # Single price bar, assign to closest bin
                        if level_low <= bar_close <= level_high:
                            volume_by_level[j] += bar_volume

        total_volume = volume_by_level.sum()

        if total_volume <= 0:
            return {"error": f"No volume data for {symbol}"}

        # Find Point of Control (POC) - price level with highest volume
        poc_index = np.argmax(volume_by_level)
        poc_price = bin_centers[poc_index]
        poc_volume = volume_by_level[poc_index]

        # Calculate Value Area (70% of total volume centered around POC)
        target_volume = total_volume * 0.70
        value_area_volume = poc_volume
        va_low_idx = poc_index
        va_high_idx = poc_index

        # Expand value area from POC until we capture 70% of volume
        while value_area_volume < target_volume:
            # Check volume at adjacent levels
            lower_vol = volume_by_level[va_low_idx - 1] if va_low_idx > 0 else 0
            upper_vol = volume_by_level[va_high_idx + 1] if va_high_idx < num_levels - 1 else 0

            if lower_vol == 0 and upper_vol == 0:
                break

            # Expand toward the side with more volume
            if lower_vol >= upper_vol and va_low_idx > 0:
                va_low_idx -= 1
                value_area_volume += volume_by_level[va_low_idx]
            elif va_high_idx < num_levels - 1:
                va_high_idx += 1
                value_area_volume += volume_by_level[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                value_area_volume += volume_by_level[va_low_idx]
            else:
                break

        val_price = bins[va_low_idx]  # Value Area Low
        vah_price = bins[va_high_idx + 1]  # Value Area High

        current_price = close.iloc[-1]

        # Create volume profile data for visualization
        volume_profile = []
        for i in range(num_levels):
            level_volume = volume_by_level[i]
            volume_pct = (level_volume / total_volume * 100) if total_volume > 0 else 0
            volume_profile.append({
                "price_low": round(bins[i], 2),
                "price_high": round(bins[i + 1], 2),
                "price_mid": round(bin_centers[i], 2),
                "volume": int(level_volume),
                "volume_percent": round(volume_pct, 2),
                "is_poc": i == poc_index,
                "in_value_area": va_low_idx <= i <= va_high_idx,
            })

        # Generate interpretation
        signals = []
        if current_price > vah_price:
            signals.append("🟢 Price above Value Area High (potential breakout)")
        elif current_price < val_price:
            signals.append("🔴 Price below Value Area Low (potential breakdown)")
        else:
            signals.append("⚪ Price within Value Area (consolidation zone)")

        # Identify if current price is near high volume nodes (support/resistance)
        price_to_poc_dist = abs(current_price - poc_price) / poc_price * 100
        if price_to_poc_dist < 2:
            signals.append("⚠️ Price near POC (strong support/resistance)")

        return {
            "symbol": symbol.upper(),
            "period": period,
            "current_price": round(current_price, 2),
            "price_range": {
                "min": round(price_min, 2),
                "max": round(price_max, 2),
            },
            "volume_profile": {
                "poc": {
                    "price": round(poc_price, 2),
                    "volume": int(poc_volume),
                    "volume_percent": round(poc_volume / total_volume * 100, 2),
                    "description": "Point of Control - highest volume price level",
                },
                "value_area": {
                    "high": round(vah_price, 2),
                    "low": round(val_price, 2),
                    "volume_percent": round(value_area_volume / total_volume * 100, 2),
                    "description": "Value Area - price range containing ~70% of volume",
                },
                "total_volume": int(total_volume),
                "num_levels": num_levels,
            },
            "levels": volume_profile,
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except ImportError:
        return {"error": "NumPy library not available"}
    except Exception as e:
        logger.error(f"Volume profile calculation failed: {e}")
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
