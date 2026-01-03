"""Answer Synthesis Phase for StockAI Agent.

Synthesizes collected data into a coherent, actionable response.
"""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


DISCLAIMER = """
---
*Disclaimer: This analysis is for informational purposes only and should not be construed as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
"""


class AnswerPhase:
    """Handles the answer synthesis phase of agent execution.

    Responsibilities:
    - Combine all tool results into coherent response
    - Format according to UX design
    - Highlight key findings
    - Add appropriate disclaimers
    """

    def __init__(self):
        """Initialize answer phase."""
        pass

    def synthesize(
        self,
        query: str,
        symbol: str | None,
        tool_results: list[dict],
    ) -> str:
        """Synthesize collected data into final answer.

        This is the fallback synthesizer when LLM is not available.

        Args:
            query: Original user query
            symbol: Target stock symbol
            tool_results: Collected tool results

        Returns:
            Formatted answer string
        """
        sections = []

        # Title
        if symbol:
            sections.append(f"# 📊 Analysis: {symbol}")
        else:
            sections.append("# 📊 Market Analysis")

        sections.append(f"\n*Query: {query}*")
        sections.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n")

        # Extract and format data
        stock_info = self._extract_stock_info(tool_results)
        price_data = self._extract_price_data(tool_results)
        technical = self._extract_technical(tool_results)

        # Stock Info Section
        if stock_info:
            sections.append("## Company Overview")
            sections.append(self._format_stock_info(stock_info))

        # Price Section
        if price_data:
            sections.append("\n## Current Price")
            sections.append(self._format_price_data(price_data))

        # Technical Analysis Section
        if technical:
            sections.append("\n## Technical Analysis")
            sections.append(self._format_technical(technical))

        # Key Findings
        findings = self._generate_findings(stock_info, price_data, technical)
        if findings:
            sections.append("\n## Key Findings")
            for finding in findings:
                sections.append(f"- {finding}")

        # Add disclaimer
        sections.append(DISCLAIMER)

        return "\n".join(sections)

    def _extract_stock_info(self, results: list[dict]) -> dict | None:
        """Extract stock info from results."""
        for result in results:
            for tool_result in result.get("results", []):
                if tool_result.get("tool") == "get_stock_info":
                    return tool_result.get("data", {})
        return None

    def _extract_price_data(self, results: list[dict]) -> dict | None:
        """Extract price data from results."""
        for result in results:
            for tool_result in result.get("results", []):
                if tool_result.get("tool") in ["get_current_price", "get_price_history"]:
                    return tool_result.get("data", {})
        return None

    def _extract_technical(self, results: list[dict]) -> dict | None:
        """Extract technical analysis from results."""
        for result in results:
            for tool_result in result.get("results", []):
                if tool_result.get("tool") == "get_technical_indicators":
                    return tool_result.get("data", {})
        return None

    def _format_stock_info(self, info: dict) -> str:
        """Format stock info section."""
        lines = []

        name = info.get("name", "N/A")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        lines.append(f"**{name}**\n")
        lines.append(f"- Sector: {sector}")
        lines.append(f"- Industry: {industry}")

        # Market metrics
        market_cap = info.get("market_cap")
        if market_cap:
            if market_cap >= 1e12:
                cap_str = f"Rp {market_cap/1e12:.2f}T"
            else:
                cap_str = f"Rp {market_cap/1e9:.2f}B"
            lines.append(f"- Market Cap: {cap_str}")

        pe = info.get("pe_ratio")
        pb = info.get("pb_ratio")
        if pe:
            lines.append(f"- P/E Ratio: {pe:.2f}")
        if pb:
            lines.append(f"- P/B Ratio: {pb:.2f}")

        # Index membership
        if info.get("is_idx30"):
            lines.append("- 🏆 Member of IDX30")
        if info.get("is_lq45"):
            lines.append("- 🏅 Member of LQ45")

        return "\n".join(lines)

    def _format_price_data(self, data: dict) -> str:
        """Format price section."""
        lines = []

        price = data.get("price")
        change = data.get("change")
        change_pct = data.get("change_percent")

        if price:
            lines.append(f"**Current Price: Rp {price:,.0f}**")

            if change is not None and change_pct is not None:
                icon = "🟢" if change >= 0 else "🔴"
                sign = "+" if change >= 0 else ""
                lines.append(f"{icon} {sign}Rp {change:,.0f} ({sign}{change_pct:.2f}%)")

        volume = data.get("volume")
        if volume:
            lines.append(f"\nVolume: {volume:,}")

        # Summary data if from price history
        summary = data.get("summary")
        if summary:
            lines.append(f"\n**{summary.get('period', '1mo')} Performance**")
            period_change = summary.get("change_percent")
            if period_change:
                icon = "🟢" if period_change >= 0 else "🔴"
                sign = "+" if period_change >= 0 else ""
                lines.append(f"Period Change: {icon} {sign}{period_change:.2f}%")

            high = summary.get("high")
            low = summary.get("low")
            if high and low:
                lines.append(f"Range: Rp {low:,.0f} - Rp {high:,.0f}")

        return "\n".join(lines)

    def _format_technical(self, data: dict) -> str:
        """Format technical analysis section."""
        lines = []

        indicators = data.get("indicators", {})

        # RSI
        rsi = indicators.get("rsi", {})
        if rsi:
            value = rsi.get("value")
            interp = rsi.get("interpretation", "neutral")
            icon = {"oversold": "🟢", "overbought": "🔴", "neutral": "🟡"}.get(interp, "")
            lines.append(f"**RSI (14):** {value} {icon} {interp}")

        # MACD
        macd = indicators.get("macd", {})
        if macd:
            interp = macd.get("interpretation", "neutral")
            icon = "🟢" if interp == "bullish" else "🔴"
            lines.append(f"**MACD:** {icon} {interp}")

        # Signals
        signals = data.get("signals", [])
        if signals:
            lines.append("\n**Signals:**")
            for signal in signals:
                lines.append(f"- {signal}")

        return "\n".join(lines)

    def _generate_findings(
        self,
        info: dict | None,
        price: dict | None,
        technical: dict | None,
    ) -> list[str]:
        """Generate key findings from data."""
        findings = []

        # Price-based findings
        if price:
            change_pct = price.get("change_percent")
            if change_pct:
                if change_pct > 2:
                    findings.append(f"🟢 Stock is up {change_pct:.2f}% today - strong positive momentum")
                elif change_pct < -2:
                    findings.append(f"🔴 Stock is down {abs(change_pct):.2f}% today - significant selling pressure")

        # Technical-based findings
        if technical:
            indicators = technical.get("indicators", {})

            rsi = indicators.get("rsi", {})
            if rsi.get("interpretation") == "oversold":
                findings.append("🟢 RSI indicates oversold conditions - potential buying opportunity")
            elif rsi.get("interpretation") == "overbought":
                findings.append("🔴 RSI indicates overbought conditions - consider taking profits")

            macd = indicators.get("macd", {})
            if macd.get("interpretation") == "bullish":
                findings.append("🟢 MACD shows bullish crossover - positive trend signal")
            elif macd.get("interpretation") == "bearish":
                findings.append("🔴 MACD shows bearish crossover - trend may be weakening")

        # Info-based findings
        if info:
            pe = info.get("pe_ratio")
            if pe:
                if pe < 10:
                    findings.append(f"🟢 Low P/E ratio ({pe:.2f}) - potentially undervalued")
                elif pe > 30:
                    findings.append(f"🟡 High P/E ratio ({pe:.2f}) - priced for growth expectations")

            div_yield = info.get("dividend_yield")
            if div_yield and div_yield > 0.03:
                findings.append(f"🟢 Attractive dividend yield of {div_yield*100:.2f}%")

        return findings


def format_rupiah(value: float) -> str:
    """Format a number as Indonesian Rupiah.

    Args:
        value: Numeric value

    Returns:
        Formatted string like "Rp 9,500"
    """
    if value >= 1e12:
        return f"Rp {value/1e12:.2f}T"
    elif value >= 1e9:
        return f"Rp {value/1e9:.2f}B"
    elif value >= 1e6:
        return f"Rp {value/1e6:.2f}M"
    else:
        return f"Rp {value:,.0f}"
