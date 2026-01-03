"""Portfolio Analytics for StockAI.

Provides AI-powered portfolio analysis and recommendations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from stockai.data.database import session_scope
from stockai.data.models import PortfolioItem, Stock
from stockai.data.sources.yahoo import YahooFinanceSource
from stockai.core.portfolio.pnl import PnLCalculator

logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """Analyzes portfolio for insights and recommendations.

    Features:
    - Concentration risk analysis
    - Sector allocation
    - Correlation analysis
    - Diversification score
    - Volatility metrics
    - AI-powered recommendations
    """

    def __init__(self, session: Session | None = None):
        """Initialize portfolio analytics.

        Args:
            session: Optional SQLAlchemy session
        """
        self._session = session
        self._use_context_manager = session is None
        self._yahoo = YahooFinanceSource()
        self._pnl = PnLCalculator(session)

    def analyze_concentration(
        self,
        prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Analyze portfolio concentration risk.

        Args:
            prices: Optional price overrides

        Returns:
            Concentration analysis
        """
        summary = self._pnl.get_portfolio_summary(prices)
        positions = summary.get("positions", [])

        if not positions:
            return {
                "concentration_score": 0,
                "risk_level": "N/A",
                "top_holdings": [],
                "recommendations": ["Add positions to your portfolio"],
            }

        # Calculate Herfindahl-Hirschman Index (HHI)
        allocations = [p.get("allocation_percent", 0) for p in positions]
        hhi = sum(a ** 2 for a in allocations)

        # HHI interpretation
        # < 1500: Low concentration
        # 1500-2500: Moderate concentration
        # > 2500: High concentration
        if hhi < 1500:
            risk_level = "LOW"
            concentration_score = 1 - (hhi / 1500) * 0.3
        elif hhi < 2500:
            risk_level = "MODERATE"
            concentration_score = 0.7 - ((hhi - 1500) / 1000) * 0.3
        else:
            risk_level = "HIGH"
            concentration_score = max(0.1, 0.4 - ((hhi - 2500) / 5000) * 0.3)

        # Top holdings
        top_holdings = sorted(
            positions,
            key=lambda x: x.get("allocation_percent", 0),
            reverse=True,
        )[:5]

        # Generate recommendations
        recommendations = []
        if risk_level == "HIGH":
            recommendations.append(
                f"High concentration risk: Top position is {top_holdings[0]['allocation_percent']:.1f}% of portfolio"
            )
            recommendations.append("Consider reducing largest positions to improve diversification")
        if len(positions) < 5:
            recommendations.append(
                f"Low diversification: Only {len(positions)} positions. Consider adding more stocks"
            )

        return {
            "concentration_score": round(concentration_score, 2),
            "hhi_index": round(hhi, 2),
            "risk_level": risk_level,
            "position_count": len(positions),
            "top_holdings": [
                {
                    "symbol": h.get("symbol"),
                    "allocation": h.get("allocation_percent"),
                }
                for h in top_holdings
            ],
            "recommendations": recommendations,
        }

    def analyze_sector_allocation(self) -> dict[str, Any]:
        """Analyze portfolio sector allocation.

        Returns:
            Sector allocation analysis
        """

        def _execute(session: Session) -> dict:
            positions = (
                session.query(PortfolioItem)
                .join(Stock)
                .all()
            )

            if not positions:
                return {
                    "sectors": {},
                    "diversification_score": 0,
                    "recommendations": ["No positions to analyze"],
                }

            # Fetch sector info for each position
            sector_allocation = {}
            unknown_sectors = []

            for pos in positions:
                symbol = pos.stock.symbol
                shares = pos.shares
                cost = float(pos.avg_price) * shares

                # Try to get sector from database or Yahoo
                sector = pos.stock.sector
                if not sector:
                    try:
                        info = self._yahoo.get_stock_info(symbol)
                        sector = info.get("sector", "Unknown")
                    except Exception:
                        sector = "Unknown"
                        unknown_sectors.append(symbol)

                if sector not in sector_allocation:
                    sector_allocation[sector] = {
                        "value": 0,
                        "stocks": [],
                    }

                sector_allocation[sector]["value"] += cost
                sector_allocation[sector]["stocks"].append(symbol)

            # Calculate percentages
            total_value = sum(s["value"] for s in sector_allocation.values())
            for sector_data in sector_allocation.values():
                sector_data["percent"] = round(
                    (sector_data["value"] / total_value) * 100 if total_value > 0 else 0,
                    2,
                )

            # Diversification score based on sector count
            sector_count = len([s for s in sector_allocation if s != "Unknown"])
            if sector_count >= 8:
                diversification = "EXCELLENT"
                score = 0.9
            elif sector_count >= 5:
                diversification = "GOOD"
                score = 0.7
            elif sector_count >= 3:
                diversification = "MODERATE"
                score = 0.5
            else:
                diversification = "LOW"
                score = 0.3

            # Recommendations
            recommendations = []
            largest_sector = max(
                sector_allocation.items(),
                key=lambda x: x[1]["percent"],
            )
            if largest_sector[1]["percent"] > 40:
                recommendations.append(
                    f"Overweight in {largest_sector[0]}: {largest_sector[1]['percent']:.1f}%"
                )

            if sector_count < 5:
                recommendations.append(
                    f"Low sector diversification: Only {sector_count} sectors. Consider adding exposure to other sectors"
                )

            return {
                "sectors": {
                    k: {
                        "value": round(v["value"], 2),
                        "percent": v["percent"],
                        "stocks": v["stocks"],
                    }
                    for k, v in sorted(
                        sector_allocation.items(),
                        key=lambda x: x[1]["percent"],
                        reverse=True,
                    )
                },
                "sector_count": sector_count,
                "diversification_level": diversification,
                "diversification_score": score,
                "recommendations": recommendations,
                "unknown_sectors": unknown_sectors if unknown_sectors else None,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def calculate_portfolio_volatility(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Calculate portfolio volatility metrics.

        Args:
            days: Number of days for calculation

        Returns:
            Volatility metrics
        """

        def _execute(session: Session) -> dict:
            positions = (
                session.query(PortfolioItem)
                .join(Stock)
                .all()
            )

            if not positions:
                return {
                    "portfolio_volatility": 0,
                    "risk_level": "N/A",
                    "position_volatilities": [],
                }

            # Calculate individual stock volatilities
            position_vols = []
            total_value = 0

            for pos in positions:
                symbol = pos.stock.symbol
                try:
                    df = self._yahoo.get_historical_data(
                        symbol,
                        period=f"{days}d",
                    )
                    if df is not None and len(df) > 5:
                        # Calculate daily returns
                        returns = df["Close"].pct_change().dropna()
                        volatility = returns.std() * (252 ** 0.5)  # Annualized

                        value = float(pos.avg_price) * pos.shares
                        total_value += value

                        position_vols.append({
                            "symbol": symbol,
                            "volatility": round(volatility * 100, 2),
                            "value": value,
                        })
                except Exception as e:
                    logger.warning(f"Could not calculate volatility for {symbol}: {e}")

            if not position_vols:
                return {
                    "portfolio_volatility": 0,
                    "risk_level": "UNKNOWN",
                    "error": "Could not fetch price data",
                }

            # Weighted portfolio volatility (simplified - assumes no correlation)
            weighted_vol = sum(
                (p["value"] / total_value) * p["volatility"]
                for p in position_vols
            ) if total_value > 0 else 0

            # Risk level classification
            if weighted_vol < 15:
                risk_level = "LOW"
            elif weighted_vol < 25:
                risk_level = "MODERATE"
            elif weighted_vol < 40:
                risk_level = "HIGH"
            else:
                risk_level = "VERY HIGH"

            # Sort by volatility
            position_vols.sort(key=lambda x: x["volatility"], reverse=True)

            return {
                "portfolio_volatility": round(weighted_vol, 2),
                "risk_level": risk_level,
                "position_volatilities": position_vols,
                "most_volatile": position_vols[0]["symbol"] if position_vols else None,
                "least_volatile": position_vols[-1]["symbol"] if position_vols else None,
                "days_analyzed": days,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_full_analysis(
        self,
        prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive portfolio analysis.

        Args:
            prices: Optional price overrides

        Returns:
            Full analysis with all metrics
        """
        summary = self._pnl.get_portfolio_summary(prices)
        concentration = self.analyze_concentration(prices)
        sectors = self.analyze_sector_allocation()
        volatility = self.calculate_portfolio_volatility()

        # Generate overall score
        scores = [
            concentration.get("concentration_score", 0.5),
            sectors.get("diversification_score", 0.5),
            1 - (volatility.get("portfolio_volatility", 25) / 100),  # Lower vol = better
        ]
        overall_score = sum(scores) / len(scores)

        # Overall health assessment
        if overall_score >= 0.7:
            health = "EXCELLENT"
        elif overall_score >= 0.5:
            health = "GOOD"
        elif overall_score >= 0.3:
            health = "NEEDS_ATTENTION"
        else:
            health = "POOR"

        # Aggregate recommendations
        all_recommendations = []
        all_recommendations.extend(concentration.get("recommendations", []))
        all_recommendations.extend(sectors.get("recommendations", []))

        if volatility.get("risk_level") in ["HIGH", "VERY HIGH"]:
            all_recommendations.append(
                f"High portfolio volatility ({volatility['portfolio_volatility']:.1f}%). "
                "Consider adding defensive stocks"
            )

        return {
            "overall_score": round(overall_score, 2),
            "health_status": health,
            "summary": summary.get("summary", {}),
            "concentration": concentration,
            "sector_allocation": sectors,
            "volatility": volatility,
            "recommendations": all_recommendations[:5],  # Top 5 recommendations
            "analyzed_at": datetime.utcnow().isoformat(),
        }

    def generate_ai_insights(
        self,
        analysis: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate AI-powered insights from analysis.

        Args:
            analysis: Pre-computed analysis or None to compute

        Returns:
            List of insight strings
        """
        if analysis is None:
            analysis = self.get_full_analysis()

        insights = []
        summary = analysis.get("summary", {})
        concentration = analysis.get("concentration", {})
        sectors = analysis.get("sector_allocation", {})
        volatility = analysis.get("volatility", {})

        # P&L insights
        pnl = summary.get("total_unrealized_pnl", 0)
        pnl_pct = summary.get("total_pnl_percent", 0)
        if pnl > 0:
            insights.append(
                f"Portfolio is profitable with +{pnl_pct:.1f}% unrealized gains (Rp {pnl:,.0f})"
            )
        elif pnl < 0:
            insights.append(
                f"Portfolio shows -{abs(pnl_pct):.1f}% unrealized loss (Rp {abs(pnl):,.0f})"
            )

        # Concentration insights
        if concentration.get("risk_level") == "HIGH":
            top = concentration.get("top_holdings", [{}])[0]
            insights.append(
                f"High concentration risk: {top.get('symbol', 'Top stock')} represents "
                f"{top.get('allocation', 0):.1f}% of portfolio"
            )

        # Sector insights
        sector_data = sectors.get("sectors", {})
        if sector_data:
            top_sector = list(sector_data.items())[0] if sector_data else None
            if top_sector and top_sector[1].get("percent", 0) > 30:
                insights.append(
                    f"Sector overweight: {top_sector[0]} at {top_sector[1]['percent']:.1f}%"
                )

        # Volatility insights
        vol_level = volatility.get("risk_level", "")
        if vol_level in ["HIGH", "VERY HIGH"]:
            most_vol = volatility.get("most_volatile")
            insights.append(
                f"High portfolio volatility ({volatility.get('portfolio_volatility', 0):.1f}%). "
                f"Most volatile: {most_vol}"
            )

        # Position count insights
        pos_count = summary.get("position_count", 0)
        if pos_count < 5:
            insights.append(
                f"Low diversification with only {pos_count} positions. "
                "Consider adding 5-10 stocks for better risk management"
            )
        elif pos_count > 20:
            insights.append(
                f"Over-diversified with {pos_count} positions. "
                "Consider consolidating for easier management"
            )

        return insights
