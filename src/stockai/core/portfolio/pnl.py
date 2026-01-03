"""P&L Calculator for StockAI.

Calculates profit/loss, returns, and performance metrics.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from stockai.data.database import session_scope
from stockai.data.models import PortfolioItem, PortfolioTransaction, Stock
from stockai.data.sources.yahoo import YahooFinanceSource

logger = logging.getLogger(__name__)


class PnLCalculator:
    """Calculates portfolio P&L and performance metrics.

    Features:
    - Real-time P&L with current prices
    - Realized vs unrealized gains
    - Percentage returns
    - Daily/total change tracking
    """

    def __init__(self, session: Session | None = None):
        """Initialize P&L calculator.

        Args:
            session: Optional SQLAlchemy session
        """
        self._session = session
        self._use_context_manager = session is None
        self._yahoo = YahooFinanceSource()

    def _get_current_price(self, symbol: str) -> float | None:
        """Get current market price for symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            info = self._yahoo.get_stock_info(symbol)
            return info.get("current_price") or info.get("previousClose")
        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")
            return None

    def calculate_position_pnl(
        self,
        symbol: str,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        """Calculate P&L for single position.

        Args:
            symbol: Stock ticker symbol
            current_price: Optional override for current price

        Returns:
            P&L metrics dict
        """

        def _execute(session: Session) -> dict:
            position = (
                session.query(PortfolioItem)
                .join(Stock)
                .filter(Stock.symbol == symbol.upper())
                .first()
            )

            if not position:
                return {
                    "symbol": symbol.upper(),
                    "error": "Position not found",
                }

            shares = position.shares
            avg_cost = float(position.avg_price)
            cost_basis = avg_cost * shares

            # Get current price
            price = current_price or self._get_current_price(symbol)
            if price is None:
                return {
                    "symbol": symbol.upper(),
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "cost_basis": cost_basis,
                    "current_price": None,
                    "error": "Could not fetch current price",
                }

            # Calculate unrealized P&L
            market_value = price * shares
            unrealized_pnl = market_value - cost_basis
            pnl_percent = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

            return {
                "symbol": symbol.upper(),
                "name": position.stock.name,
                "shares": shares,
                "avg_cost": round(avg_cost, 2),
                "cost_basis": round(cost_basis, 2),
                "current_price": round(price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
                "is_profit": unrealized_pnl >= 0,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def calculate_portfolio_pnl(
        self,
        prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Calculate total portfolio P&L.

        Args:
            prices: Optional dict of symbol -> price overrides

        Returns:
            Portfolio P&L summary
        """
        prices = prices or {}

        def _execute(session: Session) -> dict:
            positions = (
                session.query(PortfolioItem)
                .join(Stock)
                .all()
            )

            if not positions:
                return {
                    "total_cost_basis": 0,
                    "total_market_value": 0,
                    "total_unrealized_pnl": 0,
                    "total_pnl_percent": 0,
                    "positions": [],
                    "position_count": 0,
                }

            total_cost = 0.0
            total_value = 0.0
            position_results = []
            errors = []

            for pos in positions:
                symbol = pos.stock.symbol
                shares = pos.shares
                avg_cost = float(pos.avg_price)
                cost_basis = avg_cost * shares
                total_cost += cost_basis

                # Get price
                price = prices.get(symbol) or self._get_current_price(symbol)

                if price is None:
                    errors.append(symbol)
                    # Use cost basis as fallback
                    total_value += cost_basis
                    position_results.append({
                        "symbol": symbol,
                        "shares": shares,
                        "cost_basis": round(cost_basis, 2),
                        "market_value": None,
                        "unrealized_pnl": None,
                        "error": "Price unavailable",
                    })
                else:
                    market_value = price * shares
                    total_value += market_value
                    unrealized_pnl = market_value - cost_basis
                    pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

                    position_results.append({
                        "symbol": symbol,
                        "shares": shares,
                        "avg_cost": round(avg_cost, 2),
                        "cost_basis": round(cost_basis, 2),
                        "current_price": round(price, 2),
                        "market_value": round(market_value, 2),
                        "unrealized_pnl": round(unrealized_pnl, 2),
                        "pnl_percent": round(pnl_pct, 2),
                    })

            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

            return {
                "total_cost_basis": round(total_cost, 2),
                "total_market_value": round(total_value, 2),
                "total_unrealized_pnl": round(total_pnl, 2),
                "total_pnl_percent": round(total_pnl_pct, 2),
                "is_profit": total_pnl >= 0,
                "positions": position_results,
                "position_count": len(positions),
                "errors": errors if errors else None,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_realized_pnl(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Calculate realized P&L from sell transactions.

        Args:
            symbol: Optional filter by symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Realized P&L summary
        """

        def _execute(session: Session) -> dict:
            query = (
                session.query(PortfolioTransaction)
                .join(PortfolioItem)
                .join(Stock)
                .filter(PortfolioTransaction.transaction_type == "SELL")
            )

            if symbol:
                query = query.filter(Stock.symbol == symbol.upper())
            if start_date:
                query = query.filter(PortfolioTransaction.transaction_date >= start_date)
            if end_date:
                query = query.filter(PortfolioTransaction.transaction_date <= end_date)

            sells = query.all()

            total_realized = 0.0
            transactions = []

            for sell in sells:
                # Get corresponding buys to estimate cost basis
                avg_cost = float(sell.portfolio_item.avg_price)
                cost_basis = avg_cost * sell.shares
                sale_value = float(sell.price) * sell.shares
                realized = sale_value - cost_basis

                total_realized += realized

                transactions.append({
                    "symbol": sell.portfolio_item.stock.symbol,
                    "date": sell.transaction_date.isoformat(),
                    "shares": sell.shares,
                    "sale_price": float(sell.price),
                    "sale_value": round(sale_value, 2),
                    "cost_basis": round(cost_basis, 2),
                    "realized_pnl": round(realized, 2),
                })

            return {
                "total_realized_pnl": round(total_realized, 2),
                "transaction_count": len(transactions),
                "transactions": transactions,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_portfolio_summary(
        self,
        prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive portfolio summary.

        Combines position data with P&L calculations.

        Args:
            prices: Optional price overrides

        Returns:
            Complete portfolio summary
        """
        pnl = self.calculate_portfolio_pnl(prices)
        realized = self.get_realized_pnl()

        # Sort positions by value (descending)
        positions = sorted(
            pnl.get("positions", []),
            key=lambda x: x.get("market_value") or x.get("cost_basis", 0),
            reverse=True,
        )

        # Calculate allocation percentages
        total_value = pnl.get("total_market_value", 0)
        for pos in positions:
            value = pos.get("market_value") or pos.get("cost_basis", 0)
            pos["allocation_percent"] = round(
                (value / total_value) * 100 if total_value > 0 else 0, 2
            )

        # Identify winners and losers
        winners = [p for p in positions if p.get("unrealized_pnl", 0) > 0]
        losers = [p for p in positions if p.get("unrealized_pnl", 0) < 0]

        return {
            "summary": {
                "position_count": pnl.get("position_count", 0),
                "total_cost_basis": pnl.get("total_cost_basis", 0),
                "total_market_value": pnl.get("total_market_value", 0),
                "total_unrealized_pnl": pnl.get("total_unrealized_pnl", 0),
                "total_pnl_percent": pnl.get("total_pnl_percent", 0),
                "total_realized_pnl": realized.get("total_realized_pnl", 0),
                "is_profit": pnl.get("is_profit", True),
            },
            "positions": positions,
            "winners_count": len(winners),
            "losers_count": len(losers),
            "best_performer": max(positions, key=lambda x: x.get("pnl_percent", 0)) if positions else None,
            "worst_performer": min(positions, key=lambda x: x.get("pnl_percent", 0)) if positions else None,
            "errors": pnl.get("errors"),
        }
