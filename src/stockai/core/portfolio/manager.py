"""Portfolio Manager for StockAI.

Handles portfolio operations: add, remove, update positions.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from stockai.data.database import session_scope
from stockai.data.models import PortfolioItem, PortfolioTransaction, Stock

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio operations.

    Features:
    - Add new positions or increase existing
    - Remove positions (full or partial sell)
    - Track all transactions with history
    - Calculate average cost basis
    """

    def __init__(self, session: Session | None = None):
        """Initialize portfolio manager.

        Args:
            session: Optional SQLAlchemy session
        """
        self._session = session
        self._use_context_manager = session is None

    def _get_or_create_stock(self, session: Session, symbol: str) -> Stock:
        """Get existing stock or create placeholder.

        Args:
            session: Database session
            symbol: Stock ticker symbol

        Returns:
            Stock model instance
        """
        stock = session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            # Create placeholder stock entry
            stock = Stock(
                symbol=symbol.upper(),
                name=f"{symbol.upper()} (Auto-created)",
                is_active=True,
            )
            session.add(stock)
            session.flush()
            logger.info(f"Created placeholder stock: {symbol}")
        return stock

    def add_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: datetime | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Add shares to portfolio (buy).

        If position exists, updates average price.
        Records transaction history.

        Args:
            symbol: Stock ticker symbol
            shares: Number of shares to add
            price: Purchase price per share
            date: Transaction date (default: now)
            notes: Optional transaction notes

        Returns:
            Result dict with position details
        """
        if shares <= 0:
            raise ValueError("Shares must be positive")
        if price <= 0:
            raise ValueError("Price must be positive")

        date = date or datetime.utcnow()

        def _execute(session: Session) -> dict:
            stock = self._get_or_create_stock(session, symbol)

            # Check for existing position
            position = (
                session.query(PortfolioItem)
                .filter(PortfolioItem.stock_id == stock.id)
                .first()
            )

            if position:
                # Update existing position with weighted average
                total_cost = float(position.avg_price) * position.shares + price * shares
                new_shares = position.shares + shares
                new_avg_price = total_cost / new_shares

                position.shares = new_shares
                position.avg_price = Decimal(str(round(new_avg_price, 2)))
                position.updated_at = datetime.utcnow()
            else:
                # Create new position
                position = PortfolioItem(
                    stock_id=stock.id,
                    shares=shares,
                    avg_price=Decimal(str(price)),
                    purchase_date=date,
                    notes=notes,
                )
                session.add(position)
                session.flush()

            # Record transaction
            transaction = PortfolioTransaction(
                portfolio_item_id=position.id,
                transaction_type="BUY",
                shares=shares,
                price=Decimal(str(price)),
                transaction_date=date,
                notes=notes,
            )
            session.add(transaction)

            return {
                "action": "BUY",
                "symbol": symbol.upper(),
                "shares": shares,
                "price": price,
                "total_shares": position.shares,
                "avg_price": float(position.avg_price),
                "total_cost": shares * price,
                "date": date.isoformat(),
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def remove_position(
        self,
        symbol: str,
        shares: int | None = None,
        price: float | None = None,
        date: datetime | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Remove shares from portfolio (sell).

        If shares not specified, sells entire position.

        Args:
            symbol: Stock ticker symbol
            shares: Number of shares to sell (None = all)
            price: Sale price per share
            date: Transaction date (default: now)
            notes: Optional transaction notes

        Returns:
            Result dict with sale details and P&L
        """
        date = date or datetime.utcnow()

        def _execute(session: Session) -> dict:
            stock = session.query(Stock).filter(Stock.symbol == symbol.upper()).first()
            if not stock:
                raise ValueError(f"Stock {symbol} not found")

            position = (
                session.query(PortfolioItem)
                .filter(PortfolioItem.stock_id == stock.id)
                .first()
            )
            if not position:
                raise ValueError(f"No position found for {symbol}")

            sell_shares = shares if shares is not None else position.shares

            if sell_shares <= 0:
                raise ValueError("Shares to sell must be positive")
            if sell_shares > position.shares:
                raise ValueError(
                    f"Cannot sell {sell_shares} shares, only {position.shares} owned"
                )

            avg_cost = float(position.avg_price)
            sell_price = price or avg_cost  # Default to avg cost if no price

            # Calculate realized P&L
            cost_basis = avg_cost * sell_shares
            sale_value = sell_price * sell_shares
            realized_pnl = sale_value - cost_basis
            pnl_percent = (realized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

            # Update position
            remaining_shares = position.shares - sell_shares

            if remaining_shares == 0:
                # Full sell - delete position
                session.delete(position)
                position_closed = True
            else:
                # Partial sell - update shares (avg price stays same)
                position.shares = remaining_shares
                position.updated_at = datetime.utcnow()
                position_closed = False

            # Record transaction (before deleting position)
            if remaining_shares > 0 or not position_closed:
                transaction = PortfolioTransaction(
                    portfolio_item_id=position.id,
                    transaction_type="SELL",
                    shares=sell_shares,
                    price=Decimal(str(sell_price)),
                    transaction_date=date,
                    notes=notes,
                )
                session.add(transaction)

            return {
                "action": "SELL",
                "symbol": symbol.upper(),
                "shares": sell_shares,
                "price": sell_price,
                "remaining_shares": remaining_shares,
                "position_closed": position_closed,
                "cost_basis": round(cost_basis, 2),
                "sale_value": round(sale_value, 2),
                "realized_pnl": round(realized_pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
                "date": date.isoformat(),
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all portfolio positions.

        Returns:
            List of position dicts with stock info
        """

        def _execute(session: Session) -> list[dict]:
            positions = (
                session.query(PortfolioItem)
                .join(Stock)
                .order_by(Stock.symbol)
                .all()
            )

            result = []
            for pos in positions:
                result.append({
                    "id": pos.id,
                    "symbol": pos.stock.symbol,
                    "name": pos.stock.name,
                    "shares": pos.shares,
                    "avg_price": float(pos.avg_price),
                    "cost_basis": float(pos.avg_price) * pos.shares,
                    "purchase_date": pos.purchase_date.isoformat() if pos.purchase_date else None,
                    "notes": pos.notes,
                })
            return result

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Get single portfolio position.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Position dict or None if not found
        """

        def _execute(session: Session) -> dict | None:
            position = (
                session.query(PortfolioItem)
                .join(Stock)
                .filter(Stock.symbol == symbol.upper())
                .first()
            )

            if not position:
                return None

            return {
                "id": position.id,
                "symbol": position.stock.symbol,
                "name": position.stock.name,
                "shares": position.shares,
                "avg_price": float(position.avg_price),
                "cost_basis": float(position.avg_price) * position.shares,
                "purchase_date": position.purchase_date.isoformat() if position.purchase_date else None,
                "notes": position.notes,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_transactions(
        self,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get transaction history.

        Args:
            symbol: Optional filter by symbol
            limit: Maximum transactions to return

        Returns:
            List of transaction dicts
        """

        def _execute(session: Session) -> list[dict]:
            query = (
                session.query(PortfolioTransaction)
                .join(PortfolioItem)
                .join(Stock)
            )

            if symbol:
                query = query.filter(Stock.symbol == symbol.upper())

            transactions = (
                query.order_by(PortfolioTransaction.transaction_date.desc())
                .limit(limit)
                .all()
            )

            result = []
            for txn in transactions:
                result.append({
                    "id": txn.id,
                    "symbol": txn.portfolio_item.stock.symbol,
                    "type": txn.transaction_type,
                    "shares": txn.shares,
                    "price": float(txn.price),
                    "total": float(txn.price) * txn.shares,
                    "date": txn.transaction_date.isoformat(),
                    "notes": txn.notes,
                })
            return result

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def clear_portfolio(self) -> int:
        """Clear all portfolio data (for testing).

        Returns:
            Number of positions deleted
        """

        def _execute(session: Session) -> int:
            # Delete transactions first (due to foreign key)
            session.query(PortfolioTransaction).delete()
            # Then delete positions
            count = session.query(PortfolioItem).delete()
            return count

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)
