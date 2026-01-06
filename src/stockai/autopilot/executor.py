"""Paper Trading Executor for Autopilot.

Handles paper trading execution and portfolio management.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import json
import logging
import pytz

from sqlalchemy.orm import Session

from stockai.data.database import get_session
from stockai.data.models import Base

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("Asia/Jakarta")
SHARES_PER_LOT = 100


@dataclass
class PaperPosition:
    """A paper trading position."""

    symbol: str
    lots: int
    shares: int
    avg_price: float
    current_price: float
    stop_loss: float | None
    target: float | None
    entry_date: datetime
    pnl: float = 0
    pnl_pct: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "lots": self.lots,
            "shares": self.shares,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "entry_date": self.entry_date.isoformat(),
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class PaperPortfolio:
    """Paper trading portfolio state."""

    initial_capital: float
    cash: float
    positions: dict[str, PaperPosition]
    created_at: datetime
    updated_at: datetime

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            p.shares * p.current_price for p in self.positions.values()
        )
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.total_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Calculate total P&L percentage."""
        if self.initial_capital <= 0:
            return 0
        return (self.total_pnl / self.initial_capital) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class PaperExecutor:
    """Paper trading executor.

    Manages paper trading portfolio and executes trades.
    Persists state to a JSON file for simplicity.
    """

    def __init__(self, portfolio_file: str = "paper_portfolio.json"):
        """Initialize paper executor.

        Args:
            portfolio_file: Path to portfolio state file
        """
        self.portfolio_file = portfolio_file
        self.portfolio: PaperPortfolio | None = None

    def load_portfolio(self) -> PaperPortfolio | None:
        """Load portfolio from file.

        Returns:
            PaperPortfolio or None if not found
        """
        try:
            import os
            if not os.path.exists(self.portfolio_file):
                return None

            with open(self.portfolio_file) as f:
                data = json.load(f)

            positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                positions[symbol] = PaperPosition(
                    symbol=pos_data["symbol"],
                    lots=pos_data["lots"],
                    shares=pos_data["shares"],
                    avg_price=pos_data["avg_price"],
                    current_price=pos_data["current_price"],
                    stop_loss=pos_data.get("stop_loss"),
                    target=pos_data.get("target"),
                    entry_date=datetime.fromisoformat(pos_data["entry_date"]),
                    pnl=pos_data.get("pnl", 0),
                    pnl_pct=pos_data.get("pnl_pct", 0),
                )

            self.portfolio = PaperPortfolio(
                initial_capital=data["initial_capital"],
                cash=data["cash"],
                positions=positions,
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
            )

            return self.portfolio

        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return None

    def save_portfolio(self) -> bool:
        """Save portfolio to file.

        Returns:
            True if successful
        """
        if not self.portfolio:
            return False

        try:
            self.portfolio.updated_at = datetime.now(TIMEZONE)
            data = self.portfolio.to_dict()

            with open(self.portfolio_file, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return False

    def create_portfolio(self, initial_capital: float) -> PaperPortfolio:
        """Create a new paper portfolio.

        Args:
            initial_capital: Starting capital in Rupiah

        Returns:
            New PaperPortfolio
        """
        now = datetime.now(TIMEZONE)
        self.portfolio = PaperPortfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
            positions={},
            created_at=now,
            updated_at=now,
        )
        self.save_portfolio()
        return self.portfolio

    def buy(
        self,
        symbol: str,
        lots: int,
        price: float,
        stop_loss: float | None = None,
        target: float | None = None,
    ) -> bool:
        """Execute a paper buy order.

        Args:
            symbol: Stock symbol
            lots: Number of lots
            price: Price per share
            stop_loss: Stop-loss price
            target: Target price

        Returns:
            True if successful
        """
        if not self.portfolio:
            logger.error("No portfolio loaded")
            return False

        shares = lots * SHARES_PER_LOT
        cost = shares * price

        if cost > self.portfolio.cash:
            logger.error(f"Insufficient cash: need {cost:,.0f}, have {self.portfolio.cash:,.0f}")
            return False

        # Check if already holding
        if symbol in self.portfolio.positions:
            # Average up/down
            existing = self.portfolio.positions[symbol]
            total_shares = existing.shares + shares
            total_cost = (existing.shares * existing.avg_price) + cost
            new_avg_price = total_cost / total_shares

            existing.lots = total_shares // SHARES_PER_LOT
            existing.shares = total_shares
            existing.avg_price = new_avg_price
            if stop_loss:
                existing.stop_loss = stop_loss
            if target:
                existing.target = target
        else:
            # New position
            self.portfolio.positions[symbol] = PaperPosition(
                symbol=symbol,
                lots=lots,
                shares=shares,
                avg_price=price,
                current_price=price,
                stop_loss=stop_loss,
                target=target,
                entry_date=datetime.now(TIMEZONE),
            )

        self.portfolio.cash -= cost
        self.save_portfolio()

        logger.info(f"BUY: {lots} lots {symbol} @ Rp {price:,.0f}")
        return True

    def sell(
        self,
        symbol: str,
        lots: int | None = None,
        price: float | None = None,
    ) -> float:
        """Execute a paper sell order.

        Args:
            symbol: Stock symbol
            lots: Number of lots (None = sell all)
            price: Price per share (None = use current price)

        Returns:
            Proceeds from sale, or 0 if failed
        """
        if not self.portfolio:
            logger.error("No portfolio loaded")
            return 0

        if symbol not in self.portfolio.positions:
            logger.error(f"No position in {symbol}")
            return 0

        position = self.portfolio.positions[symbol]

        if lots is None:
            lots = position.lots

        if lots > position.lots:
            logger.error(f"Insufficient lots: have {position.lots}, trying to sell {lots}")
            return 0

        sell_price = price or position.current_price
        shares_to_sell = lots * SHARES_PER_LOT
        proceeds = shares_to_sell * sell_price

        if lots == position.lots:
            # Sell entire position
            del self.portfolio.positions[symbol]
        else:
            # Partial sell
            position.lots -= lots
            position.shares -= shares_to_sell

        self.portfolio.cash += proceeds
        self.save_portfolio()

        logger.info(f"SELL: {lots} lots {symbol} @ Rp {sell_price:,.0f}")
        return proceeds

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions.

        Args:
            prices: Dict of symbol -> current price
        """
        if not self.portfolio:
            return

        for symbol, position in self.portfolio.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.pnl = (position.current_price - position.avg_price) * position.shares
                if position.avg_price > 0:
                    position.pnl_pct = ((position.current_price / position.avg_price) - 1) * 100

        self.save_portfolio()

    def get_portfolio_for_engine(self) -> dict[str, Any] | None:
        """Get portfolio in format expected by AutopilotEngine.

        Returns:
            Dict with positions and cash
        """
        if not self.portfolio:
            return None

        positions = {}
        for symbol, pos in self.portfolio.positions.items():
            positions[symbol] = {
                "lots": pos.lots,
                "shares": pos.shares,
                "avg_price": pos.avg_price,
                "stop_loss": pos.stop_loss,
                "target": pos.target,
            }

        return {
            "positions": positions,
            "cash": self.portfolio.cash,
            "initial_capital": self.portfolio.initial_capital,
        }


def format_portfolio_for_display(portfolio: PaperPortfolio) -> str:
    """Format portfolio for CLI display.

    Args:
        portfolio: PaperPortfolio to format

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        f"PAPER PORTFOLIO - {portfolio.updated_at.strftime('%d %B %Y')}",
        "=" * 60,
        "",
        "PORTFOLIO:",
    ]

    if portfolio.positions:
        # Table header
        lines.append(
            f"{'Symbol':<8} {'Lots':>6} {'Avg Cost':>12} {'Current':>12} {'P&L':>10} {'%':>8}"
        )
        lines.append("-" * 60)

        for symbol, pos in portfolio.positions.items():
            pnl_str = f"{pos.pnl:+,.0f}" if pos.pnl != 0 else "0"
            pnl_pct_str = f"{pos.pnl_pct:+.2f}%" if pos.pnl_pct != 0 else "0.00%"

            lines.append(
                f"{symbol:<8} {pos.lots:>6} Rp {pos.avg_price:>9,.0f} Rp {pos.current_price:>9,.0f} {pnl_str:>10} {pnl_pct_str:>8}"
            )

        lines.append("-" * 60)
    else:
        lines.append("   No positions")

    # Summary
    lines.extend([
        "",
        "SUMMARY:",
        f"   Total Value:  Rp {portfolio.total_value:>15,.0f}",
        f"   Total P&L:    Rp {portfolio.total_pnl:>15,.0f} ({portfolio.total_pnl_pct:+.2f}%)",
        f"   Cash:         Rp {portfolio.cash:>15,.0f}",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
