"""Paper Trading Module.

Simulated trading for practice without real money.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import pytz

import logging

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("Asia/Jakarta")


class TradeAction(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PaperTrade:
    """A single paper trade record."""
    id: str
    symbol: str
    action: TradeAction
    lots: int
    price: float
    total_value: float
    fee: float
    timestamp: datetime
    notes: str = ""
    stop_loss: float | None = None
    target: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action.value,
            "lots": self.lots,
            "price": self.price,
            "total_value": self.total_value,
            "fee": self.fee,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
            "stop_loss": self.stop_loss,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperTrade":
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            action=TradeAction(data["action"]),
            lots=data["lots"],
            price=data["price"],
            total_value=data["total_value"],
            fee=data["fee"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            notes=data.get("notes", ""),
            stop_loss=data.get("stop_loss"),
            target=data.get("target"),
        )


@dataclass
class Position:
    """A stock position in the portfolio."""
    symbol: str
    lots: int
    shares: int  # lots * 100
    avg_price: float
    total_cost: float
    current_price: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: float | None = None
    target: float | None = None

    def update_price(self, price: float) -> None:
        """Update position with current market price."""
        self.current_price = price
        self.current_value = self.shares * price
        self.unrealized_pnl = self.current_value - self.total_cost
        self.unrealized_pnl_pct = (self.unrealized_pnl / self.total_cost) * 100 if self.total_cost > 0 else 0


@dataclass
class PaperTradingAccount:
    """Paper trading account for practice."""
    initial_capital: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[PaperTrade] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(TIMEZONE))
    fee_rate: float = 0.0015  # 0.15% broker fee
    tax_rate: float = 0.001   # 0.1% sell tax

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(p.current_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        """Total profit/loss."""
        return self.portfolio_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Total profit/loss percentage."""
        return (self.total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0

    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L from closed trades."""
        pnl = 0.0
        # Track cost basis per symbol
        cost_basis: dict[str, list[tuple[int, float]]] = {}  # symbol -> [(shares, price), ...]

        for trade in self.trades:
            if trade.action == TradeAction.BUY:
                if trade.symbol not in cost_basis:
                    cost_basis[trade.symbol] = []
                cost_basis[trade.symbol].append((trade.lots * 100, trade.price))
            elif trade.action == TradeAction.SELL:
                if trade.symbol in cost_basis and cost_basis[trade.symbol]:
                    # FIFO method
                    shares_to_sell = trade.lots * 100
                    sell_price = trade.price
                    while shares_to_sell > 0 and cost_basis[trade.symbol]:
                        buy_shares, buy_price = cost_basis[trade.symbol][0]
                        if buy_shares <= shares_to_sell:
                            pnl += buy_shares * (sell_price - buy_price)
                            shares_to_sell -= buy_shares
                            cost_basis[trade.symbol].pop(0)
                        else:
                            pnl += shares_to_sell * (sell_price - buy_price)
                            cost_basis[trade.symbol][0] = (buy_shares - shares_to_sell, buy_price)
                            shares_to_sell = 0
        return pnl

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        return f"PT{datetime.now(TIMEZONE).strftime('%Y%m%d%H%M%S')}{len(self.trades):04d}"

    def _calculate_fee(self, value: float, is_sell: bool = False) -> float:
        """Calculate trading fee."""
        fee = value * self.fee_rate
        if is_sell:
            fee += value * self.tax_rate
        return round(fee, 2)

    def buy(
        self,
        symbol: str,
        lots: int,
        price: float,
        stop_loss: float | None = None,
        target: float | None = None,
        notes: str = "",
    ) -> PaperTrade | str:
        """Execute a paper buy order.

        Args:
            symbol: Stock symbol
            lots: Number of lots (1 lot = 100 shares)
            price: Buy price per share
            stop_loss: Optional stop-loss price
            target: Optional target price
            notes: Optional trade notes

        Returns:
            PaperTrade if successful, error message string if failed
        """
        symbol = symbol.upper()
        shares = lots * 100
        gross_value = shares * price
        fee = self._calculate_fee(gross_value)
        total_cost = gross_value + fee

        # Check if enough cash
        if total_cost > self.cash:
            return f"Insufficient funds. Need Rp {total_cost:,.0f}, have Rp {self.cash:,.0f}"

        # Execute trade
        self.cash -= total_cost

        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            new_shares = pos.shares + shares
            new_cost = pos.total_cost + gross_value
            pos.shares = new_shares
            pos.lots = new_shares // 100
            pos.total_cost = new_cost
            pos.avg_price = new_cost / new_shares
            if stop_loss:
                pos.stop_loss = stop_loss
            if target:
                pos.target = target
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                lots=lots,
                shares=shares,
                avg_price=price,
                total_cost=gross_value,
                current_price=price,
                current_value=gross_value,
                stop_loss=stop_loss,
                target=target,
            )

        # Record trade
        trade = PaperTrade(
            id=self._generate_trade_id(),
            symbol=symbol,
            action=TradeAction.BUY,
            lots=lots,
            price=price,
            total_value=total_cost,
            fee=fee,
            timestamp=datetime.now(TIMEZONE),
            notes=notes,
            stop_loss=stop_loss,
            target=target,
        )
        self.trades.append(trade)

        logger.info(f"Paper BUY: {lots} lot {symbol} @ Rp {price:,.0f}")
        return trade

    def sell(
        self,
        symbol: str,
        lots: int,
        price: float,
        notes: str = "",
    ) -> PaperTrade | str:
        """Execute a paper sell order.

        Args:
            symbol: Stock symbol
            lots: Number of lots to sell
            price: Sell price per share
            notes: Optional trade notes

        Returns:
            PaperTrade if successful, error message string if failed
        """
        symbol = symbol.upper()

        # Check if position exists
        if symbol not in self.positions:
            return f"No position in {symbol}"

        pos = self.positions[symbol]
        if lots > pos.lots:
            return f"Insufficient shares. Have {pos.lots} lots, trying to sell {lots}"

        shares = lots * 100
        gross_value = shares * price
        fee = self._calculate_fee(gross_value, is_sell=True)
        net_proceeds = gross_value - fee

        # Execute trade
        self.cash += net_proceeds

        # Update position
        pos.shares -= shares
        pos.lots = pos.shares // 100
        pos.total_cost = pos.shares * pos.avg_price

        # Remove position if fully closed
        if pos.shares <= 0:
            del self.positions[symbol]

        # Record trade
        trade = PaperTrade(
            id=self._generate_trade_id(),
            symbol=symbol,
            action=TradeAction.SELL,
            lots=lots,
            price=price,
            total_value=net_proceeds,
            fee=fee,
            timestamp=datetime.now(TIMEZONE),
            notes=notes,
        )
        self.trades.append(trade)

        logger.info(f"Paper SELL: {lots} lot {symbol} @ Rp {price:,.0f}")
        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update position prices with current market data.

        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])

    def check_stop_losses(self, prices: dict[str, float]) -> list[str]:
        """Check if any stop-losses are triggered.

        Returns:
            List of warning messages for triggered stop-losses
        """
        warnings = []
        for symbol, pos in self.positions.items():
            if pos.stop_loss and symbol in prices:
                current = prices[symbol]
                if current <= pos.stop_loss:
                    warnings.append(
                        f"⚠️ STOP-LOSS TRIGGERED: {symbol} at Rp {current:,.0f} "
                        f"(stop: Rp {pos.stop_loss:,.0f})"
                    )
        return warnings

    def check_targets(self, prices: dict[str, float]) -> list[str]:
        """Check if any targets are reached.

        Returns:
            List of messages for reached targets
        """
        messages = []
        for symbol, pos in self.positions.items():
            if pos.target and symbol in prices:
                current = prices[symbol]
                if current >= pos.target:
                    messages.append(
                        f"🎯 TARGET REACHED: {symbol} at Rp {current:,.0f} "
                        f"(target: Rp {pos.target:,.0f})"
                    )
        return messages

    def get_summary(self) -> dict[str, Any]:
        """Get account summary."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "positions_count": len(self.positions),
            "trades_count": len(self.trades),
            "win_rate": self._calculate_win_rate(),
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed trades."""
        sells = [t for t in self.trades if t.action == TradeAction.SELL]
        if not sells:
            return 0.0

        # Simple approximation: compare sell price to prior buy avg
        wins = 0
        for sell in sells:
            # Find corresponding buys
            buys = [t for t in self.trades
                   if t.action == TradeAction.BUY
                   and t.symbol == sell.symbol
                   and t.timestamp < sell.timestamp]
            if buys:
                avg_buy = sum(t.price for t in buys) / len(buys)
                if sell.price > avg_buy:
                    wins += 1

        return (wins / len(sells)) * 100 if sells else 0.0

    def save(self, path: Path) -> None:
        """Save account to file."""
        data = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {
                symbol: {
                    "symbol": pos.symbol,
                    "lots": pos.lots,
                    "shares": pos.shares,
                    "avg_price": pos.avg_price,
                    "total_cost": pos.total_cost,
                    "stop_loss": pos.stop_loss,
                    "target": pos.target,
                }
                for symbol, pos in self.positions.items()
            },
            "trades": [t.to_dict() for t in self.trades],
            "created_at": self.created_at.isoformat(),
            "fee_rate": self.fee_rate,
            "tax_rate": self.tax_rate,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Paper account saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "PaperTradingAccount":
        """Load account from file."""
        if not path.exists():
            raise FileNotFoundError(f"No paper account at {path}")

        data = json.loads(path.read_text())

        account = cls(
            initial_capital=data["initial_capital"],
            cash=data["cash"],
            fee_rate=data.get("fee_rate", 0.0015),
            tax_rate=data.get("tax_rate", 0.001),
        )
        account.created_at = datetime.fromisoformat(data["created_at"])

        # Load positions
        for symbol, pos_data in data.get("positions", {}).items():
            account.positions[symbol] = Position(
                symbol=pos_data["symbol"],
                lots=pos_data["lots"],
                shares=pos_data["shares"],
                avg_price=pos_data["avg_price"],
                total_cost=pos_data["total_cost"],
                stop_loss=pos_data.get("stop_loss"),
                target=pos_data.get("target"),
            )

        # Load trades
        account.trades = [PaperTrade.from_dict(t) for t in data.get("trades", [])]

        logger.info(f"Paper account loaded from {path}")
        return account


def create_paper_account(
    capital: float = 10_000_000,
    save_path: Path | None = None,
) -> PaperTradingAccount:
    """Create a new paper trading account.

    Args:
        capital: Starting capital in Rupiah
        save_path: Optional path to save account

    Returns:
        New paper trading account
    """
    account = PaperTradingAccount(
        initial_capital=capital,
        cash=capital,
    )

    if save_path:
        account.save(save_path)

    return account


def get_default_paper_path() -> Path:
    """Get default path for paper trading data."""
    return Path.home() / ".stockai" / "paper_trading.json"
