"""Database Models for StockAI.

SQLAlchemy ORM models for storing stock data, predictions, and portfolio.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Stock(Base):
    """Stock information model."""

    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Numeric(20, 2))
    is_idx30 = Column(Boolean, default=False)
    is_lq45 = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")
    portfolio_items = relationship("PortfolioItem", back_populates="stock")
    watchlist_items = relationship("WatchlistItem", back_populates="stock")

    def __repr__(self) -> str:
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"


class StockPrice(Base):
    """Daily stock price data (OHLCV)."""

    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Numeric(12, 2), nullable=False)
    high = Column(Numeric(12, 2), nullable=False)
    low = Column(Numeric(12, 2), nullable=False)
    close = Column(Numeric(12, 2), nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Numeric(12, 2))

    # Technical indicators (cached)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    atr_14 = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uix_stock_date"),
        Index("ix_stock_prices_date", "date"),
        Index("ix_stock_prices_stock_date", "stock_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<StockPrice(stock_id={self.stock_id}, date='{self.date}', close={self.close})>"


class Prediction(Base):
    """ML prediction results."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    direction = Column(String(10), nullable=False)  # UP, DOWN, NEUTRAL
    confidence = Column(Float, nullable=False)

    # Model contributions
    xgboost_prob = Column(Float)
    lstm_prob = Column(Float)
    sentiment_score = Column(Float)
    ensemble_prob = Column(Float)

    # Features used
    feature_importance = Column(Text)  # JSON string

    # Actual outcome (filled later)
    actual_direction = Column(String(10))
    actual_return = Column(Float)
    is_correct = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_stock_date", "stock_id", "prediction_date"),
        Index("ix_predictions_target", "target_date"),
    )

    def __repr__(self) -> str:
        return f"<Prediction(stock_id={self.stock_id}, direction='{self.direction}', confidence={self.confidence})>"


class PortfolioItem(Base):
    """User's portfolio holdings."""

    __tablename__ = "portfolio_items"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    shares = Column(Integer, nullable=False)
    avg_price = Column(Numeric(12, 2), nullable=False)
    purchase_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="portfolio_items")
    transactions = relationship(
        "PortfolioTransaction", back_populates="portfolio_item", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PortfolioItem(stock_id={self.stock_id}, shares={self.shares})>"


class PortfolioTransaction(Base):
    """Portfolio buy/sell transactions."""

    __tablename__ = "portfolio_transactions"

    id = Column(Integer, primary_key=True)
    portfolio_item_id = Column(Integer, ForeignKey("portfolio_items.id"), nullable=False)
    transaction_type = Column(String(10), nullable=False)  # BUY, SELL
    shares = Column(Integer, nullable=False)
    price = Column(Numeric(12, 2), nullable=False)
    transaction_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    portfolio_item = relationship("PortfolioItem", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<PortfolioTransaction(type='{self.transaction_type}', shares={self.shares})>"


class WatchlistItem(Base):
    """User's stock watchlist."""

    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    alert_price_above = Column(Numeric(12, 2))
    alert_price_below = Column(Numeric(12, 2))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="watchlist_items")

    __table_args__ = (UniqueConstraint("stock_id", name="uix_watchlist_stock"),)

    def __repr__(self) -> str:
        return f"<WatchlistItem(stock_id={self.stock_id})>"


class NewsArticle(Base):
    """News articles for sentiment analysis."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False, unique=True)
    source = Column(String(100))
    published_at = Column(DateTime)
    content = Column(Text)
    summary = Column(Text)

    # Sentiment analysis
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # POSITIVE, NEGATIVE, NEUTRAL
    sentiment_confidence = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_news_stock_date", "stock_id", "published_at"),
        Index("ix_news_published", "published_at"),
    )

    def __repr__(self) -> str:
        return f"<NewsArticle(title='{self.title[:50]}...')>"


class AgentMemory(Base):
    """Agent conversation and research memory."""

    __tablename__ = "agent_memories"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), nullable=False, index=True)
    memory_type = Column(String(50), nullable=False)  # research, conversation, insight
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    content = Column(Text, nullable=False)
    extra_data = Column(Text)  # JSON string for metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_agent_memory_session", "session_id", "memory_type"),)

    def __repr__(self) -> str:
        return f"<AgentMemory(session='{self.session_id}', type='{self.memory_type}')>"


class CacheEntry(Base):
    """Generic cache storage."""

    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    cache_value = Column(Text, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CacheEntry(key='{self.cache_key}')>"


class AutopilotRun(Base):
    """Autopilot trading run session."""

    __tablename__ = "autopilot_runs"

    id = Column(Integer, primary_key=True)
    run_date = Column(DateTime, nullable=False, index=True)
    index_scanned = Column(String(10), nullable=False)  # JII70, IDX30, LQ45, ALL
    stocks_scanned = Column(Integer, default=0)
    signals_generated = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    initial_capital = Column(Numeric(20, 2))
    final_value = Column(Numeric(20, 2))
    is_dry_run = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trades = relationship(
        "AutopilotTrade", back_populates="run", cascade="all, delete-orphan"
    )
    validations = relationship(
        "AutopilotValidation", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_autopilot_runs_date", "run_date"),
        UniqueConstraint("run_date", "index_scanned", "is_dry_run", name="uix_autopilot_run"),
    )

    def __repr__(self) -> str:
        return f"<AutopilotRun(date='{self.run_date}', index='{self.index_scanned}', trades={self.trades_executed})>"


class AutopilotTrade(Base):
    """Individual trades executed by autopilot."""

    __tablename__ = "autopilot_trades"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("autopilot_runs.id"), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    action = Column(String(4), nullable=False)  # BUY, SELL
    lots = Column(Integer, nullable=False)
    shares = Column(Integer, nullable=False)
    price = Column(Numeric(12, 2), nullable=False)
    total_value = Column(Numeric(20, 2), nullable=False)
    score = Column(Float)
    reason = Column(String(100))
    stop_loss = Column(Numeric(12, 2))
    target = Column(Numeric(12, 2))
    created_at = Column(DateTime, default=datetime.utcnow)

    # AI Validation fields
    ai_validated = Column(Boolean, default=False)
    ai_composite_score = Column(Float)
    ai_fundamental_score = Column(Float)
    ai_technical_score = Column(Float)
    ai_sentiment_score = Column(Float)
    ai_risk_score = Column(Float)
    ai_recommendation = Column(String(20))
    ai_approved = Column(Boolean)
    ai_rejection_reason = Column(String(200))

    # Relationships
    run = relationship("AutopilotRun", back_populates="trades")

    def __repr__(self) -> str:
        return f"<AutopilotTrade(symbol='{self.symbol}', action='{self.action}', lots={self.lots})>"


class AutopilotValidation(Base):
    """Track all AI validations, including rejections."""

    __tablename__ = "autopilot_validations"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("autopilot_runs.id"), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(4), nullable=False)  # BUY, SELL
    autopilot_score = Column(Float)
    ai_composite_score = Column(Float)
    ai_fundamental_score = Column(Float)
    ai_technical_score = Column(Float)
    ai_sentiment_score = Column(Float)
    ai_risk_score = Column(Float)
    ai_recommendation = Column(String(20))
    is_approved = Column(Boolean, nullable=False)
    rejection_reason = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Gate validation
    gates_passed = Column(Integer)
    total_gates = Column(Integer, default=6)
    rejection_reasons_json = Column(JSON)  # List of strings

    # Trade plan
    entry_low = Column(Float)
    entry_high = Column(Float)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    risk_reward_ratio = Column(Float)

    # Support/Resistance
    nearest_support = Column(Float)
    nearest_resistance = Column(Float)
    distance_to_support_pct = Column(Float)

    # Smart Money
    smart_money_score = Column(Float)
    smart_money_interpretation = Column(String(20))

    # ADX
    adx_value = Column(Float)
    adx_trend_strength = Column(String(20))

    # Relationships
    run = relationship("AutopilotRun", back_populates="validations")

    def __repr__(self) -> str:
        status = "APPROVED" if self.is_approved else "REJECTED"
        return f"<AutopilotValidation(symbol='{self.symbol}', {status})>"
