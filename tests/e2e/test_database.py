"""E2E Tests for Database Models and Initialization (Story 1.2)."""

import os
import tempfile
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from typer.testing import CliRunner

from stockai.cli.main import app
from stockai.data.database import DatabaseManager, get_db, init_database, session_scope
from stockai.data.models import (
    AgentMemory,
    Base,
    CacheEntry,
    NewsArticle,
    PortfolioItem,
    PortfolioTransaction,
    Prediction,
    Stock,
    StockPrice,
    WatchlistItem,
)

runner = CliRunner()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.

    Uses a unique database file per test to ensure proper isolation.
    """
    import uuid

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use unique DB file per test invocation
        unique_id = uuid.uuid4().hex[:8]
        db_path = os.path.join(tmpdir, f"test_{unique_id}.db")
        os.environ["STOCKAI_DB_PATH"] = db_path

        # Clear singleton completely before each test
        if DatabaseManager._engine is not None:
            try:
                DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._SessionLocal = None

        yield tmpdir

        # Cleanup after test
        if DatabaseManager._engine is not None:
            try:
                DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._SessionLocal = None
        if "STOCKAI_DB_PATH" in os.environ:
            del os.environ["STOCKAI_DB_PATH"]


class TestDatabaseInitialization:
    """Test database initialization."""

    def test_init_command(self, temp_db):
        """AC1.2.1: Database schema initializes correctly."""
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "initialized" in result.stdout.lower()

    def test_tables_created(self, temp_db):
        """AC1.2.2: All required tables are created."""
        init_database()
        db = get_db()

        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        required_tables = [
            "stocks",
            "stock_prices",
            "predictions",
            "portfolio_items",
            "portfolio_transactions",
            "watchlist_items",
            "news_articles",
            "agent_memories",
            "cache_entries",
        ]

        for table in required_tables:
            assert table in tables, f"Missing table: {table}"


class TestStockModel:
    """Test Stock model operations."""

    def test_create_stock(self, temp_db):
        """Test creating a stock record."""
        init_database()

        with session_scope() as session:
            stock = Stock(
                symbol="BBCA",
                name="Bank Central Asia Tbk",
                sector="Financial",
                industry="Banking",
                market_cap=Decimal("1000000000000"),
                is_idx30=True,
                is_lq45=True,
            )
            session.add(stock)
            session.flush()
            assert stock.id is not None

    def test_stock_unique_symbol(self, temp_db):
        """Test symbol uniqueness constraint."""
        init_database()
        db = get_db()

        # First insert succeeds
        session1 = db.get_session()
        try:
            stock1 = Stock(symbol="TEST1", name="Test Stock 1")
            session1.add(stock1)
            session1.commit()
        finally:
            session1.close()

        # Second insert with same symbol should fail
        session2 = db.get_session()
        try:
            stock2 = Stock(symbol="TEST1", name="Duplicate")
            session2.add(stock2)
            with pytest.raises(IntegrityError):
                session2.commit()
        finally:
            session2.rollback()
            session2.close()

    def test_query_stock_by_symbol(self, temp_db):
        """Test querying stock by symbol."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="TLKM", name="Telkom Indonesia")
            session.add(stock)

        with session_scope() as session:
            found = session.query(Stock).filter(Stock.symbol == "TLKM").first()
            assert found is not None
            assert found.name == "Telkom Indonesia"


class TestStockPriceModel:
    """Test StockPrice model operations."""

    def test_create_price_record(self, temp_db):
        """Test creating a stock price record."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="BBRI", name="Bank Rakyat Indonesia")
            session.add(stock)
            session.flush()

            price = StockPrice(
                stock_id=stock.id,
                date=datetime(2024, 1, 15),
                open=Decimal("4500"),
                high=Decimal("4600"),
                low=Decimal("4450"),
                close=Decimal("4550"),
                volume=10000000,
            )
            session.add(price)
            session.flush()
            assert price.id is not None

    def test_price_with_indicators(self, temp_db):
        """Test storing technical indicators with price."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="ASII", name="Astra International")
            session.add(stock)
            session.flush()

            price = StockPrice(
                stock_id=stock.id,
                date=datetime(2024, 1, 15),
                open=Decimal("5000"),
                high=Decimal("5100"),
                low=Decimal("4950"),
                close=Decimal("5050"),
                volume=5000000,
                rsi_14=65.5,
                macd=0.12,
                bb_upper=5200.0,
                bb_middle=5000.0,
                bb_lower=4800.0,
            )
            session.add(price)
            stock_id = stock.id

        with session_scope() as session:
            found = session.query(StockPrice).filter(StockPrice.stock_id == stock_id).first()
            assert found.rsi_14 == 65.5
            assert found.macd == 0.12


class TestPredictionModel:
    """Test Prediction model operations."""

    def test_create_prediction(self, temp_db):
        """Test creating a prediction record."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="UNVR", name="Unilever Indonesia")
            session.add(stock)
            session.flush()

            prediction = Prediction(
                stock_id=stock.id,
                prediction_date=datetime(2024, 1, 15),
                target_date=datetime(2024, 1, 22),
                direction="UP",
                confidence=0.75,
                xgboost_prob=0.72,
                lstm_prob=0.78,
                sentiment_score=0.65,
                ensemble_prob=0.75,
            )
            session.add(prediction)
            session.flush()
            assert prediction.id is not None

    def test_prediction_directions(self, temp_db):
        """Test prediction direction values."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="BMRI", name="Bank Mandiri")
            session.add(stock)
            session.flush()
            stock_id = stock.id

            for direction in ["UP", "DOWN", "NEUTRAL"]:
                pred = Prediction(
                    stock_id=stock_id,
                    prediction_date=datetime(2024, 1, 15),
                    target_date=datetime(2024, 1, 22),
                    direction=direction,
                    confidence=0.6,
                )
                session.add(pred)

        with session_scope() as session:
            count = session.query(Prediction).filter(Prediction.stock_id == stock_id).count()
            assert count == 3


class TestPortfolioModel:
    """Test Portfolio model operations."""

    def test_create_portfolio_item(self, temp_db):
        """Test creating a portfolio item."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="GOTO", name="GoTo Gojek Tokopedia")
            session.add(stock)
            session.flush()

            item = PortfolioItem(
                stock_id=stock.id,
                shares=1000,
                avg_price=Decimal("100"),
            )
            session.add(item)
            session.flush()
            assert item.id is not None

    def test_portfolio_transaction(self, temp_db):
        """Test recording portfolio transactions."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="ICBP", name="Indofood CBP")
            session.add(stock)
            session.flush()

            item = PortfolioItem(
                stock_id=stock.id,
                shares=500,
                avg_price=Decimal("10000"),
            )
            session.add(item)
            session.flush()

            tx = PortfolioTransaction(
                portfolio_item_id=item.id,
                transaction_type="BUY",
                shares=500,
                price=Decimal("10000"),
            )
            session.add(tx)

        with session_scope() as session:
            found = session.query(PortfolioTransaction).first()
            assert found.transaction_type == "BUY"
            assert found.shares == 500


class TestWatchlistModel:
    """Test Watchlist model operations."""

    def test_create_watchlist_item(self, temp_db):
        """Test adding stock to watchlist."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="KLBF", name="Kalbe Farma")
            session.add(stock)
            session.flush()

            watch = WatchlistItem(
                stock_id=stock.id,
                alert_price_above=Decimal("1800"),
                alert_price_below=Decimal("1500"),
            )
            session.add(watch)
            session.flush()
            assert watch.id is not None


class TestNewsArticleModel:
    """Test NewsArticle model operations."""

    def test_create_news_article(self, temp_db):
        """Test creating news article with sentiment."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="INDF", name="Indofood Sukses Makmur")
            session.add(stock)
            session.flush()

            article = NewsArticle(
                stock_id=stock.id,
                title="Indofood Reports Strong Q3 Earnings",
                url="https://example.com/news/123",
                source="detik.com",
                published_at=datetime(2024, 1, 15),
                sentiment_score=0.8,
                sentiment_label="POSITIVE",
                sentiment_confidence=0.92,
            )
            session.add(article)

        with session_scope() as session:
            found = session.query(NewsArticle).first()
            assert found.sentiment_label == "POSITIVE"


class TestAgentMemoryModel:
    """Test AgentMemory model operations."""

    def test_create_agent_memory(self, temp_db):
        """Test creating agent memory entry."""
        init_database()

        with session_scope() as session:
            memory = AgentMemory(
                session_id="session_123",
                memory_type="research",
                content="BBCA shows strong fundamentals with 15% ROE",
            )
            session.add(memory)

        with session_scope() as session:
            found = session.query(AgentMemory).filter(
                AgentMemory.session_id == "session_123"
            ).first()
            assert found is not None
            assert "BBCA" in found.content


class TestCacheEntryModel:
    """Test CacheEntry model operations."""

    def test_create_cache_entry(self, temp_db):
        """Test creating cache entry."""
        init_database()

        with session_scope() as session:
            cache = CacheEntry(
                cache_key="stock_price_BBCA_2024-01-15",
                cache_value='{"close": 9500}',
                expires_at=datetime(2024, 1, 16),
            )
            session.add(cache)

        with session_scope() as session:
            found = session.query(CacheEntry).filter(
                CacheEntry.cache_key == "stock_price_BBCA_2024-01-15"
            ).first()
            assert found is not None


class TestDatabaseRelationships:
    """Test database relationships."""

    def test_stock_prices_relationship(self, temp_db):
        """Test Stock -> StockPrice relationship."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="ANTM", name="Aneka Tambang")
            session.add(stock)
            session.flush()

            for i in range(5):
                price = StockPrice(
                    stock_id=stock.id,
                    date=datetime(2024, 1, 10 + i),
                    open=Decimal("1500"),
                    high=Decimal("1550"),
                    low=Decimal("1480"),
                    close=Decimal("1520"),
                    volume=1000000,
                )
                session.add(price)

        with session_scope() as session:
            stock = session.query(Stock).filter(Stock.symbol == "ANTM").first()
            assert len(stock.prices) == 5

    def test_cascade_delete(self, temp_db):
        """Test cascade delete on stock prices."""
        init_database()

        with session_scope() as session:
            stock = Stock(symbol="PTBA", name="Bukit Asam")
            session.add(stock)
            session.flush()
            stock_id = stock.id

            price = StockPrice(
                stock_id=stock_id,
                date=datetime(2024, 1, 15),
                open=Decimal("2500"),
                high=Decimal("2550"),
                low=Decimal("2480"),
                close=Decimal("2520"),
                volume=500000,
            )
            session.add(price)
            session.flush()

            # Delete stock should cascade to prices
            session.delete(stock)

        with session_scope() as session:
            prices = session.query(StockPrice).filter(StockPrice.stock_id == stock_id).all()
            assert len(prices) == 0
