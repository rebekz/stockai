"""Database initialization and session management.

Handles SQLite database setup with SQLAlchemy ORM.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from stockai.config import get_settings
from stockai.data.models import Base


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable SQLite foreign keys and WAL mode for better performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class DatabaseManager:
    """Manages database connections and sessions."""

    _instance = None
    _engine = None
    _SessionLocal = None

    def __new__(cls):
        """Singleton pattern for database manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize database manager."""
        if self._engine is None:
            self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Create database engine and session factory."""
        settings = get_settings()
        db_path = settings.db_full_path

        # Ensure data directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=settings.log_level == "DEBUG",
            connect_args={"check_same_thread": False},
        )

        # Create session factory
        self._SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine,
        )

    @property
    def engine(self) -> Engine:
        """Get database engine."""
        return self._engine

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self._engine)

    def drop_tables(self) -> None:
        """Drop all database tables (use with caution)."""
        Base.metadata.drop_all(bind=self._engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around operations.

        Usage:
            with db.session_scope() as session:
                session.query(Stock).all()
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database() -> None:
    """Initialize database and create tables.

    Call this at application startup.
    """
    db = get_db()
    db.create_tables()


def get_session() -> Session:
    """Get a new database session.

    Remember to close the session after use.
    """
    return get_db().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Get a managed database session context.

    Usage:
        with session_scope() as session:
            stocks = session.query(Stock).all()
    """
    with get_db().session_scope() as session:
        yield session
