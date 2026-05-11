"""
Database engine and session management using SQLAlchemy.
Provides both synchronous session factory and a FastAPI dependency.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import NullPool
from typing import Generator

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# Synchronous engine — used for migrations and background tasks
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,       # Verify connections before use
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,      # Log SQL only in debug mode
)


@event.listens_for(engine, "connect")
def on_connect(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
    logger.debug("New database connection established.")


# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.
    Ensures the session is always closed after the request.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_all_tables() -> None:
    """
    Create all tables defined in ORM models.

    NOTE: In production, prefer running Alembic migrations:
        alembic upgrade head
    This function is retained for local development and test environments
    where running migrations is impractical.
    """
    from app.models import dataset, experiment  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified (create_all). Use Alembic in production.")
