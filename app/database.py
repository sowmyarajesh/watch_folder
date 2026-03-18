from datetime import datetime
from contextlib import contextmanager
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine, select

from app.config import settings

# check_same_thread=False required for SQLite with multiple threads
_connect_args = {"check_same_thread": False}
engine = create_engine(
    settings.db_url,
    connect_args=_connect_args,
    echo=False,          # set True to log SQL
)


def init_db() -> None:
    """Create all tables if they don't already exist."""
    # Import models here so SQLModel metadata is populated before create_all
    import app.models  # noqa: F401
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Synchronous context-manager session.
    Use this in background threads (watcher, worker) that cannot await.
    """
    with Session(engine) as session:
        yield session


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a session per request."""
    with Session(engine) as session:
        yield session
