"""Database access: the one engine factory. Everything else queries through it."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from ml_template.paths import DATABASE

DB_PATH = DATABASE  # kept for existing imports


def get_engine(db_path: Path = DB_PATH) -> Engine:
    """SQLite engine with foreign keys enforced. Tests pass ``tmp_path / 'test.db'``."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine
