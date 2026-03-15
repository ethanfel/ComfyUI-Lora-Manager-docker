"""SQLite storage for CivitAI community stats (downloads, ratings, likes).

This is a standalone DB separate from the main model cache to avoid
upstream schema conflicts.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_stats (
    sha256 TEXT PRIMARY KEY,
    civitai_model_id INTEGER,
    civitai_version_id INTEGER,
    download_count INTEGER DEFAULT 0,
    rating REAL DEFAULT 0,
    rating_count INTEGER DEFAULT 0,
    thumbs_up_count INTEGER DEFAULT 0,
    fetched_at REAL
);
"""


def _default_db_path() -> Path:
    """Return path to civitai_stats.db inside the cache directory."""
    from ..utils.cache_paths import get_cache_base_dir
    return Path(get_cache_base_dir()) / "civitai_stats.db"


class CivitaiStatsDB:
    """Thin wrapper around a SQLite database for CivitAI stats."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        """Create the database and table if they don't exist."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.init()
        return self._conn  # type: ignore[return-value]

    def upsert(self, sha256: str, data: dict) -> None:
        """Insert or update stats for a single model."""
        conn = self._ensure_conn()
        conn.execute(
            """INSERT INTO model_stats
               (sha256, civitai_model_id, civitai_version_id,
                download_count, rating, rating_count, thumbs_up_count, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sha256) DO UPDATE SET
                 civitai_model_id = COALESCE(excluded.civitai_model_id, civitai_model_id),
                 civitai_version_id = COALESCE(excluded.civitai_version_id, civitai_version_id),
                 download_count = excluded.download_count,
                 rating = excluded.rating,
                 rating_count = excluded.rating_count,
                 thumbs_up_count = excluded.thumbs_up_count,
                 fetched_at = excluded.fetched_at
            """,
            (
                sha256,
                data.get("civitai_model_id"),
                data.get("civitai_version_id"),
                data.get("download_count", 0),
                data.get("rating", 0),
                data.get("rating_count", 0),
                data.get("thumbs_up_count", 0),
                time.time(),
            ),
        )
        conn.commit()

    def upsert_batch(self, rows: list[tuple[str, dict]]) -> None:
        """Insert or update stats for multiple models."""
        if not rows:
            return
        conn = self._ensure_conn()
        now = time.time()
        conn.executemany(
            """INSERT INTO model_stats
               (sha256, civitai_model_id, civitai_version_id,
                download_count, rating, rating_count, thumbs_up_count, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sha256) DO UPDATE SET
                 civitai_model_id = COALESCE(excluded.civitai_model_id, civitai_model_id),
                 civitai_version_id = COALESCE(excluded.civitai_version_id, civitai_version_id),
                 download_count = excluded.download_count,
                 rating = excluded.rating,
                 rating_count = excluded.rating_count,
                 thumbs_up_count = excluded.thumbs_up_count,
                 fetched_at = excluded.fetched_at
            """,
            [
                (
                    sha256,
                    d.get("civitai_model_id"),
                    d.get("civitai_version_id"),
                    d.get("download_count", 0),
                    d.get("rating", 0),
                    d.get("rating_count", 0),
                    d.get("thumbs_up_count", 0),
                    now,
                )
                for sha256, d in rows
            ],
        )
        conn.commit()

    def get_by_hashes(self, hashes: list[str]) -> dict[str, dict]:
        """Return stats keyed by sha256 for the given hashes."""
        if not hashes:
            return {}
        conn = self._ensure_conn()
        placeholders = ",".join("?" for _ in hashes)
        rows = conn.execute(
            f"SELECT * FROM model_stats WHERE sha256 IN ({placeholders})",
            hashes,
        ).fetchall()
        return {row["sha256"]: dict(row) for row in rows}

    def get_all(self) -> dict[str, dict]:
        """Return all stats keyed by sha256."""
        conn = self._ensure_conn()
        rows = conn.execute("SELECT * FROM model_stats").fetchall()
        return {row["sha256"]: dict(row) for row in rows}

    def count(self) -> int:
        """Return number of rows in model_stats."""
        conn = self._ensure_conn()
        return conn.execute("SELECT COUNT(*) FROM model_stats").fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
