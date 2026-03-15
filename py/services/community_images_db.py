"""SQLite storage for CivitAI community images (prompts, gen params, reactions).

Shares the same civitai_stats.db file as CivitaiStatsDB but uses a separate table.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS community_images (
    civitai_image_id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL,
    civitai_model_id INTEGER,
    username TEXT,
    image_url TEXT,
    local_filename TEXT,
    width INTEGER,
    height INTEGER,
    prompt TEXT,
    negative_prompt TEXT,
    steps INTEGER,
    sampler TEXT,
    cfg_scale REAL,
    seed INTEGER,
    denoise REAL,
    base_model TEXT,
    like_count INTEGER DEFAULT 0,
    heart_count INTEGER DEFAULT 0,
    laugh_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    created_at TEXT,
    fetched_at REAL
);
CREATE INDEX IF NOT EXISTS idx_community_sha256 ON community_images(sha256);
"""

_UPSERT_SQL = """
    INSERT INTO community_images
       (civitai_image_id, sha256, civitai_model_id, username,
        image_url, local_filename, width, height,
        prompt, negative_prompt, steps, sampler, cfg_scale,
        seed, denoise, base_model,
        like_count, heart_count, laugh_count, comment_count,
        created_at, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(civitai_image_id) DO UPDATE SET
      image_url = excluded.image_url,
      like_count = excluded.like_count,
      heart_count = excluded.heart_count,
      laugh_count = excluded.laugh_count,
      comment_count = excluded.comment_count,
      fetched_at = excluded.fetched_at
"""


def _default_db_path() -> Path:
    """Return path to civitai_stats.db inside the cache directory."""
    from ..utils.cache_paths import get_cache_base_dir
    return Path(get_cache_base_dir()) / "civitai_stats.db"


def _row_to_params(data: dict, now: float) -> tuple:
    """Extract a row dict into a parameter tuple for the upsert SQL."""
    return (
        data.get("civitai_image_id"),
        data.get("sha256"),
        data.get("civitai_model_id"),
        data.get("username"),
        data.get("image_url"),
        data.get("local_filename"),
        data.get("width"),
        data.get("height"),
        data.get("prompt"),
        data.get("negative_prompt"),
        data.get("steps"),
        data.get("sampler"),
        data.get("cfg_scale"),
        data.get("seed"),
        data.get("denoise"),
        data.get("base_model"),
        data.get("like_count", 0),
        data.get("heart_count", 0),
        data.get("laugh_count", 0),
        data.get("comment_count", 0),
        data.get("created_at"),
        now,
    )


class CommunityImagesDB:
    """Thin wrapper around SQLite for CivitAI community images."""

    _instance: CommunityImagesDB | None = None

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def get_instance(cls) -> CommunityImagesDB:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.init()
        return cls._instance

    def init(self) -> None:
        """Create the database and table if they don't exist."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.init()
        return self._conn  # type: ignore[return-value]

    def upsert(self, data: dict) -> None:
        """Insert or update a single community image."""
        conn = self._ensure_conn()
        conn.execute(_UPSERT_SQL, _row_to_params(data, time.time()))
        conn.commit()

    def upsert_batch(self, rows: list[dict]) -> None:
        """Insert or update multiple community images."""
        if not rows:
            return
        conn = self._ensure_conn()
        now = time.time()
        conn.executemany(_UPSERT_SQL, [_row_to_params(d, now) for d in rows])
        conn.commit()

    def get_by_hashes(self, hashes: list[str]) -> dict[str, list[dict]]:
        """Return images grouped by sha256 for the given hashes."""
        if not hashes:
            return {}
        conn = self._ensure_conn()
        result: dict[str, list[dict]] = {}
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT * FROM community_images WHERE sha256 IN ({placeholders}) "
                f"ORDER BY like_count + heart_count DESC",
                chunk,
            ).fetchall()
            for row in rows:
                sha = row["sha256"]
                result.setdefault(sha, []).append(dict(row))
        return result

    def get_hashes_with_images(self, hashes: list[str]) -> set[str]:
        """Return which of the given hashes already have community images."""
        if not hashes:
            return set()
        conn = self._ensure_conn()
        found: set[str] = set()
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT DISTINCT sha256 FROM community_images WHERE sha256 IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                found.add(row["sha256"])
        return found

    def delete_by_hash(self, sha256: str) -> None:
        """Delete all community images for a given model hash."""
        conn = self._ensure_conn()
        conn.execute("DELETE FROM community_images WHERE sha256 = ?", (sha256,))
        conn.commit()

    def count(self) -> int:
        """Return number of rows in community_images."""
        conn = self._ensure_conn()
        return conn.execute("SELECT COUNT(*) FROM community_images").fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if CommunityImagesDB._instance is self:
            CommunityImagesDB._instance = None
