"""SQLite storage for CivitAI community images (prompts, gen params, reactions).

Shares the same civitai_stats.db file as CivitaiStatsDB but uses a separate table.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
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
    has_workflow INTEGER DEFAULT 0,
    media_type TEXT DEFAULT 'image',
    resources TEXT,
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
        has_workflow, media_type, resources, created_at, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(civitai_image_id) DO UPDATE SET
      image_url = excluded.image_url,
      local_filename = COALESCE(excluded.local_filename, community_images.local_filename),
      like_count = excluded.like_count,
      heart_count = excluded.heart_count,
      laugh_count = excluded.laugh_count,
      comment_count = excluded.comment_count,
      has_workflow = excluded.has_workflow,
      media_type = excluded.media_type,
      resources = COALESCE(excluded.resources, community_images.resources),
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
        data.get("has_workflow", 0),
        data.get("media_type", "image"),
        data.get("resources"),
        data.get("created_at"),
        now,
    )


class CommunityImagesDB:
    """Thin wrapper around SQLite for CivitAI community images."""

    _instance: CommunityImagesDB | None = None
    _lock = threading.Lock()
    _db_lock = threading.Lock()

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def get_instance(cls) -> CommunityImagesDB:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst.init()
                    cls._instance = inst
        return cls._instance

    def init(self) -> None:
        """Create the database and table if they don't exist."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that may be missing from older schema versions."""
        conn = self._conn
        if conn is None:
            return
        with self._db_lock:
            # Check if has_workflow column exists
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(community_images)")
            }
            if "has_workflow" not in cols:
                conn.execute(
                    "ALTER TABLE community_images ADD COLUMN has_workflow INTEGER DEFAULT 0"
                )
            if "resources" not in cols:
                conn.execute(
                    "ALTER TABLE community_images ADD COLUMN resources TEXT"
                )
            if "media_type" not in cols:
                conn.execute(
                    "ALTER TABLE community_images ADD COLUMN media_type TEXT DEFAULT 'image'"
                )
            conn.commit()

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.init()
        return self._conn  # type: ignore[return-value]

    def upsert(self, data: dict) -> None:
        """Insert or update a single community image."""
        with self._db_lock:
            conn = self._ensure_conn()
            conn.execute(_UPSERT_SQL, _row_to_params(data, time.time()))
            conn.commit()

    def upsert_batch(self, rows: list[dict]) -> None:
        """Insert or update multiple community images."""
        if not rows:
            return
        with self._db_lock:
            conn = self._ensure_conn()
            now = time.time()
            conn.executemany(_UPSERT_SQL, [_row_to_params(d, now) for d in rows])
            conn.commit()

    def get_by_hashes(self, hashes: list[str]) -> dict[str, list[dict]]:
        """Return images grouped by sha256 for the given hashes."""
        if not hashes:
            return {}
        with self._db_lock:
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
        with self._db_lock:
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

    def get_image_counts(self, hashes: list[str]) -> dict[str, int]:
        """Return image count per sha256 for the given hashes."""
        if not hashes:
            return {}
        with self._db_lock:
            conn = self._ensure_conn()
            result: dict[str, int] = {}
            chunk_size = 500
            for i in range(0, len(hashes), chunk_size):
                chunk = hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT sha256, COUNT(*) as cnt FROM community_images "
                    f"WHERE sha256 IN ({placeholders}) GROUP BY sha256",
                    chunk,
                ).fetchall()
                for row in rows:
                    result[row["sha256"]] = row["cnt"]
        return result

    def get_models_paginated(
        self,
        allowed_hashes: list[str],
        page: int = 1,
        page_size: int = 10,
        sort: str = "reactions:desc",
    ) -> dict:
        """Return community images paginated by model (sha256).

        Returns dict with keys: models (list of sha256), total, images (sha256 -> list).
        Only includes sha256 values present in allowed_hashes that have images.
        """
        if not allowed_hashes:
            return {"models": [], "total": 0, "images": {}}

        # Build the set of sha256 values that have images AND are in allowed_hashes
        # We need to query in chunks due to SQLite variable limits
        all_model_rows: list[dict] = []
        with self._db_lock:
            conn = self._ensure_conn()
            chunk_size = 500
            for i in range(0, len(allowed_hashes), chunk_size):
                chunk = allowed_hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT sha256, "
                    f"  COUNT(*) as image_count, "
                    f"  SUM(like_count + heart_count) as total_reactions, "
                    f"  MAX(created_at) as newest, "
                    f"  MAX(fetched_at) as last_fetched "
                    f"FROM community_images "
                    f"WHERE sha256 IN ({placeholders}) "
                    f"GROUP BY sha256",
                    chunk,
                ).fetchall()
                all_model_rows.extend(dict(r) for r in rows)

        if not all_model_rows:
            return {"models": [], "total": 0, "images": {}}

        # Sort models
        key, direction = (sort.split(":") + ["desc"])[:2]
        reverse = direction != "asc"

        if key == "reactions":
            all_model_rows.sort(key=lambda r: r["total_reactions"] or 0, reverse=reverse)
        elif key == "recent":
            all_model_rows.sort(key=lambda r: r["newest"] or "", reverse=reverse)
        elif key == "fetched":
            all_model_rows.sort(key=lambda r: r["last_fetched"] or 0, reverse=reverse)
        elif key == "lora":
            # Sort by sha256 as placeholder — caller will re-sort by name
            pass
        else:
            all_model_rows.sort(key=lambda r: r["total_reactions"] or 0, reverse=True)

        total = len(all_model_rows)

        # Paginate
        offset = (page - 1) * page_size
        page_models = all_model_rows[offset : offset + page_size]
        page_hashes = [r["sha256"] for r in page_models]

        # Fetch images for this page's models only
        images = self.get_by_hashes(page_hashes)

        return {
            "models": page_hashes,
            "total": total,
            "images": images,
        }

    def delete_by_hash(self, sha256: str) -> None:
        """Delete all community images for a given model hash."""
        with self._db_lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM community_images WHERE sha256 = ?", (sha256,))
            conn.commit()

    def delete_stale(self, sha256: str, before_time: float) -> int:
        """Delete community images for a hash that were fetched before a timestamp.

        Returns the number of rows deleted.
        """
        with self._db_lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM community_images WHERE sha256 = ? AND fetched_at < ?",
                (sha256, before_time),
            )
            conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        """Return number of rows in community_images."""
        with self._db_lock:
            conn = self._ensure_conn()
            return conn.execute("SELECT COUNT(*) FROM community_images").fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        with self._db_lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        if CommunityImagesDB._instance is self:
            CommunityImagesDB._instance = None
