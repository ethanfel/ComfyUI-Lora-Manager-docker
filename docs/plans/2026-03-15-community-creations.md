# Community Creations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Community Creations" tab showing top community-generated images per LoRA with full generation metadata, fetched from CivitAI and stored locally for permanence.

**Architecture:** New `community_images` table in `civitai_stats.db`, a fetch service that calls CivitAI's `/images` API per model, downloads images to `{example_images_path}/{sha256}/community/`, and a new page tab with card grid UI grouped by LoRA. Follows existing patterns from `civitai_stats_db.py`, `civitai_stats_routes.py`, and `stats_routes.py`.

**Tech Stack:** Python 3 (aiohttp, aiohttp.ClientSession, sqlite3), Jinja2 templates, vanilla JS (IIFE pattern), CSS.

---

### Task 1: Community Images DB Layer

**Files:**
- Create: `py/services/community_images_db.py`
- Test: `tests/test_community_images_db.py`

**Context:** Follow the exact pattern from `py/services/civitai_stats_db.py` — singleton with `get_instance()`, lazy init, WAL mode, upsert with ON CONFLICT, chunked `get_by_hashes()`. The DB file is the same `civitai_stats.db` (shared with stats), so use `_default_db_path()` from `civitai_stats_db.py` or import the same helper. **Important:** Since two singleton classes share the same DB file with separate connections, set `PRAGMA busy_timeout=5000` after WAL mode to avoid `SQLITE_BUSY` errors during concurrent writes.

**Step 1: Write failing tests**

```python
# tests/test_community_images_db.py
"""Tests for CommunityImagesDB."""
import pytest
from py.services.community_images_db import CommunityImagesDB


@pytest.fixture
def db(tmp_path):
    instance = CommunityImagesDB(db_path=tmp_path / "test.db")
    instance.init()
    yield instance
    instance.close()


def test_upsert_and_get_by_hash(db):
    """Should store and retrieve a community image by model hash."""
    db.upsert({
        "civitai_image_id": 12345,
        "sha256": "abc123",
        "civitai_model_id": 42,
        "username": "testuser",
        "image_url": "https://example.com/img.jpg",
        "local_filename": "12345.jpg",
        "width": 1024,
        "height": 768,
        "prompt": "a beautiful landscape with mountains",
        "negative_prompt": "blurry",
        "steps": 30,
        "sampler": "DPM++ 2M Karras",
        "cfg_scale": 7.0,
        "seed": 123456,
        "denoise": 0.75,
        "base_model": "Pony",
        "like_count": 10,
        "heart_count": 5,
        "laugh_count": 2,
        "comment_count": 1,
        "created_at": "2026-01-15T10:00:00Z",
    })
    result = db.get_by_hashes(["abc123"])
    assert "abc123" in result
    images = result["abc123"]
    assert len(images) == 1
    assert images[0]["civitai_image_id"] == 12345
    assert images[0]["prompt"] == "a beautiful landscape with mountains"
    assert images[0]["like_count"] == 10


def test_upsert_batch(db):
    """Should store multiple images at once."""
    rows = [
        {
            "civitai_image_id": 1,
            "sha256": "hash_a",
            "civitai_model_id": 10,
            "username": "user1",
            "prompt": "test prompt one for image",
            "like_count": 5,
        },
        {
            "civitai_image_id": 2,
            "sha256": "hash_a",
            "civitai_model_id": 10,
            "username": "user2",
            "prompt": "test prompt two for image",
            "like_count": 3,
        },
        {
            "civitai_image_id": 3,
            "sha256": "hash_b",
            "civitai_model_id": 20,
            "username": "user3",
            "prompt": "another test prompt here",
            "like_count": 8,
        },
    ]
    db.upsert_batch(rows)
    result = db.get_by_hashes(["hash_a", "hash_b"])
    assert len(result["hash_a"]) == 2
    assert len(result["hash_b"]) == 1


def test_upsert_updates_existing(db):
    """Upserting same image_id should update, not duplicate."""
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "xyz",
        "like_count": 5,
        "prompt": "original prompt for testing",
    })
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "xyz",
        "like_count": 15,
        "prompt": "original prompt for testing",
    })
    result = db.get_by_hashes(["xyz"])
    assert len(result["xyz"]) == 1
    assert result["xyz"][0]["like_count"] == 15


def test_count(db):
    """Should count total images."""
    assert db.count() == 0
    db.upsert({"civitai_image_id": 1, "sha256": "a", "prompt": "test prompt long enough"})
    db.upsert({"civitai_image_id": 2, "sha256": "b", "prompt": "test prompt long enough"})
    assert db.count() == 2


def test_get_by_hashes_empty(db):
    """Empty hash list should return empty dict."""
    assert db.get_by_hashes([]) == {}


def test_get_hashes_with_images(db):
    """Should return set of sha256 values that have images."""
    db.upsert({"civitai_image_id": 1, "sha256": "aaa", "prompt": "test"})
    db.upsert({"civitai_image_id": 2, "sha256": "bbb", "prompt": "test"})
    result = db.get_hashes_with_images(["aaa", "bbb", "ccc"])
    assert result == {"aaa", "bbb"}


def test_delete_by_hash(db):
    """Should delete all images for a given hash."""
    db.upsert({"civitai_image_id": 1, "sha256": "del_me", "prompt": "test"})
    db.upsert({"civitai_image_id": 2, "sha256": "del_me", "prompt": "test"})
    db.upsert({"civitai_image_id": 3, "sha256": "keep_me", "prompt": "test"})
    db.delete_by_hash("del_me")
    assert db.get_by_hashes(["del_me"]) == {}
    assert len(db.get_by_hashes(["keep_me"])["keep_me"]) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_community_images_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'py.services.community_images_db'`

**Step 3: Write the implementation**

```python
# py/services/community_images_db.py
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


def _default_db_path() -> Path:
    """Return path to civitai_stats.db inside the cache directory."""
    from ..utils.cache_paths import get_cache_base_dir
    return Path(get_cache_base_dir()) / "civitai_stats.db"


class CommunityImagesDB:
    """Thin wrapper around SQLite for CivitAI community images."""

    _instance: CommunityImagesDB | None = None

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def get_instance(cls) -> CommunityImagesDB:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.init()
        return cls._instance

    def init(self) -> None:
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
        conn.execute(
            """INSERT INTO community_images
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
            """,
            (
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
                time.time(),
            ),
        )
        conn.commit()

    def upsert_batch(self, rows: list[dict]) -> None:
        """Insert or update multiple community images."""
        if not rows:
            return
        conn = self._ensure_conn()
        now = time.time()
        conn.executemany(
            """INSERT INTO community_images
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
            """,
            [
                (
                    d.get("civitai_image_id"),
                    d.get("sha256"),
                    d.get("civitai_model_id"),
                    d.get("username"),
                    d.get("image_url"),
                    d.get("local_filename"),
                    d.get("width"),
                    d.get("height"),
                    d.get("prompt"),
                    d.get("negative_prompt"),
                    d.get("steps"),
                    d.get("sampler"),
                    d.get("cfg_scale"),
                    d.get("seed"),
                    d.get("denoise"),
                    d.get("base_model"),
                    d.get("like_count", 0),
                    d.get("heart_count", 0),
                    d.get("laugh_count", 0),
                    d.get("comment_count", 0),
                    d.get("created_at"),
                    now,
                )
                for d in rows
            ],
        )
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
        conn = self._ensure_conn()
        return conn.execute("SELECT COUNT(*) FROM community_images").fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        if CommunityImagesDB._instance is self:
            CommunityImagesDB._instance = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_community_images_db.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add py/services/community_images_db.py tests/test_community_images_db.py
git commit -m "feat: add CommunityImagesDB for storing community creation metadata"
```

---

### Task 2: Community Images Fetch Service

**Files:**
- Create: `py/services/community_images_service.py`
- Test: `tests/test_community_images_service.py`

**Context:** This service calls `GET /api/v1/images?modelId={id}&sort=Most Reactions&limit=20` for each model, filters results (skip author, skip short/missing prompts), downloads image files, and stores metadata via `CommunityImagesDB`. Follow the pattern from `civitai_stats_service.py` — aiohttp session with rate limiting, progress callback, batch processing.

**Step 1: Write failing tests**

```python
# tests/test_community_images_service.py
"""Tests for CivitAI community images fetch service."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from py.services.community_images_service import (
    filter_community_images,
    CommunityImagesFetchService,
)
from py.services.community_images_db import CommunityImagesDB


# --- Sync tests for filter_community_images ---

def test_filter_excludes_author():
    """Should exclude images by the model author."""
    images = [
        {"username": "AuthorName", "meta": {"meta": {"prompt": "a detailed prompt with many words"}}, "stats": {}},
        {"username": "someone_else", "meta": {"meta": {"prompt": "another detailed prompt here"}}, "stats": {}},
    ]
    result = filter_community_images(images, author_username="AuthorName")
    assert len(result) == 1
    assert result[0]["username"] == "someone_else"


def test_filter_excludes_author_case_insensitive():
    """Author filtering should be case-insensitive."""
    images = [
        {"username": "AUTHORNAME", "meta": {"meta": {"prompt": "a detailed prompt with words"}}, "stats": {}},
    ]
    result = filter_community_images(images, author_username="authorname")
    assert len(result) == 0


def test_filter_excludes_missing_prompt():
    """Should exclude images without meta.meta.prompt."""
    images = [
        {"username": "user1", "meta": {}, "stats": {}},
        {"username": "user2", "meta": None, "stats": {}},
        {"username": "user3", "stats": {}},
    ]
    result = filter_community_images(images, author_username="nobody")
    assert len(result) == 0


def test_filter_excludes_short_prompt():
    """Should exclude images with prompt shorter than 20 chars."""
    images = [
        {"username": "user1", "meta": {"meta": {"prompt": "short"}}, "stats": {}},
        {"username": "user2", "meta": {"meta": {"prompt": "this is a long enough prompt for filtering"}}, "stats": {}},
    ]
    result = filter_community_images(images, author_username="nobody")
    assert len(result) == 1
    assert result[0]["username"] == "user2"


def test_filter_limits_to_10():
    """Should return at most 10 images."""
    images = [
        {"username": f"user{i}", "meta": {"meta": {"prompt": f"detailed prompt number {i} with enough length"}}, "stats": {}}
        for i in range(15)
    ]
    result = filter_community_images(images, author_username="nobody")
    assert len(result) == 10


# --- Async tests for CommunityImagesFetchService ---

@pytest.fixture
def img_db(tmp_path):
    db = CommunityImagesDB(db_path=tmp_path / "test.db")
    db.init()
    yield db
    db.close()


@pytest.mark.asyncio
async def test_fetch_images_for_model(img_db, tmp_path):
    """Should fetch images from API, filter, and store in DB."""
    api_response = {
        "items": [
            {
                "id": 999,
                "url": "https://image.civitai.com/test/image.jpg",
                "username": "community_user",
                "width": 1024,
                "height": 768,
                "baseModel": "SDXL",
                "createdAt": "2026-01-01T00:00:00Z",
                "stats": {"likeCount": 10, "heartCount": 5, "laughCount": 0, "commentCount": 2},
                "meta": {
                    "meta": {
                        "prompt": "a beautiful landscape with rolling hills and sunset",
                        "negativePrompt": "blurry, ugly",
                        "steps": 30,
                        "sampler": "DPM++ 2M Karras",
                        "cfgScale": 7,
                        "seed": 42,
                        "denoise": 0.75,
                    }
                },
            },
            {
                "id": 998,
                "url": "https://image.civitai.com/test/author.jpg",
                "username": "lora_author",
                "width": 512,
                "height": 512,
                "stats": {"likeCount": 100},
                "meta": {"meta": {"prompt": "author's own image with long prompt text"}},
            },
        ],
        "metadata": {},
    }

    service = CommunityImagesFetchService(db=img_db)

    with patch.object(service, "_fetch_images_api", new_callable=AsyncMock, return_value=api_response):
        with patch.object(service, "_download_image", new_callable=AsyncMock, return_value="local_hash/community/999.jpg"):
            count = await service.fetch_images_for_model(
                sha256="local_hash",
                civitai_model_id=42,
                author_username="lora_author",
            )

    assert count == 1  # Only community_user's image, not author's
    result = img_db.get_by_hashes(["local_hash"])
    assert len(result["local_hash"]) == 1
    img = result["local_hash"][0]
    assert img["civitai_image_id"] == 999
    assert img["username"] == "community_user"
    assert img["prompt"] == "a beautiful landscape with rolling hills and sunset"
    assert img["like_count"] == 10

    await service.close()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_community_images_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# py/services/community_images_service.py
"""Service for fetching CivitAI community images and storing them locally."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

from .community_images_db import CommunityImagesDB

logger = logging.getLogger(__name__)

_CIVITAI_API = "https://civitai.com/api/v1"
_RATE_LIMIT_DELAY = 1.5
_MIN_PROMPT_LENGTH = 20
_MAX_IMAGES_PER_MODEL = 10


def filter_community_images(
    items: list[dict],
    author_username: str | None = None,
) -> list[dict]:
    """Filter CivitAI image items: remove author, missing/short prompts.

    Returns at most _MAX_IMAGES_PER_MODEL images.
    """
    author_lower = (author_username or "").lower()
    filtered = []
    for item in items:
        # Skip author's own images
        username = (item.get("username") or "").lower()
        if author_lower and username == author_lower:
            continue
        # Must have meta.meta.prompt
        meta = item.get("meta")
        if not meta or not isinstance(meta, dict):
            continue
        inner = meta.get("meta")
        if not inner or not isinstance(inner, dict):
            continue
        prompt = inner.get("prompt") or ""
        if len(prompt) < _MIN_PROMPT_LENGTH:
            continue
        filtered.append(item)
        if len(filtered) >= _MAX_IMAGES_PER_MODEL:
            break
    return filtered


def _extract_image_data(item: dict, sha256: str, civitai_model_id: int) -> dict:
    """Convert a CivitAI image API item to our DB row format."""
    meta = (item.get("meta") or {}).get("meta") or {}
    stats = item.get("stats") or {}
    return {
        "civitai_image_id": item["id"],
        "sha256": sha256,
        "civitai_model_id": civitai_model_id,
        "username": item.get("username"),
        "image_url": item.get("url"),
        "width": item.get("width"),
        "height": item.get("height"),
        "prompt": meta.get("prompt"),
        "negative_prompt": meta.get("negativePrompt"),
        "steps": meta.get("steps"),
        "sampler": meta.get("sampler"),
        "cfg_scale": meta.get("cfgScale"),
        "seed": meta.get("seed"),
        "denoise": meta.get("denoise"),
        "base_model": item.get("baseModel"),
        "like_count": stats.get("likeCount", 0),
        "heart_count": stats.get("heartCount", 0),
        "laugh_count": stats.get("laughCount", 0),
        "comment_count": stats.get("commentCount", 0),
        "created_at": item.get("createdAt"),
    }


class CommunityImagesFetchService:
    """Fetches community images from CivitAI and stores them locally."""

    def __init__(
        self,
        db: CommunityImagesDB,
        api_key: str | None = None,
    ):
        self.db = db
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    async def _fetch_images_api(
        self, model_id: int, _retries: int = 2
    ) -> dict | None:
        """GET /api/v1/images?modelId={id}&sort=Most Reactions&limit=20"""
        url = f"{_CIVITAI_API}/images"
        params = {
            "modelId": str(model_id),
            "sort": "Most Reactions",
            "limit": "20",
        }
        session = await self._get_session()
        for attempt in range(_retries + 1):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning(
                            "CivitAI rate limited on images, backing off (attempt %d)",
                            attempt + 1,
                        )
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        logger.debug(
                            "CivitAI images returned %d for model %d",
                            resp.status,
                            model_id,
                        )
                        return None
                    return await resp.json()
            except Exception as exc:
                logger.warning(
                    "CivitAI images fetch failed for model %d: %s", model_id, exc
                )
                return None
        return None

    async def _download_image(
        self, image_url: str, sha256: str, image_id: int
    ) -> str | None:
        """Download an image file to {model_folder}/community/{image_id}.jpg.

        Uses get_model_folder() to resolve the correct path (respects
        library-scoped folders). Returns the local filename on success,
        None on failure.
        """
        from ..utils.example_images_paths import get_model_folder

        model_dir = get_model_folder(sha256)
        if not model_dir:
            return None
        dest_dir = Path(model_dir) / "community"
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{image_id}.jpg"
        dest_path = dest_dir / filename
        # Build relative path from static mount for URL construction
        from ..utils.example_images_paths import get_model_relative_path

        rel_path = get_model_relative_path(sha256)
        if not rel_path:
            return None
        relative_filename = f"{rel_path}/community/{filename}"
        if dest_path.exists():
            return relative_filename
        session = await self._get_session()
        try:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                dest_path.write_bytes(data)
                return relative_filename
        except Exception as exc:
            logger.debug("Failed to download image %d: %s", image_id, exc)
            return None

    async def fetch_images_for_model(
        self,
        sha256: str,
        civitai_model_id: int,
        author_username: str | None = None,
    ) -> int:
        """Fetch and store community images for a single model.

        Returns number of images stored.
        """
        data = await self._fetch_images_api(civitai_model_id)
        if not data:
            return 0
        items = data.get("items") or []
        filtered = filter_community_images(items, author_username=author_username)
        if not filtered:
            return 0

        rows = []
        for item in filtered:
            row = _extract_image_data(item, sha256, civitai_model_id)
            # Download image file
            image_url = item.get("url")
            if image_url:
                local_fn = await self._download_image(image_url, sha256, item["id"])
                row["local_filename"] = local_fn
            rows.append(row)

        self.db.upsert_batch(rows)
        return len(rows)

    async def fetch_all(
        self,
        models: list[dict],
        progress_callback: Optional[Callable] = None,
    ) -> int:
        """Bulk fetch community images for a list of models.

        Args:
            models: list of dicts with keys:
                sha256, civitai_model_id, civitai_creator_username (optional).
            progress_callback: optional async callable(current, total).

        Returns:
            Total number of images stored.
        """
        total = len(models)
        total_stored = 0
        api_success = 0
        api_fail = 0

        logger.info("Fetching community images for %d models", total)

        for i, model in enumerate(models):
            sha256 = model.get("sha256")
            model_id = model.get("civitai_model_id")
            author = model.get("civitai_creator_username")

            if not sha256 or not model_id:
                continue

            count = await self.fetch_images_for_model(
                sha256=sha256,
                civitai_model_id=model_id,
                author_username=author,
            )
            if count > 0:
                api_success += 1
                total_stored += count
            else:
                api_fail += 1

            if progress_callback:
                await progress_callback(i + 1, total)

            if i < total - 1:
                await asyncio.sleep(_RATE_LIMIT_DELAY)

        logger.info(
            "Community images fetch complete: %d/%d models had images, %d images stored",
            api_success,
            total,
            total_stored,
        )
        return total_stored

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_community_images_service.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add py/services/community_images_service.py tests/test_community_images_service.py
git commit -m "feat: add CommunityImagesFetchService for fetching community creations from CivitAI"
```

---

### Task 3: Community Images Route Handlers

**Files:**
- Create: `py/routes/community_images_routes.py`
- Modify: `py/lora_manager.py:170-173` — add route registration
- Modify: `standalone.py:325-342` — add route registration

**Context:** Follow the pattern from `civitai_stats_routes.py` — static methods on a class, `setup_routes(cls, app)`, WebSocket progress broadcast. The page route follows `stats_routes.py` pattern — Jinja2 template rendering with i18n.

**Step 1: Write the route handlers**

```python
# py/routes/community_images_routes.py
"""Route registrar for Community Creations endpoints."""
from __future__ import annotations

import logging
import jinja2
from aiohttp import web

from ..config import config
from ..services.community_images_db import CommunityImagesDB
from ..services.community_images_service import CommunityImagesFetchService
from ..services.service_registry import ServiceRegistry
from ..services.settings_manager import get_settings_manager
from ..services.server_i18n import server_i18n
from ..services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)


class CommunityImagesRoutes:
    """Route handlers for Community Creations page and API."""

    _template_env: jinja2.Environment | None = None

    @classmethod
    def _get_template_env(cls) -> jinja2.Environment:
        if cls._template_env is None:
            cls._template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(config.templates_path),
                autoescape=True,
            )
        return cls._template_env

    @staticmethod
    async def handle_page(request: web.Request) -> web.Response:
        """GET /community — render the Community Creations page."""
        try:
            settings = get_settings_manager()
            user_language = settings.get("language", "en")
            server_i18n.set_locale(user_language)

            env = CommunityImagesRoutes._get_template_env()
            if not hasattr(env, "_i18n_filter_added"):
                env.filters["t"] = server_i18n.create_template_filter()
                env._i18n_filter_added = True

            template = env.get_template("community_creations.html")
            rendered = template.render(
                request=request,
                settings=settings,
                t=server_i18n.get_translation,
                is_initializing=False,
            )
            return web.Response(text=rendered, content_type="text/html")
        except Exception as exc:
            logger.exception("Error rendering community creations page")
            return web.Response(text="Error loading page", status=500)

    @staticmethod
    async def handle_fetch(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/fetch — trigger bulk fetch."""
        try:
            db = CommunityImagesDB.get_instance()
            settings = get_settings_manager()
            api_key = settings.get("civitai_api_key", "")

            # Collect models with civitai data from all scanners
            models = []
            scanner_names = ["lora", "checkpoint", "embedding"]
            for name, getter in zip(
                scanner_names,
                [
                    ServiceRegistry.get_lora_scanner,
                    ServiceRegistry.get_checkpoint_scanner,
                    ServiceRegistry.get_embedding_scanner,
                ],
            ):
                try:
                    scanner = await getter()
                    cache = await scanner.get_cached_data()
                    for item in cache.raw_data:
                        if not item:
                            continue
                        civitai = item.get("civitai") or {}
                        model_id = civitai.get("modelId")
                        sha256 = item.get("sha256")
                        creator = (civitai.get("creator") or {}).get("username")
                        if model_id and sha256:
                            models.append({
                                "sha256": sha256,
                                "civitai_model_id": model_id,
                                "civitai_creator_username": creator,
                            })
                except Exception as exc:
                    logger.info("Failed to get %s scanner data: %s", name, exc)

            if not models:
                return web.json_response({"success": True, "stored": 0, "total": 0})

            # Check for force refresh option
            try:
                body = await request.json() if request.can_read_body else {}
            except Exception:
                body = {}
            force = body.get("force", False)

            # Skip models that already have community images (unless force)
            existing = set()
            if not force:
                all_hashes = [m["sha256"] for m in models]
                existing = db.get_hashes_with_images(all_hashes)
                models = [m for m in models if m["sha256"] not in existing]

            logger.info(
                "Fetching community images for %d models (%d already have images)",
                len(models),
                len(existing),
            )

            if not models:
                return web.json_response({
                    "success": True,
                    "stored": 0,
                    "total": 0,
                    "skipped": len(existing),
                })

            service = CommunityImagesFetchService(db=db, api_key=api_key)

            async def progress_callback(current: int, total: int) -> None:
                await ws_manager.broadcast({
                    "type": "community_images_progress",
                    "current": current,
                    "total": total,
                    "progress": round(current / total * 100) if total else 0,
                })

            try:
                stored = await service.fetch_all(models, progress_callback=progress_callback)
            finally:
                await service.close()

            return web.json_response({
                "success": True,
                "stored": stored,
                "total": len(models),
                "skipped": len(existing),
            })
        except Exception as exc:
            logger.exception("Community images fetch failed")
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    @staticmethod
    async def handle_by_hashes(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/by-hashes — get images for given hashes."""
        try:
            body = await request.json()
            hashes = body.get("hashes", [])
        except Exception:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"}, status=400
            )
        if not isinstance(hashes, list) or len(hashes) > 5000:
            return web.json_response(
                {"success": False, "error": "Invalid or too many hashes (max 5000)"},
                status=400,
            )
        if not hashes:
            return web.json_response({"success": True, "images": {}})

        db = CommunityImagesDB.get_instance()
        images = db.get_by_hashes(hashes)

        # Clean for JSON — remove fetched_at
        clean: dict[str, list[dict]] = {}
        for sha, img_list in images.items():
            clean[sha] = [
                {k: v for k, v in img.items() if k != "fetched_at"}
                for img in img_list
            ]
        return web.json_response({"success": True, "images": clean})

    @staticmethod
    async def handle_status(request: web.Request) -> web.Response:
        """GET /api/lm/community-images/status — DB count."""
        db = CommunityImagesDB.get_instance()
        return web.json_response({
            "success": True,
            "count": db.count(),
        })

    @classmethod
    def setup_routes(cls, app: web.Application) -> None:
        """Register community images routes."""
        app.router.add_get("/community", cls.handle_page)
        app.router.add_post("/api/lm/community-images/fetch", cls.handle_fetch)
        app.router.add_post("/api/lm/community-images/by-hashes", cls.handle_by_hashes)
        app.router.add_get("/api/lm/community-images/status", cls.handle_status)

        async def cleanup(app):
            instance = CommunityImagesDB._instance
            if instance:
                instance.close()
                CommunityImagesDB._instance = None

        app.on_shutdown.append(cleanup)
        logger.info("Community images routes registered")
```

**Step 2: Register routes in lora_manager.py**

In `py/lora_manager.py`, after the CivitaiStatsRoutes import (around line 96), add:
```python
from .routes.community_images_routes import CommunityImagesRoutes
```

After the CivitaiStatsRoutes.setup_routes(app) block (around line 173), add:
```python
try:
    CommunityImagesRoutes.setup_routes(app)
except Exception:
    logger.exception("Failed to register community images routes")
```

**Step 3: Register routes in standalone.py**

In `standalone.py`, after the CivitaiStatsRoutes import (around line 325), add:
```python
from py.routes.community_images_routes import CommunityImagesRoutes
```

After `CivitaiStatsRoutes.setup_routes(app)` (around line 342), add:
```python
CommunityImagesRoutes.setup_routes(app)
```

**Step 4: Run existing tests to verify nothing broke**

Run: `pytest tests/ -v --ignore=tests/frontend -x`
Expected: All existing tests PASS

**Step 5: Commit**

```bash
git add py/routes/community_images_routes.py py/lora_manager.py standalone.py
git commit -m "feat: add community images route handlers and register in both server modes"
```

---

### Task 4: Community Creations HTML Template

**Files:**
- Create: `templates/community_creations.html`
- Modify: `templates/components/header.html:25-44` — add nav tab

**Context:** Follow the pattern from `templates/recipes.html` — extends `base.html`, sets `page_id`, includes page CSS, has a card grid container. The header nav needs a new tab entry matching the existing pattern.

**Step 1: Add nav tab to header**

In `templates/components/header.html`, after the statistics nav item (line 43-44), add:
```html
<a href="/community" class="nav-item{% if current_path.startswith('/community') %} active{% endif %}"
  id="communityNavItem">
  <i class="fas fa-images"></i> <span>Community</span>
</a>
```

Also update the `current_page` detection block (lines 10-19) to add before `{% else %}`:
```
{% elif current_path.startswith('/community') %}
{% set current_page = 'community' %}
```

Also update the `search_disabled` line (line 21) to disable search on the community page:
```
{% set search_disabled = current_page == 'statistics' or current_page == 'community' %}
```

**Step 2: Create the page template**

```html
{# templates/community_creations.html #}
{% extends "base.html" %}

{% block title %}Community Creations{% endblock %}
{% block page_id %}community{% endblock %}

{% block page_css %}
<link rel="stylesheet" href="/loras_static/css/community_creations.css?v={{ version }}">
{% endblock %}

{% block content %}
<div class="controls">
    <div class="actions">
        <div class="action-buttons">
            <div class="control-group">
                <button id="fetchCommunityBtn" data-action="fetch-community"
                        title="Fetch community images from CivitAI">
                    <i class="fas fa-images"></i> <span>Fetch Community Images</span>
                </button>
            </div>
            <div class="control-group">
                <select id="communitySortSelect" title="Sort community images">
                    <optgroup label="Sort by">
                        <option value="reactions:desc">Most Liked</option>
                        <option value="reactions:asc">Least Liked</option>
                        <option value="recent:desc">Most Recent</option>
                        <option value="lora:asc">LoRA Name (A-Z)</option>
                        <option value="lora:desc">LoRA Name (Z-A)</option>
                    </optgroup>
                </select>
            </div>
        </div>
    </div>
</div>

<div id="communityGrid" class="community-grid">
    <!-- Cards rendered by JS -->
</div>

<div id="communityEmpty" class="community-empty" style="display:none;">
    <i class="fas fa-images"></i>
    <h3>No community images yet</h3>
    <p>Click "Fetch Community Images" to download top community creations for your LoRAs.</p>
</div>
{% endblock %}

{% block main_script %}
<script type="module" src="/loras_static/js/community_creations.js?v={{ version }}"></script>
{% endblock %}
```

**Step 3: Commit**

```bash
git add templates/community_creations.html templates/components/header.html
git commit -m "feat: add Community Creations page template and nav tab"
```

---

### Task 5: Community Creations CSS

**Files:**
- Create: `static/css/community_creations.css`

**Context:** Style the card grid, image cards, prompt sections, and reaction badges. Follow the visual style from existing `card.css` but adapted for image-focused cards with prompt text.

**Step 1: Create the stylesheet**

```css
/* static/css/community_creations.css */

/* ── Grid layout ─────────────────────────────────────────── */
.community-grid {
    padding: 16px;
}

.community-lora-group {
    margin-bottom: 32px;
}

.community-lora-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 4px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.1));
}

.community-lora-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary, #fff);
}

.community-lora-header .lora-link {
    color: var(--text-secondary, rgba(255,255,255,0.6));
    font-size: 12px;
    text-decoration: none;
    cursor: pointer;
}

.community-lora-header .lora-link:hover {
    color: var(--accent-color, #7c3aed);
}

.community-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
}

/* ── Image card ──────────────────────────────────────────── */
.community-card {
    background: var(--card-bg, #1e1e2e);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-color, rgba(255,255,255,0.08));
    transition: transform 0.15s, box-shadow 0.15s;
    cursor: pointer;
}

.community-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.community-card-image {
    width: 100%;
    aspect-ratio: 4/3;
    object-fit: cover;
    display: block;
    background: var(--bg-secondary, #111);
}

.community-card-body {
    padding: 10px 12px;
}

.community-card-prompt {
    font-size: 12px;
    color: var(--text-secondary, rgba(255,255,255,0.7));
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 8px;
    word-break: break-word;
}

.community-card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}

.community-meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    background: rgba(255,255,255,0.06);
    color: var(--text-secondary, rgba(255,255,255,0.5));
}

.community-card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 6px;
    border-top: 1px solid var(--border-color, rgba(255,255,255,0.06));
}

.community-card-reactions {
    display: flex;
    gap: 8px;
}

.community-reaction {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 11px;
    color: var(--text-secondary, rgba(255,255,255,0.5));
}

.community-reaction i {
    font-size: 10px;
}

.community-card-user {
    font-size: 10px;
    color: var(--text-secondary, rgba(255,255,255,0.4));
}

/* ── Expanded card / modal ───────────────────────────────── */
.community-detail-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.8);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
}

.community-detail {
    background: var(--card-bg, #1e1e2e);
    border-radius: 12px;
    max-width: 900px;
    max-height: 90vh;
    overflow-y: auto;
    width: 100%;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}

.community-detail-image {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #000;
    border-radius: 12px 0 0 12px;
    max-height: 80vh;
}

.community-detail-info {
    padding: 20px;
    overflow-y: auto;
}

.community-detail-info h4 {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--text-primary, #fff);
}

.community-detail-prompt {
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-secondary, rgba(255,255,255,0.7));
    background: rgba(0,0,0,0.2);
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 16px;
    white-space: pre-wrap;
    word-break: break-word;
    position: relative;
}

.community-detail-prompt .copy-btn {
    position: absolute;
    top: 6px;
    right: 6px;
    background: rgba(255,255,255,0.1);
    border: none;
    color: rgba(255,255,255,0.5);
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
}

.community-detail-prompt .copy-btn:hover {
    background: rgba(255,255,255,0.2);
    color: #fff;
}

.community-detail-params {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-bottom: 16px;
}

.community-detail-param {
    font-size: 12px;
    color: var(--text-secondary, rgba(255,255,255,0.5));
}

.community-detail-param strong {
    color: var(--text-primary, rgba(255,255,255,0.7));
}

/* ── Empty state ─────────────────────────────────────────── */
.community-empty {
    text-align: center;
    padding: 80px 20px;
    color: var(--text-secondary, rgba(255,255,255,0.4));
}

.community-empty i {
    font-size: 48px;
    margin-bottom: 16px;
    display: block;
}

.community-empty h3 {
    margin: 0 0 8px;
    color: var(--text-primary, rgba(255,255,255,0.6));
}

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 768px) {
    .community-detail {
        grid-template-columns: 1fr;
    }
    .community-detail-image {
        border-radius: 12px 12px 0 0;
        max-height: 40vh;
    }
}
```

**Step 2: Commit**

```bash
git add static/css/community_creations.css
git commit -m "feat: add Community Creations page styles"
```

---

### Task 6: Community Creations JavaScript

**Files:**
- Create: `static/js/community_creations.js`

**Context:** ES module loaded by the template. Fetches community images from `/api/lm/community-images/by-hashes` (collecting hashes from scanner data via a list endpoint), renders cards grouped by LoRA name, handles detail modal on click, fetch button, sort. Follow the pattern from `civitai_stats_ui.js` for fetch button states and `recipes.js` for page initialization.

**Step 1: Create the JS module**

```javascript
// static/js/community_creations.js
/**
 * Community Creations page — card grid of community images grouped by LoRA.
 */

// ── State ────────────────────────────────────────────────
let _allImages = {};   // sha256 → [{image}, ...]
let _modelNames = {};  // sha256 → model_name
let _sortKey = "reactions:desc";

// ── Init ─────────────────────────────────────────────────
async function init() {
    setupFetchButton();
    setupSortSelect();
    await loadImages();
}

// ── Load images from API ─────────────────────────────────
async function loadImages() {
    try {
        // Paginate through all lora list pages to collect hashes + names
        // (server caps page_size at 100)
        const hashes = [];
        let page = 1;
        while (true) {
            const listResp = await fetch(`/api/lm/loras/list?page=${page}&page_size=100`);
            const listData = await listResp.json();
            const items = listData.items || [];
            if (items.length === 0) break;

            for (const item of items) {
                if (item.sha256) {
                    hashes.push(item.sha256);
                    _modelNames[item.sha256] = item.model_name || item.file_name || "Unknown";
                }
            }

            // Stop if we got fewer than page_size (last page)
            if (items.length < 100) break;
            page++;
        }

        if (hashes.length === 0) {
            showEmpty();
            return;
        }

        // Fetch community images for all hashes
        const resp = await fetch("/api/lm/community-images/by-hashes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ hashes }),
        });
        const data = await resp.json();
        if (data.success && data.images) {
            _allImages = data.images;
        }

        if (Object.keys(_allImages).length === 0) {
            showEmpty();
        } else {
            renderGrid();
        }
    } catch (err) {
        console.error("[Community] Failed to load images:", err);
        showEmpty();
    }
}

// ── Render grid ──────────────────────────────────────────
function renderGrid() {
    const grid = document.getElementById("communityGrid");
    const empty = document.getElementById("communityEmpty");
    if (!grid) return;

    grid.innerHTML = "";
    empty.style.display = "none";

    // Build sorted groups
    const groups = buildSortedGroups();

    if (groups.length === 0) {
        showEmpty();
        return;
    }

    for (const group of groups) {
        const section = document.createElement("div");
        section.className = "community-lora-group";

        // Header
        const header = document.createElement("div");
        header.className = "community-lora-header";
        header.innerHTML = `<h3>${escapeHtml(group.name)}</h3>
            <span class="lora-link">${group.images.length} image${group.images.length !== 1 ? "s" : ""}</span>`;
        section.appendChild(header);

        // Cards
        const cardsDiv = document.createElement("div");
        cardsDiv.className = "community-cards";

        for (const img of group.images) {
            cardsDiv.appendChild(createCard(img, group.sha256));
        }

        section.appendChild(cardsDiv);
        grid.appendChild(section);
    }
}

function buildSortedGroups() {
    const groups = [];
    for (const [sha256, images] of Object.entries(_allImages)) {
        if (!images || images.length === 0) continue;
        const name = _modelNames[sha256] || "Unknown";

        // Sort images within group by reactions
        const sorted = [...images].sort((a, b) => {
            const ra = (a.like_count || 0) + (a.heart_count || 0);
            const rb = (b.like_count || 0) + (b.heart_count || 0);
            return rb - ra;
        });

        groups.push({ sha256, name, images: sorted });
    }

    // Sort groups
    const [key, dir] = _sortKey.split(":");
    const asc = dir === "asc" ? 1 : -1;

    if (key === "reactions") {
        groups.sort((a, b) => {
            const ra = a.images.reduce((s, i) => s + (i.like_count || 0) + (i.heart_count || 0), 0);
            const rb = b.images.reduce((s, i) => s + (i.like_count || 0) + (i.heart_count || 0), 0);
            return (rb - ra) * asc;
        });
    } else if (key === "recent") {
        groups.sort((a, b) => {
            const da = a.images[0]?.created_at || "";
            const db_ = b.images[0]?.created_at || "";
            return db_.localeCompare(da) * asc;
        });
    } else if (key === "lora") {
        groups.sort((a, b) => a.name.localeCompare(b.name) * asc);
    }

    return groups;
}

// ── Card creation ────────────────────────────────────────
function createCard(img, sha256) {
    const card = document.createElement("div");
    card.className = "community-card";
    card.addEventListener("click", () => showDetail(img, sha256));

    // Image
    const imgUrl = img.local_filename
        ? `/example_images_static/${img.local_filename}`
        : img.image_url || "";

    card.innerHTML = `
        <img class="community-card-image" src="${escapeHtml(imgUrl)}"
             alt="Community creation" loading="lazy"
             onerror="this.style.display='none'">
        <div class="community-card-body">
            <div class="community-card-prompt">${escapeHtml(img.prompt || "")}</div>
            <div class="community-card-meta">
                ${img.sampler ? `<span class="community-meta-tag">${escapeHtml(img.sampler)}</span>` : ""}
                ${img.steps ? `<span class="community-meta-tag">${img.steps} steps</span>` : ""}
                ${img.cfg_scale ? `<span class="community-meta-tag">CFG ${img.cfg_scale}</span>` : ""}
                ${img.base_model ? `<span class="community-meta-tag">${escapeHtml(img.base_model)}</span>` : ""}
            </div>
            <div class="community-card-footer">
                <div class="community-card-reactions">
                    ${img.like_count ? `<span class="community-reaction"><i class="fas fa-thumbs-up"></i> ${img.like_count}</span>` : ""}
                    ${img.heart_count ? `<span class="community-reaction"><i class="fas fa-heart"></i> ${img.heart_count}</span>` : ""}
                    ${img.comment_count ? `<span class="community-reaction"><i class="fas fa-comment"></i> ${img.comment_count}</span>` : ""}
                </div>
                <span class="community-card-user">${escapeHtml(img.username || "")}</span>
            </div>
        </div>
    `;
    return card;
}

// ── Detail modal ─────────────────────────────────────────
function showDetail(img, sha256) {
    // Remove existing overlay
    const existing = document.querySelector(".community-detail-overlay");
    if (existing) existing.remove();

    const imgUrl = img.local_filename
        ? `/example_images_static/${img.local_filename}`
        : img.image_url || "";

    const overlay = document.createElement("div");
    overlay.className = "community-detail-overlay";
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.remove();
    });

    overlay.innerHTML = `
        <div class="community-detail">
            <img class="community-detail-image" src="${escapeHtml(imgUrl)}" alt="Community creation">
            <div class="community-detail-info">
                <h4>Prompt</h4>
                <div class="community-detail-prompt">
                    <button class="copy-btn" title="Copy prompt"><i class="fas fa-copy"></i> Copy</button>
                    ${escapeHtml(img.prompt || "")}
                </div>
                ${img.negative_prompt ? `
                    <h4>Negative Prompt</h4>
                    <div class="community-detail-prompt">${escapeHtml(img.negative_prompt)}</div>
                ` : ""}
                <h4>Parameters</h4>
                <div class="community-detail-params">
                    ${img.steps ? `<div class="community-detail-param"><strong>Steps:</strong> ${img.steps}</div>` : ""}
                    ${img.sampler ? `<div class="community-detail-param"><strong>Sampler:</strong> ${escapeHtml(img.sampler)}</div>` : ""}
                    ${img.cfg_scale ? `<div class="community-detail-param"><strong>CFG Scale:</strong> ${img.cfg_scale}</div>` : ""}
                    ${img.seed ? `<div class="community-detail-param"><strong>Seed:</strong> ${img.seed}</div>` : ""}
                    ${img.denoise ? `<div class="community-detail-param"><strong>Denoise:</strong> ${img.denoise}</div>` : ""}
                    ${img.base_model ? `<div class="community-detail-param"><strong>Base Model:</strong> ${escapeHtml(img.base_model)}</div>` : ""}
                    ${img.width && img.height ? `<div class="community-detail-param"><strong>Size:</strong> ${img.width}x${img.height}</div>` : ""}
                </div>
                <div class="community-card-reactions" style="margin-top:12px;">
                    ${img.like_count ? `<span class="community-reaction"><i class="fas fa-thumbs-up"></i> ${img.like_count}</span>` : ""}
                    ${img.heart_count ? `<span class="community-reaction"><i class="fas fa-heart"></i> ${img.heart_count}</span>` : ""}
                    ${img.laugh_count ? `<span class="community-reaction"><i class="fas fa-laugh"></i> ${img.laugh_count}</span>` : ""}
                    ${img.comment_count ? `<span class="community-reaction"><i class="fas fa-comment"></i> ${img.comment_count}</span>` : ""}
                </div>
                <div class="community-card-user" style="margin-top:8px;">
                    by ${escapeHtml(img.username || "unknown")}
                    ${img.created_at ? ` &middot; ${new Date(img.created_at).toLocaleDateString()}` : ""}
                </div>
            </div>
        </div>
    `;

    // Copy button handler
    document.body.appendChild(overlay);
    const copyBtn = overlay.querySelector(".copy-btn");
    if (copyBtn) {
        copyBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(img.prompt || "").then(() => {
                copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                }, 2000);
            });
        });
    }

    // Close on Escape
    const escHandler = (e) => {
        if (e.key === "Escape") {
            overlay.remove();
            document.removeEventListener("keydown", escHandler);
        }
    };
    document.addEventListener("keydown", escHandler);
}

// ── Fetch button ─────────────────────────────────────────
function setupFetchButton() {
    const btn = document.getElementById("fetchCommunityBtn");
    if (!btn) return;

    btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Fetching...</span>';

        try {
            const resp = await fetch("/api/lm/community-images/fetch", { method: "POST" });
            const data = await resp.json();
            if (data.success) {
                const count = data.stored || 0;
                btn.innerHTML = `<i class="fas fa-check"></i> <span>${count} images saved</span>`;
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-images"></i> <span>Fetch Community Images</span>';
                    btn.disabled = false;
                }, 3000);
                // Reload images
                await loadImages();
            } else {
                throw new Error(data.error || "Unknown error");
            }
        } catch (err) {
            btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Error</span>';
            btn.title = err.message || String(err);
            console.error("[Community] Fetch failed:", err);
            setTimeout(() => {
                btn.innerHTML = '<i class="fas fa-images"></i> <span>Fetch Community Images</span>';
                btn.title = "Fetch community images from CivitAI";
                btn.disabled = false;
            }, 5000);
        }
    });
}

// ── Sort select ──────────────────────────────────────────
function setupSortSelect() {
    const select = document.getElementById("communitySortSelect");
    if (!select) return;
    select.addEventListener("change", () => {
        _sortKey = select.value;
        renderGrid();
    });
}

// ── Helpers ──────────────────────────────────────────────
function showEmpty() {
    const grid = document.getElementById("communityGrid");
    const empty = document.getElementById("communityEmpty");
    if (grid) grid.innerHTML = "";
    if (empty) empty.style.display = "";
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Start ────────────────────────────────────────────────
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
```

**Step 2: Commit**

```bash
git add static/js/community_creations.js
git commit -m "feat: add Community Creations page JavaScript"
```

---

### Task 7: Integration Test — End-to-End Verification

**Files:**
- No new files — manual verification steps

**Step 1: Run all backend tests**

Run: `pytest tests/ -v --ignore=tests/frontend -x`
Expected: All tests PASS

**Step 2: Run frontend tests**

Run: `npm test`
Expected: All tests PASS

**Step 3: Start standalone server and verify**

Run: `python standalone.py --port 8188`

Verify in browser:
1. Navigate to `http://localhost:8188/community` — page should load with empty state
2. Header nav should show "Community" tab highlighted
3. Click "Fetch Community Images" — should start fetching with spinner
4. After fetch, cards should appear grouped by LoRA name
5. Click a card — detail modal should show full prompt, params, reactions
6. Copy button should copy prompt to clipboard
7. Sort dropdown should reorder groups

**Step 4: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: integration fixes for community creations feature"
```
