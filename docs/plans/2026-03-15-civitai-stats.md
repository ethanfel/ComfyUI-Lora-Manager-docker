# CivitAI Stats Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show CivitAI community stats (downloads, rating, thumbs up) on model cards with sorting support, ported from LM-Remote into the main codebase.

**Architecture:** Isolated layer of new files (separate SQLite DB, dedicated service/routes, self-contained JS) with 3 small changes to existing upstream files. Frontend uses fetch() interception and MutationObserver to avoid modifying upstream card rendering code. Backend sort support via small `elif` addition in `model_cache.py`.

**Tech Stack:** Python/aiohttp backend, SQLite, CivitAI API v1, vanilla JS frontend

---

### Task 1: CivitAI Stats SQLite DB Layer

**Files:**
- Create: `py/services/civitai_stats_db.py`
- Test: `tests/test_civitai_stats_db.py`

**Step 1: Write the failing test**

Create `tests/test_civitai_stats_db.py`:

```python
"""Tests for CivitAI stats SQLite database layer."""
import pytest
from pathlib import Path
from py.services.civitai_stats_db import CivitaiStatsDB


@pytest.fixture
def stats_db(tmp_path):
    db = CivitaiStatsDB(db_path=tmp_path / "test_stats.db")
    db.init()
    yield db
    db.close()


def test_init_creates_table(stats_db):
    """DB should create model_stats table on init."""
    assert stats_db.count() == 0


def test_upsert_and_get(stats_db):
    """Single upsert should be retrievable by hash."""
    stats_db.upsert("abc123", {
        "civitai_model_id": 1,
        "civitai_version_id": 10,
        "download_count": 500,
        "rating": 4.5,
        "rating_count": 100,
        "thumbs_up_count": 80,
    })
    result = stats_db.get_by_hashes(["abc123"])
    assert "abc123" in result
    assert result["abc123"]["download_count"] == 500
    assert result["abc123"]["rating"] == 4.5


def test_upsert_updates_existing(stats_db):
    """Upserting same hash should update values."""
    stats_db.upsert("abc123", {"download_count": 100})
    stats_db.upsert("abc123", {"download_count": 200})
    result = stats_db.get_by_hashes(["abc123"])
    assert result["abc123"]["download_count"] == 200


def test_upsert_batch(stats_db):
    """Batch upsert should insert multiple rows."""
    rows = [
        ("hash1", {"download_count": 10, "civitai_model_id": 1}),
        ("hash2", {"download_count": 20, "civitai_model_id": 2}),
    ]
    stats_db.upsert_batch(rows)
    assert stats_db.count() == 2
    result = stats_db.get_by_hashes(["hash1", "hash2"])
    assert result["hash1"]["download_count"] == 10
    assert result["hash2"]["download_count"] == 20


def test_upsert_batch_empty(stats_db):
    """Batch upsert with empty list should be a no-op."""
    stats_db.upsert_batch([])
    assert stats_db.count() == 0


def test_get_all(stats_db):
    """get_all should return all rows."""
    stats_db.upsert("a", {"download_count": 1})
    stats_db.upsert("b", {"download_count": 2})
    all_stats = stats_db.get_all()
    assert len(all_stats) == 2
    assert "a" in all_stats
    assert "b" in all_stats


def test_get_by_hashes_empty(stats_db):
    """get_by_hashes with empty list should return empty dict."""
    assert stats_db.get_by_hashes([]) == {}


def test_count(stats_db):
    """count should return number of rows."""
    assert stats_db.count() == 0
    stats_db.upsert("x", {"download_count": 1})
    assert stats_db.count() == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_civitai_stats_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'py.services.civitai_stats_db'`

**Step 3: Write minimal implementation**

Create `py/services/civitai_stats_db.py` — adapted from `custom_nodes/ComfyUI-LM-Remote/stats_db.py`:

```python
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
```

Note: The test imports `from py.services.civitai_stats_db import CivitaiStatsDB`. The existing test infrastructure in `tests/conftest.py` already handles `sys.path` setup. The `_default_db_path()` uses a lazy import of `get_cache_base_dir()` so it's only called when no explicit `db_path` is provided — test fixtures always pass an explicit `db_path` to avoid this.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_civitai_stats_db.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add py/services/civitai_stats_db.py tests/test_civitai_stats_db.py
git commit -m "feat: add CivitAI stats SQLite database layer"
```

---

### Task 2: CivitAI Stats Fetch Service

**Files:**
- Create: `py/services/civitai_stats_service.py`
- Test: `tests/test_civitai_stats_service.py`

**Step 1: Write the failing test**

Create `tests/test_civitai_stats_service.py`:

```python
"""Tests for CivitAI stats fetch service."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from py.services.civitai_stats_service import extract_version_stats, CivitaiStatsFetchService
from py.services.civitai_stats_db import CivitaiStatsDB


# --- Sync tests for extract_version_stats ---

def test_extract_version_stats_basic():
    """Should extract sha256 and stats from model response."""
    model_data = {
        "id": 42,
        "modelVersions": [
            {
                "id": 100,
                "files": [{"hashes": {"SHA256": "ABCDEF1234567890"}}],
                "stats": {
                    "downloadCount": 1500,
                    "rating": 4.8,
                    "ratingCount": 200,
                    "thumbsUpCount": 180,
                },
            }
        ],
    }
    result = extract_version_stats(model_data)
    assert len(result) == 1
    sha, stats = result[0]
    assert sha == "abcdef1234567890"  # lowercased
    assert stats["download_count"] == 1500
    assert stats["rating"] == 4.8
    assert stats["civitai_model_id"] == 42
    assert stats["civitai_version_id"] == 100


def test_extract_version_stats_no_hash():
    """Versions without SHA256 should be skipped."""
    model_data = {
        "id": 1,
        "modelVersions": [
            {"id": 10, "files": [{"hashes": {}}], "stats": {"downloadCount": 5}},
        ],
    }
    assert extract_version_stats(model_data) == []


def test_extract_version_stats_multiple_versions():
    """Should extract one entry per version with a hash."""
    model_data = {
        "id": 1,
        "modelVersions": [
            {
                "id": 10,
                "files": [{"hashes": {"SHA256": "AAA"}}],
                "stats": {"downloadCount": 10},
            },
            {
                "id": 20,
                "files": [{"hashes": {"SHA256": "BBB"}}],
                "stats": {"downloadCount": 20},
            },
        ],
    }
    result = extract_version_stats(model_data)
    assert len(result) == 2


# --- Async tests for CivitaiStatsFetchService ---

@pytest.fixture
def stats_db(tmp_path):
    db = CivitaiStatsDB(db_path=tmp_path / "test.db")
    db.init()
    yield db
    db.close()


@pytest.mark.asyncio
async def test_fetch_stats_for_models(stats_db):
    """Should fetch from CivitAI API and store stats in DB."""
    mock_response = {
        "id": 42,
        "modelVersions": [
            {
                "id": 100,
                "files": [{"hashes": {"SHA256": "ABC123"}}],
                "stats": {"downloadCount": 999, "rating": 4.5, "ratingCount": 50, "thumbsUpCount": 40},
            }
        ],
    }

    service = CivitaiStatsFetchService(db=stats_db)

    with patch.object(service, "_fetch_model", new_callable=AsyncMock, return_value=mock_response):
        updated = await service.fetch_stats_for_models(
            [{"sha256": "abc123", "civitai_model_id": 42}]
        )

    assert updated == 1
    result = stats_db.get_by_hashes(["abc123"])
    assert result["abc123"]["download_count"] == 999

    await service.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_civitai_stats_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'py.services.civitai_stats_service'`

**Step 3: Write minimal implementation**

Create `py/services/civitai_stats_service.py` — adapted from `custom_nodes/ComfyUI-LM-Remote/stats_service.py`:

```python
"""Service for fetching CivitAI community stats and storing them locally."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import aiohttp

from .civitai_stats_db import CivitaiStatsDB

logger = logging.getLogger(__name__)

_CIVITAI_API = "https://civitai.com/api/v1"
_RATE_LIMIT_DELAY = 1.5  # seconds between requests


def extract_version_stats(model_data: dict) -> list[tuple[str, dict]]:
    """Extract (sha256, stats_dict) pairs from a CivitAI model response.

    Each model has multiple versions; each version has files with hashes.
    Returns one entry per version that has a SHA256 hash.
    """
    model_id = model_data.get("id")
    results = []

    for version in model_data.get("modelVersions", []):
        version_id = version.get("id")
        sha256 = None
        for f in version.get("files", []):
            sha256 = (f.get("hashes") or {}).get("SHA256")
            if sha256:
                break
        if not sha256:
            continue

        stats = version.get("stats") or {}
        results.append((
            sha256.lower(),
            {
                "civitai_model_id": model_id,
                "civitai_version_id": version_id,
                "download_count": stats.get("downloadCount", 0),
                "rating": stats.get("rating", 0),
                "rating_count": stats.get("ratingCount", 0),
                "thumbs_up_count": stats.get("thumbsUpCount", 0),
            },
        ))

    return results


class CivitaiStatsFetchService:
    """Fetches CivitAI stats for models and stores them in CivitaiStatsDB."""

    def __init__(self, db: CivitaiStatsDB, api_key: str | None = None):
        self.db = db
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=timeout
            )
        return self._session

    async def _fetch_model(self, model_id: int, _retries: int = 2) -> dict | None:
        """Fetch a single model's data from CivitAI API."""
        url = f"{_CIVITAI_API}/models/{model_id}"
        session = await self._get_session()
        for attempt in range(_retries + 1):
            try:
                async with session.get(url) as resp:
                    if resp.status == 429:
                        logger.warning("CivitAI rate limited, backing off (attempt %d)", attempt + 1)
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        logger.debug("CivitAI returned %d for model %d", resp.status, model_id)
                        return None
                    return await resp.json()
            except Exception as exc:
                logger.warning("CivitAI fetch failed for model %d: %s", model_id, exc)
                return None
        return None

    async def fetch_stats_for_models(
        self,
        models: list[dict],
        progress_callback: Optional[Callable] = None,
    ) -> int:
        """Fetch stats from CivitAI for a list of models.

        Args:
            models: list of dicts with 'sha256' and 'civitai_model_id' keys.
            progress_callback: optional async callable(current, total).

        Returns:
            Number of model versions successfully updated.
        """
        # Deduplicate by model_id
        seen_model_ids: set[int] = set()
        unique_models: list[dict] = []
        for m in models:
            mid = m.get("civitai_model_id")
            if mid and mid not in seen_model_ids:
                seen_model_ids.add(mid)
                unique_models.append(m)

        total = len(unique_models)
        updated = 0

        for i, model in enumerate(unique_models):
            model_id = model["civitai_model_id"]
            data = await self._fetch_model(model_id)
            if data:
                rows = extract_version_stats(data)
                if rows:
                    self.db.upsert_batch(rows)
                    updated += len(rows)

            if progress_callback:
                await progress_callback(i + 1, total)

            # Rate limiting between requests
            if i < total - 1:
                await asyncio.sleep(_RATE_LIMIT_DELAY)

        return updated

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_civitai_stats_service.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add py/services/civitai_stats_service.py tests/test_civitai_stats_service.py
git commit -m "feat: add CivitAI stats fetch service"
```

---

### Task 3: CivitAI Stats Route Registrar

**Files:**
- Create: `py/routes/civitai_stats_routes.py`
- Modify: `py/lora_manager.py:25` (add import)
- Modify: `py/lora_manager.py:163` (register routes)

**Step 1: Write the route registrar**

Create `py/routes/civitai_stats_routes.py`:

```python
"""Route registrar for CivitAI community stats endpoints."""
from __future__ import annotations

import logging
from aiohttp import web

from ..services.civitai_stats_db import CivitaiStatsDB
from ..services.civitai_stats_service import CivitaiStatsFetchService
from ..services.service_registry import ServiceRegistry
from ..services.settings_manager import get_settings_manager

logger = logging.getLogger(__name__)

# Module-level singleton
_stats_db: CivitaiStatsDB | None = None


def _get_stats_db() -> CivitaiStatsDB:
    global _stats_db
    if _stats_db is None:
        _stats_db = CivitaiStatsDB()
        _stats_db.init()
    return _stats_db


class CivitaiStatsRoutes:
    """Route handlers for CivitAI stats fetch and status endpoints."""

    @staticmethod
    async def handle_fetch_stats(request: web.Request) -> web.Response:
        """POST /api/lm/civitai-stats/fetch — trigger bulk CivitAI stats fetch."""
        db = _get_stats_db()
        settings = get_settings_manager()
        api_key = settings.get("civitai_api_key", "")

        # Collect all models with modelId from all scanners
        models = []
        for getter in [
            ServiceRegistry.get_lora_scanner,
            ServiceRegistry.get_checkpoint_scanner,
            ServiceRegistry.get_embedding_scanner,
        ]:
            try:
                scanner = await getter()
                cache = await scanner.get_cached_data()
                for item in cache.raw_data:
                    civitai = item.get("civitai", {})
                    model_id = civitai.get("modelId")
                    sha256 = item.get("sha256")
                    if model_id and sha256:
                        models.append({
                            "sha256": sha256,
                            "civitai_model_id": model_id,
                        })
            except Exception as exc:
                logger.debug("Failed to get scanner data: %s", exc)

        if not models:
            return web.json_response({"success": True, "updated": 0, "total": 0})

        service = CivitaiStatsFetchService(db=db, api_key=api_key)
        try:
            updated = await service.fetch_stats_for_models(models)
        finally:
            await service.close()

        return web.json_response({"success": True, "updated": updated, "total": len(models)})

    @staticmethod
    async def handle_stats_status(request: web.Request) -> web.Response:
        """GET /api/lm/civitai-stats/status — check stats DB count."""
        db = _get_stats_db()
        return web.json_response({"success": True, "count": db.count()})

    @staticmethod
    async def handle_enrich(request: web.Request) -> web.Response:
        """POST /api/lm/civitai-stats/by-hashes — get stats for given hashes.

        Accepts JSON body: {"hashes": ["abc", "def"]}
        Uses POST because the hash list can be large (thousands of 64-char SHA256).
        """
        try:
            body = await request.json()
            hashes = body.get("hashes", [])
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)
        if not hashes:
            return web.json_response({"success": True, "stats": {}})
        db = _get_stats_db()
        stats = db.get_by_hashes(hashes)
        # Convert for JSON (remove fetched_at internal field)
        clean = {}
        for sha, data in stats.items():
            clean[sha] = {
                "download_count": data.get("download_count", 0),
                "rating": data.get("rating", 0),
                "rating_count": data.get("rating_count", 0),
                "thumbs_up_count": data.get("thumbs_up_count", 0),
            }
        return web.json_response({"success": True, "stats": clean})

    @classmethod
    def setup_routes(cls, app: web.Application) -> None:
        """Register CivitAI stats routes."""
        app.router.add_post("/api/lm/civitai-stats/fetch", cls.handle_fetch_stats)
        app.router.add_get("/api/lm/civitai-stats/status", cls.handle_stats_status)
        app.router.add_post("/api/lm/civitai-stats/by-hashes", cls.handle_enrich)

        # Cleanup on shutdown
        async def cleanup(app):
            global _stats_db
            if _stats_db:
                _stats_db.close()
                _stats_db = None

        app.on_shutdown.append(cleanup)
        logger.info("CivitAI stats routes registered")
```

**Step 2: Modify `py/lora_manager.py` to register routes**

At line 25 (imports section), add:

```python
from .routes.civitai_stats_routes import CivitaiStatsRoutes
```

At line 168 (after `PreviewRoutes.setup_routes(app)`), add:

```python
        CivitaiStatsRoutes.setup_routes(app)
```

**Step 3: Run existing tests to verify no breakage**

Run: `pytest tests/ -v --ignore=tests/frontend`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add py/routes/civitai_stats_routes.py py/lora_manager.py
git commit -m "feat: add CivitAI stats API routes"
```

---

### Task 4: Frontend — Stats UI JavaScript

**Files:**
- Create: `static/js/civitai_stats_ui.js`
- Modify: `templates/base.html:106` (add script tag)

**Step 1: Create the stats UI JS**

Create `static/js/civitai_stats_ui.js` — adapted from `custom_nodes/ComfyUI-LM-Remote/static/lm_stats_ui.js`.

Key changes from LM-Remote version:
- API endpoint is `/api/lm/civitai-stats/fetch` (not `/api/lm-extra/fetch-stats`)
- Stats are fetched via `/api/lm/civitai-stats/by-hashes` after list API responses (instead of being embedded in list response)
- No proxy enrichment — JS fetches stats separately

```javascript
// static/js/civitai_stats_ui.js
/**
 * CivitAI Stats UI — card badges, sort dropdown options, fetch button.
 *
 * Self-contained script that patches model cards with CivitAI community
 * stats (downloads, rating, likes). Fetches stats from the local
 * /api/lm/civitai-stats/* endpoints.
 */
(function () {
    "use strict";

    // ── Compact number formatting ──────────────────────────────────
    function formatCompact(n) {
        if (n == null) return null;
        if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
        if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "k";
        return String(n);
    }

    // ── Badge creation ─────────────────────────────────────────────
    function createStatBadge(icon, value, title) {
        if (value == null) return null;
        const badge = document.createElement("span");
        badge.className = "lm-stat-badge";
        badge.title = title;
        badge.innerHTML = `<i class="fas fa-${icon}"></i> ${formatCompact(value)}`;
        return badge;
    }

    // ── Inject CSS ─────────────────────────────────────────────────
    function injectStyles() {
        if (document.getElementById("lm-civitai-stats-styles")) return;
        const style = document.createElement("style");
        style.id = "lm-civitai-stats-styles";
        style.textContent = `
            .lm-stat-badges {
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
                padding: 2px 6px;
            }
            .lm-stat-badge {
                display: inline-flex;
                align-items: center;
                gap: 3px;
                font-size: 11px;
                padding: 1px 5px;
                border-radius: 4px;
                background: rgba(255,255,255,0.1);
                color: rgba(255,255,255,0.8);
                white-space: nowrap;
            }
            .lm-stat-badge i {
                font-size: 10px;
                opacity: 0.7;
            }
        `;
        document.head.appendChild(style);
    }

    // ── Stats cache ────────────────────────────────────────────────
    const _statsMap = {};

    // Fetch stats for a batch of hashes from local DB
    async function fetchStatsForHashes(hashes) {
        if (!hashes.length) return;
        try {
            const resp = await _origFetch("/api/lm/civitai-stats/by-hashes", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ hashes }),
            });
            const data = await resp.json();
            if (data.success && data.stats) {
                Object.assign(_statsMap, data.stats);
            }
        } catch (e) {
            console.debug("[CivitAI Stats] Failed to fetch stats:", e);
        }
    }

    // ── Intercept list API to collect hashes ───────────────────────
    const _origFetch = window.fetch;
    window.fetch = async function (...args) {
        const response = await _origFetch.apply(this, args);
        const url = typeof args[0] === "string" ? args[0] : args[0]?.url;
        if (url && url.includes("/api/lm/") && url.includes("/list")) {
            try {
                const clone = response.clone();
                const data = await clone.json();
                const hashes = (data.items || [])
                    .map((item) => item.sha256)
                    .filter((h) => h && !_statsMap[h]);
                if (hashes.length > 0) {
                    // Fire and forget — cards will be patched when data arrives
                    fetchStatsForHashes(hashes).then(() => patchCards());
                }
            } catch (e) { /* ignore parse errors */ }
        }
        return response;
    };

    // ── Patch model cards ──────────────────────────────────────────
    function patchCards() {
        const cards = document.querySelectorAll(".model-card:not([data-stats-patched])");
        cards.forEach((card) => {
            const sha = card.dataset.sha256;
            if (!sha || !_statsMap[sha]) return;

            card.setAttribute("data-stats-patched", "1");

            const stats = _statsMap[sha];
            const container = document.createElement("div");
            container.className = "lm-stat-badges";

            const dlBadge = createStatBadge("download", stats.download_count, "Downloads");
            const ratingBadge = createStatBadge("star",
                stats.rating ? Number(stats.rating.toFixed(1)) : null, "Rating");
            const thumbsBadge = createStatBadge("thumbs-up", stats.thumbs_up_count, "Likes");

            [dlBadge, ratingBadge, thumbsBadge].forEach((b) => {
                if (b) container.appendChild(b);
            });

            if (container.children.length > 0) {
                const modelInfo = card.querySelector(".model-info");
                if (modelInfo) {
                    modelInfo.appendChild(container);
                }
            }
        });
    }

    // ── Stats sort keys ──────────────────────────────────────────
    const _STATS_SORT_KEYS = new Set(["downloads", "rating", "thumbsup"]);

    // ── Patch sort dropdown ────────────────────────────────────────
    function patchSortDropdown() {
        const select = document.getElementById("sortSelect");
        if (!select || select.querySelector('[value="downloads:desc"]')) return;

        const group = document.createElement("optgroup");
        group.label = "CivitAI Stats";

        const options = [
            ["downloads:desc", "Most downloaded"],
            ["downloads:asc", "Least downloaded"],
            ["rating:desc", "Highest rated"],
            ["rating:asc", "Lowest rated"],
            ["thumbsup:desc", "Most liked"],
            ["thumbsup:asc", "Least liked"],
        ];

        options.forEach(([value, label]) => {
            const opt = document.createElement("option");
            opt.value = value;
            opt.textContent = label;
            group.appendChild(opt);
        });

        select.appendChild(group);

        // Restore persisted stats sort — our options didn't exist when
        // PageControls.loadSortPreference() ran, so the saved value
        // was silently ignored. Re-apply it now.
        const path = window.location.pathname;
        const pageType = path.includes("checkpoints") ? "checkpoints"
            : path.includes("embeddings") ? "embeddings" : "loras";
        const saved = localStorage.getItem("lora_manager_" + pageType + "_sort")
            || localStorage.getItem(pageType + "_sort");  // legacy fallback
        if (saved && _STATS_SORT_KEYS.has(saved.split(":")[0]) && select.value !== saved) {
            select.value = saved;
            select.dispatchEvent(new Event("change"));
        }
    }

    // ── Toolbar "Fetch Stats" button ───────────────────────────────
    function addFetchStatsButton() {
        const toolbar = document.querySelector(".action-buttons");
        if (!toolbar || document.getElementById("fetchStatsBtn")) return;

        const group = document.createElement("div");
        group.className = "control-group";
        group.innerHTML = `
            <button id="fetchStatsBtn" data-action="fetch-stats"
                    title="Fetch CivitAI stats (downloads, ratings, likes)">
                <i class="fas fa-chart-bar"></i> <span>Fetch Stats</span>
            </button>
        `;

        const bulkBtn = document.getElementById("bulkOperationsBtn");
        if (bulkBtn && bulkBtn.closest(".control-group")) {
            toolbar.insertBefore(group, bulkBtn.closest(".control-group"));
        } else {
            toolbar.appendChild(group);
        }

        group.querySelector("button").addEventListener("click", async () => {
            const btn = document.getElementById("fetchStatsBtn");
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Fetching...</span>';

            try {
                const resp = await _origFetch("/api/lm/civitai-stats/fetch", { method: "POST" });
                const data = await resp.json();
                if (data.success) {
                    const count = parseInt(data.updated, 10) || 0;
                    btn.innerHTML = `<i class="fas fa-check"></i> <span>${count} updated</span>`;
                    setTimeout(() => {
                        btn.innerHTML = '<i class="fas fa-chart-bar"></i> <span>Fetch Stats</span>';
                        btn.disabled = false;
                    }, 3000);
                    // Trigger page reload to show new stats
                    const sortSelect = document.getElementById("sortSelect");
                    if (sortSelect) {
                        sortSelect.dispatchEvent(new Event("change"));
                    }
                } else {
                    throw new Error(data.error || "Unknown error");
                }
            } catch (err) {
                btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Error</span>';
                console.error("[CivitAI Stats] Fetch failed:", err);
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-chart-bar"></i> <span>Fetch Stats</span>';
                    btn.disabled = false;
                }, 3000);
            }
        });
    }

    // ── Observe DOM for card rendering ─────────────────────────────
    // Stats sorting is handled server-side via model_cache._sort_data()
    // so no client-side sort interception is needed. The sort dropdown
    // options added by patchSortDropdown() use keys (downloads, rating,
    // thumbsup) that the backend recognizes.

    function startObserver() {
        injectStyles();
        patchSortDropdown();
        addFetchStatsButton();
        patchCards();

        let debounceTimer = null;
        const observer = new MutationObserver(() => {
            if (debounceTimer) return;
            debounceTimer = setTimeout(() => {
                debounceTimer = null;
                patchCards();
                patchSortDropdown();
                addFetchStatsButton();

            }, 200);
        });

        const grid = document.getElementById("modelGrid");
        observer.observe(grid || document.body, { childList: true, subtree: true });
    }

    // ── Init ───────────────────────────────────────────────────────
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startObserver);
    } else {
        startObserver();
    }
})();
```

**Step 2: Modify `templates/base.html` to load the script**

At line 106 (inside the `{% block additional_scripts %}` block, which is inside the `{% if not is_initializing %}` conditional in child templates), add:

In `templates/base.html`, change line 106 from:
```html
    {% block additional_scripts %}{% endblock %}
```
to:
```html
    {% block additional_scripts %}{% endblock %}
    <script src="/loras_static/js/civitai_stats_ui.js?v={{ version }}"></script>
```

Note: The script loads after all page-specific scripts. It's outside the `is_initializing` guard but handles this gracefully — if no `#modelGrid` or `.model-card` elements exist, it simply does nothing.

**Step 3: Verify by running the dev server**

Run: `LORA_MANAGER_STANDALONE=1 python standalone.py --port 8188`

Verify:
- Page loads without JS errors
- "Fetch Stats" button appears in toolbar
- Sort dropdown has "CivitAI Stats" optgroup
- After clicking "Fetch Stats", badges appear on cards that have CivitAI metadata

**Step 4: Commit**

```bash
git add static/js/civitai_stats_ui.js templates/base.html
git commit -m "feat: add CivitAI stats frontend UI with badges and sort"
```

---

### Task 5: Backend Sort Support for Stats Keys

**Files:**
- Modify: `py/services/model_cache.py:244` (add elif branch in `_sort_data()`)

**Step 1: Add stats sort keys to `_sort_data()`**

In `py/services/model_cache.py`, after the `elif sort_key == 'usage':` block (around line 253) and before the `else:` fallback, add:

```python
        elif sort_key in ('downloads', 'rating', 'thumbsup'):
            # Sort by CivitAI stats from separate stats DB
            from .civitai_stats_db import CivitaiStatsDB
            stats_db = CivitaiStatsDB()
            try:
                all_stats = stats_db.get_all()
            finally:
                stats_db.close()
            stats_field = {
                'downloads': 'download_count',
                'rating': 'rating',
                'thumbsup': 'thumbs_up_count',
            }[sort_key]
            with_stats = [item for item in data if item.get('sha256', '') in all_stats]
            without_stats = [item for item in data if item.get('sha256', '') not in all_stats]
            with_stats.sort(
                key=lambda x: all_stats.get(x.get('sha256', ''), {}).get(stats_field, 0),
                reverse=reverse,
            )
            result = with_stats + without_stats
```

**Step 2: Run existing tests to verify no breakage**

Run: `pytest tests/ -v --ignore=tests/frontend -x`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add py/services/model_cache.py
git commit -m "feat: add CivitAI stats sort keys to model cache"
```

---

### Task 6: Add .gitignore entry

**Files:**
- Modify: `.gitignore` (add civitai_stats.db)

**Step 1: Add gitignore entry**

Add to `.gitignore`:

```
civitai_stats.db
```

**Step 2: Run all tests**

Run: `pytest tests/ -v --ignore=tests/frontend`
Expected: All tests PASS

Run: `npm test` (if frontend test infrastructure is set up)
Expected: All tests PASS (no frontend test files created for this feature use DOM APIs that need manual testing)

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add civitai_stats.db to gitignore"
```

---

### Task 7: Integration verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing + new civitai stats tests)

**Step 2: Run standalone server and manual test**

Run: `LORA_MANAGER_STANDALONE=1 python standalone.py --port 8188`

Manual verification checklist:
- [ ] Page loads, no console errors
- [ ] "Fetch Stats" button visible in toolbar
- [ ] Click "Fetch Stats" → spinner → count → resets
- [ ] Cards with CivitAI data show download/rating/thumbs-up badges
- [ ] Sort dropdown has "CivitAI Stats" group with 6 options
- [ ] Selecting "Most downloaded" sorts cards by download count
- [ ] `/api/lm/civitai-stats/status` returns `{"success": true, "count": N}`
- [ ] `POST /api/lm/civitai-stats/by-hashes` with `{"hashes":["abc"]}` returns stats

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address integration test findings"
```
