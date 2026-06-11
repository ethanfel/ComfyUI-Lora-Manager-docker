# CivitAI Stats Integration — Design

## Goal

Show CivitAI community stats (downloads, rating, thumbs up) on model cards in the standalone Docker LoRA Manager, with sorting support. Port the feature from the LM-Remote proxy shim into the main codebase with minimal upstream conflict risk.

## Constraints

- **Minimal upstream file changes**: Only touch 2 existing files with trivial additions (1 script tag, 1 route registration line). All other code lives in new files.
- **Same UX as LM-Remote**: Badge row on cards, sort dropdown optgroup, "Fetch Stats" button in toolbar.
- **Separate SQLite DB**: `civitai_stats.db` — avoids schema migration on the main model cache.

## Architecture

### New Files (no upstream conflict)

| File | Purpose |
|------|---------|
| `py/services/civitai_stats_db.py` | SQLite wrapper for `civitai_stats.db` — schema, upsert, query by sha256 |
| `py/services/civitai_stats_service.py` | Fetches stats from CivitAI API v1 `/models/{id}`, stores via stats DB |
| `py/routes/civitai_stats_routes.py` | Route registrar: `POST /api/lm/civitai-stats/fetch`, `GET /api/lm/civitai-stats/status` |
| `static/js/civitai_stats_ui.js` | Badge rendering, sort dropdown patching, fetch button (adapted from LM-Remote) |
| `static/css/civitai_stats.css` | Stats badge styling |

### Existing Files Modified (3 total, minimal changes)

| File | Change |
|------|--------|
| `templates/base.html` | Add 1 `<script>` tag |
| `py/lora_manager.py` | Add import + 3 lines to register `CivitaiStatsRoutes` |
| `py/services/model_cache.py` | Add `elif` branch in `_sort_data()` for stats sort keys (~15 lines) |

### Data Flow

```
User clicks "Fetch Stats" button
  → POST /api/lm/civitai-stats/fetch
  → CivitaiStatsService extracts unique modelIds from scanner cache
  → Fetches /api/v1/models/{id} for each (rate-limited, with 429 retry)
  → Extracts per-version stats keyed by sha256
  → Upserts into civitai_stats.db
  → WebSocket progress broadcast
  → Returns {success, updated, total}

Page loads / sorts / paginates
  → GET /api/lm/loras/list?sort_by=name:asc&page=1
  → Normal LoRA Manager response
  → Frontend JS intercepts fetch() response
  → Merges stats from captured data onto model cards
  → If sort is stats-based, JS fetches all items and sorts client-side
```

### Frontend Strategy (same as LM-Remote)

The JS file uses the same approach as LM-Remote to avoid touching upstream code:

1. **fetch() interceptor** — captures stats from list API responses into a local map
2. **MutationObserver** (debounced 200ms, scoped to `#modelGrid`) — patches new cards with badge elements
3. **Sort dropdown patching** — adds "CivitAI Stats" optgroup dynamically
4. **Fetch button injection** — adds button to `.action-buttons` toolbar

For stats-based sorting, the frontend overrides the normal pagination:
- Fetches all items (page_size=9999) from the list API
- Sorts by the stats column client-side
- Renders the appropriate page slice

### Backend Components

**CivitaiStatsDB** (adapted from LM-Remote `stats_db.py`):
- Separate `civitai_stats.db` file in the app's data directory
- Schema: `sha256 TEXT PRIMARY KEY, civitai_model_id INT, civitai_version_id INT, download_count INT, rating REAL, rating_count INT, thumbs_up_count INT, fetched_at TEXT`
- WAL mode, `check_same_thread=False`
- Methods: `init()`, `upsert_batch()`, `get_by_hashes()`, `get_all()`, `count()`, `close()`

**CivitaiStatsService** (adapted from LM-Remote `stats_service.py`):
- Uses `aiohttp.ClientSession` with CivitAI API key from settings
- `extract_version_stats(response_json)` → list of `(sha256, stats_dict)`
- `fetch_stats_for_models(models, progress_callback)` — deduplicates by modelId, rate-limits at 1.5s between requests, retries on 429
- Gets model list from scanner cache (all model types)

**CivitaiStatsRoutes**:
- `POST /api/lm/civitai-stats/fetch` — triggers bulk fetch, broadcasts progress via WebSocket
- `GET /api/lm/civitai-stats/status` — returns `{count}` from stats DB
- Registered in `lora_manager.py` alongside existing `StatsRoutes`

### Testing

- `tests/test_civitai_stats_db.py` — DB CRUD operations
- `tests/test_civitai_stats_service.py` — API response parsing, fetch with mocked HTTP
- `tests/test_civitai_stats_routes.py` — Route handler integration tests
- `tests/frontend/civitai_stats_ui.test.js` — Frontend JS unit tests

### Cleanup

- Remove LM-Remote stats code after this is working (stats_db.py, stats_service.py, lm_stats_ui.js, proxy.py stats sections)
