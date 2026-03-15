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


def extract_version_stats(model_data: dict) -> dict[int, dict]:
    """Extract {version_id: stats_dict} from a CivitAI model response.

    Returns a dict keyed by version ID so callers can map stats
    to local sha256 hashes (which may differ from CivitAI's file hashes).
    """
    model_id = model_data.get("id")
    results = {}

    for version in model_data.get("modelVersions", []):
        version_id = version.get("id")
        if not version_id:
            continue
        stats = version.get("stats") or {}
        results[version_id] = {
            "civitai_model_id": model_id,
            "civitai_version_id": version_id,
            "download_count": stats.get("downloadCount", 0),
            "rating": stats.get("rating", 0),
            "rating_count": stats.get("ratingCount", 0),
            "thumbs_up_count": stats.get("thumbsUpCount", 0),
        }

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
        # Group local models by civitai_model_id, collecting (sha256, version_id) pairs
        model_id_to_locals: dict[int, list[dict]] = {}
        for m in models:
            mid = m.get("civitai_model_id")
            if mid:
                model_id_to_locals.setdefault(mid, []).append(m)

        unique_model_ids = list(model_id_to_locals.keys())
        total = len(unique_model_ids)
        updated = 0

        for i, model_id in enumerate(unique_model_ids):
            data = await self._fetch_model(model_id)
            if data:
                version_stats = extract_version_stats(data)
                if version_stats:
                    # Map local sha256 → stats using version ID match
                    rows = []
                    for local_model in model_id_to_locals[model_id]:
                        local_sha = local_model.get("sha256")
                        local_vid = local_model.get("civitai_version_id")
                        if local_sha and local_vid and local_vid in version_stats:
                            rows.append((local_sha, version_stats[local_vid]))
                    # Fallback: if no version ID match but only one version, use it
                    if not rows and len(version_stats) == 1:
                        only_stats = next(iter(version_stats.values()))
                        for local_model in model_id_to_locals[model_id]:
                            local_sha = local_model.get("sha256")
                            if local_sha:
                                rows.append((local_sha, only_stats))
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
