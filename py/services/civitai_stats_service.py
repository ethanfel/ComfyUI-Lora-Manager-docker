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
