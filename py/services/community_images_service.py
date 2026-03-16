"""Service for fetching CivitAI community images and storing them locally."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

import io
import json

import aiohttp
from PIL import Image

from .community_images_db import CommunityImagesDB
from .websocket_manager import ws_manager
from ..utils.example_images_paths import get_model_folder, get_model_relative_path

logger = logging.getLogger(__name__)

_CIVITAI_API = "https://civitai.com/api/v1"
_RATE_LIMIT_DELAY = 1.5  # seconds between requests
_MIN_PROMPT_LENGTH = 20
_MAX_IMAGES_PER_MODEL = 10
_WEBP_QUALITY = 85
_MAX_IMAGE_DIMENSION = 1280  # resize longest side


def filter_community_images(
    items: list[dict], author_username: str
) -> list[dict]:
    """Filter CivitAI image items, excluding author and low-quality prompts.

    - Skip images by author (case-insensitive username match)
    - Skip images without meta.meta.prompt
    - Skip images with prompt < 20 chars
    - Return at most 10 images
    """
    author_lower = (author_username or "").lower()
    result: list[dict] = []

    for item in items:
        # Skip author's own images
        username = (item.get("username") or "").lower()
        if username and author_lower and username == author_lower:
            continue

        # Skip images without prompt (double-nested meta)
        meta = item.get("meta")
        if not meta or not isinstance(meta, dict):
            continue
        inner_meta = meta.get("meta")
        if not inner_meta or not isinstance(inner_meta, dict):
            continue
        prompt = inner_meta.get("prompt") or ""
        if len(prompt) < _MIN_PROMPT_LENGTH:
            continue

        result.append(item)
        if len(result) >= _MAX_IMAGES_PER_MODEL:
            break

    return result


def _extract_image_data(
    item: dict,
    sha256: str,
    civitai_model_id: int,
    version_cache: dict[int, dict] | None = None,
) -> dict:
    """Convert a CivitAI API image item to DB row format."""
    stats = item.get("stats") or {}
    meta = item.get("meta") or {}
    inner_meta = (meta.get("meta") or {}) if isinstance(meta, dict) else {}

    # Build enriched resources list from civitaiResources
    resources_json = None
    raw_resources = inner_meta.get("civitaiResources") or []
    if raw_resources:
        enriched = []
        cache = version_cache or {}
        for res in raw_resources:
            entry = {
                "type": res.get("type"),
                "weight": res.get("weight"),
                "modelVersionId": res.get("modelVersionId"),
            }
            vid = res.get("modelVersionId")
            if vid and vid in cache:
                info = cache[vid]
                entry["name"] = info.get("name")
                entry["modelId"] = info.get("modelId")
            enriched.append(entry)
        resources_json = json.dumps(enriched)

    return {
        "civitai_image_id": item.get("id"),
        "sha256": sha256,
        "civitai_model_id": civitai_model_id,
        "username": item.get("username"),
        "image_url": item.get("url"),
        "local_filename": None,
        "width": item.get("width"),
        "height": item.get("height"),
        "prompt": inner_meta.get("prompt"),
        "negative_prompt": inner_meta.get("negativePrompt"),
        "steps": inner_meta.get("steps"),
        "sampler": inner_meta.get("sampler"),
        "cfg_scale": inner_meta.get("cfgScale"),
        "seed": inner_meta.get("seed"),
        "denoise": inner_meta.get("denoise"),
        "base_model": item.get("baseModel"),
        "like_count": stats.get("likeCount", 0),
        "heart_count": stats.get("heartCount", 0),
        "laugh_count": stats.get("laughCount", 0),
        "comment_count": stats.get("commentCount", 0),
        "media_type": item.get("type", "image"),
        "resources": resources_json,
        "created_at": item.get("createdAt"),
    }


class CommunityImagesFetchService:
    """Fetches CivitAI community images and stores them via CommunityImagesDB."""

    def __init__(self, db: CommunityImagesDB, api_key: str | None = None):
        self.db = db
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        self._cancelled = False
        # modelVersionId -> {"name": str, "modelId": int}
        self._version_cache: dict[int, dict] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy aiohttp session with auth header."""
        if self._session is None or self._session.closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=timeout
            )
        return self._session

    async def _resolve_version(self, version_id: int) -> dict | None:
        """Resolve a CivitAI modelVersionId to name + modelId. Cached."""
        if version_id in self._version_cache:
            return self._version_cache[version_id]

        session = await self._get_session()
        try:
            async with session.get(
                f"{_CIVITAI_API}/model-versions/{version_id}"
            ) as resp:
                if resp.status == 429:
                    await ws_manager.broadcast({
                        "type": "community_images_warning",
                        "message": "Rate limited resolving resource names, some names may be missing",
                    })
                    return None
                if resp.status != 200:
                    return None
                data = await resp.json()
                model = data.get("model") or {}
                name = model.get("name") or data.get("name")
                model_id = model.get("id")
                if name:
                    info = {"name": name, "modelId": model_id}
                    self._version_cache[version_id] = info
                    return info
                return None
        except Exception:
            return None

    async def _resolve_resources(self, items: list[dict]) -> None:
        """Resolve modelVersionId → name for all civitaiResources across items.

        Best-effort: failures are silently skipped (resources still stored
        without names). Batches unique IDs and caches results.
        """
        version_ids: set[int] = set()
        for item in items:
            meta = item.get("meta")
            if not meta or not isinstance(meta, dict):
                continue
            inner_meta = (meta.get("meta") or {}) if isinstance(meta, dict) else {}
            for res in inner_meta.get("civitaiResources") or []:
                vid = res.get("modelVersionId")
                if vid and vid not in self._version_cache:
                    version_ids.add(vid)

        if not version_ids:
            return

        resolved = 0
        for vid in version_ids:
            info = await self._resolve_version(vid)
            if info:
                resolved += 1
            await asyncio.sleep(0.2)

        if version_ids:
            logger.debug(
                "Resolved %d/%d resource names", resolved, len(version_ids)
            )

    async def _fetch_images_api(
        self, model_id: int, retries: int = 2
    ) -> dict | None:
        """GET /api/v1/images with retries and 429 backoff."""
        url = f"{_CIVITAI_API}/images"
        params = {
            "modelId": str(model_id),
            "sort": "Most Reactions",
            "limit": "20",
        }
        session = await self._get_session()
        for attempt in range(retries + 1):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(
                            "CivitAI rate limited, backing off %ds (attempt %d)",
                            wait_time,
                            attempt + 1,
                        )
                        await ws_manager.broadcast({
                            "type": "community_images_warning",
                            "message": f"Rate limited by CivitAI, waiting {wait_time}s...",
                        })
                        await asyncio.sleep(wait_time)
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
                    "CivitAI images fetch failed for model %d: %s",
                    model_id,
                    exc,
                )
                return None
        return None

    async def _download_media(
        self, image_url: str, sha256: str, image_id: int, media_type: str = "image"
    ) -> tuple[str | None, bool]:
        """Download image or video, storing locally.

        Images are converted to WebP with resize. Videos are saved as-is (mp4).
        If a ComfyUI workflow is found in PNG metadata, saves it as
        {image_id}.workflow.json alongside the media.

        Returns (relative_path, has_workflow) or (None, False) on failure.
        """
        model_folder = get_model_folder(sha256)
        rel_path = get_model_relative_path(sha256)
        if not model_folder or not rel_path:
            logger.warning("No model folder for hash %s, skipping download", sha256)
            return None, False

        community_dir = os.path.join(model_folder, "community")
        os.makedirs(community_dir, exist_ok=True)

        has_workflow = False

        try:
            session = await self._get_session()
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    logger.debug(
                        "Failed to download media %d: HTTP %d",
                        image_id,
                        resp.status,
                    )
                    return None, False
                data = await resp.read()

            if media_type == "video":
                # Save video as mp4
                filepath = os.path.join(community_dir, f"{image_id}.mp4")
                with open(filepath, "wb") as f:
                    f.write(data)
                return f"{rel_path}/community/{image_id}.mp4", False

            # Image path: convert to WebP
            filepath = os.path.join(community_dir, f"{image_id}.webp")

            img = Image.open(io.BytesIO(data))
            try:
                # Extract ComfyUI workflow from PNG tEXt chunks
                # PNG text chunks store JSON as strings — parse them to avoid double-encoding
                workflow_data = {}
                if hasattr(img, "info") and isinstance(img.info, dict):
                    for key in ("workflow", "prompt"):
                        if key in img.info:
                            raw = img.info[key]
                            try:
                                workflow_data[key] = json.loads(raw) if isinstance(raw, str) else raw
                            except (json.JSONDecodeError, TypeError):
                                workflow_data[key] = raw

                if workflow_data:
                    workflow_path = os.path.join(
                        community_dir, f"{image_id}.workflow.json"
                    )
                    with open(workflow_path, "w", encoding="utf-8") as f:
                        json.dump(workflow_data, f)
                    has_workflow = True

                # Convert to WebP with resize
                img.thumbnail(
                    (_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION),
                    Image.LANCZOS,
                )
                img.save(filepath, "webp", quality=_WEBP_QUALITY)
            finally:
                img.close()

            # Remove old .jpg if it exists (migration from pre-WebP format)
            old_jpg = os.path.join(community_dir, f"{image_id}.jpg")
            if os.path.exists(old_jpg):
                os.remove(old_jpg)
        except Exception as exc:
            logger.warning("Failed to download media %d: %s", image_id, exc)
            return None, False

        return f"{rel_path}/community/{image_id}.webp", has_workflow

    async def fetch_images_for_model(
        self,
        sha256: str,
        civitai_model_id: int,
        author_username: str,
    ) -> int:
        """Fetch, filter, download, and store community images for one model.

        Returns the number of images stored.
        """
        response = await self._fetch_images_api(civitai_model_id)
        if not response:
            return 0

        items = response.get("items", [])
        filtered = filter_community_images(items, author_username)
        if not filtered:
            return 0

        # Resolve resource names (batched, cached)
        await self._resolve_resources(filtered)

        rows: list[dict] = []
        for item in filtered:
            image_id = item.get("id")
            image_url = item.get("url")
            item_type = item.get("type", "image")
            if not image_id or not image_url:
                continue

            local_path, has_workflow = await self._download_media(
                image_url, sha256, image_id, media_type=item_type
            )

            row = _extract_image_data(
                item, sha256, civitai_model_id, self._version_cache
            )
            row["local_filename"] = local_path
            row["has_workflow"] = 1 if has_workflow else 0
            rows.append(row)

        if rows:
            self.db.upsert_batch(rows)

        return len(rows)

    def cancel(self) -> None:
        """Signal the fetch loop to stop after the current model."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    async def fetch_all(
        self,
        models: list[dict],
        progress_callback: Callable | None = None,
    ) -> int:
        """Bulk fetch community images with rate limiting.

        Args:
            models: list of dicts with 'sha256', 'civitai_model_id',
                     and 'author_username' keys.
            progress_callback: optional async callable(current, total).

        Returns:
            Total number of images stored.
        """
        self._cancelled = False
        total = len(models)
        total_stored = 0

        logger.info("Fetching community images for %d models", total)

        for i, model in enumerate(models):
            if self._cancelled:
                logger.info("Community images fetch cancelled at %d/%d", i, total)
                break

            sha256 = model.get("sha256")
            model_id = model.get("civitai_model_id")
            author = model.get("author_username", "")

            if not sha256 or not model_id:
                if progress_callback:
                    await progress_callback(i + 1, total)
                continue

            count = await self.fetch_images_for_model(sha256, model_id, author)
            total_stored += count

            if progress_callback:
                await progress_callback(i + 1, total)

            # Rate limiting between requests
            if i < total - 1:
                await asyncio.sleep(_RATE_LIMIT_DELAY)

        logger.info(
            "Community images fetch %s: %d images stored for %d models",
            "cancelled" if self._cancelled else "complete",
            total_stored,
            total,
        )
        return total_stored

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
