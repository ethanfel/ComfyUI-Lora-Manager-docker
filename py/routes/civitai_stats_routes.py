"""Route registrar for CivitAI community stats endpoints."""
from __future__ import annotations

import asyncio
import logging
from aiohttp import web

from ..services.civitai_stats_db import CivitaiStatsDB
from ..services.civitai_stats_service import CivitaiStatsFetchService
from ..services.service_registry import ServiceRegistry
from ..services.settings_manager import get_settings_manager
from ..services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

_fetch_lock = asyncio.Lock()


class CivitaiStatsRoutes:
    """Route handlers for CivitAI stats fetch and retrieval endpoints."""

    @staticmethod
    async def handle_fetch_stats(request: web.Request) -> web.Response:
        """POST /api/lm/civitai-stats/fetch — trigger bulk CivitAI stats fetch."""
        if _fetch_lock.locked():
            return web.json_response(
                {"success": False, "error": "Stats fetch already in progress"},
                status=409,
            )
        async with _fetch_lock:
            try:
                db = CivitaiStatsDB.get_instance()
                settings = get_settings_manager()
                api_key = settings.get("civitai_api_key", "")

                # Collect all models with modelId from all scanners
                models = []
                for name, getter in [
                    ("lora", ServiceRegistry.get_lora_scanner),
                    ("checkpoint", ServiceRegistry.get_checkpoint_scanner),
                    ("embedding", ServiceRegistry.get_embedding_scanner),
                ]:
                    try:
                        scanner = await getter()
                        cache = await scanner.get_cached_data()
                        for item in cache.raw_data:
                            if not item:
                                continue
                            civitai = item.get("civitai") or {}
                            model_id = civitai.get("modelId")
                            version_id = civitai.get("id")
                            sha256 = item.get("sha256")
                            if model_id and sha256:
                                models.append({
                                    "sha256": sha256,
                                    "civitai_model_id": model_id,
                                    "civitai_version_id": version_id,
                                })
                    except Exception as exc:
                        logger.warning("Failed to get %s scanner data: %s", name, exc)

                logger.info("Collected %d models for stats fetch", len(models))

                if not models:
                    return web.json_response({"success": True, "updated": 0, "total": 0})

                service = CivitaiStatsFetchService(db=db, api_key=api_key)

                async def progress_callback(current: int, total: int) -> None:
                    await ws_manager.broadcast({
                        "type": "civitai_stats_progress",
                        "current": current,
                        "total": total,
                        "progress": round(current / total * 100) if total else 0,
                    })

                try:
                    updated = await service.fetch_stats_for_models(models, progress_callback=progress_callback)
                finally:
                    await service.close()

                return web.json_response({"success": True, "updated": updated, "total": len(models)})
            except Exception as exc:
                logger.exception("CivitAI stats fetch failed")
                return web.json_response(
                    {"success": False, "error": str(exc)},
                    status=500,
                )

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
        if not isinstance(hashes, list) or len(hashes) > 5000:
            return web.json_response(
                {"success": False, "error": "Invalid or too many hashes (max 5000)"},
                status=400,
            )
        # Filter to valid string hashes only
        hashes = [h for h in hashes if isinstance(h, str) and h]
        if not hashes:
            return web.json_response({"success": True, "stats": {}})
        db = CivitaiStatsDB.get_instance()
        stats = db.get_by_hashes(hashes)
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
        app.router.add_post("/api/lm/civitai-stats/by-hashes", cls.handle_enrich)

        # Cleanup on shutdown
        async def cleanup(app):
            instance = CivitaiStatsDB._instance
            if instance:
                instance.close()
                CivitaiStatsDB._instance = None

        app.on_shutdown.append(cleanup)
        logger.info("CivitAI stats routes registered")
