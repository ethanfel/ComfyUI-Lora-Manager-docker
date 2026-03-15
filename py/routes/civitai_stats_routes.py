"""Route registrar for CivitAI community stats endpoints."""
from __future__ import annotations

import logging
from aiohttp import web

from ..services.civitai_stats_db import CivitaiStatsDB
from ..services.civitai_stats_service import CivitaiStatsFetchService
from ..services.service_registry import ServiceRegistry
from ..services.settings_manager import get_settings_manager
from ..services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)


class CivitaiStatsRoutes:
    """Route handlers for CivitAI stats fetch and status endpoints."""

    @staticmethod
    async def handle_fetch_stats(request: web.Request) -> web.Response:
        """POST /api/lm/civitai-stats/fetch — trigger bulk CivitAI stats fetch."""
        db = CivitaiStatsDB.get_instance()
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

    @staticmethod
    async def handle_stats_status(request: web.Request) -> web.Response:
        """GET /api/lm/civitai-stats/status — check stats DB count."""
        db = CivitaiStatsDB.get_instance()
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
        db = CivitaiStatsDB.get_instance()
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
            instance = CivitaiStatsDB._instance
            if instance:
                instance.close()
                CivitaiStatsDB._instance = None

        app.on_shutdown.append(cleanup)
        logger.info("CivitAI stats routes registered")
