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
        try:
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
                        version_id = civitai.get("id")  # civitai version id
                        sha256 = item.get("sha256")
                        if model_id and sha256:
                            models.append({
                                "sha256": sha256,
                                "civitai_model_id": model_id,
                                "civitai_version_id": version_id,
                            })
                except Exception as exc:
                    logger.debug("Failed to get scanner data: %s", exc)

            # Log collection summary for debugging
            with_vid = sum(1 for m in models if m.get("civitai_version_id"))
            without_vid = len(models) - with_vid
            logger.info("Collected %d models for stats fetch (%d with version_id, %d without)",
                        len(models), with_vid, without_vid)
            if models:
                sample = models[:3]
                logger.info("Sample collected sha256 values: %s",
                            [(m["sha256"][:16], m["civitai_model_id"]) for m in sample])

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
    async def handle_stats_status(request: web.Request) -> web.Response:
        """GET /api/lm/civitai-stats/status — check stats DB count."""
        db = CivitaiStatsDB.get_instance()
        all_stats = db.get_all()
        db_hashes = list(all_stats.keys())[:5]

        # Debug: find a model WITH civitai data and check if its hash is in the DB
        debug = {}
        try:
            scanner = await ServiceRegistry.get_lora_scanner()
            cache = await scanner.get_cached_data()
            # Find first model with civitai modelId
            for item in cache.raw_data:
                if not item:
                    continue
                civitai = item.get("civitai") or {}
                mid = civitai.get("modelId")
                if mid:
                    item_sha = item.get("sha256", "")
                    db_match = db.get_by_hashes([item_sha])
                    debug = {
                        "scanner_sha256": item_sha,
                        "model_name": item.get("model_name", "")[:40],
                        "civitai_model_id": mid,
                        "civitai_version_id": civitai.get("id"),
                        "in_stats_db": bool(db_match),
                    }
                    break
            # Reverse lookup: pick a DB entry, find same model in scanner
            conn = db._ensure_conn()
            db_row = conn.execute(
                "SELECT sha256, civitai_model_id, civitai_version_id FROM model_stats LIMIT 1"
            ).fetchone()
            if db_row:
                db_mid = db_row["civitai_model_id"]
                db_vid = db_row["civitai_version_id"]
                db_sha = db_row["sha256"]
                # Find this model_id in scanner cache
                scanner_match = None
                for item in cache.raw_data:
                    if not item:
                        continue
                    c = item.get("civitai") or {}
                    if c.get("modelId") == db_mid:
                        scanner_match = {
                            "sha256": item.get("sha256", ""),
                            "name": item.get("model_name", "")[:40],
                        }
                        break
                debug["db_entry"] = {
                    "sha256": db_sha,
                    "civitai_model_id": db_mid,
                    "civitai_version_id": db_vid,
                }
                debug["scanner_match_for_db_model"] = scanner_match
        except Exception as exc:
            debug = {"error": str(exc)}

        return web.json_response({
            "success": True,
            "count": db.count(),
            "db_sample_hashes": [h[:16] for h in db_hashes],
            "debug": debug,
        })

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
        if not hashes:
            return web.json_response({"success": True, "stats": {}})
        db = CivitaiStatsDB.get_instance()
        logger.info("by-hashes query: %d hashes, sample=%s", len(hashes),
                     [h[:16] for h in hashes[:3]])
        stats = db.get_by_hashes(hashes)
        logger.info("by-hashes result: %d matches out of %d queried", len(stats), len(hashes))
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
