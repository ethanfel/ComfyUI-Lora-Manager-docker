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
            cls._template_env.filters["t"] = server_i18n.create_template_filter()
        return cls._template_env

    @staticmethod
    async def handle_page(request: web.Request) -> web.Response:
        """GET /community — render the Community Creations page."""
        try:
            settings = get_settings_manager()
            user_language = settings.get("language", "en")
            server_i18n.set_locale(user_language)

            env = CommunityImagesRoutes._get_template_env()
            template = env.get_template("community_creations.html")
            rendered = template.render(
                request=request,
                settings=settings,
                t=server_i18n.get_translation,
                is_initializing=False,
            )
            return web.Response(text=rendered, content_type="text/html")
        except Exception:
            logger.exception("Error rendering community creations page")
            return web.Response(text="Error loading page", status=500)

    @staticmethod
    async def handle_fetch(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/fetch — trigger bulk fetch."""
        try:
            db = CommunityImagesDB.get_instance()
            settings = get_settings_manager()
            api_key = settings.get("civitai_api_key", "")

            # Collect LoRA models with civitai data
            models = []
            try:
                scanner = await ServiceRegistry.get_lora_scanner()
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
                            "author_username": creator or "",
                        })
            except Exception as exc:
                logger.info("Failed to get lora scanner data: %s", exc)

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
        try:
            db = CommunityImagesDB.get_instance()
            return web.json_response({
                "success": True,
                "count": db.count(),
            })
        except Exception as exc:
            logger.exception("Community images status check failed")
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

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
