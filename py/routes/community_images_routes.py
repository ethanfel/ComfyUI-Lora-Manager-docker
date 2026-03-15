"""Route registrar for Community Creations endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import os

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

# Module-level fetch state — prevents concurrent fetches and supports cancel
_fetch_lock = asyncio.Lock()
_active_service: CommunityImagesFetchService | None = None
_fetch_task: asyncio.Task | None = None


def _clean_image(img: dict) -> dict:
    """Clean an image dict for JSON response — drop fetched_at, parse resources."""
    out = {k: v for k, v in img.items() if k != "fetched_at"}
    # Parse resources JSON string into list
    if isinstance(out.get("resources"), str):
        try:
            out["resources"] = json.loads(out["resources"])
        except (json.JSONDecodeError, TypeError):
            out["resources"] = []
    return out


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
        global _active_service, _fetch_task

        if _fetch_lock.locked():
            return web.json_response(
                {"success": False, "error": "Fetch already in progress"},
                status=409,
            )

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

            async with _fetch_lock:
                service = CommunityImagesFetchService(db=db, api_key=api_key)
                _active_service = service

                async def progress_callback(current: int, total: int) -> None:
                    await ws_manager.broadcast({
                        "type": "community_images_progress",
                        "current": current,
                        "total": total,
                        "progress": round(current / total * 100) if total else 0,
                    })

                try:
                    stored = await service.fetch_all(
                        models, progress_callback=progress_callback
                    )
                finally:
                    await service.close()
                    _active_service = None

                return web.json_response({
                    "success": True,
                    "stored": stored,
                    "total": len(models),
                    "skipped": len(existing),
                    "cancelled": service.cancelled,
                })
        except Exception as exc:
            logger.exception("Community images fetch failed")
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    @staticmethod
    async def handle_cancel(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/cancel — cancel in-progress fetch."""
        if _active_service is not None:
            _active_service.cancel()
            return web.json_response({"success": True, "message": "Cancel requested"})
        return web.json_response({"success": True, "message": "No fetch in progress"})

    @staticmethod
    async def _get_lora_metadata() -> tuple[list[str], dict[str, str], dict[str, str]]:
        """Return (hashes, name_map, base_model_map) from the lora scanner."""
        lora_hashes: list[str] = []
        name_map: dict[str, str] = {}
        base_model_map: dict[str, str] = {}
        try:
            scanner = await ServiceRegistry.get_lora_scanner()
            cache = await scanner.get_cached_data()
            for item in cache.raw_data:
                if not item:
                    continue
                sha256 = item.get("sha256")
                if sha256:
                    lora_hashes.append(sha256)
                    name_map[sha256] = (
                        item.get("model_name")
                        or item.get("file_name")
                        or "Unknown"
                    )
                    base_model_map[sha256] = item.get("base_model") or ""
        except Exception as exc:
            logger.info("Failed to get lora scanner data: %s", exc)
        return lora_hashes, name_map, base_model_map

    @staticmethod
    async def handle_by_models(request: web.Request) -> web.Response:
        """GET /api/lm/community-images/by-models — paginated models with images."""
        try:
            page = int(request.query.get("page", "1"))
            page_size = int(request.query.get("page_size", "10"))
            sort = request.query.get("sort", "reactions:desc")
        except (ValueError, TypeError):
            page, page_size, sort = 1, 10, "reactions:desc"

        page_size = min(max(page_size, 1), 50)
        page = max(page, 1)
        base_model_filter = request.query.get("base_model", "")

        try:
            lora_hashes, name_map, base_model_map = (
                await CommunityImagesRoutes._get_lora_metadata()
            )

            # Filter by base model if requested
            if base_model_filter:
                lora_hashes = [
                    h for h in lora_hashes
                    if base_model_map.get(h, "") == base_model_filter
                ]

            db = CommunityImagesDB.get_instance()
            result = db.get_models_paginated(
                allowed_hashes=lora_hashes,
                page=page,
                page_size=page_size,
                sort=sort,
            )

            # Sort by lora name if requested (DB can't do this — needs name_map)
            sort_key = sort.split(":")[0]
            if sort_key == "lora" and result["models"]:
                direction = sort.split(":")[-1] if ":" in sort else "asc"
                reverse = direction != "asc"
                all_hashes = []
                for i in range(0, len(lora_hashes), 500):
                    chunk = lora_hashes[i : i + 500]
                    has_images = db.get_hashes_with_images(chunk)
                    all_hashes.extend(has_images)
                all_hashes.sort(
                    key=lambda h: name_map.get(h, "Unknown").lower(),
                    reverse=reverse,
                )
                total = len(all_hashes)
                offset = (page - 1) * page_size
                page_hashes = all_hashes[offset : offset + page_size]
                images = db.get_by_hashes(page_hashes)
                result = {"models": page_hashes, "total": total, "images": images}

            # Build response with model names
            models_out = []
            for sha in result["models"]:
                model_images = result["images"].get(sha, [])
                clean_images = [_clean_image(img) for img in model_images]
                models_out.append({
                    "sha256": sha,
                    "model_name": name_map.get(sha, "Unknown"),
                    "base_model": base_model_map.get(sha, ""),
                    "image_count": len(clean_images),
                    "images": clean_images,
                })

            # Collect available base models (only those with community images)
            hashes_with_images = db.get_hashes_with_images(
                list(base_model_map.keys())
            )
            base_model_counts: dict[str, int] = {}
            for h in hashes_with_images:
                bm = base_model_map.get(h, "")
                if bm:
                    base_model_counts[bm] = base_model_counts.get(bm, 0) + 1

            return web.json_response({
                "success": True,
                "models": models_out,
                "page": page,
                "page_size": page_size,
                "total_models": result["total"],
                "total_pages": (result["total"] + page_size - 1) // page_size,
                "base_models": base_model_counts,
            })

        except Exception as exc:
            logger.exception("Failed to get paginated community models")
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
            clean[sha] = [_clean_image(img) for img in img_list]
        return web.json_response({"success": True, "images": clean})

    @staticmethod
    async def handle_workflow(request: web.Request) -> web.Response:
        """GET /api/lm/community-images/workflow/{image_id} — return workflow JSON."""
        try:
            image_id = int(request.match_info["image_id"])
        except (KeyError, ValueError):
            return web.json_response(
                {"success": False, "error": "Invalid image ID"}, status=400
            )

        db = CommunityImagesDB.get_instance()
        conn = db._ensure_conn()
        row = conn.execute(
            "SELECT sha256 FROM community_images WHERE civitai_image_id = ?",
            (image_id,),
        ).fetchone()
        if not row:
            return web.json_response(
                {"success": False, "error": "Image not found"}, status=404
            )

        from ..utils.example_images_paths import get_model_folder
        model_folder = get_model_folder(row["sha256"])
        if not model_folder:
            return web.json_response(
                {"success": False, "error": "Model folder not found"}, status=404
            )

        workflow_path = os.path.join(
            model_folder, "community", f"{image_id}.workflow.json"
        )
        if not os.path.exists(workflow_path):
            return web.json_response(
                {"success": False, "error": "No workflow found"}, status=404
            )

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)

        return web.json_response({"success": True, "data": workflow_data})

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
        app.router.add_post("/api/lm/community-images/cancel", cls.handle_cancel)
        app.router.add_get("/api/lm/community-images/by-models", cls.handle_by_models)
        app.router.add_post("/api/lm/community-images/by-hashes", cls.handle_by_hashes)
        app.router.add_get(
            "/api/lm/community-images/workflow/{image_id}", cls.handle_workflow
        )
        app.router.add_get("/api/lm/community-images/status", cls.handle_status)

        async def cleanup(app):
            instance = CommunityImagesDB._instance
            if instance:
                instance.close()
                CommunityImagesDB._instance = None

        app.on_shutdown.append(cleanup)
        logger.info("Community images routes registered")
