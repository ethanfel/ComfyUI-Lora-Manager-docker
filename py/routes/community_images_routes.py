"""Route registrar for Community Creations endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

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
_fetch_in_progress = False
_active_service: CommunityImagesFetchService | None = None
_fetch_cancel_requested = False
_refresh_model_in_progress = False
_inventory_refresh_task: asyncio.Task | None = None
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


async def _get_refreshed_lora_cache(scanner):
    """Coalesce scanner refreshes and avoid racing background initialization."""
    global _inventory_refresh_task

    task = _inventory_refresh_task
    if task is None or task.done():
        async def refresh_cache():
            is_initializing = getattr(scanner, "is_initializing", None)
            while callable(is_initializing) and is_initializing():
                await asyncio.sleep(0.1)

            # Reconciliation mutates the scanner's shared cache incrementally.
            # Let it finish even if the image fetch is cancelled so other pages
            # never observe a partially reconciled inventory.
            return await scanner.get_cached_data(
                force_refresh=True,
                rebuild_cache=getattr(scanner, "_cache", None) is None,
            )

        task = asyncio.create_task(refresh_cache())
        _inventory_refresh_task = task

    try:
        return await asyncio.shield(task)
    finally:
        if task.done() and _inventory_refresh_task is task:
            _inventory_refresh_task = None


def _normalize_hashes(value: object) -> list[str]:
    """Return unique, normalized model hashes from a settings/API value."""
    if not isinstance(value, list):
        return []

    hashes: list[str] = []
    seen: set[str] = set()
    for raw_hash in value:
        if not isinstance(raw_hash, str):
            continue
        sha256 = raw_hash.strip().lower()
        if not sha256 or sha256 in seen:
            continue
        seen.add(sha256)
        hashes.append(sha256)
    return hashes


def _get_hidden_model_hashes() -> set[str]:
    """Return community-only hidden model hashes from persistent settings."""
    settings = get_settings_manager()
    return set(_normalize_hashes(settings.get("community_hidden_model_hashes", [])))


def _clean_image(img: dict) -> dict:
    """Clean an image dict for JSON response — drop fetched_at, parse resources.

    Adds ``preview_url`` using the same /api/lm/previews endpoint as the main
    lora tab, so the browser gets the correct Content-Type headers.
    """
    out = {k: v for k, v in img.items() if k != "fetched_at"}
    # Parse resources JSON string into list
    if isinstance(out.get("resources"), str):
        try:
            out["resources"] = json.loads(out["resources"])
        except (json.JSONDecodeError, TypeError):
            out["resources"] = []
    # Build preview_url through the previews endpoint (same as main lora tab)
    local = out.get("local_filename")
    if local:
        from ..utils.example_images_paths import get_example_images_root
        import urllib.parse
        root = get_example_images_root()
        if root:
            full_path = os.path.join(root, local)
            encoded = urllib.parse.quote(full_path, safe="")
            out["preview_url"] = f"/api/lm/previews?path={encoded}"
            # Thumbnail URL for grid cards (smaller, faster loading)
            if local.endswith(".webp"):
                thumb_path = full_path.replace(".webp", "_thumb.webp")
                if os.path.exists(thumb_path):
                    thumb_encoded = urllib.parse.quote(thumb_path, safe="")
                    out["thumbnail_url"] = f"/api/lm/previews?path={thumb_encoded}"
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
            from ..utils.version import get_app_version
            rendered = template.render(
                request=request,
                settings=settings,
                t=server_i18n.get_translation,
                is_initializing=False,
                version=get_app_version(),
            )
            return web.Response(text=rendered, content_type="text/html")
        except Exception:
            logger.exception("Error rendering community creations page")
            return web.Response(text="Error loading page", status=500)

    @staticmethod
    async def _get_lora_models(
        force_refresh: bool = False,
    ) -> list[dict]:
        """Return the active LoRA inventory, including models without images."""
        try:
            scanner = await ServiceRegistry.get_lora_scanner()
            if force_refresh:
                cache = await _get_refreshed_lora_cache(scanner)
            else:
                cache = await scanner.get_cached_data()

            models: list[dict] = []
            models_by_hash: dict[str, dict] = {}
            for item in cache.raw_data if cache else []:
                if not item:
                    continue
                sha256 = item.get("sha256")
                if not isinstance(sha256, str) or not sha256.strip():
                    continue
                normalized_hash = sha256.strip().lower()

                civitai = item.get("civitai") or {}
                model_name = (
                    item.get("model_name")
                    or item.get("file_name")
                    or sha256[:8]
                )
                base_model = (
                    item.get("base_model")
                    or civitai.get("baseModel")
                    or ""
                )
                candidate = {
                    "sha256": sha256,
                    "civitai_model_id": civitai.get("modelId"),
                    "civitai_version_id": civitai.get("id"),
                    "author_username": (
                        (civitai.get("creator") or {}).get("username") or ""
                    ),
                    "model_name": str(model_name),
                    "base_model": str(base_model),
                }

                existing = models_by_hash.get(normalized_hash)
                if existing is None:
                    models_by_hash[normalized_hash] = candidate
                    models.append(candidate)
                    continue

                # Duplicate files may share a content hash but have different
                # sidecar completeness. Prefer the copy with CivitAI metadata so
                # the shared model remains fetchable.
                if (
                    not existing.get("civitai_model_id")
                    and candidate.get("civitai_model_id")
                ):
                    original_hash = existing["sha256"]
                    original_base_model = existing.get("base_model")
                    existing.update(candidate)
                    existing["sha256"] = original_hash
                    if not existing.get("base_model") and original_base_model:
                        existing["base_model"] = original_base_model
                elif not existing.get("base_model") and candidate.get("base_model"):
                    existing["base_model"] = candidate["base_model"]
            return models
        except Exception as exc:
            logger.exception("Failed to get lora scanner data: %s", exc)
            raise

    @staticmethod
    async def handle_fetch(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/fetch — fetch all or selected LoRAs."""
        global _active_service, _fetch_cancel_requested, _fetch_in_progress

        try:
            body = await request.json() if request.can_read_body else {}
        except Exception:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"success": False, "error": "JSON body must be an object"},
                status=400,
            )

        force = body.get("force") is True
        requested_hashes: list[str] | None = None
        if "hashes" in body:
            raw_hashes = body.get("hashes")
            if not isinstance(raw_hashes, list) or len(raw_hashes) > 5000:
                return web.json_response(
                    {
                        "success": False,
                        "error": "Invalid or too many hashes (max 5000)",
                    },
                    status=400,
                )
            requested_hashes = _normalize_hashes(raw_hashes)
            if len(requested_hashes) != len(raw_hashes) or not requested_hashes:
                return web.json_response(
                    {
                        "success": False,
                        "error": "Hashes must be unique, non-empty strings",
                    },
                    status=400,
                )

        if _refresh_model_in_progress:
            return web.json_response(
                {"success": False, "error": "A model refresh is already running"},
                status=409,
            )

        # Scanner reconciliation happens before the fetch service exists. Reject
        # overlapping requests during that preparation window so two bulk loops
        # can never run while sharing the module-level cancellation state.
        if _fetch_in_progress and _active_service is None:
            return web.json_response(
                {"success": False, "error": "A community fetch is starting"},
                status=409,
            )

        # If a fetch is already running, cancel it and let this one take over.
        if _fetch_in_progress and _active_service is not None:
            logger.info("Cancelling previous fetch to start a new one")
            _active_service.cancel()
            # Wait briefly for the old fetch loop to notice the cancel
            for _ in range(20):
                if not _fetch_in_progress:
                    break
                await asyncio.sleep(0.25)
            if _fetch_in_progress:
                return web.json_response(
                    {"success": False, "error": "Previous fetch still stopping, try again"},
                    status=409,
                )

        # A per-model refresh can start while the replacement fetch above is
        # waiting for an older bulk fetch to stop. Recheck after that await so
        # both fetch paths remain mutually exclusive.
        if _refresh_model_in_progress:
            return web.json_response(
                {"success": False, "error": "A model refresh is already running"},
                status=409,
            )
        _fetch_in_progress = True
        _fetch_cancel_requested = False

        try:
            db = CommunityImagesDB.get_instance()
            settings = get_settings_manager()
            api_key = settings.get("civitai_api_key", "")

            # Bulk actions reconcile the scanner so newly added LoRAs are picked
            # up. Selected actions use the inventory already shown by the picker,
            # avoiding a full filesystem walk before a targeted fetch starts.
            inventory = await CommunityImagesRoutes._get_lora_models(
                force_refresh=requested_hashes is None,
            )
            if _fetch_cancel_requested:
                return web.json_response({
                    "success": True,
                    "stored": 0,
                    "total": 0,
                    "skipped": 0,
                    "cancelled": True,
                })
            inventory_by_hash = {
                model["sha256"].lower(): model for model in inventory
            }

            if requested_hashes is not None:
                unknown = [
                    sha256 for sha256 in requested_hashes
                    if sha256 not in inventory_by_hash
                ]
                if unknown:
                    return web.json_response(
                        {
                            "success": False,
                            "error": "One or more selected models are no longer available",
                            "unknown_hashes": unknown,
                        },
                        status=400,
                    )
                models = [inventory_by_hash[sha256] for sha256 in requested_hashes]
                unavailable = [
                    model["sha256"] for model in models
                    if not model.get("civitai_model_id")
                ]
                if unavailable:
                    return web.json_response(
                        {
                            "success": False,
                            "error": "One or more selected models have no CivitAI metadata",
                            "unavailable_hashes": unavailable,
                        },
                        status=400,
                    )
            else:
                hidden_hashes = _get_hidden_model_hashes()
                models = [
                    model for model in inventory
                    if model.get("civitai_model_id")
                    and model["sha256"].lower() not in hidden_hashes
                ]

            if not models:
                return web.json_response({
                    "success": True,
                    "stored": 0,
                    "total": 0,
                    "skipped": 0,
                })

            # Skip models that already have enough community images (unless force)
            skipped = 0
            if not force:
                all_hashes = [m["sha256"] for m in models]
                counts = db.get_image_counts(all_hashes)
                models = [m for m in models if counts.get(m["sha256"], 0) < 10]
                skipped = len(all_hashes) - len(models)

            logger.info(
                "Fetching community images for %d models (%d already have 10+ images)",
                len(models),
                skipped,
            )

            if not models:
                return web.json_response({
                    "success": True,
                    "stored": 0,
                    "total": 0,
                    "skipped": skipped,
                })

            service = CommunityImagesFetchService(db=db, api_key=api_key)
            _active_service = service

            async def progress_callback(current: int, total: int, stored: int) -> None:
                await ws_manager.broadcast({
                    "type": "community_images_progress",
                    "current": current,
                    "total": total,
                    "stored": stored,
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
                "skipped": skipped,
                "cancelled": service.cancelled,
            })
        except Exception:
            logger.exception("Community images fetch failed")
            return web.json_response(
                {"success": False, "error": "Community images fetch failed"},
                status=500,
            )
        finally:
            _fetch_in_progress = False
            _fetch_cancel_requested = False

    @staticmethod
    async def handle_cancel(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/cancel — cancel in-progress fetch."""
        global _fetch_cancel_requested

        if _active_service is not None:
            _active_service.cancel()
            return web.json_response({"success": True, "message": "Cancel requested"})
        if _fetch_in_progress:
            _fetch_cancel_requested = True
            return web.json_response({
                "success": True,
                "message": "Fetch cancellation requested",
            })
        return web.json_response({"success": True, "message": "No fetch in progress"})

    @staticmethod
    async def _get_lora_metadata() -> tuple[list[str], dict[str, str], dict[str, str]]:
        """Return visible hashes and display metadata from the LoRA inventory."""
        lora_hashes: list[str] = []
        name_map: dict[str, str] = {}
        base_model_map: dict[str, str] = {}
        hidden_hashes = _get_hidden_model_hashes()
        for model in await CommunityImagesRoutes._get_lora_models():
            sha256 = model["sha256"]
            if sha256.lower() in hidden_hashes:
                continue
            lora_hashes.append(sha256)
            name_map[sha256] = model["model_name"]
            base_model_map[sha256] = model["base_model"]
        return lora_hashes, name_map, base_model_map

    @staticmethod
    async def handle_models(request: web.Request) -> web.Response:
        """GET /models — list all LoRAs, including new and hidden models."""
        refresh = request.query.get("refresh", "").lower() in {
            "1", "true", "yes",
        }
        try:
            inventory = await CommunityImagesRoutes._get_lora_models(
                force_refresh=refresh,
            )
            db = CommunityImagesDB.get_instance()
            counts = db.get_image_counts(
                [model["sha256"] for model in inventory]
            )
            hidden_hashes = _get_hidden_model_hashes()

            models_out = []
            for model in inventory:
                sha256 = model["sha256"]
                fetchable = bool(model.get("civitai_model_id"))
                models_out.append({
                    "sha256": sha256,
                    "model_name": model["model_name"],
                    "base_model": model["base_model"],
                    "image_count": counts.get(sha256, 0),
                    "hidden": sha256.lower() in hidden_hashes,
                    "fetchable": fetchable,
                    "unavailable_reason": (
                        "No CivitAI metadata" if not fetchable else ""
                    ),
                })

            models_out.sort(
                key=lambda model: (
                    not model["hidden"],
                    model["model_name"].lower(),
                    model["sha256"].lower(),
                )
            )
            return web.json_response({
                "success": True,
                "models": models_out,
                "total_models": len(models_out),
                "hidden_count": sum(1 for model in models_out if model["hidden"]),
            })
        except Exception:
            logger.exception("Failed to get community model inventory")
            return web.json_response(
                {"success": False, "error": "Internal server error"}, status=500
            )

    @staticmethod
    async def handle_visibility(request: web.Request) -> web.Response:
        """POST /visibility — hide or show a model on the Community page."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"success": False, "error": "JSON body must be an object"},
                status=400,
            )

        raw_hash = body.get("sha256")
        hidden = body.get("hidden")
        if not isinstance(raw_hash, str) or not raw_hash.strip():
            return web.json_response(
                {"success": False, "error": "Missing sha256"}, status=400
            )
        if not _SHA256_RE.fullmatch(raw_hash.strip()):
            return web.json_response(
                {"success": False, "error": "Invalid sha256"}, status=400
            )
        if not isinstance(hidden, bool):
            return web.json_response(
                {"success": False, "error": "hidden must be a boolean"},
                status=400,
            )

        sha256 = raw_hash.strip().lower()
        settings = get_settings_manager()
        hidden_hashes = set(_normalize_hashes(
            settings.get("community_hidden_model_hashes", [])
        ))
        if hidden:
            hidden_hashes.add(sha256)
        else:
            hidden_hashes.discard(sha256)
        settings.set("community_hidden_model_hashes", sorted(hidden_hashes))

        return web.json_response({
            "success": True,
            "sha256": raw_hash.strip(),
            "hidden": hidden,
            "hidden_count": len(hidden_hashes),
        })

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
        search_query = request.query.get("search", "").strip().lower()

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

            # Filter by search query (match against model name)
            if search_query:
                lora_hashes = [
                    h for h in lora_hashes
                    if search_query in name_map.get(h, "").lower()
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

            # Include installed base models even before their first Community
            # fetch. Their zero count makes newly added model families visible
            # in the tab bar without changing the total of populated models.
            hashes_with_images = db.get_hashes_with_images(
                list(base_model_map.keys())
            )
            base_model_counts: dict[str, int] = {
                base_model: 0
                for base_model in set(base_model_map.values())
                if base_model
            }
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
                {"success": False, "error": "Internal server error"}, status=500
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
        hashes = [h for h in hashes if isinstance(h, str) and h]
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
        sha256 = request.query.get("sha256", "").strip()
        if sha256:
            row = conn.execute(
                "SELECT sha256 FROM community_images "
                "WHERE civitai_image_id = ? AND lower(sha256) = lower(?)",
                (image_id, sha256),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT sha256 FROM community_images WHERE civitai_image_id = ? "
                "ORDER BY sha256 LIMIT 1",
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

        workflow_path = os.path.realpath(
            os.path.join(model_folder, "community", f"{image_id}.workflow.json")
        )
        # Ensure resolved path stays inside the model folder
        if not workflow_path.startswith(os.path.realpath(model_folder) + os.sep):
            return web.json_response(
                {"success": False, "error": "Invalid path"}, status=400
            )
        if not os.path.exists(workflow_path):
            return web.json_response(
                {"success": False, "error": "No workflow found"}, status=404
            )

        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read workflow file %s: %s", workflow_path, exc)
            return web.json_response(
                {"success": False, "error": "Workflow file is corrupted"}, status=500
            )

        return web.json_response({"success": True, "data": workflow_data})

    @staticmethod
    async def handle_refresh_model(request: web.Request) -> web.Response:
        """POST /api/lm/community-images/refresh-model — re-fetch for one model."""
        try:
            body = await request.json()
            sha256 = body.get("sha256")
        except Exception:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"}, status=400
            )

        if not sha256:
            return web.json_response(
                {"success": False, "error": "Missing sha256"}, status=400
            )

        global _refresh_model_in_progress
        if _fetch_in_progress or _refresh_model_in_progress:
            return web.json_response(
                {"success": False, "error": "Another community fetch is running"},
                status=409,
            )

        _refresh_model_in_progress = True
        try:
            return await CommunityImagesRoutes._refresh_model(sha256)
        finally:
            _refresh_model_in_progress = False

    @staticmethod
    async def _refresh_model(sha256: str) -> web.Response:
        """Fetch fresh community images for one known model hash."""
        # Use the normalized inventory so duplicate files with the same hash
        # benefit from its metadata merge instead of stopping at an incomplete
        # first scanner entry.
        matched_sha256 = sha256
        civitai_model_id = None
        civitai_version_id = None
        author_username = ""
        model_name = ""
        try:
            for model in await CommunityImagesRoutes._get_lora_models():
                if model["sha256"].lower() == sha256.lower():
                    matched_sha256 = model["sha256"]
                    civitai_model_id = model.get("civitai_model_id")
                    civitai_version_id = model.get("civitai_version_id")
                    author_username = model.get("author_username", "")
                    model_name = model.get("model_name", "")
                    break
        except Exception as exc:
            logger.info("Failed to get lora scanner data: %s", exc)

        if not civitai_model_id:
            return web.json_response(
                {"success": False, "error": "No CivitAI data for this model"},
                status=404,
            )

        try:
            db = CommunityImagesDB.get_instance()
            settings = get_settings_manager()
            api_key = settings.get("civitai_api_key", "")
            service = CommunityImagesFetchService(db=db, api_key=api_key)
            try:
                # Fetch first — only delete old data after we have new results
                before_fetch = time.time()
                count = await service.fetch_images_for_model(
                    matched_sha256, civitai_model_id, author_username,
                    civitai_version_id=civitai_version_id,
                    model_name=model_name,
                )
            finally:
                await service.close()

            if count == 0:
                return web.json_response({
                    "success": False,
                    "error": "CivitAI returned no images for this version",
                })

            # Remove stale images that were not part of the fresh fetch
            stale = db.delete_stale(matched_sha256, before_fetch)
            if stale:
                logger.info(
                    "Removed %d stale community images for %s",
                    stale,
                    matched_sha256,
                )

            # Return the refreshed images
            images = db.get_by_hashes([matched_sha256])
            clean_images = [
                _clean_image(img)
                for img in images.get(matched_sha256, [])
            ]

            return web.json_response({
                "success": True,
                "stored": count,
                "images": clean_images,
            })
        except Exception:
            logger.exception("Failed to refresh community images for %s", sha256)
            return web.json_response(
                {"success": False, "error": "Refresh failed"}, status=500
            )

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
                {"success": False, "error": "Internal server error"}, status=500
            )

    @staticmethod
    async def handle_test_api(request: web.Request) -> web.Response:
        """GET /api/lm/community-images/test-api — test CivitAI API connectivity."""
        import aiohttp as _aiohttp

        settings = get_settings_manager()
        api_key = settings.get("civitai_api_key", "")

        result = {
            "has_api_key": bool(api_key),
            "api_key_prefix": "(set)" if api_key else "(empty)",
        }

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = _aiohttp.ClientTimeout(total=15)
        try:
            async with _aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                # Test 1: simple images endpoint with small limit
                url = "https://civitai.com/api/v1/images"
                params = {"limit": "1", "sort": "Most Reactions"}
                async with session.get(url, params=params) as resp:
                    result["images_endpoint"] = {
                        "status": resp.status,
                        "ok": resp.status == 200,
                    }
                    # Capture rate limit headers
                    for h in resp.headers:
                        hl = h.lower()
                        if "rate" in hl or "limit" in hl or "retry" in hl or "remaining" in hl:
                            result.setdefault("rate_limit_headers", {})[h] = resp.headers[h]

                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        result["images_endpoint"]["returned_items"] = len(items)
                    elif resp.status == 429:
                        result["images_endpoint"]["error"] = "RATE LIMITED"
                        body = await resp.text()
                        if body:
                            result["images_endpoint"]["body"] = body[:500]
                    else:
                        body = await resp.text()
                        result["images_endpoint"]["error"] = body[:500]

        except _aiohttp.ClientError as exc:
            result["connection_error"] = str(exc)
        except Exception as exc:
            result["error"] = str(exc)

        result["success"] = True
        return web.json_response(result)

    @classmethod
    def setup_routes(cls, app: web.Application) -> None:
        """Register community images routes."""
        app.router.add_get("/community", cls.handle_page)
        app.router.add_post("/api/lm/community-images/fetch", cls.handle_fetch)
        app.router.add_post("/api/lm/community-images/cancel", cls.handle_cancel)
        app.router.add_get("/api/lm/community-images/models", cls.handle_models)
        app.router.add_post(
            "/api/lm/community-images/visibility", cls.handle_visibility
        )
        app.router.add_get("/api/lm/community-images/by-models", cls.handle_by_models)
        app.router.add_post("/api/lm/community-images/by-hashes", cls.handle_by_hashes)
        app.router.add_get(
            "/api/lm/community-images/workflow/{image_id}", cls.handle_workflow
        )
        app.router.add_post(
            "/api/lm/community-images/refresh-model", cls.handle_refresh_model
        )
        app.router.add_get("/api/lm/community-images/status", cls.handle_status)
        app.router.add_get("/api/lm/community-images/test-api", cls.handle_test_api)

        async def cleanup(app):
            instance = CommunityImagesDB._instance
            if instance:
                instance.close()
                CommunityImagesDB._instance = None

        app.on_shutdown.append(cleanup)
        logger.info("Community images routes registered")
