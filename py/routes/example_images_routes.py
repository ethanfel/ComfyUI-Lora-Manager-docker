from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable, Mapping

from aiohttp import web

from .example_images_route_registrar import ExampleImagesRouteRegistrar
from .handlers.example_images_handlers import (
    ExampleImagesDownloadHandler,
    ExampleImagesFileHandler,
    ExampleImagesHandlerSet,
    ExampleImagesManagementHandler,
)
from ..services.use_cases.example_images import (
    DownloadExampleImagesUseCase,
    ImportExampleImagesUseCase,
)
from ..utils.example_images_download_manager import (
    DownloadManager,
    get_default_download_manager,
)
from ..utils.example_images_file_manager import ExampleImagesFileManager
from ..utils.example_images_processor import ExampleImagesProcessor
from ..services.example_images_cleanup_service import ExampleImagesCleanupService

logger = logging.getLogger(__name__)


class ExampleImagesRoutes:
    """Route controller for example image endpoints."""

    def __init__(
        self,
        *,
        ws_manager,
        download_manager: DownloadManager | None = None,
        processor=ExampleImagesProcessor,
        file_manager=ExampleImagesFileManager,
        cleanup_service: ExampleImagesCleanupService | None = None,
    ) -> None:
        if ws_manager is None:
            raise ValueError("ws_manager is required")
        self._download_manager = download_manager or get_default_download_manager(ws_manager)
        self._processor = processor
        self._file_manager = file_manager
        self._cleanup_service = cleanup_service or ExampleImagesCleanupService()
        self._handler_set: ExampleImagesHandlerSet | None = None
        self._handler_mapping: Mapping[str, Callable[[web.Request], web.StreamResponse]] | None = None

    @classmethod
    def setup_routes(cls, app: web.Application, *, ws_manager) -> None:
        """Register routes on the given aiohttp application using default wiring."""

        controller = cls(ws_manager=ws_manager)
        controller.register(app)

    def register(self, app: web.Application) -> None:
        """Bind the controller's handlers to the aiohttp router."""

        registrar = ExampleImagesRouteRegistrar(app)
        registrar.register_routes(self.to_route_mapping())

        # Workflow download endpoint (simple file read, not part of handler set)
        app.router.add_get(
            "/api/lm/example-images/workflow",
            self._handle_workflow,
        )

        # Scan existing PNGs for embedded workflows
        app.router.add_post(
            "/api/lm/example-images/scan-workflows",
            self._handle_scan_workflows,
        )

    @staticmethod
    async def _handle_workflow(request: web.Request) -> web.Response:
        """GET /api/lm/example-images/workflow?model_hash=X&filename=Y"""
        import re
        from ..utils.example_images_paths import get_model_folder

        model_hash = request.query.get("model_hash")
        filename = request.query.get("filename")
        if not model_hash or not filename:
            return web.json_response(
                {"success": False, "error": "Missing model_hash or filename"},
                status=400,
            )

        # Validate model_hash is a hex string
        if not re.fullmatch(r"[a-fA-F0-9]{64}", model_hash):
            return web.json_response(
                {"success": False, "error": "Invalid model_hash"}, status=400
            )

        # Reject path traversal attempts
        base_name = os.path.basename(filename)
        if not base_name or base_name != filename:
            return web.json_response(
                {"success": False, "error": "Invalid filename"}, status=400
            )

        model_folder = get_model_folder(model_hash)
        if not model_folder:
            return web.json_response(
                {"success": False, "error": "Model folder not found"}, status=404
            )

        stem = os.path.splitext(base_name)[0]
        workflow_path = os.path.realpath(
            os.path.join(model_folder, f"{stem}.workflow.json")
        )
        # Ensure resolved path stays inside the model folder
        if not workflow_path.startswith(os.path.realpath(model_folder) + os.sep):
            return web.json_response(
                {"success": False, "error": "Invalid filename"}, status=400
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
    async def _handle_scan_workflows(request: web.Request) -> web.Response:
        """POST /api/lm/example-images/scan-workflows — scan existing media for workflows.

        Phase 1: Extract workflows from PNG metadata (embedded in file).
        Phase 2: For non-PNG files (videos, WebP, etc.), look up CivitAI
        metadata and extract workflow from the API ``meta.comfy`` field.
        """
        from ..utils.example_images_paths import get_example_images_root
        from ..services.service_registry import ServiceRegistry
        from ..utils.metadata_manager import MetadataManager

        root = get_example_images_root()
        if not root or not os.path.isdir(root):
            return web.json_response(
                {"success": False, "error": "No example images path configured or path not found"},
                status=400,
            )

        # Phase 1: PNG embedded workflows
        result = await asyncio.to_thread(
            ExampleImagesProcessor.scan_existing_workflows, root
        )
        scanned = result["scanned"]
        found = result["found"]
        errors = result["errors"]

        # Phase 2: Non-PNG files — extract workflow from CivitAI API metadata
        files_by_hash = await asyncio.to_thread(
            ExampleImagesProcessor.collect_files_needing_metadata_workflow, root
        )
        if files_by_hash:
            scanners = [
                await ServiceRegistry.get_lora_scanner(),
                await ServiceRegistry.get_checkpoint_scanner(),
                await ServiceRegistry.get_embedding_scanner(),
            ]
            for model_hash, file_list in files_by_hash.items():
                # Find model data in scanner caches
                model_data = None
                for scanner in scanners:
                    if scanner.has_hash(model_hash):
                        cache = await scanner.get_cached_data()
                        for item in cache.raw_data:
                            if item.get("sha256") == model_hash:
                                model_data = item
                                break
                    if model_data:
                        break
                if not model_data:
                    continue

                await MetadataManager.hydrate_model_data(model_data)
                civitai_images = (model_data.get("civitai") or {}).get("images") or []

                # Build index lookup: image_N -> civitai_images[N]
                meta_by_index: dict[int, dict] = {}
                for idx, ci in enumerate(civitai_images):
                    meta = ci.get("meta")
                    if meta and isinstance(meta, dict):
                        meta_by_index[idx] = meta

                for file_path, filename in file_list:
                    scanned += 1
                    # Match image_N.ext -> index N
                    stem = os.path.splitext(filename)[0]
                    if stem.startswith("image_"):
                        try:
                            idx = int(stem[6:])
                        except ValueError:
                            continue
                    else:
                        continue
                    meta = meta_by_index.get(idx)
                    if not meta:
                        continue
                    try:
                        if ExampleImagesProcessor._extract_workflow_from_api_meta(meta, file_path):
                            found += 1
                            logger.info("Extracted workflow from API meta for %s", file_path)
                    except Exception:
                        errors += 1
                        logger.error("Error extracting API workflow for %s", file_path, exc_info=True)

        return web.json_response({
            "success": True,
            "scanned": scanned,
            "found": found,
            "errors": errors,
        })

    def to_route_mapping(self) -> Mapping[str, Callable[[web.Request], web.StreamResponse]]:
        """Return the registrar-compatible mapping of handler names to callables."""

        if self._handler_mapping is None:
            handler_set = self._build_handler_set()
            self._handler_set = handler_set
            self._handler_mapping = handler_set.to_route_mapping()
        return self._handler_mapping

    def _build_handler_set(self) -> ExampleImagesHandlerSet:
        logger.debug("Building ExampleImagesHandlerSet with %s, %s, %s", self._download_manager, self._processor, self._file_manager)
        download_use_case = DownloadExampleImagesUseCase(download_manager=self._download_manager)
        download_handler = ExampleImagesDownloadHandler(download_use_case, self._download_manager)
        import_use_case = ImportExampleImagesUseCase(processor=self._processor)
        management_handler = ExampleImagesManagementHandler(
            import_use_case,
            self._processor,
            self._cleanup_service,
        )
        file_handler = ExampleImagesFileHandler(self._file_manager)
        return ExampleImagesHandlerSet(
            download=download_handler,
            management=management_handler,
            files=file_handler,
        )
