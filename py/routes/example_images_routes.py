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
        from ..utils.example_images_paths import get_model_folder

        model_hash = request.query.get("model_hash")
        filename = request.query.get("filename")
        if not model_hash or not filename:
            return web.json_response(
                {"success": False, "error": "Missing model_hash or filename"},
                status=400,
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
        """POST /api/lm/example-images/scan-workflows — scan existing PNGs for embedded workflows."""
        from ..utils.example_images_paths import get_example_images_root

        root = get_example_images_root()
        if not root or not os.path.isdir(root):
            return web.json_response(
                {"success": False, "error": "No example images path configured or path not found"},
                status=400,
            )

        result = await asyncio.to_thread(
            ExampleImagesProcessor.scan_existing_workflows, root
        )
        return web.json_response({
            "success": True,
            "scanned": result["scanned"],
            "found": result["found"],
            "errors": result["errors"],
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
