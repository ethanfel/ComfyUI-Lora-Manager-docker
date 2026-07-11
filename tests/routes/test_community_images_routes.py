"""Route tests for Community Creations model inventory and selection."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import py.routes.community_images_routes as community_routes
from py.routes.community_images_routes import CommunityImagesRoutes
from py.services.community_images_db import CommunityImagesDB


HASH_ALPHA = "a" * 64
HASH_BETA = "b" * 64
HASH_HIDDEN = "c" * 64
HASH_NO_CIVITAI = "d" * 64
HASH_UNKNOWN = "e" * 64


def _model(
    sha256: str,
    name: str,
    *,
    model_id: int | None = 1,
    version_id: int | None = 10,
    base_model: str = "SDXL 1.0",
) -> dict:
    civitai = {}
    if model_id is not None:
        civitai = {
            "modelId": model_id,
            "id": version_id,
            "creator": {"username": f"author-{name.lower()}"},
        }
    return {
        "sha256": sha256,
        "model_name": name,
        "file_name": f"{name.lower()}.safetensors",
        "base_model": base_model,
        "civitai": civitai,
    }


def _store_image(db: CommunityImagesDB, image_id: int, sha256: str) -> None:
    db.upsert({
        "civitai_image_id": image_id,
        "sha256": sha256,
        "civitai_model_id": image_id * 10,
        "username": "community-user",
        "image_url": f"https://example.com/{image_id}.png",
        "prompt": f"Community image prompt {image_id}",
        "like_count": image_id,
    })


class FakeSettings:
    def __init__(self, **values: object) -> None:
        self.values = {
            "civitai_api_key": "",
            "community_hidden_model_hashes": [],
            **values,
        }
        self.set_calls: list[tuple[str, object]] = []

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value
        self.set_calls.append((key, value))


class FakeScanner:
    def __init__(
        self,
        cached_models: list[dict],
        *,
        refreshed_models: list[dict] | None = None,
    ) -> None:
        self._cache = SimpleNamespace(raw_data=list(cached_models))
        self._refreshed_models = (
            list(refreshed_models) if refreshed_models is not None else None
        )
        self.calls: list[tuple[bool, bool]] = []
        self.cancel_calls = 0

    async def get_cached_data(
        self,
        force_refresh: bool = False,
        rebuild_cache: bool = False,
    ) -> SimpleNamespace:
        self.calls.append((force_refresh, rebuild_cache))
        if force_refresh and self._refreshed_models is not None:
            self._cache = SimpleNamespace(raw_data=list(self._refreshed_models))
        return self._cache

    def cancel_task(self) -> None:
        self.cancel_calls += 1


class RecordingFetchService:
    instances: list["RecordingFetchService"] = []

    def __init__(self, db: CommunityImagesDB, api_key: str | None = None) -> None:
        self.db = db
        self.api_key = api_key
        self.models: list[dict] = []
        self.cancelled = False
        self.closed = False
        self.__class__.instances.append(self)

    async def fetch_all(self, models: list[dict], progress_callback=None) -> int:
        self.models = [dict(model) for model in models]
        return len(models) * 2

    async def close(self) -> None:
        self.closed = True

    def cancel(self) -> None:
        self.cancelled = True


class RecordingRefreshService(RecordingFetchService):
    refresh_calls: list[dict] = []

    async def fetch_images_for_model(
        self,
        sha256: str,
        civitai_model_id: int,
        author_username: str,
        *,
        civitai_version_id: int | None = None,
        model_name: str = "",
    ) -> int:
        self.__class__.refresh_calls.append({
            "sha256": sha256,
            "civitai_model_id": civitai_model_id,
            "author_username": author_username,
            "civitai_version_id": civitai_version_id,
            "model_name": model_name,
        })
        return 1


@pytest.fixture
def community_db(tmp_path) -> CommunityImagesDB:
    db = CommunityImagesDB(db_path=tmp_path / "community-routes.db")
    db.init()
    yield db
    db.close()


@pytest.fixture(autouse=True)
def reset_fetch_state():
    previous_in_progress = community_routes._fetch_in_progress
    previous_service = community_routes._active_service
    previous_cancel_requested = community_routes._fetch_cancel_requested
    previous_refresh_in_progress = community_routes._refresh_model_in_progress
    previous_inventory_task = community_routes._inventory_refresh_task
    community_routes._fetch_in_progress = False
    community_routes._active_service = None
    community_routes._fetch_cancel_requested = False
    community_routes._refresh_model_in_progress = False
    community_routes._inventory_refresh_task = None
    RecordingFetchService.instances.clear()
    RecordingRefreshService.refresh_calls.clear()
    yield
    community_routes._fetch_in_progress = previous_in_progress
    community_routes._active_service = previous_service
    community_routes._fetch_cancel_requested = previous_cancel_requested
    community_routes._refresh_model_in_progress = previous_refresh_in_progress
    community_routes._inventory_refresh_task = previous_inventory_task


@pytest.fixture
async def route_client_factory(monkeypatch, community_db):
    clients: list[TestClient] = []

    async def create(
        scanner: FakeScanner,
        *,
        settings: FakeSettings | None = None,
        fetch_service=RecordingFetchService,
    ) -> tuple[TestClient, FakeSettings]:
        fake_settings = settings or FakeSettings()

        async def get_lora_scanner() -> FakeScanner:
            return scanner

        monkeypatch.setattr(
            community_routes,
            "ServiceRegistry",
            SimpleNamespace(get_lora_scanner=get_lora_scanner),
        )
        monkeypatch.setattr(
            community_routes,
            "CommunityImagesDB",
            SimpleNamespace(get_instance=lambda: community_db, _instance=None),
        )
        monkeypatch.setattr(
            community_routes,
            "get_settings_manager",
            lambda: fake_settings,
        )
        monkeypatch.setattr(
            community_routes,
            "CommunityImagesFetchService",
            fetch_service,
        )

        app = web.Application()
        CommunityImagesRoutes.setup_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client, fake_settings

    yield create

    for client in clients:
        await client.close()


@pytest.mark.asyncio
async def test_model_inventory_refreshes_scanner_and_includes_zero_image_models(
    route_client_factory,
    community_db,
):
    existing = _model(HASH_ALPHA, "Alpha", model_id=11, version_id=101)
    new_model = _model(HASH_BETA, "Beta", model_id=22, version_id=202)
    scanner = FakeScanner([existing], refreshed_models=[existing, new_model])
    _store_image(community_db, 1, HASH_ALPHA)
    client, _ = await route_client_factory(scanner)

    response = await client.get("/api/lm/community-images/models?refresh=true")
    payload = await response.json()

    assert response.status == 200
    assert payload["success"] is True
    assert payload["total_models"] == 2
    by_hash = {model["sha256"]: model for model in payload["models"]}
    assert by_hash[HASH_ALPHA]["image_count"] == 1
    assert by_hash[HASH_BETA]["image_count"] == 0
    assert by_hash[HASH_BETA]["fetchable"] is True
    assert scanner.calls == [(True, False)]


@pytest.mark.asyncio
async def test_concurrent_inventory_refreshes_share_one_scanner_task(
    route_client_factory,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    started = asyncio.Event()
    release = asyncio.Event()
    original_get_cached_data = scanner.get_cached_data

    async def slow_get_cached_data(
        force_refresh: bool = False,
        rebuild_cache: bool = False,
    ):
        if force_refresh:
            started.set()
            await release.wait()
        return await original_get_cached_data(force_refresh, rebuild_cache)

    scanner.get_cached_data = slow_get_cached_data
    client, _ = await route_client_factory(scanner)

    first = asyncio.create_task(
        client.get("/api/lm/community-images/models?refresh=true")
    )
    await started.wait()
    second = asyncio.create_task(
        client.get("/api/lm/community-images/models?refresh=true")
    )
    await asyncio.sleep(0.05)
    release.set()

    first_response, second_response = await asyncio.gather(first, second)
    assert first_response.status == 200
    assert second_response.status == 200
    assert scanner.calls == [(True, False)]


@pytest.mark.asyncio
async def test_inventory_refresh_reconciles_after_scanner_initialization():
    existing = _model(HASH_ALPHA, "Alpha")
    new_model = _model(HASH_BETA, "Beta")
    scanner = FakeScanner([existing], refreshed_models=[existing, new_model])
    initializing = True

    def is_initializing() -> bool:
        return initializing

    scanner.is_initializing = is_initializing
    refresh_task = asyncio.create_task(
        community_routes._get_refreshed_lora_cache(scanner)
    )

    await asyncio.sleep(0.05)
    assert scanner.calls == []
    initializing = False
    cache = await refresh_task

    assert scanner.calls == [(True, False)]
    assert cache.raw_data == [existing, new_model]


@pytest.mark.asyncio
async def test_model_inventory_prefers_metadata_bearing_duplicate(
    route_client_factory,
):
    without_metadata = _model(HASH_ALPHA, "Local Copy", model_id=None)
    with_metadata = _model(HASH_ALPHA.upper(), "CivitAI Copy", model_id=42)
    scanner = FakeScanner([without_metadata, with_metadata])
    client, _ = await route_client_factory(scanner)

    response = await client.get("/api/lm/community-images/models")
    payload = await response.json()

    assert response.status == 200
    assert len(payload["models"]) == 1
    assert payload["models"][0]["sha256"] == HASH_ALPHA
    assert payload["models"][0]["model_name"] == "CivitAI Copy"
    assert payload["models"][0]["fetchable"] is True


@pytest.mark.asyncio
async def test_model_refresh_uses_metadata_bearing_duplicate(
    route_client_factory,
):
    without_metadata = _model(
        HASH_ALPHA,
        "Local Copy",
        model_id=None,
        base_model="Pony",
    )
    with_metadata = _model(
        HASH_ALPHA.upper(),
        "CivitAI Copy",
        model_id=42,
        version_id=420,
        base_model="",
    )
    scanner = FakeScanner([without_metadata, with_metadata])
    client, _ = await route_client_factory(
        scanner,
        fetch_service=RecordingRefreshService,
    )

    response = await client.post(
        "/api/lm/community-images/refresh-model",
        json={"sha256": HASH_ALPHA},
    )
    payload = await response.json()

    assert response.status == 200
    assert payload["success"] is True
    assert RecordingRefreshService.refresh_calls == [{
        "sha256": HASH_ALPHA,
        "civitai_model_id": 42,
        "author_username": "author-civitai copy",
        "civitai_version_id": 420,
        "model_name": "CivitAI Copy",
    }]

    inventory_response = await client.get("/api/lm/community-images/models")
    inventory = await inventory_response.json()
    assert inventory["models"][0]["base_model"] == "Pony"


@pytest.mark.asyncio
async def test_model_inventory_surfaces_scanner_failures(
    route_client_factory,
):
    scanner = FakeScanner([])

    async def fail_get_cached_data(*_args, **_kwargs):
        raise RuntimeError("scanner failed")

    scanner.get_cached_data = fail_get_cached_data
    client, _ = await route_client_factory(scanner)

    response = await client.get("/api/lm/community-images/models")
    payload = await response.json()

    assert response.status == 500
    assert payload == {"success": False, "error": "Internal server error"}


@pytest.mark.asyncio
async def test_hidden_models_are_flagged_in_inventory_and_excluded_from_grid(
    route_client_factory,
    community_db,
):
    visible = _model(HASH_ALPHA, "Alpha", model_id=11)
    hidden = _model(HASH_HIDDEN, "Hidden", model_id=33)
    scanner = FakeScanner([visible, hidden])
    settings = FakeSettings(
        community_hidden_model_hashes=[HASH_HIDDEN.upper()]
    )
    _store_image(community_db, 1, HASH_ALPHA)
    _store_image(community_db, 2, HASH_HIDDEN)
    client, _ = await route_client_factory(scanner, settings=settings)

    inventory_response = await client.get("/api/lm/community-images/models")
    inventory = await inventory_response.json()
    grid_response = await client.get("/api/lm/community-images/by-models")
    grid = await grid_response.json()

    inventory_by_hash = {
        model["sha256"]: model for model in inventory["models"]
    }
    assert inventory_response.status == 200
    assert inventory_by_hash[HASH_ALPHA]["hidden"] is False
    assert inventory_by_hash[HASH_HIDDEN]["hidden"] is True
    assert inventory["hidden_count"] == 1
    assert grid_response.status == 200
    assert [model["sha256"] for model in grid["models"]] == [HASH_ALPHA]
    assert grid["total_models"] == 1


@pytest.mark.asyncio
async def test_grid_includes_zero_image_base_models_in_tabs(
    route_client_factory,
    community_db,
):
    populated = _model(
        HASH_ALPHA,
        "Alpha",
        model_id=11,
        base_model="SDXL 1.0",
    )
    new_family = _model(
        HASH_BETA,
        "Krea Style",
        model_id=22,
        base_model="Krea 2",
    )
    scanner = FakeScanner([populated, new_family])
    _store_image(community_db, 1, HASH_ALPHA)
    client, _ = await route_client_factory(scanner)

    response = await client.get("/api/lm/community-images/by-models")
    payload = await response.json()

    assert response.status == 200
    assert payload["total_models"] == 1
    assert payload["base_models"] == {
        "SDXL 1.0": 1,
        "Krea 2": 0,
    }


@pytest.mark.asyncio
async def test_visibility_update_persists_normalized_hash(
    route_client_factory,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    settings = FakeSettings()
    client, _ = await route_client_factory(scanner, settings=settings)

    response = await client.post(
        "/api/lm/community-images/visibility",
        json={"sha256": f"  {HASH_ALPHA.upper()}  ", "hidden": True},
    )
    payload = await response.json()
    inventory_response = await client.get("/api/lm/community-images/models")
    inventory = await inventory_response.json()

    assert response.status == 200
    assert payload["hidden"] is True
    assert settings.values["community_hidden_model_hashes"] == [HASH_ALPHA]
    assert settings.set_calls == [
        ("community_hidden_model_hashes", [HASH_ALPHA])
    ]
    assert inventory["models"][0]["hidden"] is True


@pytest.mark.asyncio
async def test_visibility_rejects_invalid_hash(route_client_factory):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    settings = FakeSettings()
    client, _ = await route_client_factory(scanner, settings=settings)

    response = await client.post(
        "/api/lm/community-images/visibility",
        json={"sha256": "not-a-sha256", "hidden": True},
    )
    payload = await response.json()

    assert response.status == 400
    assert payload == {"success": False, "error": "Invalid sha256"}
    assert settings.set_calls == []


@pytest.mark.asyncio
async def test_selected_fetch_passes_only_requested_model(
    route_client_factory,
):
    scanner = FakeScanner([
        _model(HASH_ALPHA, "Alpha", model_id=11, version_id=101),
        _model(HASH_BETA, "Beta", model_id=22, version_id=202),
    ])
    client, _ = await route_client_factory(scanner)

    response = await client.post(
        "/api/lm/community-images/fetch",
        json={"hashes": [HASH_BETA.upper()], "force": True},
    )
    payload = await response.json()

    assert response.status == 200
    assert payload["success"] is True
    assert payload["total"] == 1
    assert payload["stored"] == 2
    assert len(RecordingFetchService.instances) == 1
    service = RecordingFetchService.instances[0]
    assert [model["sha256"] for model in service.models] == [HASH_BETA]
    assert service.closed is True
    assert scanner.calls == [(False, False)]


@pytest.mark.asyncio
async def test_fetch_rejects_overlap_while_another_request_is_preparing(
    route_client_factory,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    client, _ = await route_client_factory(scanner)
    community_routes._fetch_in_progress = True
    community_routes._active_service = None

    response = await client.post(
        "/api/lm/community-images/fetch",
        json={"hashes": [HASH_ALPHA]},
    )
    payload = await response.json()

    assert response.status == 409
    assert payload == {
        "success": False,
        "error": "A community fetch is starting",
    }
    assert scanner.calls == []
    assert RecordingFetchService.instances == []


@pytest.mark.asyncio
async def test_replacement_fetch_rechecks_model_refresh_after_wait(
    route_client_factory,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    client, _ = await route_client_factory(scanner)
    cancelled = False
    cancel_called = asyncio.Event()

    def cancel_old_fetch() -> None:
        nonlocal cancelled
        cancelled = True
        cancel_called.set()

    community_routes._fetch_in_progress = True
    community_routes._active_service = SimpleNamespace(cancel=cancel_old_fetch)

    async def start_model_refresh() -> None:
        await cancel_called.wait()
        community_routes._fetch_in_progress = False
        community_routes._active_service = None
        community_routes._refresh_model_in_progress = True

    refresh_task = asyncio.create_task(start_model_refresh())
    response = await client.post(
        "/api/lm/community-images/fetch",
        json={"hashes": [HASH_ALPHA]},
    )
    await refresh_task
    payload = await response.json()

    assert cancelled is True
    assert response.status == 409
    assert payload == {
        "success": False,
        "error": "A model refresh is already running",
    }
    assert RecordingFetchService.instances == []


@pytest.mark.asyncio
async def test_cancel_does_not_interrupt_scanner_preparation(
    route_client_factory,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    client, _ = await route_client_factory(scanner)
    community_routes._fetch_in_progress = True
    community_routes._active_service = None

    response = await client.post("/api/lm/community-images/cancel")
    payload = await response.json()

    assert response.status == 200
    assert payload == {
        "success": True,
        "message": "Fetch cancellation requested",
    }
    assert community_routes._fetch_cancel_requested is True
    assert scanner.cancel_calls == 0


@pytest.mark.asyncio
async def test_cancelled_fetch_allows_joined_inventory_refresh_to_finish(
    route_client_factory,
):
    existing = _model(HASH_ALPHA, "Alpha")
    new_model = _model(HASH_BETA, "Beta")
    scanner = FakeScanner([existing], refreshed_models=[existing, new_model])
    started = asyncio.Event()
    release = asyncio.Event()
    original_get_cached_data = scanner.get_cached_data

    async def slow_get_cached_data(
        force_refresh: bool = False,
        rebuild_cache: bool = False,
    ):
        if force_refresh:
            started.set()
            await release.wait()
        return await original_get_cached_data(force_refresh, rebuild_cache)

    scanner.get_cached_data = slow_get_cached_data
    client, _ = await route_client_factory(scanner)

    fetch_task = asyncio.create_task(
        client.post("/api/lm/community-images/fetch")
    )
    await started.wait()
    inventory_task = asyncio.create_task(
        client.get("/api/lm/community-images/models?refresh=true")
    )
    await asyncio.sleep(0.05)
    cancel_response = await client.post("/api/lm/community-images/cancel")
    release.set()
    fetch_response, inventory_response = await asyncio.gather(
        fetch_task,
        inventory_task,
    )
    cancel_payload = await cancel_response.json()
    fetch_payload = await fetch_response.json()
    inventory_payload = await inventory_response.json()

    assert cancel_payload["message"] == "Fetch cancellation requested"
    assert fetch_payload["cancelled"] is True
    assert inventory_payload["total_models"] == 2
    assert scanner.cancel_calls == 0
    assert scanner.calls == [(True, False)]
    assert RecordingFetchService.instances == []


@pytest.mark.asyncio
async def test_cancelled_scanner_preparation_does_not_start_fetch_service(
    route_client_factory,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    started = asyncio.Event()
    release = asyncio.Event()
    original_get_cached_data = scanner.get_cached_data

    async def slow_get_cached_data(
        force_refresh: bool = False,
        rebuild_cache: bool = False,
    ):
        if force_refresh:
            started.set()
            await release.wait()
        return await original_get_cached_data(force_refresh, rebuild_cache)

    scanner.get_cached_data = slow_get_cached_data
    client, _ = await route_client_factory(scanner)

    fetch_task = asyncio.create_task(
        client.post("/api/lm/community-images/fetch")
    )
    await started.wait()
    cancel_response = await client.post("/api/lm/community-images/cancel")
    release.set()
    fetch_response = await fetch_task
    cancel_payload = await cancel_response.json()
    fetch_payload = await fetch_response.json()

    assert cancel_payload["message"] == "Fetch cancellation requested"
    assert fetch_payload["cancelled"] is True
    assert fetch_payload["stored"] == 0
    assert scanner.cancel_calls == 0
    assert RecordingFetchService.instances == []


@pytest.mark.asyncio
async def test_model_refresh_is_rejected_during_bulk_fetch(route_client_factory):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    client, _ = await route_client_factory(scanner)
    community_routes._fetch_in_progress = True

    response = await client.post(
        "/api/lm/community-images/refresh-model",
        json={"sha256": HASH_ALPHA},
    )
    payload = await response.json()

    assert response.status == 409
    assert payload == {
        "success": False,
        "error": "Another community fetch is running",
    }
    assert scanner.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"hashes": []},
        {"hashes": HASH_ALPHA},
        {"hashes": [HASH_ALPHA, HASH_ALPHA.upper()]},
        {"hashes": [123]},
    ],
)
async def test_invalid_selected_fetch_payloads_are_rejected(
    route_client_factory,
    payload,
):
    scanner = FakeScanner([_model(HASH_ALPHA, "Alpha")])
    client, _ = await route_client_factory(scanner)

    response = await client.post(
        "/api/lm/community-images/fetch",
        json=payload,
    )
    response_payload = await response.json()

    assert response.status == 400
    assert response_payload["success"] is False
    assert RecordingFetchService.instances == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_hash", "response_field"),
    [
        (HASH_UNKNOWN, "unknown_hashes"),
        (HASH_NO_CIVITAI, "unavailable_hashes"),
    ],
)
async def test_unknown_or_unavailable_selected_models_are_rejected(
    route_client_factory,
    selected_hash,
    response_field,
):
    scanner = FakeScanner([
        _model(HASH_ALPHA, "Alpha", model_id=11),
        _model(HASH_NO_CIVITAI, "No Metadata", model_id=None),
    ])
    client, _ = await route_client_factory(scanner)

    response = await client.post(
        "/api/lm/community-images/fetch",
        json={"hashes": [selected_hash]},
    )
    payload = await response.json()

    assert response.status == 400
    assert payload["success"] is False
    assert payload[response_field] == [selected_hash]
    assert RecordingFetchService.instances == []
