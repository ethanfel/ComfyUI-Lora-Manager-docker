import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from py.routes.lora_routes import LoraRoutes
from server import PromptServer  # pyright: ignore[reportMissingImports]


class DummyRequest:
    def __init__(self, *, query=None, match_info=None, json_data=None):
        self.query = query or {}
        self.match_info = match_info or {}
        self._json_data = json_data or {}

    async def json(self):
        return self._json_data


class StubLoraService:
    def __init__(self):
        self.trigger_words = {}
        self.usage_tips = {}
        self.models = []
        self.resolve_args = None
        self.annotated_models = None

    async def get_lora_trigger_words(self, name):
        return self.trigger_words.get(name, [])

    async def get_lora_usage_tips_by_relative_path(self, path):
        return self.usage_tips.get(path)

    async def find_models_by_name(self, name, *, base_model=None):
        self.resolve_args = (name, base_model)
        return self.models

    async def format_response(self, model):
        return None if model.get("corrupt") else model

    async def annotate_update_flags(self, models):
        self.annotated_models = models
        return [{**model, "update_available": True} for model in models]


@pytest.fixture
def routes():
    handler = LoraRoutes()
    handler.service = StubLoraService()  # pyright: ignore[reportAttributeAccessIssue]
    return handler


async def test_get_lora_trigger_words_success(routes):
    routes.service.trigger_words["demo"] = ["trigger"]
    response = await routes.get_lora_trigger_words(DummyRequest(query={"name": "demo"}))
    payload = json.loads(response.text)
    assert payload == {"success": True, "trigger_words": ["trigger"]}


async def test_get_lora_trigger_words_missing_name(routes):
    response = await routes.get_lora_trigger_words(DummyRequest())
    assert response.status == 400


async def test_get_lora_trigger_words_error(routes):
    async def failing(*_args, **_kwargs):
        raise RuntimeError("fail")

    routes.service.get_lora_trigger_words = failing

    response = await routes.get_lora_trigger_words(DummyRequest(query={"name": "demo"}))
    payload = json.loads(response.text)
    assert response.status == 500
    assert payload["success"] is False


async def test_get_usage_tips_success(routes):
    routes.service.usage_tips["path"] = "tips"
    response = await routes.get_lora_usage_tips_by_path(DummyRequest(query={"relative_path": "path"}))
    payload = json.loads(response.text)
    assert payload == {"success": True, "usage_tips": "tips"}


async def test_get_usage_tips_missing_param(routes):
    response = await routes.get_lora_usage_tips_by_path(DummyRequest())
    assert response.status == 400


async def test_get_usage_tips_error(routes):
    async def failing(*_args, **_kwargs):
        raise RuntimeError("bad")

    routes.service.get_lora_usage_tips_by_relative_path = failing
    response = await routes.get_lora_usage_tips_by_path(DummyRequest(query={"relative_path": "path"}))
    payload = json.loads(response.text)
    assert response.status == 500
    assert payload["success"] is False


async def test_resolve_lora_returns_card_record(routes):
    routes.service.models = [
        {
            "file_name": "styles/example.safetensors",
            "model_name": "Example Style",
            "preview_url": "/example.png",
        }
    ]

    response = await routes.resolve_lora(
        DummyRequest(
            query={"name": "styles/example.safetensors", "base_model": "Krea 2"}
        )
    )
    payload = json.loads(response.text)

    assert payload["success"] is True
    assert payload["found"] is True
    assert payload["model"]["model_name"] == "Example Style"
    assert payload["model"]["update_available"] is True
    assert routes.service.annotated_models == routes.service.models
    assert routes.service.resolve_args == (
        "styles/example.safetensors",
        "Krea 2",
    )


async def test_resolve_lora_reports_missing_model_without_error(routes):
    response = await routes.resolve_lora(
        DummyRequest(query={"name": "missing.safetensors"})
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload == {
        "success": True,
        "found": False,
        "query": "missing.safetensors",
    }


async def test_resolve_lora_reports_ambiguous_matches(routes):
    routes.service.models = [
        {"file_name": "duplicate.safetensors", "folder": "one"},
        {"file_name": "duplicate.safetensors", "folder": "two"},
    ]

    response = await routes.resolve_lora(
        DummyRequest(query={"name": "duplicate.safetensors"})
    )
    payload = json.loads(response.text)

    assert payload["success"] is True
    assert payload["found"] is False
    assert payload["ambiguous"] is True
    assert len(payload["candidates"]) == 2


async def test_resolve_lora_requires_name(routes):
    response = await routes.resolve_lora(DummyRequest())
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload == {"success": False, "error": "LoRA name is required"}


async def test_resolve_lora_ignores_corrupt_formatted_matches(routes):
    routes.service.models = [{"corrupt": True}]

    response = await routes.resolve_lora(
        DummyRequest(query={"name": "corrupt.safetensors"})
    )
    payload = json.loads(response.text)

    assert payload == {
        "success": True,
        "found": False,
        "query": "corrupt.safetensors",
    }


async def test_get_trigger_words_broadcasts(monkeypatch, routes):
    send_mock = MagicMock()
    PromptServer.instance = SimpleNamespace(send_sync=send_mock)

    monkeypatch.setattr("py.routes.lora_routes.get_lora_info", lambda name: (f"path/{name}", [f"trigger-{name}"]))

    request = DummyRequest(json_data={"lora_names": ["one"], "node_ids": [{"node_id": "node", "graph_id": "graph-1"}]})

    response = await routes.get_trigger_words(request)
    payload = json.loads(response.text)

    assert payload == {"success": True}
    send_mock.assert_called_once_with(
        "trigger_word_update",
        {"id": "node", "graph_id": "graph-1", "message": "trigger-one"},
    )


async def test_get_trigger_words_error(monkeypatch, routes):
    async def failing_json():
        raise RuntimeError("bad json")

    request = DummyRequest(json_data=None)
    request.json = failing_json

    response = await routes.get_trigger_words(request)
    payload = json.loads(response.text)
    assert response.status == 500
    assert payload["success"] is False
