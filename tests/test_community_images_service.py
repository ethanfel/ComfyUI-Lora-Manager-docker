"""Tests for community images fetch service."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from py.services.community_images_service import (
    filter_community_images,
    _extract_image_data,
    CommunityImagesFetchService,
)
from py.services.community_images_db import CommunityImagesDB


def _make_image_item(
    image_id=1,
    username="someone",
    prompt="A beautiful landscape painting with mountains and rivers in the background",
    url="https://example.com/img.jpg",
    width=512,
    height=512,
    like_count=10,
    heart_count=5,
    laugh_count=2,
    comment_count=1,
    base_model="SD 1.5",
    cfg_scale=7,
    negative_prompt="bad quality",
    steps=20,
    sampler="Euler",
    seed=12345,
    denoise=0.7,
    created_at="2024-01-01T00:00:00Z",
):
    """Build a realistic CivitAI image item with double-nested meta."""
    return {
        "id": image_id,
        "url": url,
        "width": width,
        "height": height,
        "username": username,
        "baseModel": base_model,
        "createdAt": created_at,
        "stats": {
            "likeCount": like_count,
            "heartCount": heart_count,
            "laughCount": laugh_count,
            "commentCount": comment_count,
        },
        "meta": {
            "meta": {
                "prompt": prompt,
                "negativePrompt": negative_prompt,
                "cfgScale": cfg_scale,
                "steps": steps,
                "sampler": sampler,
                "seed": seed,
                "denoise": denoise,
            }
        },
    }


# --- filter_community_images tests ---


def test_filter_excludes_author():
    """Images by the exact author username should be excluded."""
    items = [
        _make_image_item(image_id=1, username="AuthorName"),
        _make_image_item(image_id=2, username="OtherUser"),
    ]
    result = filter_community_images(items, "AuthorName")
    assert len(result) == 1
    assert result[0]["id"] == 2


def test_filter_excludes_author_case_insensitive():
    """Author matching should be case-insensitive."""
    items = [
        _make_image_item(image_id=1, username="authorname"),
        _make_image_item(image_id=2, username="AUTHORNAME"),
        _make_image_item(image_id=3, username="AuthorName"),
        _make_image_item(image_id=4, username="OtherUser"),
    ]
    result = filter_community_images(items, "AuthorName")
    assert len(result) == 1
    assert result[0]["id"] == 4


def test_filter_excludes_missing_prompt():
    """Images without meta, with meta=None, or with empty meta should be excluded."""
    items = [
        # No meta key at all
        {"id": 1, "username": "u1", "stats": {}},
        # meta is None
        {"id": 2, "username": "u2", "meta": None, "stats": {}},
        # meta.meta is empty dict
        {"id": 3, "username": "u3", "meta": {"meta": {}}, "stats": {}},
        # meta.meta is None
        {"id": 4, "username": "u4", "meta": {"meta": None}, "stats": {}},
        # Valid item
        _make_image_item(image_id=5, username="u5"),
    ]
    result = filter_community_images(items, "nobody")
    assert len(result) == 1
    assert result[0]["id"] == 5


def test_filter_excludes_short_prompt():
    """Prompts shorter than 20 characters should be excluded."""
    items = [
        _make_image_item(image_id=1, prompt="short"),  # 5 chars
        _make_image_item(image_id=2, prompt="a" * 19),  # 19 chars
        _make_image_item(image_id=3, prompt="a" * 20),  # 20 chars - should pass
    ]
    result = filter_community_images(items, "nobody")
    assert len(result) == 1
    assert result[0]["id"] == 3


def test_filter_limits_to_10():
    """Should return at most 10 images even if more pass the filter."""
    items = [_make_image_item(image_id=i, username=f"user{i}") for i in range(15)]
    result = filter_community_images(items, "nobody")
    assert len(result) == 10


# --- _extract_image_data tests ---


def test_extract_image_data_maps_fields():
    """Should map CivitAI API fields to DB row format."""
    item = _make_image_item(
        image_id=42,
        username="testuser",
        prompt="A long prompt for testing purposes with enough characters",
        url="https://example.com/img.jpg",
        width=768,
        height=1024,
        like_count=100,
        heart_count=50,
        cfg_scale=7.5,
        negative_prompt="bad",
        steps=25,
        sampler="DPM++",
        seed=999,
        denoise=0.65,
        base_model="SDXL 1.0",
    )
    result = _extract_image_data(item, "abc123hash", 77)
    assert result["civitai_image_id"] == 42
    assert result["sha256"] == "abc123hash"
    assert result["civitai_model_id"] == 77
    assert result["username"] == "testuser"
    assert result["image_url"] == "https://example.com/img.jpg"
    assert result["width"] == 768
    assert result["height"] == 1024
    assert result["like_count"] == 100
    assert result["heart_count"] == 50
    assert result["cfg_scale"] == 7.5
    assert result["negative_prompt"] == "bad"
    assert result["steps"] == 25
    assert result["sampler"] == "DPM++"
    assert result["seed"] == 999
    assert result["denoise"] == 0.65
    assert result["base_model"] == "SDXL 1.0"


# --- Async tests for CommunityImagesFetchService ---


@pytest.fixture
def community_db(tmp_path):
    db = CommunityImagesDB(db_path=tmp_path / "test_community.db")
    db.init()
    yield db
    db.close()


@pytest.mark.asyncio
async def test_fetch_images_for_model(community_db, tmp_path):
    """Should fetch from API, filter, download images, and store in DB."""
    api_items = [
        _make_image_item(image_id=1, username="model_author"),  # excluded - author
        _make_image_item(image_id=2, username="community_user"),  # included
        _make_image_item(image_id=3, username="another_user", prompt="short"),  # excluded - short prompt
    ]
    api_response = {"items": api_items}

    service = CommunityImagesFetchService(db=community_db)

    with patch.object(
        service, "_fetch_images_api", new_callable=AsyncMock, return_value=api_response
    ), patch.object(
        service, "_download_image", new_callable=AsyncMock, return_value=("abc123/community/2.webp", False)
    ):
        count = await service.fetch_images_for_model(
            sha256="abc123",
            civitai_model_id=42,
            author_username="model_author",
        )

    # Only image_id=2 should pass (author excluded, short prompt excluded)
    assert count == 1

    # Verify stored in DB
    result = community_db.get_by_hashes(["abc123"])
    assert "abc123" in result
    assert len(result["abc123"]) == 1
    assert result["abc123"][0]["civitai_image_id"] == 2
    assert result["abc123"][0]["local_filename"] == "abc123/community/2.webp"

    await service.close()


def test_extract_image_data_missing_fields():
    """Should handle items with missing stats, meta, or None values gracefully."""
    item = {"id": 99, "username": "u", "url": "https://x.com/i.jpg"}
    result = _extract_image_data(item, "hash1", 10)
    assert result["civitai_image_id"] == 99
    assert result["prompt"] is None
    assert result["like_count"] == 0
    assert result["steps"] is None
    assert result["base_model"] is None


@pytest.mark.asyncio
async def test_fetch_all(community_db):
    """Should iterate models, skip invalid entries, rate limit, and report progress."""
    models = [
        {"sha256": "h1", "civitai_model_id": 1, "author_username": "a1"},
        {"sha256": None, "civitai_model_id": 2, "author_username": "a2"},  # skipped
        {"sha256": "h3", "civitai_model_id": 3, "author_username": "a3"},
    ]
    service = CommunityImagesFetchService(db=community_db)
    progress_calls = []

    async def track_progress(current, total):
        progress_calls.append((current, total))

    with patch.object(
        service, "fetch_images_for_model", new_callable=AsyncMock, return_value=2
    ), patch("py.services.community_images_service.asyncio.sleep", new_callable=AsyncMock):
        total = await service.fetch_all(models, progress_callback=track_progress)

    # h1 and h3 fetched (2 each), skipped model returns 0 but progress still called
    assert total == 4
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]

    await service.close()
