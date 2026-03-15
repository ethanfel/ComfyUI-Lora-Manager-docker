"""Tests for CommunityImagesDB."""
import pytest
from py.services.community_images_db import CommunityImagesDB


@pytest.fixture
def db(tmp_path):
    instance = CommunityImagesDB(db_path=tmp_path / "test.db")
    instance.init()
    yield instance
    instance.close()


def test_upsert_and_get_by_hash(db):
    """Should store and retrieve a community image by model hash."""
    db.upsert({
        "civitai_image_id": 12345,
        "sha256": "abc123",
        "civitai_model_id": 42,
        "username": "testuser",
        "image_url": "https://example.com/img.jpg",
        "local_filename": "12345.jpg",
        "width": 1024,
        "height": 768,
        "prompt": "a beautiful landscape with mountains",
        "negative_prompt": "blurry",
        "steps": 30,
        "sampler": "DPM++ 2M Karras",
        "cfg_scale": 7.0,
        "seed": 123456,
        "denoise": 0.75,
        "base_model": "Pony",
        "like_count": 10,
        "heart_count": 5,
        "laugh_count": 2,
        "comment_count": 1,
        "created_at": "2026-01-15T10:00:00Z",
    })
    result = db.get_by_hashes(["abc123"])
    assert "abc123" in result
    images = result["abc123"]
    assert len(images) == 1
    assert images[0]["civitai_image_id"] == 12345
    assert images[0]["prompt"] == "a beautiful landscape with mountains"
    assert images[0]["like_count"] == 10


def test_upsert_batch(db):
    """Should store multiple images at once."""
    rows = [
        {
            "civitai_image_id": 1,
            "sha256": "hash_a",
            "civitai_model_id": 10,
            "username": "user1",
            "prompt": "test prompt one for image",
            "like_count": 5,
        },
        {
            "civitai_image_id": 2,
            "sha256": "hash_a",
            "civitai_model_id": 10,
            "username": "user2",
            "prompt": "test prompt two for image",
            "like_count": 3,
        },
        {
            "civitai_image_id": 3,
            "sha256": "hash_b",
            "civitai_model_id": 20,
            "username": "user3",
            "prompt": "another test prompt here",
            "like_count": 8,
        },
    ]
    db.upsert_batch(rows)
    result = db.get_by_hashes(["hash_a", "hash_b"])
    assert len(result["hash_a"]) == 2
    assert len(result["hash_b"]) == 1


def test_upsert_updates_existing(db):
    """Upserting same image_id should update, not duplicate."""
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "xyz",
        "like_count": 5,
        "prompt": "original prompt for testing",
    })
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "xyz",
        "like_count": 15,
        "prompt": "original prompt for testing",
    })
    result = db.get_by_hashes(["xyz"])
    assert len(result["xyz"]) == 1
    assert result["xyz"][0]["like_count"] == 15


def test_count(db):
    """Should count total images."""
    assert db.count() == 0
    db.upsert({"civitai_image_id": 1, "sha256": "a", "prompt": "test prompt long enough"})
    db.upsert({"civitai_image_id": 2, "sha256": "b", "prompt": "test prompt long enough"})
    assert db.count() == 2


def test_get_by_hashes_empty(db):
    """Empty hash list should return empty dict."""
    assert db.get_by_hashes([]) == {}


def test_get_hashes_with_images(db):
    """Should return set of sha256 values that have images."""
    db.upsert({"civitai_image_id": 1, "sha256": "aaa", "prompt": "test"})
    db.upsert({"civitai_image_id": 2, "sha256": "bbb", "prompt": "test"})
    result = db.get_hashes_with_images(["aaa", "bbb", "ccc"])
    assert result == {"aaa", "bbb"}


def test_delete_by_hash(db):
    """Should delete all images for a given hash."""
    db.upsert({"civitai_image_id": 1, "sha256": "del_me", "prompt": "test"})
    db.upsert({"civitai_image_id": 2, "sha256": "del_me", "prompt": "test"})
    db.upsert({"civitai_image_id": 3, "sha256": "keep_me", "prompt": "test"})
    db.delete_by_hash("del_me")
    assert db.get_by_hashes(["del_me"]) == {}
    assert len(db.get_by_hashes(["keep_me"])["keep_me"]) == 1
