"""Tests for CommunityImagesDB."""
import sqlite3

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
    """Upserting the same image/model association should update, not duplicate."""
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


def test_same_image_id_can_be_stored_for_multiple_hashes(db):
    """A shared CivitAI image should retain a separate path for each model."""
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "hash_a",
        "local_filename": "hash_a/community/100.webp",
        "like_count": 5,
    })
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "hash_b",
        "local_filename": "hash_b/community/100.webp",
        "like_count": 8,
    })

    result = db.get_by_hashes(["hash_a", "hash_b"])

    assert db.count() == 2
    assert result["hash_a"][0]["local_filename"] == "hash_a/community/100.webp"
    assert result["hash_b"][0]["local_filename"] == "hash_b/community/100.webp"

    # A metadata-only refresh for one association must not replace either path.
    db.upsert({
        "civitai_image_id": 100,
        "sha256": "hash_a",
        "local_filename": None,
        "like_count": 10,
    })
    refreshed = db.get_by_hashes(["hash_a", "hash_b"])
    assert refreshed["hash_a"][0]["local_filename"] == "hash_a/community/100.webp"
    assert refreshed["hash_b"][0]["local_filename"] == "hash_b/community/100.webp"
    assert refreshed["hash_a"][0]["like_count"] == 10
    assert refreshed["hash_b"][0]["like_count"] == 8


def test_migrates_legacy_single_primary_key_schema(tmp_path):
    """Legacy rows should survive migration to the composite primary key."""
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE community_images (
                civitai_image_id INTEGER PRIMARY KEY,
                sha256 TEXT NOT NULL,
                civitai_model_id INTEGER,
                username TEXT,
                image_url TEXT,
                local_filename TEXT,
                width INTEGER,
                height INTEGER,
                prompt TEXT,
                negative_prompt TEXT,
                steps INTEGER,
                sampler TEXT,
                cfg_scale REAL,
                seed INTEGER,
                denoise REAL,
                base_model TEXT,
                like_count INTEGER DEFAULT 0,
                heart_count INTEGER DEFAULT 0,
                laugh_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                has_workflow INTEGER DEFAULT 0,
                media_type TEXT DEFAULT 'image',
                resources TEXT,
                created_at TEXT,
                fetched_at REAL
            );
        """)
        conn.execute(
            "INSERT INTO community_images "
            "(civitai_image_id, sha256, local_filename, prompt, like_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                42,
                "legacy_hash",
                "legacy_hash/community/42.webp",
                "legacy prompt",
                7,
            ),
        )

    instance = CommunityImagesDB(db_path=db_path)
    try:
        instance.init()

        table_info = instance._conn.execute(
            "PRAGMA table_info(community_images)"
        ).fetchall()
        primary_key = [
            row["name"]
            for row in sorted(table_info, key=lambda row: row["pk"])
            if row["pk"]
        ]
        assert primary_key == ["civitai_image_id", "sha256"]

        legacy = instance.get_by_hashes(["legacy_hash"])["legacy_hash"][0]
        assert legacy["civitai_image_id"] == 42
        assert legacy["local_filename"] == "legacy_hash/community/42.webp"
        assert legacy["like_count"] == 7

        instance.upsert({
            "civitai_image_id": 42,
            "sha256": "second_hash",
            "local_filename": "second_hash/community/42.webp",
        })
        migrated = instance.get_by_hashes(["legacy_hash", "second_hash"])
        assert migrated["legacy_hash"][0]["local_filename"] == (
            "legacy_hash/community/42.webp"
        )
        assert migrated["second_hash"][0]["local_filename"] == (
            "second_hash/community/42.webp"
        )
    finally:
        instance.close()


def test_workflow_backfill_targets_image_and_hash(db, tmp_path, monkeypatch):
    """Workflow backfill should not mark another model sharing the image ID."""
    model_folders = {
        "hash_a": tmp_path / "hash_a",
        "hash_b": tmp_path / "hash_b",
    }
    workflow_dir = model_folders["hash_a"] / "community"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "100.workflow.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "py.utils.example_images_paths.get_model_folder",
        lambda sha256: str(model_folders[sha256]),
    )
    db.upsert_batch([
        {
            "civitai_image_id": 100,
            "sha256": "hash_a",
            "local_filename": "hash_a/community/100.webp",
        },
        {
            "civitai_image_id": 100,
            "sha256": "hash_b",
            "local_filename": "hash_b/community/100.webp",
        },
    ])

    db._backfill_has_workflow()

    result = db.get_by_hashes(["hash_a", "hash_b"])
    assert result["hash_a"][0]["has_workflow"] == 1
    assert result["hash_b"][0]["has_workflow"] == 0


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


def test_get_models_paginated(db):
    """Should return paginated models with their images."""
    # Insert images for 3 models
    db.upsert_batch([
        {"civitai_image_id": 1, "sha256": "model_a", "like_count": 10, "heart_count": 5, "created_at": "2026-01-01"},
        {"civitai_image_id": 2, "sha256": "model_a", "like_count": 3, "heart_count": 1, "created_at": "2026-01-02"},
        {"civitai_image_id": 3, "sha256": "model_b", "like_count": 20, "heart_count": 10, "created_at": "2026-02-01"},
        {"civitai_image_id": 4, "sha256": "model_c", "like_count": 1, "heart_count": 0, "created_at": "2026-03-01"},
    ])

    # Page 1, size 2 — should get 2 models, sorted by reactions desc
    result = db.get_models_paginated(
        allowed_hashes=["model_a", "model_b", "model_c"],
        page=1, page_size=2, sort="reactions:desc",
    )
    assert result["total"] == 3
    assert len(result["models"]) == 2
    # model_b has 30 total reactions, model_a has 19 — both should be on page 1
    assert result["models"][0] == "model_b"
    assert result["models"][1] == "model_a"
    assert "model_b" in result["images"]
    assert "model_a" in result["images"]

    # Page 2 — should get remaining model
    result2 = db.get_models_paginated(
        allowed_hashes=["model_a", "model_b", "model_c"],
        page=2, page_size=2, sort="reactions:desc",
    )
    assert len(result2["models"]) == 1
    assert result2["models"][0] == "model_c"

    # Only allowed hashes are returned
    result3 = db.get_models_paginated(
        allowed_hashes=["model_a"],
        page=1, page_size=10, sort="reactions:desc",
    )
    assert result3["total"] == 1
    assert result3["models"] == ["model_a"]


def test_get_models_paginated_empty(db):
    """Should handle empty allowed_hashes."""
    result = db.get_models_paginated(allowed_hashes=[], page=1, page_size=10)
    assert result == {"models": [], "total": 0, "images": {}}


def test_delete_by_hash(db):
    """Should delete all images for a given hash."""
    db.upsert({"civitai_image_id": 1, "sha256": "del_me", "prompt": "test"})
    db.upsert({"civitai_image_id": 2, "sha256": "del_me", "prompt": "test"})
    db.upsert({"civitai_image_id": 3, "sha256": "keep_me", "prompt": "test"})
    db.delete_by_hash("del_me")
    assert db.get_by_hashes(["del_me"]) == {}
    assert len(db.get_by_hashes(["keep_me"])["keep_me"]) == 1
