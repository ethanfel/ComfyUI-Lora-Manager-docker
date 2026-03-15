"""Tests for CivitAI stats SQLite database layer."""
import pytest
from pathlib import Path
from py.services.civitai_stats_db import CivitaiStatsDB


@pytest.fixture
def stats_db(tmp_path):
    db = CivitaiStatsDB(db_path=tmp_path / "test_stats.db")
    db.init()
    yield db
    db.close()


def test_init_creates_table(stats_db):
    """DB should create model_stats table on init."""
    assert stats_db.count() == 0


def test_upsert_and_get(stats_db):
    """Single upsert should be retrievable by hash."""
    stats_db.upsert("abc123", {
        "civitai_model_id": 1,
        "civitai_version_id": 10,
        "download_count": 500,
        "rating": 4.5,
        "rating_count": 100,
        "thumbs_up_count": 80,
    })
    result = stats_db.get_by_hashes(["abc123"])
    assert "abc123" in result
    assert result["abc123"]["download_count"] == 500
    assert result["abc123"]["rating"] == 4.5


def test_upsert_updates_existing(stats_db):
    """Upserting same hash should update values."""
    stats_db.upsert("abc123", {"download_count": 100})
    stats_db.upsert("abc123", {"download_count": 200})
    result = stats_db.get_by_hashes(["abc123"])
    assert result["abc123"]["download_count"] == 200


def test_upsert_batch(stats_db):
    """Batch upsert should insert multiple rows."""
    rows = [
        ("hash1", {"download_count": 10, "civitai_model_id": 1}),
        ("hash2", {"download_count": 20, "civitai_model_id": 2}),
    ]
    stats_db.upsert_batch(rows)
    assert stats_db.count() == 2
    result = stats_db.get_by_hashes(["hash1", "hash2"])
    assert result["hash1"]["download_count"] == 10
    assert result["hash2"]["download_count"] == 20


def test_upsert_batch_empty(stats_db):
    """Batch upsert with empty list should be a no-op."""
    stats_db.upsert_batch([])
    assert stats_db.count() == 0


def test_get_all(stats_db):
    """get_all should return all rows."""
    stats_db.upsert("a", {"download_count": 1})
    stats_db.upsert("b", {"download_count": 2})
    all_stats = stats_db.get_all()
    assert len(all_stats) == 2
    assert "a" in all_stats
    assert "b" in all_stats


def test_get_by_hashes_empty(stats_db):
    """get_by_hashes with empty list should return empty dict."""
    assert stats_db.get_by_hashes([]) == {}


def test_count(stats_db):
    """count should return number of rows."""
    assert stats_db.count() == 0
    stats_db.upsert("x", {"download_count": 1})
    assert stats_db.count() == 1
