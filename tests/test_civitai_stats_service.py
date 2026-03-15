"""Tests for CivitAI stats fetch service."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from py.services.civitai_stats_service import extract_version_stats, CivitaiStatsFetchService
from py.services.civitai_stats_db import CivitaiStatsDB


# --- Sync tests for extract_version_stats ---

def test_extract_version_stats_basic():
    """Should extract sha256 and stats from model response."""
    model_data = {
        "id": 42,
        "modelVersions": [
            {
                "id": 100,
                "files": [{"hashes": {"SHA256": "ABCDEF1234567890"}}],
                "stats": {
                    "downloadCount": 1500,
                    "rating": 4.8,
                    "ratingCount": 200,
                    "thumbsUpCount": 180,
                },
            }
        ],
    }
    result = extract_version_stats(model_data)
    assert len(result) == 1
    sha, stats = result[0]
    assert sha == "abcdef1234567890"  # lowercased
    assert stats["download_count"] == 1500
    assert stats["rating"] == 4.8
    assert stats["civitai_model_id"] == 42
    assert stats["civitai_version_id"] == 100


def test_extract_version_stats_no_hash():
    """Versions without SHA256 should be skipped."""
    model_data = {
        "id": 1,
        "modelVersions": [
            {"id": 10, "files": [{"hashes": {}}], "stats": {"downloadCount": 5}},
        ],
    }
    assert extract_version_stats(model_data) == []


def test_extract_version_stats_multiple_versions():
    """Should extract one entry per version with a hash."""
    model_data = {
        "id": 1,
        "modelVersions": [
            {
                "id": 10,
                "files": [{"hashes": {"SHA256": "AAA"}}],
                "stats": {"downloadCount": 10},
            },
            {
                "id": 20,
                "files": [{"hashes": {"SHA256": "BBB"}}],
                "stats": {"downloadCount": 20},
            },
        ],
    }
    result = extract_version_stats(model_data)
    assert len(result) == 2


# --- Async tests for CivitaiStatsFetchService ---

@pytest.fixture
def stats_db(tmp_path):
    db = CivitaiStatsDB(db_path=tmp_path / "test.db")
    db.init()
    yield db
    db.close()


@pytest.mark.asyncio
async def test_fetch_stats_for_models(stats_db):
    """Should fetch from CivitAI API and store stats in DB."""
    mock_response = {
        "id": 42,
        "modelVersions": [
            {
                "id": 100,
                "files": [{"hashes": {"SHA256": "ABC123"}}],
                "stats": {"downloadCount": 999, "rating": 4.5, "ratingCount": 50, "thumbsUpCount": 40},
            }
        ],
    }

    service = CivitaiStatsFetchService(db=stats_db)

    with patch.object(service, "_fetch_model", new_callable=AsyncMock, return_value=mock_response):
        updated = await service.fetch_stats_for_models(
            [{"sha256": "abc123", "civitai_model_id": 42}]
        )

    assert updated == 1
    result = stats_db.get_by_hashes(["abc123"])
    assert result["abc123"]["download_count"] == 999

    await service.close()
