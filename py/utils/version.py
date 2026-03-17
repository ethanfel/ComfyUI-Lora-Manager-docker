"""Shared app version helper for cache-busting template variables."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_cached_version: str | None = None


def get_app_version() -> str:
    """Return a version string like '1.2.3-abc1234' for cache busting."""
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    version = "1.0.0"
    short_hash = "stable"
    try:
        import toml

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pyproject_path = os.path.join(root_dir, "pyproject.toml")

        if os.path.exists(pyproject_path):
            with open(pyproject_path, "r", encoding="utf-8") as f:
                data = toml.load(f)
                version = data.get("project", {}).get("version", "1.0.0").replace("v", "")

        git_dir = os.path.join(root_dir, ".git")
        if os.path.exists(git_dir):
            try:
                import git
                repo = git.Repo(root_dir)
                short_hash = repo.head.commit.hexsha[:7]
            except Exception:
                pass
    except Exception as e:
        logger.debug("Failed to read version info: %s", e)

    _cached_version = f"{version}-{short_hash}"
    return _cached_version
