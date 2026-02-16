"""
Local filesystem storage for uploaded meshes and computed results.

In production, replace with S3 / GCS with pre-signed upload URLs.
The interface is kept minimal so swapping storage backends is
straightforward.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import config

logger = logging.getLogger(__name__)


def _scan_dir(scan_id: str) -> Path:
    d = config.storage.upload_dir / scan_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(scan_id: str, filename: str, content: bytes) -> Path:
    """Persist an uploaded mesh file.  Returns the saved path."""
    dest = _scan_dir(scan_id) / filename
    dest.write_bytes(content)
    logger.info("Saved upload: %s (%d bytes)", dest, len(content))
    return dest


def get_upload_path(scan_id: str) -> Path | None:
    """Return the path of the first mesh file for this scan, or None."""
    d = config.storage.upload_dir / scan_id
    if not d.exists():
        return None
    for ext in (".obj", ".ply", ".stl"):
        matches = list(d.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def save_result(scan_id: str, data: dict) -> Path:
    """Persist measurement results as JSON."""
    dest = config.storage.result_dir / f"{scan_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, default=str))
    return dest


def load_result(scan_id: str) -> dict | None:
    """Load cached measurement results, or None if not found."""
    path = config.storage.result_dir / f"{scan_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
