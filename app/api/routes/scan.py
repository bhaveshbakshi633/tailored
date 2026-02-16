"""
Scan upload endpoint.

Accepts .obj / .ply / .stl mesh files, assigns a scan_id, and persists
the raw file for downstream processing.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config import config
from app.models.schemas import ScanUploadResponse, ErrorResponse
from app.storage.mesh_store import save_upload

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".obj", ".ply", ".stl", ".usdz"}


@router.post(
    "/upload",
    response_model=ScanUploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_scan(file: UploadFile = File(...)):
    """Upload a 3D mesh scan file."""

    # Validate extension
    if file.filename is None:
        raise HTTPException(400, "Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read and check size
    content = await file.read()
    max_bytes = config.storage.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            413,
            f"File too large ({len(content)} bytes). Max: {max_bytes} bytes.",
        )

    # Assign ID and persist
    scan_id = uuid.uuid4().hex[:16]
    save_upload(scan_id, file.filename, content)

    logger.info("Scan uploaded: id=%s, file=%s, size=%d bytes", scan_id, file.filename, len(content))
    return ScanUploadResponse(scan_id=scan_id, status="uploaded")
