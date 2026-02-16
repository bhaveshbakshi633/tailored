"""
Measurement endpoint.

Takes a scan_id, runs the full processing + measurement pipeline,
and returns structured body measurements with confidence scores.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException

from app.config import config
from app.core.mesh_processing import process_mesh
from app.core.measurement_engine import measure_all
from app.models.schemas import (
    MeasurementRequest,
    MeasurementResponse,
    ErrorResponse,
)
from app.storage.mesh_store import get_upload_path, save_result

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=MeasurementResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def compute_measurements(req: MeasurementRequest):
    """Process a scan and return body measurements."""

    mesh_path = get_upload_path(req.scan_id)
    if mesh_path is None or not mesh_path.exists():
        raise HTTPException(404, f"Scan '{req.scan_id}' not found")

    t0 = time.perf_counter()

    try:
        mesh, alignment_info = process_mesh(mesh_path)
        measurements = measure_all(mesh)
    except Exception as exc:
        logger.exception("Measurement failed for scan %s", req.scan_id)
        raise HTTPException(422, f"Measurement pipeline error: {exc}") from exc

    elapsed = time.perf_counter() - t0

    # Persist result
    result_data = measurements.model_dump()
    result_data["processing_time_s"] = round(elapsed, 3)
    result_data["alignment_info"] = alignment_info
    save_result(req.scan_id, result_data)

    logger.info("Measurements complete for %s in %.2f s", req.scan_id, elapsed)

    return MeasurementResponse(
        scan_id=req.scan_id,
        measurements=measurements,
        processing_time_s=round(elapsed, 3),
    )
