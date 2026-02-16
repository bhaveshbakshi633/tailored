"""
Size recommendation endpoint.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException

from app.config import config
from app.core.mesh_processing import process_mesh
from app.core.measurement_engine import measure_all
from app.core.size_recommendation import recommend_size
from app.models.schemas import (
    SizingRequest,
    SizingResponse,
    ErrorResponse,
)
from app.storage.mesh_store import get_upload_path, load_result

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=SizingResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_sizing(req: SizingRequest):
    """Get size recommendation for a previously measured scan."""

    # Try to load cached measurements first
    cached = load_result(req.scan_id)
    if cached is not None:
        from app.models.schemas import BodyMeasurements
        measurements = BodyMeasurements(**cached)
    else:
        # Fall back to computing from scratch
        mesh_path = get_upload_path(req.scan_id)
        if mesh_path is None or not mesh_path.exists():
            raise HTTPException(404, f"Scan '{req.scan_id}' not found")

        mesh, _ = process_mesh(mesh_path)
        measurements = measure_all(mesh)

    recommendation = recommend_size(
        measurements=measurements,
        garment_type=req.garment_type,
        brand=req.brand,
        fit_preference=req.fit_preference,
        stretch_factor=req.stretch_factor,
    )

    return SizingResponse(
        scan_id=req.scan_id,
        recommendation=recommendation,
    )
