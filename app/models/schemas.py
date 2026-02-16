"""
Pydantic models for API request/response and internal data transfer.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class FitPreference(str, Enum):
    tight = "tight"
    regular = "regular"
    loose = "loose"


class GarmentType(str, Enum):
    shirt = "shirt"
    pants = "pants"
    jacket = "jacket"
    dress = "dress"


# ── Measurement primitives ─────────────────────────────────────────────

class ConfidenceScore(BaseModel):
    overall: float = Field(..., ge=0.0, le=1.0)
    slice_consistency: float = Field(..., ge=0.0, le=1.0)
    symmetry: float = Field(..., ge=0.0, le=1.0)
    density: float = Field(..., ge=0.0, le=1.0)
    smoothness: float = Field(..., ge=0.0, le=1.0)


class SingleMeasurement(BaseModel):
    """One body measurement with value in centimeters and confidence."""
    name: str
    value_cm: float
    confidence: ConfidenceScore
    slice_values_cm: list[float] = Field(default_factory=list, description="Raw per-slice values used for averaging")
    measurement_height_m: float | None = None


class LandmarkSet(BaseModel):
    """Detected anatomical landmark heights in meters from floor."""
    neck_height: float
    shoulder_height: float
    chest_height: float
    waist_height: float
    hip_height: float
    crotch_height: float
    knee_height: float | None = None
    ankle_height: float | None = None
    total_height: float
    detection_method: str  # "heuristic", "curvature", "combined"


class BodyMeasurements(BaseModel):
    """Complete set of body measurements."""
    chest_circumference: SingleMeasurement
    waist_circumference: SingleMeasurement
    hip_circumference: SingleMeasurement
    shoulder_width: SingleMeasurement
    inseam: SingleMeasurement
    neck_circumference: SingleMeasurement
    body_height_cm: float
    landmarks: LandmarkSet


# ── Cross-section data ─────────────────────────────────────────────────

class CrossSectionResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    height_m: float
    perimeter_m: float
    n_contour_points: int
    n_triangle_intersections: int
    is_valid: bool


# ── Size recommendation ────────────────────────────────────────────────

class SizeScore(BaseModel):
    size_label: str
    score: float = Field(..., ge=0.0, le=1.0)
    per_measurement_scores: dict[str, float]


class SizeRecommendation(BaseModel):
    garment_type: GarmentType
    brand: str
    recommended_size: str
    fit_preference: FitPreference
    stretch_factor: float
    all_scores: list[SizeScore]
    warnings: list[str] = Field(default_factory=list)


# ── API request / response ─────────────────────────────────────────────

class ScanUploadResponse(BaseModel):
    scan_id: str
    status: str


class MeasurementRequest(BaseModel):
    scan_id: str


class MeasurementResponse(BaseModel):
    scan_id: str
    measurements: BodyMeasurements
    processing_time_s: float


class SizingRequest(BaseModel):
    scan_id: str
    garment_type: GarmentType
    brand: str = "generic"
    fit_preference: FitPreference = FitPreference.regular
    stretch_factor: float = 1.0


class SizingResponse(BaseModel):
    scan_id: str
    recommendation: SizeRecommendation


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
