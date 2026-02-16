"""
Body measurement engine.

Every measurement function:
  1. Takes multi-slice samples in a narrow band around the target height.
  2. Computes circumference (or width / length) for each slice.
  3. Uses the **trimmed mean** (drop highest and lowest) as the final value.
  4. Returns value + raw slice values for downstream confidence scoring.

Multi-slice averaging rationale
───────────────────────────────
LiDAR mesh noise is ~±1 cm.  A single cross-section at one height is
sensitive to local defects (holes, doubled surfaces, noise spikes).

By sampling  k  slices spaced Δh apart in a band  [h − w, h + w]  and
averaging, we reduce variance by  ~1/√k  while the circumference changes
only by  O(Δh × dC/dh) — negligible for small  w  (we use w = 1 cm).

Trimmed mean (drop min & max) rejects outliers caused by:
  • Arm–torso merge at chest height
  • Scan holes creating open contours
  • Clothing folds

Measurement definitions
───────────────────────
• Chest:     max circumference in the chest height band
• Waist:     min circumference in the waist band (anatomical waist)
• Hip:       max circumference in the hip band
• Neck:      min circumference in the neck band
• Shoulder:  lateral distance between the leftmost and rightmost points
             of the cross-section at shoulder height
• Inseam:    vertical distance from crotch height to floor
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from app.config import config
from app.core.cross_section import extract_cross_section, CrossSectionResult
from app.core.landmark_detection import (
    detect_landmarks,
    compute_circumference_profile,
    CircumferenceProfile,
)
from app.core.confidence import compute_confidence
from app.models.schemas import (
    SingleMeasurement,
    BodyMeasurements,
    LandmarkSet,
    ConfidenceScore,
)

logger = logging.getLogger(__name__)

mcfg = config.measurement


# ── Helpers ────────────────────────────────────────────────────────────

def _trimmed_mean(values: np.ndarray) -> float:
    """Mean after dropping the single highest and lowest values."""
    if len(values) <= 2:
        return float(np.mean(values))
    sorted_v = np.sort(values)
    return float(np.mean(sorted_v[1:-1]))


def multi_slice_circumference(
    mesh: trimesh.Trimesh,
    center_height: float,
    n_slices: int | None = None,
    band_width: float | None = None,
) -> tuple[float, list[float], list[CrossSectionResult]]:
    """
    Measure circumference with multi-slice averaging.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    center_height : float — target height in meters
    n_slices : int — number of slices in the band
    band_width : float — half-width of the band in meters

    Returns
    -------
    value : float — circumference in meters
    slice_values : list[float] — per-slice circumferences
    slice_results : list[CrossSectionResult] — full results for confidence
    """
    n_slices = n_slices or mcfg.n_averaging_slices
    band_width = band_width or mcfg.averaging_band_width_m

    heights = np.linspace(
        center_height - band_width,
        center_height + band_width,
        n_slices,
    )

    slice_values: list[float] = []
    slice_results: list[CrossSectionResult] = []

    for h in heights:
        cs = extract_cross_section(mesh, h, axis=1)
        slice_results.append(cs)
        if cs.is_valid and cs.primary_perimeter > 0:
            slice_values.append(cs.primary_perimeter)

    if not slice_values:
        return 0.0, [], slice_results

    value = _trimmed_mean(np.array(slice_values))
    return value, slice_values, slice_results


def search_circumference_extremum(
    mesh: trimesh.Trimesh,
    h_min: float,
    h_max: float,
    n_search: int | None = None,
    mode: str = "min",
) -> tuple[float, float]:
    """
    Search for the height that gives the min or max circumference in [h_min, h_max].

    Returns (best_height, circumference_at_best_height).
    """
    n_search = n_search or mcfg.search_resolution

    heights = np.linspace(h_min, h_max, n_search)
    circs = np.zeros(n_search)

    for i, h in enumerate(heights):
        cs = extract_cross_section(mesh, h, axis=1)
        circs[i] = cs.primary_perimeter if cs.is_valid else 0.0

    # Mask out invalid (zero) values
    valid = circs > 0
    if not np.any(valid):
        mid = (h_min + h_max) / 2
        return mid, 0.0

    if mode == "min":
        # Among valid slices, find minimum
        valid_circs = np.where(valid, circs, np.inf)
        best_idx = np.argmin(valid_circs)
    else:
        best_idx = np.argmax(circs)

    return float(heights[best_idx]), float(circs[best_idx])


# ── Individual measurement functions ───────────────────────────────────

def measure_chest(
    mesh: trimesh.Trimesh, landmarks: LandmarkSet,
) -> SingleMeasurement:
    """
    Chest circumference.

    Search for the maximum circumference in the chest band, then
    multi-slice average around that height.
    """
    h = landmarks.total_height
    h_min = h * config.landmarks.chest_min
    h_max = h * config.landmarks.chest_max

    best_h, _ = search_circumference_extremum(mesh, h_min, h_max, mode="max")
    value, slices, results = multi_slice_circumference(mesh, best_h)

    conf = compute_confidence(mesh, best_h, slices, results)

    return SingleMeasurement(
        name="chest_circumference",
        value_cm=value * 100,
        confidence=conf,
        slice_values_cm=[v * 100 for v in slices],
        measurement_height_m=best_h,
    )


def measure_waist(
    mesh: trimesh.Trimesh, landmarks: LandmarkSet,
) -> SingleMeasurement:
    """
    Waist circumference.

    The anatomical waist is the narrowest point between ribs and iliac crest.
    We search for the minimum circumference in the waist band.
    """
    h = landmarks.total_height
    h_min = h * config.landmarks.waist_min
    h_max = h * config.landmarks.waist_max

    best_h, _ = search_circumference_extremum(mesh, h_min, h_max, mode="min")
    value, slices, results = multi_slice_circumference(mesh, best_h)

    conf = compute_confidence(mesh, best_h, slices, results)

    return SingleMeasurement(
        name="waist_circumference",
        value_cm=value * 100,
        confidence=conf,
        slice_values_cm=[v * 100 for v in slices],
        measurement_height_m=best_h,
    )


def measure_hip(
    mesh: trimesh.Trimesh, landmarks: LandmarkSet,
) -> SingleMeasurement:
    """
    Hip circumference.

    Maximum circumference in the hip band (gluteal level).
    """
    h = landmarks.total_height
    h_min = h * config.landmarks.hip_min
    h_max = h * config.landmarks.hip_max

    best_h, _ = search_circumference_extremum(mesh, h_min, h_max, mode="max")
    value, slices, results = multi_slice_circumference(mesh, best_h)

    conf = compute_confidence(mesh, best_h, slices, results)

    return SingleMeasurement(
        name="hip_circumference",
        value_cm=value * 100,
        confidence=conf,
        slice_values_cm=[v * 100 for v in slices],
        measurement_height_m=best_h,
    )


def measure_shoulder_width(
    mesh: trimesh.Trimesh, landmarks: LandmarkSet,
) -> SingleMeasurement:
    """
    Bi-acromial (shoulder) width.

    At shoulder height, extract the cross-section and measure the X-axis
    extent (distance between leftmost and rightmost points).

    Multi-slice averaging: sample several slices and average the widths.
    """
    center_h = landmarks.shoulder_height
    n = mcfg.n_averaging_slices
    bw = mcfg.averaging_band_width_m

    heights = np.linspace(center_h - bw, center_h + bw, n)
    widths: list[float] = []
    results: list[CrossSectionResult] = []

    for h in heights:
        cs = extract_cross_section(mesh, h, axis=1)
        results.append(cs)
        if cs.is_valid and cs.contours:
            primary = max(cs.contours, key=lambda c: c.perimeter)
            x_coords = primary.points[:, 0]
            width = float(x_coords.max() - x_coords.min())
            widths.append(width)

    if not widths:
        value = 0.0
    else:
        value = _trimmed_mean(np.array(widths))

    conf = compute_confidence(mesh, center_h, widths, results)

    return SingleMeasurement(
        name="shoulder_width",
        value_cm=value * 100,
        confidence=conf,
        slice_values_cm=[w * 100 for w in widths],
        measurement_height_m=center_h,
    )


def measure_inseam(
    mesh: trimesh.Trimesh, landmarks: LandmarkSet,
) -> SingleMeasurement:
    """
    Inside leg length (inseam).

    Defined as the vertical distance from the crotch to the floor.
    The crotch is detected in landmark detection.

    Confidence is based on how reliably we detected the crotch point
    (i.e., the contour-count transition from 1 to 2).
    """
    crotch_h = landmarks.crotch_height
    floor_h = mesh.vertices[:, 1].min()
    inseam = crotch_h - floor_h

    # Synthetic confidence: we trust the crotch detection to ±2 cm,
    # which corresponds to ~1–2% error for typical inseam lengths.
    # We generate pseudo-slices by sampling crotch detection at nearby heights.
    n = mcfg.n_averaging_slices
    bw = mcfg.averaging_band_width_m
    heights_around_crotch = np.linspace(crotch_h - bw, crotch_h + bw, n)

    inseam_values: list[float] = []
    results: list[CrossSectionResult] = []

    for h in heights_around_crotch:
        cs = extract_cross_section(mesh, h, axis=1)
        results.append(cs)
        # At crotch: transition from 1 to 2 contours.
        # Use height where we still have 1 contour as upper bound.
        if cs.n_contours == 1 and cs.is_valid:
            inseam_values.append(h - floor_h)

    if not inseam_values:
        inseam_values = [inseam]

    value = _trimmed_mean(np.array(inseam_values))
    conf = compute_confidence(mesh, crotch_h, inseam_values, results)

    return SingleMeasurement(
        name="inseam",
        value_cm=value * 100,
        confidence=conf,
        slice_values_cm=[v * 100 for v in inseam_values],
        measurement_height_m=crotch_h,
    )


def measure_neck(
    mesh: trimesh.Trimesh, landmarks: LandmarkSet,
) -> SingleMeasurement:
    """
    Neck circumference.

    Search for the minimum circumference in the neck band and multi-slice
    average around that point.
    """
    h = landmarks.total_height
    h_min = h * config.landmarks.neck_min
    h_max = h * config.landmarks.neck_max

    best_h, _ = search_circumference_extremum(mesh, h_min, h_max, mode="min")
    value, slices, results = multi_slice_circumference(mesh, best_h)

    conf = compute_confidence(mesh, best_h, slices, results)

    return SingleMeasurement(
        name="neck_circumference",
        value_cm=value * 100,
        confidence=conf,
        slice_values_cm=[v * 100 for v in slices],
        measurement_height_m=best_h,
    )


# ── Full measurement pipeline ─────────────────────────────────────────

def measure_all(mesh: trimesh.Trimesh) -> BodyMeasurements:
    """
    Run the complete measurement pipeline on a processed mesh.

    Steps:
      1. Compute circumference profile
      2. Detect anatomical landmarks
      3. Measure each body dimension
      4. Package results
    """
    profile = compute_circumference_profile(mesh)
    landmarks = detect_landmarks(mesh, profile)

    total_height_m = landmarks.total_height
    logger.info("Total height: %.3f m", total_height_m)
    logger.info("Landmarks: %s", landmarks)

    chest = measure_chest(mesh, landmarks)
    waist = measure_waist(mesh, landmarks)
    hip = measure_hip(mesh, landmarks)
    shoulder = measure_shoulder_width(mesh, landmarks)
    inseam = measure_inseam(mesh, landmarks)
    neck = measure_neck(mesh, landmarks)

    return BodyMeasurements(
        chest_circumference=chest,
        waist_circumference=waist,
        hip_circumference=hip,
        shoulder_width=shoulder,
        inseam=inseam,
        neck_circumference=neck,
        body_height_cm=total_height_m * 100,
        landmarks=landmarks,
    )
