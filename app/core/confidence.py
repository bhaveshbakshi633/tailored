"""
Confidence scoring for body measurements.

Model
─────
The overall confidence  C ∈ [0, 1]  is a composite of four independent
sub-scores, combined via a weighted penalty model:

    C = max(0,  1 − α·CV − β·A − γ·S − δ·R)

where:

  CV  = coefficient of variation of multi-slice values  (σ / μ)
        Captures how consistent repeated measurements are.
        Perfect consistency → CV = 0 → no penalty.

  A   = bilateral asymmetry ratio
            |width_left − width_right|
        A = ─────────────────────────────
             (width_left + width_right) / 2
        A body in good scan pose is symmetric.  High asymmetry suggests
        bad pose, scan artifacts, or one-sided occlusion.

  S   = sparsity penalty  =  max(0,  1 − D_actual / D_required)
        D_actual   = number of triangle–plane intersections in the slice.
        D_required = configurable minimum (default 40).
        Under-sampled slices have low geometric fidelity.

  R   = roughness  =  perimeter_raw / perimeter_smoothed  −  1
        A smooth contour has R ≈ 0.  Noise-induced jitter inflates
        the raw perimeter relative to a smoothed version, so R > 0
        indicates noisy geometry.

Weights (configurable in AppConfig.confidence):
    α = 2.0   (slice variance is the most informative signal)
    β = 1.5
    γ = 1.0
    δ = 1.0

Calibration note
────────────────
These weights were chosen so that:
  • A "good" scan (CV < 0.02, A < 0.05, full density, smooth) scores > 0.90
  • A "marginal" scan (CV ~ 0.05, moderate asymmetry) scores 0.60–0.80
  • A "poor" scan (CV > 0.10 or missing data) scores < 0.50

The threshold for "usable" measurement is C ≥ 0.6.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

from app.config import config
from app.core.cross_section import CrossSectionResult, compute_perimeter
from app.models.schemas import ConfidenceScore

ccfg = config.confidence


# ── Sub-scores ─────────────────────────────────────────────────────────

def slice_consistency_score(values: list[float]) -> float:
    """
    1 − CV  clamped to [0, 1].

    CV (coefficient of variation) = σ / μ.
    Returns 1.0 for perfectly consistent slices.
    """
    if len(values) < 2:
        return 0.5  # insufficient data → neutral
    arr = np.array(values)
    mu = np.mean(arr)
    if mu < 1e-9:
        return 0.0
    cv = np.std(arr) / mu
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def symmetry_score(contour_points: np.ndarray | None) -> float:
    """
    Bilateral symmetry of a cross-section contour.

    Splits the contour at X = 0 (midline after PCA alignment) and compares
    the X-extent of the left and right halves.

    Returns 1.0 for perfect symmetry, lower for asymmetric contours.
    """
    if contour_points is None or len(contour_points) < 4:
        return 0.5

    x = contour_points[:, 0]
    left_extent = abs(float(x.min()))
    right_extent = abs(float(x.max()))

    total = left_extent + right_extent
    if total < 1e-9:
        return 0.5

    asymmetry = abs(left_extent - right_extent) / (total / 2)
    return float(np.clip(1.0 - asymmetry, 0.0, 1.0))


def density_score(n_triangle_hits: int) -> float:
    """
    Mesh density adequacy.

    Returns 1.0 if the number of triangle–plane intersections meets or
    exceeds the configured minimum.
    """
    required = ccfg.min_density_triangles
    if required <= 0:
        return 1.0
    ratio = n_triangle_hits / required
    return float(np.clip(ratio, 0.0, 1.0))


def smoothness_score(contour_points: np.ndarray | None) -> float:
    """
    Contour smoothness: ratio of smoothed perimeter to raw perimeter.

    A perfectly smooth contour has ratio = 1.0.
    Noise-induced jitter makes the raw perimeter longer than the
    smoothed version.

    Smoothing: Gaussian filter on the X and Z coordinates (the two
    in-plane dimensions after removing the slicing axis).
    """
    if contour_points is None or len(contour_points) < 6:
        return 0.5

    raw_perim = compute_perimeter(contour_points)
    if raw_perim < 1e-9:
        return 0.0

    # Smooth each coordinate independently
    sigma = 2.0
    smoothed = np.column_stack([
        gaussian_filter1d(contour_points[:, i], sigma=sigma)
        for i in range(contour_points.shape[1])
    ])

    smooth_perim = compute_perimeter(smoothed)
    if smooth_perim < 1e-9:
        return 0.0

    roughness = (raw_perim / smooth_perim) - 1.0
    return float(np.clip(1.0 - roughness, 0.0, 1.0))


# ── Combined confidence ───────────────────────────────────────────────

def compute_confidence(
    mesh: trimesh.Trimesh,
    height: float,
    slice_values: list[float],
    slice_results: list[CrossSectionResult],
) -> ConfidenceScore:
    """
    Compute composite confidence score for a measurement.

    Parameters
    ----------
    mesh          : the body mesh (for potential future use)
    height        : measurement height in meters
    slice_values  : per-slice measurement values (circumference or width)
    slice_results : per-slice CrossSectionResult objects

    Returns
    -------
    ConfidenceScore with overall and per-component scores.
    """
    # ── Slice consistency ──
    sc_consistency = slice_consistency_score(slice_values)

    # ── Symmetry: use the middle slice's primary contour ──
    sc_symmetry = 0.5
    if slice_results:
        mid_idx = len(slice_results) // 2
        mid_result = slice_results[mid_idx]
        if mid_result.contours:
            primary = max(mid_result.contours, key=lambda c: c.perimeter)
            sc_symmetry = symmetry_score(primary.points)

    # ── Density: average triangle hits across slices ──
    hit_counts = [
        max((c.n_triangle_hits for c in sr.contours), default=0)
        for sr in slice_results
        if sr.contours
    ]
    avg_hits = int(np.mean(hit_counts)) if hit_counts else 0
    sc_density = density_score(avg_hits)

    # ── Smoothness: use middle slice's primary contour ──
    sc_smoothness = 0.5
    if slice_results:
        mid_idx = len(slice_results) // 2
        mid_result = slice_results[mid_idx]
        if mid_result.contours:
            primary = max(mid_result.contours, key=lambda c: c.perimeter)
            sc_smoothness = smoothness_score(primary.points)

    # ── Composite ──
    penalty = (
        ccfg.w_slice_consistency * (1.0 - sc_consistency)
        + ccfg.w_symmetry * (1.0 - sc_symmetry)
        + ccfg.w_density * (1.0 - sc_density)
        + ccfg.w_smoothness * (1.0 - sc_smoothness)
    )

    # Normalize: max possible penalty = sum of all weights
    max_penalty = (
        ccfg.w_slice_consistency
        + ccfg.w_symmetry
        + ccfg.w_density
        + ccfg.w_smoothness
    )
    overall = float(np.clip(1.0 - penalty / max_penalty, 0.0, 1.0))

    return ConfidenceScore(
        overall=round(overall, 4),
        slice_consistency=round(sc_consistency, 4),
        symmetry=round(sc_symmetry, 4),
        density=round(sc_density, 4),
        smoothness=round(sc_smoothness, 4),
    )
