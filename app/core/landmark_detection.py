"""
Anatomical landmark detection: two approaches.

═══════════════════════════════════════════════════════════════
Approach A — Heuristic height-percentage slicing
═══════════════════════════════════════════════════════════════

Anthropometric studies (ISO 7250, CAESAR database) show that anatomical
landmark heights, normalized by stature, cluster tightly:

  Landmark       │  % of stature  │  σ (std dev)
  ───────────────┼────────────────┼─────────────
  Neck (C7)      │  82–86 %       │  ±1.5 %
  Shoulder       │  ~81 %         │  ±1.8 %
  Chest (nipple) │  72–76 %       │  ±2.0 %
  Waist (navel)  │  58–63 %       │  ±2.5 %
  Hip (max)      │  49–53 %       │  ±2.0 %
  Crotch         │  44–48 %       │  ±2.0 %
  Knee           │  ~27 %         │  ±1.5 %
  Ankle          │  ~4 %          │  ±0.5 %

Pros:
  + Extremely fast (O(1) per landmark)
  + Works on very noisy / low-resolution meshes
  + No failure modes — always returns a result

Cons:
  − Assumes "average" body proportions
  − Insensitive to actual body shape
  − Systematic bias for outlier body types
    (e.g., long torso / short legs or vice-versa)

═══════════════════════════════════════════════════════════════
Approach B — Curvature-based profile analysis
═══════════════════════════════════════════════════════════════

Compute the circumference profile  C(h)  — the cross-section perimeter at
each height  h.  This profile has characteristic features:

  C(h) local maxima → chest, hips
  C(h) local minima → waist, neck
  Contour count transition 1 → 2 → crotch

Detection algorithm:
  1. Sample  C(h)  at N uniformly spaced heights.
  2. Smooth with a Gaussian kernel (σ ~ 5 slices) to suppress LiDAR noise.
  3. Compute  dC/dh  (first derivative).
  4. Find zero-crossings of  dC/dh  → extrema.
  5. Classify extrema by their height range + sign of second derivative.

Pros:
  + Adapts to actual body shape
  + More accurate for non-standard proportions
  + Provides a rich diagnostic signal (the profile itself)

Cons:
  − Requires a reasonably complete mesh
  − Can be confused by arms close to the torso
  − Needs Gaussian smoothing to suppress noise
  − May fail on very low-resolution scans
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

import trimesh

from app.config import config
from app.core.cross_section import extract_cross_section
from app.models.schemas import LandmarkSet

logger = logging.getLogger(__name__)

lcfg = config.landmarks


# ── Circumference profile ──────────────────────────────────────────────

@dataclass
class CircumferenceProfile:
    heights: np.ndarray         # (N,) meters
    circumferences: np.ndarray  # (N,) meters
    smoothed: np.ndarray        # (N,) smoothed circumferences
    n_contours: np.ndarray      # (N,) int — number of loops at each height


def compute_circumference_profile(
    mesh: trimesh.Trimesh,
    n_slices: int | None = None,
) -> CircumferenceProfile:
    """
    Compute the horizontal circumference profile of the body mesh.

    Samples cross-sections at  n_slices  uniformly spaced heights and
    records the primary contour perimeter at each.
    """
    n_slices = n_slices or lcfg.profile_n_slices
    y_min = mesh.vertices[:, 1].min()
    y_max = mesh.vertices[:, 1].max()

    # Avoid slicing exactly at min/max (edge degenerate)
    margin = (y_max - y_min) * 0.005
    heights = np.linspace(y_min + margin, y_max - margin, n_slices)

    circs = np.zeros(n_slices)
    n_contours = np.zeros(n_slices, dtype=int)

    for i, h in enumerate(heights):
        cs = extract_cross_section(mesh, h, axis=1)
        circs[i] = cs.primary_perimeter
        n_contours[i] = cs.n_contours

    smoothed = gaussian_filter1d(circs, sigma=lcfg.profile_smooth_sigma)

    return CircumferenceProfile(
        heights=heights,
        circumferences=circs,
        smoothed=smoothed,
        n_contours=n_contours,
    )


# ── Approach A: Heuristic ──────────────────────────────────────────────

def detect_landmarks_heuristic(total_height: float) -> LandmarkSet:
    """
    Detect landmarks using fixed percentages of total stature.

    These percentages come from ISO 7250 anthropometric norms and are
    used as fallback when the curvature-based method fails.
    """
    h = total_height

    return LandmarkSet(
        neck_height=h * (lcfg.neck_min + lcfg.neck_max) / 2,
        shoulder_height=h * lcfg.shoulder_height,
        chest_height=h * (lcfg.chest_min + lcfg.chest_max) / 2,
        waist_height=h * (lcfg.waist_min + lcfg.waist_max) / 2,
        hip_height=h * (lcfg.hip_min + lcfg.hip_max) / 2,
        crotch_height=h * (lcfg.crotch_min + lcfg.crotch_max) / 2,
        knee_height=h * lcfg.knee_height,
        ankle_height=h * lcfg.ankle_height,
        total_height=h,
        detection_method="heuristic",
    )


# ── Approach B: Curvature-based ────────────────────────────────────────

def _find_extrema_in_range(
    heights: np.ndarray,
    values: np.ndarray,
    h_min: float,
    h_max: float,
    mode: str,  # "min" or "max"
) -> float | None:
    """Find the height of the local extremum within [h_min, h_max]."""
    mask = (heights >= h_min) & (heights <= h_max)
    if not np.any(mask):
        return None

    sub_vals = values[mask]
    sub_heights = heights[mask]

    if mode == "min":
        idx = np.argmin(sub_vals)
    else:
        idx = np.argmax(sub_vals)

    return float(sub_heights[idx])


def _find_crotch_height(profile: CircumferenceProfile) -> float | None:
    """
    Detect crotch height as the lowest point where the number of contours
    transitions from 2 (two legs) to 1 (single torso).
    """
    heights = profile.heights
    nc = profile.n_contours

    # Scan from bottom up; find first height where n_contours drops to 1
    # after being ≥ 2.
    in_legs = False
    for i in range(len(nc)):
        if nc[i] >= 2:
            in_legs = True
        elif in_legs and nc[i] == 1:
            return float(heights[i])

    return None


def detect_landmarks_curvature(
    mesh: trimesh.Trimesh,
    profile: CircumferenceProfile | None = None,
) -> LandmarkSet | None:
    """
    Detect anatomical landmarks from the circumference profile.

    Uses the smoothed profile to find local extrema and classifies
    them based on their height range.

    Returns None if detection fails (insufficient extrema found).
    """
    total_height = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()

    if profile is None:
        profile = compute_circumference_profile(mesh)

    h = profile.heights
    c = profile.smoothed

    # ── Detect crotch via contour-count transition ──
    crotch = _find_crotch_height(profile)
    if crotch is None:
        # Fallback: use heuristic percentage
        crotch = total_height * (lcfg.crotch_min + lcfg.crotch_max) / 2

    # ── Waist: local minimum of circumference between crotch and chest ──
    waist = _find_extrema_in_range(
        h, c,
        total_height * lcfg.waist_min,
        total_height * lcfg.waist_max,
        mode="min",
    )

    # ── Hip: local maximum below waist ──
    hip = _find_extrema_in_range(
        h, c,
        total_height * lcfg.hip_min,
        total_height * lcfg.hip_max,
        mode="max",
    )

    # ── Chest: local maximum above waist ──
    chest = _find_extrema_in_range(
        h, c,
        total_height * lcfg.chest_min,
        total_height * lcfg.chest_max,
        mode="max",
    )

    # ── Neck: local minimum above chest ──
    neck = _find_extrema_in_range(
        h, c,
        total_height * lcfg.neck_min,
        total_height * lcfg.neck_max,
        mode="min",
    )

    # ── Shoulder: where circumference starts to drop sharply above chest ──
    #    Approximated as a fixed offset above chest.
    shoulder = total_height * lcfg.shoulder_height

    # Validate: all critical landmarks must be found
    if any(v is None for v in [waist, hip, chest, neck]):
        logger.warning("Curvature-based detection failed — missing landmarks")
        return None

    return LandmarkSet(
        neck_height=neck,
        shoulder_height=shoulder,
        chest_height=chest,
        waist_height=waist,
        hip_height=hip,
        crotch_height=crotch,
        knee_height=total_height * lcfg.knee_height,
        ankle_height=total_height * lcfg.ankle_height,
        total_height=total_height,
        detection_method="curvature",
    )


# ── Combined: curvature with heuristic fallback ───────────────────────

def detect_landmarks(
    mesh: trimesh.Trimesh,
    profile: CircumferenceProfile | None = None,
) -> LandmarkSet:
    """
    Detect landmarks using curvature-based analysis with heuristic fallback.

    Tries the curvature approach first.  If it fails (insufficient features),
    falls back to height-percentage heuristics.
    """
    total_height = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()

    if profile is None:
        profile = compute_circumference_profile(mesh)

    result = detect_landmarks_curvature(mesh, profile)

    if result is not None:
        logger.info("Landmarks detected via curvature analysis")
        return result

    logger.info("Falling back to heuristic landmark detection")
    return detect_landmarks_heuristic(total_height)
