from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
import trimesh


def compute_circumference(
    mesh: trimesh.Trimesh, height: float,
) -> tuple[float, int, np.ndarray | None]:
    """
    Extract cross-section at height, select largest closed loop,
    return (perimeter_m, n_segments, contour_points).
    """
    origin = np.array([0.0, height, 0.0])
    normal = np.array([0.0, 1.0, 0.0])

    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return 0.0, 0, None

    best_perimeter = 0.0
    best_points: np.ndarray | None = None
    total_segments = 0

    # Pass 1: prefer closed loops
    for entity in section.entities:
        pts = section.vertices[entity.points]
        total_segments += len(entity.points)

        if len(pts) < 3:
            continue

        gap = float(np.linalg.norm(pts[-1] - pts[0]))
        is_closed = gap < 0.01

        if not is_closed:
            continue

        diffs = np.diff(pts, axis=0)
        perimeter = float(np.sum(np.linalg.norm(diffs, axis=1))) + gap

        if perimeter > best_perimeter:
            best_perimeter = perimeter
            best_points = pts

    # Pass 2: fallback to largest open loop if no closed loops found
    if best_points is None:
        for entity in section.entities:
            pts = section.vertices[entity.points]
            if len(pts) < 3:
                continue
            diffs = np.diff(pts, axis=0)
            gap = float(np.linalg.norm(pts[-1] - pts[0]))
            perimeter = float(np.sum(np.linalg.norm(diffs, axis=1))) + gap
            if perimeter > best_perimeter:
                best_perimeter = perimeter
                best_points = pts

    return best_perimeter, total_segments, best_points


def find_waist_height(
    mesh: trimesh.Trimesh,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Search 0.57H–0.65H for the minimum circumference.
    Returns (waist_height_m, sample_heights, smoothed_circumferences).
    """
    y_min = float(mesh.vertices[:, 1].min())
    y_max = float(mesh.vertices[:, 1].max())
    total_height = y_max - y_min

    h_lo = y_min + 0.57 * total_height
    h_hi = y_min + 0.65 * total_height

    heights = np.linspace(h_lo, h_hi, 80)
    circs = np.zeros(80)

    for i, h in enumerate(heights):
        circs[i], _, _ = compute_circumference(mesh, float(h))

    raw_valid = circs > 0
    n_valid = int(np.sum(raw_valid))
    if n_valid < 10:
        raise ValueError(
            f"Only {n_valid}/80 valid cross-sections in waist band"
        )

    smoothed = gaussian_filter1d(circs, sigma=3.0)

    masked = np.where(raw_valid, smoothed, np.inf)
    min_idx = int(np.argmin(masked))

    return float(heights[min_idx]), heights, smoothed


def multi_slice_average(
    mesh: trimesh.Trimesh, center_height: float,
) -> tuple[float, list[float]]:
    """
    7 slices in ±1 cm window around center_height.
    Trimmed mean (drop highest + lowest).
    Returns (waist_cm, slice_values_cm).
    """
    band = 0.01
    slice_heights = np.linspace(center_height - band, center_height + band, 7)

    values: list[float] = []
    for h in slice_heights:
        circ, _, _ = compute_circumference(mesh, float(h))
        if circ > 0:
            values.append(circ)

    if len(values) < 3:
        raise ValueError(
            f"Only {len(values)}/7 valid slices for multi-slice averaging"
        )

    arr = np.sort(np.array(values))
    if len(arr) > 2:
        trimmed = arr[1:-1]
    else:
        trimmed = arr

    waist_m = float(np.mean(trimmed))
    return waist_m * 100.0, [v * 100.0 for v in values]
