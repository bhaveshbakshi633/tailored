"""
Plane–mesh intersection and cross-section contour extraction.

Mathematical foundation
──────────────────────
A plane  P  is defined by a point  p₀  and unit normal  n̂.
A point  x  lies on  P  iff  n̂ · (x − p₀) = 0.

For a triangle with vertices  (a, b, c),  compute signed distances:
    dₐ = n̂ · (a − p₀)
    d_b = n̂ · (b − p₀)
    d_c = n̂ · (c − p₀)

If all three have the same sign → no intersection.
Otherwise the plane crosses exactly two edges of the triangle.
For an edge from vertex  vᵢ  (distance dᵢ)  to  vⱼ  (distance dⱼ)
with  dᵢ  and  dⱼ  of opposite sign:

    t = dᵢ / (dᵢ − dⱼ)          ∈ (0, 1)
    intersection = vᵢ + t (vⱼ − vᵢ)

Each intersected triangle contributes one line segment.  For a watertight
mesh, these segments chain into closed loops (each shared edge yields the
same intersection point from both adjacent triangles).

Segment chaining
────────────────
We match segment endpoints by proximity (tolerance ≤ 1e-8 m for exact
floating-point match on shared edges).  A greedy walk produces closed
polylines.

Contour selection
─────────────────
At a given height the plane may intersect the mesh in multiple loops:
  • Above the crotch: one loop (torso)
  • Below the crotch: two loops (left / right leg)
  • Arms away from body: additional small loops

For circumference measurements we select the loop with the **largest
perimeter** (the torso loop, or the single merged loop).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

from app.config import config

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class ContourResult:
    """Result of a single cross-section contour."""
    points: np.ndarray        # (M, 3) ordered contour vertices
    perimeter: float          # meters
    is_closed: bool
    n_triangle_hits: int      # how many triangles the plane intersected


@dataclass
class CrossSectionResult:
    """All contours at one height."""
    height_m: float
    contours: list[ContourResult]
    primary_perimeter: float  # perimeter of the selected (largest) contour
    n_contours: int
    is_valid: bool


# ── Raw plane–mesh intersection (from scratch, NumPy only) ─────────────

def intersect_plane_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Intersect a plane with a triangle mesh.  Pure NumPy implementation.

    Parameters
    ----------
    vertices : (N, 3) float64
    faces    : (F, 3) int — vertex indices
    plane_origin : (3,)
    plane_normal : (3,) — will be normalized internally

    Returns
    -------
    segments : list of (p1, p2) where p1, p2 are (3,) arrays
    """
    normal = plane_normal / np.linalg.norm(plane_normal)

    # Signed distance from every vertex to the plane
    # d_i = n̂ · (v_i − p₀)
    dist = (vertices - plane_origin) @ normal  # (N,)

    # Per-face distances  (F, 3)
    fd = dist[faces]

    # A triangle is intersected iff vertices are not all on the same side.
    # Equivalent: max(d) > 0  and  min(d) < 0   (ignoring exact-on-plane).
    fmax = fd.max(axis=1)
    fmin = fd.min(axis=1)
    hit_mask = (fmax > 0) & (fmin < 0)

    hit_faces = faces[hit_mask]
    hit_dist = fd[hit_mask]  # (H, 3)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    edges = [(0, 1), (1, 2), (2, 0)]

    for fi in range(len(hit_faces)):
        d = hit_dist[fi]
        v = vertices[hit_faces[fi]]
        pts: list[np.ndarray] = []

        for i, j in edges:
            if (d[i] > 0) != (d[j] > 0):
                t = d[i] / (d[i] - d[j])
                p = v[i] + t * (v[j] - v[i])
                pts.append(p)
            if len(pts) == 2:
                break

        if len(pts) == 2:
            segments.append((pts[0], pts[1]))

    return segments


# ── Segment chaining ──────────────────────────────────────────────────

def chain_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    tolerance: float = 1e-8,
) -> list[np.ndarray]:
    """
    Chain unordered line segments into closed polyline loops.

    For a watertight mesh, every segment endpoint is shared with exactly one
    other segment (on the adjacent triangle), so the chains always close.

    Returns list of (K, 3) arrays — each a closed contour.
    """
    if not segments:
        return []

    n = len(segments)
    used = np.zeros(n, dtype=bool)
    loops: list[np.ndarray] = []

    # Pre-stack endpoints for fast distance computation
    starts = np.array([s[0] for s in segments])  # (n, 3)
    ends = np.array([s[1] for s in segments])     # (n, 3)

    while True:
        # Find first unused segment
        remaining = np.where(~used)[0]
        if len(remaining) == 0:
            break

        idx = remaining[0]
        used[idx] = True
        chain = [segments[idx][0], segments[idx][1]]

        for _ in range(n):  # at most n iterations
            current = chain[-1]

            # Vectorized distance to all unused segment endpoints
            unused_idx = np.where(~used)[0]
            if len(unused_idx) == 0:
                break

            dist_starts = np.linalg.norm(starts[unused_idx] - current, axis=1)
            dist_ends = np.linalg.norm(ends[unused_idx] - current, axis=1)

            min_s_idx = np.argmin(dist_starts)
            min_e_idx = np.argmin(dist_ends)
            min_s_dist = dist_starts[min_s_idx]
            min_e_dist = dist_ends[min_e_idx]

            if min_s_dist < min_e_dist:
                if min_s_dist > tolerance:
                    break
                real_idx = unused_idx[min_s_idx]
                chain.append(segments[real_idx][1])
            else:
                if min_e_dist > tolerance:
                    break
                real_idx = unused_idx[min_e_idx]
                chain.append(segments[real_idx][0])

            used[real_idx] = True

            # Check closure
            if np.linalg.norm(chain[-1] - chain[0]) < tolerance:
                break

        loops.append(np.array(chain))

    return loops


# ── Perimeter computation ──────────────────────────────────────────────

def compute_perimeter(contour: np.ndarray) -> float:
    """Sum of edge lengths along a polyline.  Handles open/closed."""
    diffs = np.diff(contour, axis=0)
    perimeter = np.sum(np.linalg.norm(diffs, axis=1))
    # Close the loop if not already closed
    gap = np.linalg.norm(contour[-1] - contour[0])
    if gap > 1e-8:
        perimeter += gap
    return float(perimeter)


# ── High-level: extract cross-section using trimesh (production path) ──

def extract_cross_section_trimesh(
    mesh: trimesh.Trimesh,
    height: float,
    axis: int = 1,
) -> CrossSectionResult:
    """
    Extract cross-section using trimesh's section() method.

    This is the production code path — it is faster and handles edge cases
    that the raw implementation may miss.
    """
    origin = np.zeros(3)
    origin[axis] = height
    normal = np.zeros(3)
    normal[axis] = 1.0

    section = mesh.section(plane_origin=origin, plane_normal=normal)

    if section is None:
        return CrossSectionResult(
            height_m=height, contours=[], primary_perimeter=0.0,
            n_contours=0, is_valid=False,
        )

    contour_results: list[ContourResult] = []

    for entity in section.entities:
        points = section.vertices[entity.points]
        perimeter = compute_perimeter(points)
        is_closed = np.linalg.norm(points[0] - points[-1]) < 1e-6
        contour_results.append(ContourResult(
            points=points,
            perimeter=perimeter,
            is_closed=is_closed,
            n_triangle_hits=len(entity.points),
        ))

    if not contour_results:
        return CrossSectionResult(
            height_m=height, contours=[], primary_perimeter=0.0,
            n_contours=0, is_valid=False,
        )

    # Select primary contour: largest perimeter
    primary = max(contour_results, key=lambda c: c.perimeter)

    return CrossSectionResult(
        height_m=height,
        contours=contour_results,
        primary_perimeter=primary.perimeter,
        n_contours=len(contour_results),
        is_valid=primary.n_triangle_hits >= config.measurement.min_contour_points,
    )


# ── High-level: extract cross-section using raw algorithm ──────────────

def extract_cross_section_raw(
    mesh: trimesh.Trimesh,
    height: float,
    axis: int = 1,
) -> CrossSectionResult:
    """
    Extract cross-section using our from-scratch plane–mesh intersection.

    Useful for validation against the trimesh-based path and for environments
    where trimesh's section() has issues.
    """
    origin = np.zeros(3)
    origin[axis] = height
    normal = np.zeros(3)
    normal[axis] = 1.0

    segments = intersect_plane_mesh(
        mesh.vertices, mesh.faces, origin, normal,
    )

    if not segments:
        return CrossSectionResult(
            height_m=height, contours=[], primary_perimeter=0.0,
            n_contours=0, is_valid=False,
        )

    loops = chain_segments(segments)

    contour_results = []
    for loop in loops:
        perimeter = compute_perimeter(loop)
        is_closed = np.linalg.norm(loop[0] - loop[-1]) < 1e-6
        contour_results.append(ContourResult(
            points=loop,
            perimeter=perimeter,
            is_closed=is_closed,
            n_triangle_hits=len(loop),
        ))

    primary = max(contour_results, key=lambda c: c.perimeter)

    return CrossSectionResult(
        height_m=height,
        contours=contour_results,
        primary_perimeter=primary.perimeter,
        n_contours=len(contour_results),
        is_valid=primary.n_triangle_hits >= config.measurement.min_contour_points,
    )


# ── Default entry point ───────────────────────────────────────────────

def extract_cross_section(
    mesh: trimesh.Trimesh,
    height: float,
    axis: int = 1,
    backend: str = "trimesh",
) -> CrossSectionResult:
    """Extract cross-section using the specified backend."""
    if backend == "raw":
        return extract_cross_section_raw(mesh, height, axis)
    return extract_cross_section_trimesh(mesh, height, axis)
