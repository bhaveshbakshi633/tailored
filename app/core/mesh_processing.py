"""
Mesh processing pipeline: load → clean → align → smooth.

Mathematical foundation
──────────────────────
**PCA-based alignment** places the body into a canonical frame by computing the
covariance matrix of the vertex cloud and extracting its eigenvectors.

Given  N  vertices  V = {v₁, …, vₙ} ∈ ℝ³:

  1. Centroid:   μ = (1/N) Σ vᵢ
  2. Centering:  V' = V − μ
  3. Covariance: C = (1/N) V'ᵀ V'           (3×3 symmetric positive semi-definite)
  4. Eigen-decomposition:  C = Q Λ Qᵀ
     where  Λ = diag(λ₁, λ₂, λ₃),  λ₁ ≥ λ₂ ≥ λ₃
     and columns of Q are the corresponding eigenvectors.

For a standing human:
  • λ₁ (largest variance)  → vertical axis  (height ≈ 1.7 m spread)
  • λ₂                      → lateral axis   (shoulder width ≈ 0.45 m)
  • λ₃ (smallest)           → sagittal axis  (front-to-back depth)

Sign disambiguation:
  • Vertical:  the half of the point cloud with smaller lateral spread is the
    head.  We orient the vertical axis so that half points upward.
  • Lateral:   arbitrary (left/right symmetry).
  • Sagittal:  right-hand rule from vertical × lateral.

**Floor removal** trims the flat support surface captured by LiDAR.  We
identify the floor as the cluster of vertices whose Y coordinate falls below
the `floor_percentile`-th percentile, then remove all faces that have *any*
vertex below  (floor_y + margin).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from app.config import config

logger = logging.getLogger(__name__)

cfg = config.mesh


# ── Loading ────────────────────────────────────────────────────────────

def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load an .obj / .ply / .stl mesh and ensure it is a single Trimesh."""
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single Trimesh, got {type(mesh).__name__}")
    logger.info(
        "Loaded mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces)
    )
    return mesh


# ── Largest connected component ────────────────────────────────────────

def extract_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Keep only the largest connected component.

    LiDAR scans often contain floating debris, pedestal fragments, or
    disconnected patches.  We use face adjacency to find connected
    components and retain the one with the most faces.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    largest = max(components, key=lambda m: len(m.faces))
    removed = len(mesh.faces) - len(largest.faces)
    logger.info(
        "Extracted largest component: kept %d / %d faces (removed %d in %d fragments)",
        len(largest.faces),
        len(mesh.faces),
        removed,
        len(components) - 1,
    )
    return largest


# ── Floor removal ──────────────────────────────────────────────────────

def remove_floor(
    mesh: trimesh.Trimesh,
    percentile: float | None = None,
    margin: float | None = None,
) -> trimesh.Trimesh:
    """
    Remove the floor plane from the mesh.

    Strategy:
      1. Find the `percentile`-th percentile of vertex Y coordinates.
         For a body scan this captures the floor / ground surface.
      2. Remove every face that has ANY vertex below  (floor_y + margin).

    This is cheaper than RANSAC plane fitting and works reliably when the
    mesh is already roughly upright (which we guarantee after PCA alignment).
    """
    percentile = percentile or cfg.floor_percentile
    margin = margin or cfg.floor_margin_m

    y_coords = mesh.vertices[:, 1]
    floor_y = np.percentile(y_coords, percentile)
    cut_y = floor_y + margin

    # Identify vertices below cut
    below = y_coords < cut_y

    # Remove faces where any vertex is below
    face_below = np.any(below[mesh.faces], axis=1)
    keep_faces = ~face_below

    if not np.any(keep_faces):
        logger.warning("Floor removal would delete all faces — skipping")
        return mesh

    cleaned = mesh.submesh([np.where(keep_faces)[0]], append=True)
    logger.info(
        "Floor removal: cut at y=%.4f m, removed %d faces",
        cut_y,
        int(np.sum(face_below)),
    )
    return cleaned


# ── PCA alignment ─────────────────────────────────────────────────────

def align_mesh_pca(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """
    Align mesh to canonical coordinates using PCA.

    Returns
    -------
    aligned_mesh : trimesh.Trimesh
        Mesh in canonical frame (Y up, X lateral, Z sagittal), feet at Y≈0.
    info : dict
        eigenvalues, rotation_matrix, centroid, total_height
    """
    verts = np.array(mesh.vertices, dtype=np.float64)

    # 1. Centroid & centering
    centroid = verts.mean(axis=0)
    centered = verts - centroid

    # 2. Covariance matrix  (3 × 3)
    cov = np.cov(centered, rowvar=False)  # rowvar=False → each row is an observation

    # 3. Eigen-decomposition (eigh returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending by eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # e0 = vertical (largest variance), e1 = lateral, e2 = sagittal
    e_vertical = eigenvectors[:, 0].copy()
    e_lateral = eigenvectors[:, 1].copy()
    e_sagittal = eigenvectors[:, 2].copy()

    # 4. Sign disambiguation for vertical axis
    #    The "head" side has smaller lateral spread than the "foot" side.
    proj_v = centered @ e_vertical
    median_v = np.median(proj_v)
    upper = centered[proj_v > median_v]
    lower = centered[proj_v <= median_v]

    spread_upper = np.std(upper @ e_lateral)
    spread_lower = np.std(lower @ e_lateral)

    if spread_upper > spread_lower:
        # Upper half is wider ⇒ that's the torso/hips, not the head.
        # Flip so the narrower half (head) is "up".
        e_vertical = -e_vertical

    # 5. Ensure right-handed frame
    cross = np.cross(e_lateral, e_vertical)
    if np.dot(cross, e_sagittal) < 0:
        e_sagittal = -e_sagittal

    # 6. Build rotation: rows map original axes → canonical axes
    #    Row 0 → X (lateral), Row 1 → Y (vertical up), Row 2 → Z (sagittal)
    rotation = np.vstack([e_lateral, e_vertical, e_sagittal])  # (3, 3)

    # Guarantee proper rotation (det = +1)
    if np.linalg.det(rotation) < 0:
        rotation[2, :] = -rotation[2, :]

    # 7. Apply
    aligned_verts = (rotation @ centered.T).T  # (N, 3)

    # 8. Shift so min(Y) = 0  (feet on the ground)
    y_min = aligned_verts[:, 1].min()
    aligned_verts[:, 1] -= y_min

    total_height = aligned_verts[:, 1].max() - aligned_verts[:, 1].min()

    aligned_mesh = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=mesh.faces.copy(),
        process=False,
    )

    info = {
        "eigenvalues": eigenvalues.tolist(),
        "rotation_matrix": rotation.tolist(),
        "centroid": centroid.tolist(),
        "total_height_m": float(total_height),
    }
    logger.info("PCA alignment complete — total height: %.3f m", total_height)
    return aligned_mesh, info


# ── Smoothing ──────────────────────────────────────────────────────────

def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int | None = None,
    lamb: float | None = None,
) -> trimesh.Trimesh:
    """
    Taubin smoothing (λ|μ scheme) to reduce LiDAR noise without shrinkage.

    Standard Laplacian smoothing with factor λ > 0 shrinks the mesh.
    Taubin's fix alternates a shrinking step (λ) with an inflation step (μ < 0)
    each iteration, preserving volume.

    We use trimesh's built-in Laplacian smooth as the base and apply the
    λ/μ alternation ourselves.
    """
    iterations = iterations or cfg.smoothing_iterations
    lamb = lamb or cfg.smoothing_lambda
    mu = -lamb - 0.01  # μ slightly more negative than −λ  (Taubin's rule of thumb)

    verts = mesh.vertices.copy().astype(np.float64)
    adj = mesh.vertex_neighbors  # list of lists

    for _ in range(iterations):
        for factor in (lamb, mu):
            new_verts = verts.copy()
            for i, neighbors in enumerate(adj):
                if len(neighbors) == 0:
                    continue
                neighbor_mean = verts[neighbors].mean(axis=0)
                new_verts[i] = verts[i] + factor * (neighbor_mean - verts[i])
            verts = new_verts

    smoothed = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    logger.info("Taubin smoothing: %d iterations, λ=%.3f, μ=%.3f", iterations, lamb, mu)
    return smoothed


# ── Full pipeline ──────────────────────────────────────────────────────

def process_mesh(path: str | Path) -> tuple[trimesh.Trimesh, dict]:
    """
    Full mesh processing pipeline.

    Steps:
      1. Load mesh from file
      2. Extract largest connected component
      3. PCA alignment to canonical frame
      4. Floor removal
      5. Taubin smoothing
      6. Re-center feet at Y = 0

    Returns (processed_mesh, alignment_info).
    """
    mesh = load_mesh(path)
    mesh = extract_largest_component(mesh)
    mesh, info = align_mesh_pca(mesh)
    mesh = remove_floor(mesh)

    # After floor removal, re-shift feet to Y = 0
    y_min = mesh.vertices[:, 1].min()
    mesh.vertices[:, 1] -= y_min

    mesh = smooth_mesh(mesh)

    info["total_height_m"] = float(mesh.vertices[:, 1].max())
    logger.info("Pipeline complete — final height: %.3f m", info["total_height_m"])
    return mesh, info
