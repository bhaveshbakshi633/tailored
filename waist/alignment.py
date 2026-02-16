from __future__ import annotations

import numpy as np
import trimesh


def pca_align(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    verts = mesh.vertices.astype(np.float64)

    centroid = verts.mean(axis=0)
    centered = verts - centroid

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if eigenvalues[1] < 1e-12:
        raise ValueError("Degenerate mesh — second eigenvalue is near zero")

    ratio = float(eigenvalues[0] / eigenvalues[1])
    if ratio < 2.0:
        raise ValueError(
            f"Eigenvalue ratio λ1/λ2 = {ratio:.2f} < 2.0 — scan pose rejected"
        )

    e_vert = eigenvectors[:, 0].copy()
    e_lat = eigenvectors[:, 1].copy()
    e_sag = eigenvectors[:, 2].copy()

    proj = centered @ e_vert
    median_proj = np.median(proj)
    upper_mask = proj > median_proj
    lower_mask = ~upper_mask

    if np.any(upper_mask) and np.any(lower_mask):
        upper_spread = np.std(centered[upper_mask] @ e_lat)
        lower_spread = np.std(centered[lower_mask] @ e_lat)
        if upper_spread > lower_spread:
            e_vert = -e_vert

    R = np.vstack([e_lat, e_vert, e_sag])

    if np.linalg.det(R) < 0:
        R[2, :] = -R[2, :]

    aligned_verts = (R @ centered.T).T
    aligned_verts[:, 1] -= aligned_verts[:, 1].min()

    aligned_mesh = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=mesh.faces.copy(),
        process=False,
    )

    return aligned_mesh, {
        "eigenvalues": eigenvalues.tolist(),
        "ratio": ratio,
        "rotation": R.tolist(),
    }


def remove_floor(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    y = mesh.vertices[:, 1]
    floor_y = float(np.percentile(y, 2))
    cut_y = floor_y + 0.005

    below = y < cut_y
    face_below = np.any(below[mesh.faces], axis=1)
    keep = ~face_below

    if not np.any(keep):
        return mesh

    cleaned = mesh.submesh([np.where(keep)[0]], append=True)
    cleaned.vertices[:, 1] -= cleaned.vertices[:, 1].min()
    return cleaned
