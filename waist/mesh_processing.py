from __future__ import annotations

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load as single Trimesh, got {type(mesh).__name__}")
    return mesh


def remove_degenerate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    areas = mesh.area_faces
    mask = areas > 1e-12
    if not np.all(mask):
        mesh.update_faces(mask)
    return mesh


def remove_duplicate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    sorted_faces = np.sort(mesh.faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    if len(unique_idx) < len(mesh.faces):
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[unique_idx] = True
        mesh.update_faces(mask)
    return mesh


def remove_unreferenced_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()
    return mesh


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    return max(components, key=lambda m: len(m.faces))


def fill_small_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    trimesh.repair.fill_holes(mesh)
    return mesh


def validate_height(mesh: trimesh.Trimesh) -> None:
    height = float(mesh.extents.max())
    if not (1.3 <= height <= 2.2):
        raise ValueError(
            f"Mesh height {height:.3f}m outside valid range [1.3, 2.2]m"
        )


def process_mesh(path: str) -> trimesh.Trimesh:
    mesh = load_mesh(path)
    mesh = remove_degenerate_faces(mesh)
    mesh = remove_duplicate_faces(mesh)
    mesh = remove_unreferenced_vertices(mesh)
    mesh = keep_largest_component(mesh)
    mesh = fill_small_holes(mesh)
    validate_height(mesh)
    return mesh
