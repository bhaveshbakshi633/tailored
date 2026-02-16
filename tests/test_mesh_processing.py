"""
Tests for the mesh processing pipeline.
"""

import numpy as np
import trimesh
import pytest

from app.core.mesh_processing import (
    extract_largest_component,
    align_mesh_pca,
    smooth_mesh,
    remove_floor,
)


class TestExtractLargestComponent:

    def test_single_component_unchanged(self, unit_cube):
        result = extract_largest_component(unit_cube)
        assert len(result.faces) == len(unit_cube.faces)

    def test_removes_small_fragments(self, unit_cube):
        # Add a tiny cube far away
        small = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        small.apply_translation([10, 10, 10])
        combined = trimesh.util.concatenate([unit_cube, small])

        result = extract_largest_component(combined)
        assert len(result.faces) == len(unit_cube.faces)


class TestPCAAlignment:

    def test_vertical_axis_is_y(self, body_mesh):
        aligned, info = align_mesh_pca(body_mesh)
        height = info["total_height_m"]
        # A body should be roughly 1.5-2.0 m tall
        assert 1.0 < height < 2.5

    def test_feet_at_y_zero(self, body_mesh):
        aligned, _ = align_mesh_pca(body_mesh)
        y_min = aligned.vertices[:, 1].min()
        assert abs(y_min) < 0.01  # feet should be at Y ≈ 0

    def test_rotation_is_proper(self, body_mesh):
        _, info = align_mesh_pca(body_mesh)
        R = np.array(info["rotation_matrix"])
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6  # det(R) = 1 for proper rotation


class TestSmoothing:

    def test_preserves_face_count(self, unit_cube):
        smoothed = smooth_mesh(unit_cube, iterations=1)
        assert len(smoothed.faces) == len(unit_cube.faces)

    def test_preserves_vertex_count(self, unit_cube):
        smoothed = smooth_mesh(unit_cube, iterations=1)
        assert len(smoothed.vertices) == len(unit_cube.vertices)


class TestFloorRemoval:

    def test_removes_bottom_faces(self, cylinder_mesh):
        # Add a floor plane at Y=0
        floor = trimesh.creation.box(extents=[2, 0.01, 2])
        floor.apply_translation([0, -0.005, 0])
        combined = trimesh.util.concatenate([cylinder_mesh, floor])

        result = remove_floor(combined, percentile=2, margin=0.02)
        # Should have fewer faces than the combined mesh
        assert len(result.faces) < len(combined.faces)
