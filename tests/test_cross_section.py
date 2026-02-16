"""
Tests for cross-section extraction.
"""

import numpy as np
import trimesh
import pytest

from app.core.cross_section import (
    intersect_plane_mesh,
    chain_segments,
    compute_perimeter,
    extract_cross_section,
)


class TestIntersectPlaneMesh:

    def test_horizontal_plane_cuts_cube(self, unit_cube):
        origin = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])

        segments = intersect_plane_mesh(
            unit_cube.vertices, unit_cube.faces, origin, normal,
        )

        # A horizontal plane through the center of a cube should produce segments
        assert len(segments) > 0

    def test_no_intersection_above_mesh(self, unit_cube):
        origin = np.array([0.0, 10.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])

        segments = intersect_plane_mesh(
            unit_cube.vertices, unit_cube.faces, origin, normal,
        )
        assert len(segments) == 0


class TestChainSegments:

    def test_chains_square(self):
        """Four segments forming a square should chain into one loop."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([1, 0, 0.0]), np.array([1, 1, 0.0])),
            (np.array([1, 1, 0.0]), np.array([0, 1, 0.0])),
            (np.array([0, 1, 0.0]), np.array([0, 0, 0.0])),
        ]
        loops = chain_segments(segments)
        assert len(loops) == 1
        assert len(loops[0]) == 5  # 4 segments → 5 points (closed)

    def test_two_separate_loops(self):
        """Two separate squares should produce two loops."""
        sq1 = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([1, 0, 0.0]), np.array([1, 1, 0.0])),
            (np.array([1, 1, 0.0]), np.array([0, 1, 0.0])),
            (np.array([0, 1, 0.0]), np.array([0, 0, 0.0])),
        ]
        sq2 = [
            (np.array([5, 0, 0.0]), np.array([6, 0, 0.0])),
            (np.array([6, 0, 0.0]), np.array([6, 1, 0.0])),
            (np.array([6, 1, 0.0]), np.array([5, 1, 0.0])),
            (np.array([5, 1, 0.0]), np.array([5, 0, 0.0])),
        ]
        loops = chain_segments(sq1 + sq2)
        assert len(loops) == 2


class TestPerimeter:

    def test_square_perimeter(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],  # closed
        ], dtype=float)
        assert abs(compute_perimeter(points) - 4.0) < 1e-9


class TestCylinderCrossSection:

    def test_cylinder_circumference(self, cylinder_mesh):
        """Cross-section of a cylinder should give circumference = 2πr."""
        cs = extract_cross_section(cylinder_mesh, height=0.5, axis=1)
        expected = 2 * np.pi * 0.1  # r = 0.1 m
        # Allow 5% tolerance due to discretization
        assert abs(cs.primary_perimeter - expected) / expected < 0.05
