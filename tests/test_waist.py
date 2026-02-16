import numpy as np
import trimesh
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waist.mesh_processing import (
    remove_degenerate_faces,
    remove_duplicate_faces,
    remove_unreferenced_vertices,
    keep_largest_component,
    fill_small_holes,
    validate_height,
    process_mesh,
)
from waist.alignment import pca_align, remove_floor
from waist.detection import compute_circumference, find_waist_height, multi_slice_average
from waist.confidence import (
    slice_variance_penalty,
    symmetry_penalty,
    density_penalty,
    compute_confidence,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def tall_cylinder() -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=0.1, height=1.75, sections=64)
    mesh.vertices[:, 1] += 0.875
    return mesh


@pytest.fixture
def unit_sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


@pytest.fixture
def aligned_cylinder(tall_cylinder) -> trimesh.Trimesh:
    aligned, _ = pca_align(tall_cylinder)
    return aligned


# ── Mesh processing ───────────────────────────────────────────────────

class TestRemoveDegenerateFaces:

    def test_removes_zero_area_face(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        result = remove_degenerate_faces(mesh)
        assert len(result.faces) == 1

    def test_keeps_all_when_no_degenerate(self, tall_cylinder):
        n_before = len(tall_cylinder.faces)
        result = remove_degenerate_faces(tall_cylinder)
        assert len(result.faces) == n_before


class TestRemoveDuplicateFaces:

    def test_removes_exact_duplicate(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        result = remove_duplicate_faces(mesh)
        assert len(result.faces) == 1

    def test_removes_reordered_duplicate(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [2, 0, 1]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        result = remove_duplicate_faces(mesh)
        assert len(result.faces) == 1


class TestKeepLargestComponent:

    def test_removes_small_fragment(self, tall_cylinder):
        small = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        small.apply_translation([10, 10, 10])
        combined = trimesh.util.concatenate([tall_cylinder, small])
        result = keep_largest_component(combined)
        assert len(result.faces) == len(tall_cylinder.faces)

    def test_single_component_unchanged(self, tall_cylinder):
        result = keep_largest_component(tall_cylinder)
        assert len(result.faces) == len(tall_cylinder.faces)


class TestValidateHeight:

    def test_valid_height(self, tall_cylinder):
        validate_height(tall_cylinder)

    def test_too_short(self):
        mesh = trimesh.creation.cylinder(radius=0.1, height=0.5, sections=16)
        with pytest.raises(ValueError, match="outside valid range"):
            validate_height(mesh)

    def test_too_tall(self):
        mesh = trimesh.creation.cylinder(radius=0.1, height=3.0, sections=16)
        with pytest.raises(ValueError, match="outside valid range"):
            validate_height(mesh)


# ── PCA alignment ─────────────────────────────────────────────────────

class TestPCAAlign:

    def test_y_axis_is_tallest(self, tall_cylinder):
        aligned, info = pca_align(tall_cylinder)
        y_extent = float(np.ptp(aligned.vertices[:, 1]))
        x_extent = float(np.ptp(aligned.vertices[:, 0]))
        z_extent = float(np.ptp(aligned.vertices[:, 2]))
        assert y_extent > x_extent
        assert y_extent > z_extent

    def test_feet_at_y_zero(self, tall_cylinder):
        aligned, _ = pca_align(tall_cylinder)
        assert abs(aligned.vertices[:, 1].min()) < 1e-6

    def test_rotation_det_positive(self, tall_cylinder):
        _, info = pca_align(tall_cylinder)
        R = np.array(info["rotation"])
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_eigenvalue_ratio_stored(self, tall_cylinder):
        _, info = pca_align(tall_cylinder)
        assert info["ratio"] >= 2.0

    def test_rejects_sphere(self, unit_sphere):
        with pytest.raises(ValueError, match="Eigenvalue ratio"):
            pca_align(unit_sphere)


# ── Floor removal ─────────────────────────────────────────────────────

class TestRemoveFloor:

    def test_removes_floor_faces(self, tall_cylinder):
        floor = trimesh.creation.box(extents=[1.0, 0.01, 1.0])
        floor.apply_translation([0, -0.005, 0])
        combined = trimesh.util.concatenate([tall_cylinder, floor])

        aligned, _ = pca_align(combined)
        cleaned = remove_floor(aligned)
        assert len(cleaned.faces) < len(aligned.faces)

    def test_rebaselines_y(self, aligned_cylinder):
        cleaned = remove_floor(aligned_cylinder)
        assert abs(cleaned.vertices[:, 1].min()) < 1e-6


# ── Cross-section ─────────────────────────────────────────────────────

class TestComputeCircumference:

    def test_cylinder_circumference(self, aligned_cylinder):
        mid_y = aligned_cylinder.vertices[:, 1].mean()
        circ, n_seg, pts = compute_circumference(aligned_cylinder, mid_y)
        expected = 2.0 * np.pi * 0.1
        assert abs(circ - expected) / expected < 0.05

    def test_returns_zero_outside_mesh(self, aligned_cylinder):
        circ, n_seg, pts = compute_circumference(aligned_cylinder, 999.0)
        assert circ == 0.0
        assert pts is None

    def test_returns_contour_points(self, aligned_cylinder):
        mid_y = aligned_cylinder.vertices[:, 1].mean()
        _, _, pts = compute_circumference(aligned_cylinder, mid_y)
        assert pts is not None
        assert pts.shape[1] == 3


# ── Waist detection ───────────────────────────────────────────────────

class TestFindWaistHeight:

    def test_waist_in_band(self, aligned_cylinder):
        total_h = float(np.ptp(aligned_cylinder.vertices[:, 1]))
        waist_h, _, _ = find_waist_height(aligned_cylinder)
        y_min = float(aligned_cylinder.vertices[:, 1].min())
        ratio = (waist_h - y_min) / total_h
        assert 0.56 <= ratio <= 0.66

    def test_returns_80_samples(self, aligned_cylinder):
        _, heights, smoothed = find_waist_height(aligned_cylinder)
        assert len(heights) == 80
        assert len(smoothed) == 80


class TestMultiSliceAverage:

    def test_returns_seven_or_fewer_values(self, aligned_cylinder):
        mid = float(aligned_cylinder.vertices[:, 1].mean())
        _, slices = multi_slice_average(aligned_cylinder, mid)
        assert 3 <= len(slices) <= 7

    def test_value_in_cm(self, aligned_cylinder):
        mid = float(aligned_cylinder.vertices[:, 1].mean())
        waist_cm, _ = multi_slice_average(aligned_cylinder, mid)
        expected_cm = 2.0 * np.pi * 0.1 * 100.0
        assert abs(waist_cm - expected_cm) / expected_cm < 0.10

    def test_trimmed_mean_rejects_outliers(self):
        raw = [80.0, 80.1, 80.2, 79.9, 80.0, 50.0, 110.0]
        arr = np.sort(np.array(raw))
        trimmed = arr[1:-1]
        result = float(np.mean(trimmed))
        assert 79.0 < result < 81.0


# ── Confidence ─────────────────────────────────────────────────────────

class TestSliceVariancePenalty:

    def test_low_variance(self):
        cv = slice_variance_penalty([80.0, 80.1, 79.9, 80.0, 80.2])
        assert cv < 0.01

    def test_high_variance(self):
        cv = slice_variance_penalty([50.0, 100.0, 60.0, 90.0, 70.0])
        assert cv > 0.1

    def test_single_value(self):
        cv = slice_variance_penalty([80.0])
        assert cv == 1.0


class TestSymmetryPenalty:

    def test_symmetric_contour(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        contour = np.column_stack([
            0.1 * np.cos(theta),
            np.zeros(100),
            0.1 * np.sin(theta),
        ])
        asym = symmetry_penalty(contour)
        assert asym < 0.05

    def test_asymmetric_contour(self):
        pts = np.array([
            [-0.05, 0, 0],
            [0.20, 0, 0],
            [0.10, 0, 0.1],
            [-0.05, 0, -0.1],
        ])
        asym = symmetry_penalty(pts)
        assert asym > 0.5

    def test_none_contour(self):
        assert symmetry_penalty(None) == 1.0


class TestDensityPenalty:

    def test_sufficient_segments(self):
        assert density_penalty(60) == 0.0

    def test_exactly_required(self):
        assert density_penalty(40) == 0.0

    def test_insufficient_segments(self):
        p = density_penalty(20)
        assert 0.4 < p < 0.6

    def test_zero_segments(self):
        assert density_penalty(0) == 1.0


class TestComputeConfidence:

    def test_high_confidence(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        contour = np.column_stack([
            0.1 * np.cos(theta),
            np.zeros(100),
            0.1 * np.sin(theta),
        ])
        c = compute_confidence([80.0, 80.1, 79.9, 80.0, 80.2], contour, 60)
        assert c > 0.8

    def test_low_confidence_bad_data(self):
        c = compute_confidence([50.0, 100.0, 60.0, 90.0, 70.0], None, 5)
        assert c == 0.0

    def test_clamps_to_zero(self):
        c = compute_confidence([10.0, 200.0], None, 0)
        assert c == 0.0
