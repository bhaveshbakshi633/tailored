"""Integration tests using the parametric body generator with known ground truth."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.parametric_body import (
    BodyProfile,
    generate_body,
    add_lidar_noise,
    add_random_rotation,
    make_variants,
)
from waist.alignment import pca_align, remove_floor
from waist.detection import find_waist_height, multi_slice_average
from waist.confidence import compute_confidence
from waist.detection import compute_circumference


# ── Helpers ────────────────────────────────────────────────────────────

def _measure_waist(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    """Run alignment + detection pipeline, return (waist_cm, confidence, height_ratio)."""
    aligned, _ = pca_align(mesh)
    no_floor = remove_floor(aligned)
    total_h = float(np.ptp(no_floor.vertices[:, 1]))
    waist_h, _, _ = find_waist_height(no_floor)
    waist_cm, slices = multi_slice_average(no_floor, waist_h)
    _, n_seg, contour = compute_circumference(no_floor, waist_h)
    conf = compute_confidence(slices, contour, n_seg)
    ratio = waist_h / total_h if total_h > 0 else 0.0
    return waist_cm, conf, ratio


# ── Default body ──────────────────────────────────────────────────────

class TestDefaultBody:

    @pytest.fixture(scope="class")
    def body(self):
        mesh, gt = generate_body()
        return mesh, gt

    def test_waist_within_1cm(self, body):
        mesh, gt = body
        waist_cm, _, _ = _measure_waist(mesh)
        assert abs(waist_cm - gt["waist_circumference_cm"]) < 1.0, (
            f"measured={waist_cm:.2f}, gt={gt['waist_circumference_cm']:.2f}"
        )

    def test_confidence_above_095(self, body):
        mesh, gt = body
        _, conf, _ = _measure_waist(mesh)
        assert conf > 0.95

    def test_waist_height_ratio(self, body):
        mesh, gt = body
        _, _, ratio = _measure_waist(mesh)
        assert 0.55 <= ratio <= 0.65, f"ratio={ratio:.4f} outside [0.55, 0.65]"


# ── Body variants ────────────────────────────────────────────────────

class TestBodyVariants:

    @pytest.fixture(scope="class")
    def variants(self):
        return make_variants(5, seed=42)

    def test_all_variants_within_2cm(self, variants):
        errors = []
        for mesh, gt in variants:
            waist_cm, _, _ = _measure_waist(mesh)
            err = abs(waist_cm - gt["waist_circumference_cm"])
            errors.append(err)
        mae = np.mean(errors)
        assert mae < 2.0, f"MAE={mae:.2f} cm across {len(variants)} variants"

    def test_all_variants_confidence_above_09(self, variants):
        for mesh, gt in variants:
            _, conf, _ = _measure_waist(mesh)
            assert conf > 0.9, f"variant {gt.get('variant_id')}: conf={conf:.4f}"

    def test_no_variant_exceeds_3cm_error(self, variants):
        for mesh, gt in variants:
            waist_cm, _, _ = _measure_waist(mesh)
            err = abs(waist_cm - gt["waist_circumference_cm"])
            assert err < 3.0, (
                f"variant {gt.get('variant_id')}: err={err:.2f}, "
                f"measured={waist_cm:.2f}, gt={gt['waist_circumference_cm']:.2f}"
            )


# ── Rotation robustness ──────────────────────────────────────────────

class TestRotationRobustness:

    @pytest.fixture(scope="class")
    def body(self):
        return generate_body()

    @pytest.mark.parametrize("angle_deg", [5, 15, 30, 45, 90])
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_rotation_invariant(self, body, angle_deg, seed):
        mesh, gt = body
        rotated = add_random_rotation(mesh, max_angle_deg=angle_deg, seed=seed)
        waist_cm, _, _ = _measure_waist(rotated)
        err = abs(waist_cm - gt["waist_circumference_cm"])
        assert err < 1.0, f"angle={angle_deg}, seed={seed}: err={err:.2f} cm"


# ── Noise robustness ─────────────────────────────────────────────────

class TestNoiseRobustness:

    @pytest.fixture(scope="class")
    def body(self):
        return generate_body()

    def test_1mm_noise(self, body):
        mesh, gt = body
        noisy = add_lidar_noise(mesh, noise_std=0.001, seed=42)
        waist_cm, _, _ = _measure_waist(noisy)
        err = abs(waist_cm - gt["waist_circumference_cm"])
        assert err < 2.0, f"1mm noise: err={err:.2f} cm"

    def test_2mm_noise(self, body):
        mesh, gt = body
        noisy = add_lidar_noise(mesh, noise_std=0.002, seed=42)
        waist_cm, _, _ = _measure_waist(noisy)
        err = abs(waist_cm - gt["waist_circumference_cm"])
        assert err < 4.0, f"2mm noise: err={err:.2f} cm"

    def test_noise_plus_rotation(self, body):
        mesh, gt = body
        noisy = add_lidar_noise(mesh, noise_std=0.001, seed=42)
        rotated = add_random_rotation(noisy, max_angle_deg=30, seed=42)
        waist_cm, _, _ = _measure_waist(rotated)
        err = abs(waist_cm - gt["waist_circumference_cm"])
        assert err < 2.0, f"1mm noise + 30deg rot: err={err:.2f} cm"


# ── PCA alignment orientation ────────────────────────────────────────

class TestAlignmentOrientation:

    def test_body_not_inverted(self):
        """After alignment, 0.50H (hips) must be wider than 0.25H (knees)."""
        mesh, _ = generate_body()
        aligned, _ = pca_align(mesh)
        from waist.alignment import _lateral_extent_at_fraction

        w_knee = _lateral_extent_at_fraction(aligned.vertices, 0.25)
        w_hip = _lateral_extent_at_fraction(aligned.vertices, 0.50)
        w_shoulder = _lateral_extent_at_fraction(aligned.vertices, 0.75)
        assert w_hip > w_knee, "hips must be wider than knees"
        assert w_shoulder > w_knee, "shoulders must be wider than knees"

    def test_pre_rotated_body_not_inverted(self):
        mesh, _ = generate_body()
        rotated = add_random_rotation(mesh, max_angle_deg=90, seed=42)
        aligned, _ = pca_align(rotated)
        from waist.alignment import _lateral_extent_at_fraction

        w_knee = _lateral_extent_at_fraction(aligned.vertices, 0.25)
        w_shoulder = _lateral_extent_at_fraction(aligned.vertices, 0.75)
        assert w_shoulder > w_knee


# ── CLI end-to-end ───────────────────────────────────────────────────

class TestCLI:

    @pytest.fixture(scope="class")
    def obj_file(self, tmp_path_factory):
        mesh, gt = generate_body()
        path = tmp_path_factory.mktemp("cli") / "body.obj"
        mesh.export(str(path))
        return str(path), gt

    def test_cli_returns_valid_json(self, obj_file):
        path, gt = obj_file
        result = subprocess.run(
            [sys.executable, "waist_engine.py", path],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "waist_cm" in data
        assert "confidence" in data

    def test_cli_accuracy(self, obj_file):
        path, gt = obj_file
        result = subprocess.run(
            [sys.executable, "waist_engine.py", path],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(result.stdout)
        err = abs(data["waist_cm"] - gt["waist_circumference_cm"])
        assert err < 1.0, f"CLI err={err:.2f} cm"

    def test_cli_missing_file(self):
        result = subprocess.run(
            [sys.executable, "waist_engine.py", "/tmp/nonexistent.obj"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
