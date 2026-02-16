"""
Tests for the full measurement pipeline on a synthetic body.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

from app.core.mesh_processing import align_mesh_pca, smooth_mesh, extract_largest_component
from app.core.measurement_engine import measure_all
from scripts.generate_test_mesh import GROUND_TRUTH


@pytest.fixture
def processed_body(body_mesh):
    """Body mesh after the full processing pipeline."""
    mesh = extract_largest_component(body_mesh)
    mesh, _ = align_mesh_pca(mesh)
    mesh = smooth_mesh(mesh, iterations=2)
    mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()
    return mesh


class TestFullMeasurement:

    def test_pipeline_runs(self, processed_body):
        """The pipeline should complete without error."""
        result = measure_all(processed_body)
        assert result.body_height_cm > 0

    def test_height_reasonable(self, processed_body):
        result = measure_all(processed_body)
        assert 150 < result.body_height_cm < 200

    def test_chest_in_range(self, processed_body):
        result = measure_all(processed_body)
        gt = GROUND_TRUTH["chest_circumference_cm"]
        # Allow ±15% tolerance on synthetic mesh (geometry is approximate)
        assert abs(result.chest_circumference.value_cm - gt) / gt < 0.15

    def test_waist_in_range(self, processed_body):
        result = measure_all(processed_body)
        gt = GROUND_TRUTH["waist_circumference_cm"]
        assert abs(result.waist_circumference.value_cm - gt) / gt < 0.15

    def test_confidence_scores_valid(self, processed_body):
        result = measure_all(processed_body)
        for m in [
            result.chest_circumference,
            result.waist_circumference,
            result.hip_circumference,
            result.neck_circumference,
        ]:
            assert 0.0 <= m.confidence.overall <= 1.0
            assert 0.0 <= m.confidence.slice_consistency <= 1.0
            assert 0.0 <= m.confidence.symmetry <= 1.0
            assert 0.0 <= m.confidence.density <= 1.0
            assert 0.0 <= m.confidence.smoothness <= 1.0

    def test_landmarks_detected(self, processed_body):
        result = measure_all(processed_body)
        lm = result.landmarks
        assert lm.neck_height > lm.chest_height
        assert lm.chest_height > lm.waist_height
        assert lm.waist_height > lm.hip_height
        assert lm.hip_height > lm.crotch_height
