#!/usr/bin/env python3
"""
End-to-end benchmark: generate test mesh → process → measure → evaluate.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.generate_test_mesh import generate_body_mesh, GROUND_TRUTH
from app.core.mesh_processing import (
    extract_largest_component,
    align_mesh_pca,
    smooth_mesh,
)
from app.core.measurement_engine import measure_all
from app.core.landmark_detection import compute_circumference_profile


def main():
    print("=" * 70)
    print("GeomCalc Benchmark")
    print("=" * 70)

    # ── Generate test mesh ──
    print("\n[1] Generating synthetic body mesh...")
    mesh = generate_body_mesh()
    print(f"    Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"    Height:   {mesh.vertices[:, 1].ptp():.3f} m")

    # ── Process ──
    print("\n[2] Processing mesh...")
    t0 = time.perf_counter()

    mesh = extract_largest_component(mesh)
    mesh, info = align_mesh_pca(mesh)
    mesh = smooth_mesh(mesh, iterations=2)

    # Re-baseline Y
    mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()

    t_process = time.perf_counter() - t0
    print(f"    Processing time: {t_process:.2f} s")
    print(f"    Aligned height:  {info['total_height_m']:.3f} m")

    # ── Measure ──
    print("\n[3] Running measurement pipeline...")
    t1 = time.perf_counter()
    measurements = measure_all(mesh)
    t_measure = time.perf_counter() - t1
    print(f"    Measurement time: {t_measure:.2f} s")

    # ── Compare to ground truth ──
    print("\n[4] Results vs Ground Truth")
    print("-" * 70)
    print(f"{'Measurement':<30} {'Predicted':>10} {'Truth':>10} {'Error':>10} {'Conf':>8}")
    print("-" * 70)

    errors = []
    result_map = {
        "chest_circumference": measurements.chest_circumference,
        "waist_circumference": measurements.waist_circumference,
        "hip_circumference": measurements.hip_circumference,
        "shoulder_width": measurements.shoulder_width,
        "inseam": measurements.inseam,
        "neck_circumference": measurements.neck_circumference,
    }

    for name, m in result_map.items():
        gt_key = f"{name}_cm"
        if gt_key in GROUND_TRUTH:
            gt = GROUND_TRUTH[gt_key]
            err = m.value_cm - gt
            errors.append(abs(err))
            print(
                f"{name:<30} {m.value_cm:>10.1f} {gt:>10.1f} {err:>+10.1f} "
                f"{m.confidence.overall:>8.3f}"
            )

    print("-" * 70)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"\n    MAE:  {mae:.2f} cm")
    print(f"    RMSE: {rmse:.2f} cm")
    print(f"    Target MAE < 2.0 cm: {'PASS' if mae < 2.0 else 'FAIL'}")
    print(f"    Total time: {t_process + t_measure:.2f} s")

    # ── Body height ──
    print(f"\n    Body height: {measurements.body_height_cm:.1f} cm "
          f"(truth: {GROUND_TRUTH['height_cm']:.1f} cm, "
          f"error: {measurements.body_height_cm - GROUND_TRUTH['height_cm']:+.1f} cm)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
