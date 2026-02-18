#!/usr/bin/env python3
"""
Parametric human body mesh generator with analytically known ground truth.

Builds a watertight mesh from elliptical cross-sections that vary smoothly
along the vertical axis via cubic spline interpolation.

Every circumference is computable via Ramanujan's approximation (error < 0.05%),
giving exact ground truth for validating measurement engines.

Usage:
    python scripts/parametric_body.py [output_dir]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
import trimesh


def ellipse_perimeter(a: float, b: float) -> float:
    """
    Ramanujan's second approximation for ellipse circumference.
    a, b = semi-axes.  Error < 0.05% for all aspect ratios.
    """
    h = ((a - b) / (a + b)) ** 2 if (a + b) > 1e-12 else 0.0
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))


@dataclass
class BodyProfile:
    """Human body shape defined by height-parameterized elliptical cross-sections."""

    total_height: float = 1.75

    # (height_fraction, lateral_radius, sagittal_radius)  — all in meters
    control_points: list[tuple[float, float, float]] = field(default_factory=list)

    ankle_frac: float = 0.05
    knee_frac: float = 0.27
    crotch_frac: float = 0.45
    hip_frac: float = 0.50
    waist_frac: float = 0.58
    chest_frac: float = 0.72
    shoulder_frac: float = 0.80
    neck_base_frac: float = 0.84
    neck_top_frac: float = 0.87
    head_center_frac: float = 0.93
    crown_frac: float = 1.00

    def __post_init__(self):
        if not self.control_points:
            self.control_points = self._default_male()

    def _default_male(self) -> list[tuple[float, float, float]]:
        """
        Average adult male.  Implied ground truth (Ramanujan):
          Chest ≈ 95.6 cm, Waist ≈ 83.8 cm, Hip ≈ 97.8 cm, Neck ≈ 37.7 cm
        """
        return [
            (0.000, 0.000, 0.000),
            (self.ankle_frac, 0.040, 0.035),
            (0.15, 0.050, 0.045),
            (self.knee_frac, 0.055, 0.050),
            (0.35, 0.065, 0.058),
            (self.crotch_frac - 0.01, 0.085, 0.075),
            (self.crotch_frac, 0.090, 0.078),
            (self.hip_frac, 0.175, 0.135),
            (0.54, 0.160, 0.125),
            (self.waist_frac, 0.150, 0.115),
            (0.63, 0.155, 0.120),
            (0.68, 0.165, 0.128),
            (self.chest_frac, 0.170, 0.130),
            (0.76, 0.165, 0.125),
            (self.shoulder_frac, 0.155, 0.110),
            (self.neck_base_frac, 0.062, 0.058),
            (self.neck_top_frac, 0.058, 0.055),
            (0.89, 0.085, 0.090),
            (self.head_center_frac, 0.090, 0.095),
            (0.97, 0.070, 0.075),
            (self.crown_frac, 0.000, 0.000),
        ]


def generate_body(
    profile: BodyProfile | None = None,
    n_vertical: int = 200,
    n_azimuthal: int = 64,
) -> tuple[trimesh.Trimesh, dict]:
    """
    Generate watertight body mesh + ground truth dict.

    Returns (mesh, ground_truth) where ground_truth has keys like
    "waist_circumference_cm", "chest_circumference_cm", etc.
    """
    if profile is None:
        profile = BodyProfile()

    cp = np.array(profile.control_points)
    h_frac = cp[:, 0]
    r_lat = cp[:, 1]
    r_sag = cp[:, 2]

    cs_lat = CubicSpline(h_frac, r_lat, bc_type="clamped")
    cs_sag = CubicSpline(h_frac, r_sag, bc_type="clamped")

    t = np.linspace(0.002, 0.998, n_vertical)
    heights = t * profile.total_height
    rl = np.maximum(cs_lat(t), 0.001)
    rs = np.maximum(cs_sag(t), 0.001)

    theta = np.linspace(0, 2 * np.pi, n_azimuthal, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vertices = np.zeros((n_vertical, n_azimuthal, 3))
    vertices[:, :, 0] = rl[:, None] * cos_t[None, :]
    vertices[:, :, 2] = rs[:, None] * sin_t[None, :]
    vertices[:, :, 1] = heights[:, None]

    verts_flat = vertices.reshape(-1, 3)

    faces = []
    for i in range(n_vertical - 1):
        for j in range(n_azimuthal):
            j_next = (j + 1) % n_azimuthal
            v00 = i * n_azimuthal + j
            v01 = i * n_azimuthal + j_next
            v10 = (i + 1) * n_azimuthal + j
            v11 = (i + 1) * n_azimuthal + j_next
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    # Bottom cap
    bc = len(verts_flat)
    verts_flat = np.vstack([verts_flat, [[0.0, heights[0], 0.0]]])
    for j in range(n_azimuthal):
        j_next = (j + 1) % n_azimuthal
        faces.append([bc, j_next, j])

    # Top cap
    tc = len(verts_flat)
    verts_flat = np.vstack([verts_flat, [[0.0, heights[-1], 0.0]]])
    top_start = (n_vertical - 1) * n_azimuthal
    for j in range(n_azimuthal):
        j_next = (j + 1) % n_azimuthal
        faces.append([tc, top_start + j, top_start + j_next])

    faces_arr = np.array(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts_flat, faces=faces_arr, process=True)

    gt = _compute_ground_truth(profile, cs_lat, cs_sag)
    return mesh, gt


def _compute_ground_truth(
    profile: BodyProfile, cs_lat: CubicSpline, cs_sag: CubicSpline
) -> dict:
    H = profile.total_height

    landmarks = {
        "chest": profile.chest_frac,
        "waist": profile.waist_frac,
        "hip": profile.hip_frac,
        "neck": (profile.neck_base_frac + profile.neck_top_frac) / 2,
    }

    gt: dict = {"height_cm": round(H * 100, 1)}

    for name, frac in landmarks.items():
        rl = max(float(cs_lat(frac)), 1e-6)
        rs = max(float(cs_sag(frac)), 1e-6)
        circ = ellipse_perimeter(rl, rs)
        gt[f"{name}_circumference_cm"] = round(circ * 100, 2)

    gt["inseam_cm"] = round(profile.crotch_frac * H * 100, 1)

    shoulder_r = max(float(cs_lat(profile.shoulder_frac)), 1e-6)
    gt["shoulder_width_cm"] = round(shoulder_r * 2 * 100, 1)

    # Also store the waist height ratio for validation
    gt["waist_height_ratio"] = round(profile.waist_frac, 4)

    return gt


def add_lidar_noise(
    mesh: trimesh.Trimesh, noise_std: float = 0.003, seed: int = 42
) -> trimesh.Trimesh:
    """Add Gaussian noise along vertex normals to simulate LiDAR depth noise."""
    rng = np.random.RandomState(seed)
    normals = mesh.vertex_normals
    noise = rng.randn(len(mesh.vertices), 1) * noise_std
    noisy_verts = mesh.vertices + normals * noise
    return trimesh.Trimesh(vertices=noisy_verts, faces=mesh.faces.copy(), process=True)


def add_random_rotation(
    mesh: trimesh.Trimesh, max_angle_deg: float = 15.0, seed: int = 42
) -> trimesh.Trimesh:
    """Apply a random rotation to test PCA alignment robustness."""
    rng = np.random.RandomState(seed)
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.radians(max_angle_deg))
    R = trimesh.transformations.rotation_matrix(angle, axis)
    rotated = mesh.copy()
    rotated.apply_transform(R)
    return rotated


def make_variants(n: int = 5, seed: int = 42) -> list[tuple[trimesh.Trimesh, dict]]:
    """Generate n distinct body shapes with random proportional variation."""
    rng = np.random.RandomState(seed)
    variants = []

    for i in range(n):
        height = rng.uniform(1.55, 1.95)
        body_scale = rng.uniform(0.85, 1.25)
        waist_scale = rng.uniform(0.80, 1.10)
        hip_scale = rng.uniform(0.90, 1.15)
        chest_scale = rng.uniform(0.90, 1.15)

        profile = BodyProfile(total_height=height)
        new_cp = []
        for h_frac, rl, rs in profile.control_points:
            s = body_scale
            if 0.55 <= h_frac <= 0.62:
                s *= waist_scale
            elif 0.48 <= h_frac <= 0.52:
                s *= hip_scale
            elif 0.68 <= h_frac <= 0.76:
                s *= chest_scale
            new_cp.append((h_frac, rl * s, rs * s))

        profile.control_points = new_cp
        mesh, gt = generate_body(profile)
        gt["variant_id"] = i
        variants.append((mesh, gt))

    return variants


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/geomcalc_test")
    out.mkdir(parents=True, exist_ok=True)

    mesh, gt = generate_body()
    path = out / "body_default.obj"
    mesh.export(str(path))
    print(f"Default body: {path}")
    print(f"  Vertices={len(mesh.vertices)}, Faces={len(mesh.faces)}, Watertight={mesh.is_watertight}")
    print(f"  Ground truth: {json.dumps(gt, indent=2)}")

    variants = make_variants(5)
    all_gt = [gt]
    for m, g in variants:
        vid = g["variant_id"]
        vp = out / f"body_v{vid}.obj"
        m.export(str(vp))
        all_gt.append(g)
        print(f"  Variant {vid}: h={g['height_cm']:.0f}cm  waist={g['waist_circumference_cm']:.1f}cm")

    gt_path = out / "ground_truth.json"
    gt_path.write_text(json.dumps(all_gt, indent=2))
    print(f"\nGround truth: {gt_path}")
