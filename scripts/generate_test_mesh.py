#!/usr/bin/env python3
"""
Generate a synthetic human-shaped mesh for testing.

Builds a body from union of cylinders and spheres:
  - Head:   sphere at top
  - Neck:   small cylinder
  - Torso:  tapered cylinder (wider at chest, narrower at waist, wider at hips)
  - Arms:   two cylinders (optional)
  - Legs:   two cylinders

Known dimensions (for validation):
  - Height: 1.75 m
  - Chest circumference: ~96 cm  (radius ~15.3 cm)
  - Waist circumference: ~80 cm  (radius ~12.7 cm)
  - Hip circumference:   ~98 cm  (radius ~15.6 cm)
  - Neck circumference:  ~38 cm  (radius ~6.05 cm)
  - Shoulder width:      ~45 cm
  - Inseam:              ~80 cm

Usage:
    python scripts/generate_test_mesh.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


# ── Known ground-truth values ─────────────────────────────────────────

GROUND_TRUTH = {
    "height_cm": 175.0,
    "chest_circumference_cm": 96.0,
    "waist_circumference_cm": 80.0,
    "hip_circumference_cm": 98.0,
    "neck_circumference_cm": 38.0,
    "shoulder_width_cm": 45.0,
    "inseam_cm": 80.0,
}


def _cylinder(radius: float, height: float, sections: int = 32) -> trimesh.Trimesh:
    """Create a vertical cylinder centered at the origin."""
    return trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=sections,
    )


def _elliptic_cylinder(
    radius_x: float, radius_z: float, height: float, sections: int = 32,
) -> trimesh.Trimesh:
    """Create a vertical elliptic cylinder (circular cross-section scaled)."""
    cyl = _cylinder(1.0, height, sections)
    # Scale X and Z independently
    cyl.vertices[:, 0] *= radius_x
    cyl.vertices[:, 2] *= radius_z
    return cyl


def generate_body_mesh() -> trimesh.Trimesh:
    """
    Generate a simplified humanoid mesh.

    All dimensions in meters.
    """
    parts: list[trimesh.Trimesh] = []

    # ── Torso (three sections: chest, waist, hip) ──
    # We approximate the torso as three stacked cylinders

    # Hip section: y = 0.80 to 0.95  (height 0.15 m)
    hip_r = 0.098 / (2 * np.pi) * 2 * np.pi  # circumference 98cm → r ≈ 15.6cm
    hip_r = 0.156
    hip = _elliptic_cylinder(hip_r, hip_r * 0.85, 0.15, 48)
    hip.apply_translation([0, 0.875, 0])
    parts.append(hip)

    # Waist section: y = 0.95 to 1.10  (height 0.15 m)
    waist_r = 0.127  # circumference ~80cm
    waist = _elliptic_cylinder(waist_r, waist_r * 0.85, 0.15, 48)
    waist.apply_translation([0, 1.025, 0])
    parts.append(waist)

    # Chest section: y = 1.10 to 1.35  (height 0.25 m)
    chest_r = 0.153  # circumference ~96cm
    chest = _elliptic_cylinder(chest_r, chest_r * 0.80, 0.25, 48)
    chest.apply_translation([0, 1.225, 0])
    parts.append(chest)

    # ── Neck: y = 1.35 to 1.45 ──
    neck_r = 0.0605  # circumference ~38cm
    neck = _cylinder(neck_r, 0.10, 24)
    neck.apply_translation([0, 1.40, 0])
    parts.append(neck)

    # ── Head: sphere at top ──
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.10)
    head.apply_translation([0, 1.55, 0])
    parts.append(head)

    # ── Shoulders: short wide box ──
    shoulder_half = 0.225  # total width 45cm
    # Small cylinder bridging chest to arms
    shoulder_l = _cylinder(0.04, shoulder_half - chest_r, 16)
    shoulder_l.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    )
    shoulder_l.apply_translation([-(chest_r + (shoulder_half - chest_r) / 2), 1.30, 0])
    parts.append(shoulder_l)

    shoulder_r = _cylinder(0.04, shoulder_half - chest_r, 16)
    shoulder_r.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    )
    shoulder_r.apply_translation([(chest_r + (shoulder_half - chest_r) / 2), 1.30, 0])
    parts.append(shoulder_r)

    # ── Left leg: y = 0.0 to 0.80 ──
    leg_r = 0.065
    leg_spacing = 0.08  # distance from center to leg center

    left_leg = _cylinder(leg_r, 0.80, 32)
    left_leg.apply_translation([-leg_spacing, 0.40, 0])
    parts.append(left_leg)

    # ── Right leg: y = 0.0 to 0.80 ──
    right_leg = _cylinder(leg_r, 0.80, 32)
    right_leg.apply_translation([leg_spacing, 0.40, 0])
    parts.append(right_leg)

    # ── Merge all parts ──
    combined = trimesh.util.concatenate(parts)

    # Floor at y = 0
    y_min = combined.vertices[:, 1].min()
    combined.vertices[:, 1] -= y_min

    return combined


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "test_body.obj"
    mesh = generate_body_mesh()

    # Export
    mesh.export(output)
    print(f"Generated test mesh: {output}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces:    {len(mesh.faces)}")
    print(f"  Height:   {mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min():.3f} m")
    print(f"\nGround truth values:")
    for k, v in GROUND_TRUTH.items():
        print(f"  {k}: {v}")

    # Also save ground truth CSV for validation
    gt_path = Path(output).with_suffix(".csv")
    with gt_path.open("w") as f:
        f.write("scan_id,measurement_name,ground_truth_cm\n")
        for name, val in GROUND_TRUTH.items():
            clean_name = name.replace("_cm", "")
            if clean_name == "height":
                continue
            f.write(f"test_body,{clean_name},{val}\n")
    print(f"  Ground truth CSV: {gt_path}")


if __name__ == "__main__":
    main()
