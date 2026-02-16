#!/usr/bin/env python3

import json
import sys

from waist.mesh_processing import process_mesh
from waist.alignment import pca_align, remove_floor
from waist.detection import find_waist_height, multi_slice_average, compute_circumference
from waist.confidence import compute_confidence


def measure_waist(path: str) -> dict:
    mesh = process_mesh(path)
    mesh, _ = pca_align(mesh)
    mesh = remove_floor(mesh)

    waist_h, _, _ = find_waist_height(mesh)
    waist_cm, slice_values = multi_slice_average(mesh, waist_h)

    _, n_segments, contour = compute_circumference(mesh, waist_h)
    confidence = compute_confidence(slice_values, contour, n_segments)

    total_height = float(mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())

    return {
        "waist_cm": round(waist_cm, 1),
        "confidence": confidence,
        "debug": {
            "slice_values": [round(v, 2) for v in slice_values],
            "waist_height_ratio": round(waist_h / total_height, 4) if total_height > 0 else 0.0,
        },
    }


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input.obj>", file=sys.stderr)
        sys.exit(1)

    try:
        result = measure_waist(sys.argv[1])
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
