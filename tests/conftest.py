"""
Shared test fixtures.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import trimesh

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def unit_cube() -> trimesh.Trimesh:
    """1×1×1 cube centered at origin."""
    return trimesh.creation.box(extents=[1, 1, 1])


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    """Vertical cylinder: radius 0.1 m, height 1.0 m, base at Y=0."""
    mesh = trimesh.creation.cylinder(radius=0.1, height=1.0, sections=64)
    # Cylinder is centered at origin by default; shift so base is at Y=0
    mesh.vertices[:, 1] += 0.5
    return mesh


@pytest.fixture
def body_mesh():
    """Synthetic body mesh from generate_test_mesh."""
    from scripts.generate_test_mesh import generate_body_mesh
    return generate_body_mesh()
