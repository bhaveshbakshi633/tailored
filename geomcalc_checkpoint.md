# GeomCalc - Checkpoints

## Checkpoint [2026-02-16 12:30]
- **Accomplished:** Full measurement engine scaffolded (app/ directory)
  - 6 core modules, FastAPI backend, validation framework
  - Synthetic test mesh generator + benchmark script
- **Status:** Code written, not yet validated with full dependency install

## Checkpoint [2026-02-16 13:00]
- **Accomplished:** Minimal waist measurement engine (waist/ directory)
  - 4 modules: mesh_processing, alignment, detection, confidence
  - CLI tool: `python3 waist_engine.py input.obj`
  - 37/37 unit tests passing
  - Dependencies: numpy, scipy, trimesh, pytest only
- **Tests verified:** `python3 -m pytest tests/test_waist.py -v` → 37 passed
- **Known issue:** trimesh cylinder fixture has only 2 height rings — floor removal destroys it. Tests use aligned cylinder directly for waist detection.

## Checkpoint [2026-02-16 13:15]
- **Accomplished:** Status files updated, initial push to GitHub
- **Repository:** https://github.com/bhaveshbakshi633/tailored
- **Branch:** main
- **Current state:** All code committed and pushed
- **Next steps:** Test with real .obj LiDAR scan, validate full engine

## Checkpoint [2026-02-18 — Virtual Validation]
- **Accomplished:**
  - Built parametric body generator (scripts/parametric_body.py)
    - Elliptical cross-sections + cubic spline + Ramanujan perimeter
    - Default body + 5 random variants with known ground truth
    - LiDAR noise simulation (Gaussian along vertex normals)
    - Random rotation simulation
  - Fixed critical PCA alignment orientation bug
    - Root cause: initial "narrower half = head" heuristic failed for body shape
    - Fix: post-alignment sanity check comparing knee width (0.25H) vs shoulder width (0.75H)
    - Before fix: measured 38.1 cm (measuring thigh of inverted body)
    - After fix: measured 83.60 cm (ground truth: 83.62 cm, error: 0.02 cm)
  - Created integration test suite (tests/test_integration.py) — 29 tests
  - **Final test results: 66/66 pass** (37 unit + 29 integration)

- **Validation summary:**
  | Metric | Value |
  |--------|-------|
  | Default body error | 0.02 cm |
  | MAE (6 bodies) | 0.65 cm |
  | Max error | 1.96 cm |
  | Rotation invariance | Perfect |
  | 1mm noise error | 0.57 cm |
  | 2mm noise error | 2.30 cm |
  | Confidence (clean) | 0.999 |

- **Known issues:**
  - 4 tests in app/ scaffold failing (Phase 2, not yet validated)
  - Noise > 3mm causes significant inflation (mesh smoothing would help)

- **Next steps:**
  - Commit and push validation work
  - Test with real iPhone LiDAR .obj scan
  - Validate/fix app/ Phase 2 scaffold
