# GeomCalc - Project Context

## Project Overview
- **Name:** GeomCalc — 3D Human Body Measurement Engine
- **Location:** `/home/ssi/Projects/geomcalc/`
- **Repository:** https://github.com/bhaveshbakshi633/tailored
- **Created:** 2026-02-16
- **Status:** Waist engine validated against parametric ground truth (MAE 0.65cm)

## Active Scope — Waist Measurement Engine (v0.1)
Minimal production-grade waist measurement from LiDAR mesh:
- Input: `.obj` mesh in meters (iPhone LiDAR)
- Output: `waist_cm` + `confidence` + `debug` JSON
- 66/66 tests passing (37 unit + 29 integration)
- CLI: `python3 waist_engine.py input.obj`

## Pipeline
```
Load .obj → degenerate/duplicate removal → largest component → fill holes
→ height validation [1.3–2.2m] → PCA align (Y=vertical, det(R)=+1)
→ eigenvalue ratio gate (λ1/λ2 ≥ 2.0) → post-alignment orientation check
→ floor removal (2nd %ile + 5mm)
→ 80-slice waist search [0.57H–0.65H] → Gaussian smooth → find min
→ 7-slice ±1cm band → trimmed mean → confidence scoring → JSON output
```

## Validation Results (Parametric Bodies)
- **Default body:** 83.60 cm measured vs 83.62 cm ground truth (0.02 cm error)
- **6 body variants:** MAE = 0.65 cm, max error = 1.96 cm
- **Rotation robustness:** Perfect (any angle, 0.02 cm error)
- **1mm noise:** 0.57 cm error
- **2mm noise:** 2.30 cm error
- **1mm noise + 30deg rotation:** 0.57 cm error

## File Map
```
# Waist engine (active, tested)
waist/mesh_processing.py  — load, clean, validate
waist/alignment.py        — PCA align, orientation check, floor removal
waist/detection.py        — circumference extraction, waist search, multi-slice
waist/confidence.py       — cv + symmetry + density → composite score
waist_engine.py           — CLI entry point
tests/test_waist.py       — 37 unit tests
tests/test_integration.py — 29 integration tests (parametric body + noise/rotation)

# Parametric body generator
scripts/parametric_body.py — elliptical cross-sections, cubic spline, Ramanujan ground truth

# Full measurement engine (scaffolded, 4 tests failing — not yet validated)
app/core/                 — mesh processing, cross-section, landmarks, measurements
app/main.py               — FastAPI entrypoint
app/config.py             — all tunable parameters
```

## Key Decisions
1. PCA with eigenvalue ratio rejection (λ1/λ2 < 2.0 → reject)
2. Post-alignment orientation check: knee width (0.25H) vs shoulder width (0.75H)
3. Floor removal via 2nd percentile + 5mm margin (not RANSAC)
4. Gaussian σ=3.0 on 80-slice profile for noise suppression
5. Trimmed mean (drop min+max of 7 slices) for outlier rejection
6. Confidence: C = max(0, 1 − 2·cv − 1.5·asym − 1·sparse)

## Dependencies (waist engine)
- numpy, scipy, trimesh, pytest

## Open Questions
- Real-world LiDAR mesh validation
- iOS client (Swift/ARKit)
- Production deployment (Docker, S3, auth)
