# GeomCalc - Project Context

## Project Overview
- **Name:** GeomCalc — 3D Human Body Measurement Engine
- **Location:** `/home/ssi/Projects/geomcalc/`
- **Repository:** https://github.com/bhaveshbakshi633/tailored
- **Created:** 2026-02-16
- **Status:** Core waist engine implemented and tested, full measurement engine scaffolded

## Active Scope — Waist Measurement Engine (v0.1)
Minimal production-grade waist measurement from LiDAR mesh:
- Input: `.obj` mesh in meters (iPhone LiDAR)
- Output: `waist_cm` + `confidence` + `debug` JSON
- 37/37 unit tests passing
- CLI: `python3 waist_engine.py input.obj`

## Pipeline
```
Load .obj → degenerate/duplicate removal → largest component → fill holes
→ height validation [1.3–2.2m] → PCA align (Y=vertical, det(R)=+1)
→ eigenvalue ratio gate (λ1/λ2 ≥ 2.0) → floor removal (2nd %ile + 5mm)
→ 80-slice waist search [0.57H–0.65H] → Gaussian smooth → find min
→ 7-slice ±1cm band → trimmed mean → confidence scoring → JSON output
```

## File Map
```
# Waist engine (active, tested)
waist/mesh_processing.py  — load, clean, validate
waist/alignment.py        — PCA align, floor removal
waist/detection.py        — circumference extraction, waist search, multi-slice
waist/confidence.py       — cv + symmetry + density → composite score
waist_engine.py           — CLI entry point
tests/test_waist.py       — 37 unit tests

# Full measurement engine (scaffolded, not yet validated)
app/core/mesh_processing.py    — PCA alignment, cleaning, floor removal
app/core/cross_section.py      — plane-mesh intersection, contour chaining
app/core/landmark_detection.py — height-% and curvature-based landmarks
app/core/measurement_engine.py — chest/waist/hip/shoulder/inseam/neck
app/core/confidence.py         — 4-factor confidence scoring
app/core/size_recommendation.py— brand charts, trapezoidal fit scoring
app/validation/evaluation.py   — MAE/RMSE/Bland-Altman/calibration
app/main.py                    — FastAPI entrypoint
app/config.py                  — all tunable parameters
```

## Key Decisions
1. PCA with eigenvalue ratio rejection (λ1/λ2 < 2.0 → reject)
2. Floor removal via 2nd percentile + 5mm margin (not RANSAC)
3. Gaussian σ=3.0 on 80-slice profile for noise suppression
4. Trimmed mean (drop min+max of 7 slices) for outlier rejection
5. Confidence: C = max(0, 1 − 2·cv − 1.5·asym − 1·sparse)
6. Taubin smoothing over Laplacian — preserves mesh volume (full engine)
7. Dual cross-section backends — raw NumPy + trimesh (full engine)

## Dependencies (waist engine)
- numpy, scipy, trimesh, pytest

## Open Questions
- Real-world LiDAR mesh validation
- iOS client (Swift/ARKit)
- Production deployment (Docker, S3, auth)
