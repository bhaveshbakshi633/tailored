# GeomCalc - Implementation Plan

## Goals
1. Build complete 3D body measurement engine from LiDAR mesh input
2. Target MAE < 2 cm compared to tape measurement
3. Provide confidence metrics and size recommendations
4. Production-ready backend

## Phase 1 — Waist Engine MVP [COMPLETE]
- [x] Project structure
- [x] Mesh processing (load, clean, validate, largest component, fill holes)
- [x] PCA alignment (Y-up, det(R)=+1, eigenvalue ratio gate)
- [x] Floor removal (2nd percentile + 5mm)
- [x] Waist detection (80-slice search, Gaussian smooth, min circumference)
- [x] Multi-slice averaging (7 slices, ±1cm, trimmed mean)
- [x] Confidence scoring (cv + symmetry + density composite)
- [x] CLI tool (waist_engine.py)
- [x] Unit tests (37/37 passing)
- [x] Push to GitHub

## Phase 2 — Full Measurement Engine [SCAFFOLDED]
- [x] Configuration system (app/config.py)
- [x] Pydantic schemas (app/models/schemas.py)
- [x] Cross-section extraction (raw + trimesh backends)
- [x] Anatomical landmark detection (heuristic + curvature)
- [x] Measurement engine (6 measurements with multi-slice averaging)
- [x] Confidence scoring (4-factor composite model)
- [x] Size recommendation engine (trapezoidal scoring)
- [x] Validation framework (MAE/RMSE/Bland-Altman)
- [x] FastAPI routes (upload, measure, sizing)
- [x] Storage layer
- [x] Synthetic test mesh generator
- [x] Benchmark script
- [ ] Install full deps and validate app/ tests
- [ ] Run benchmark against synthetic ground truth

## Phase 3 — Production [PENDING]
- [ ] Real LiDAR scan validation (10-20 scans + tape measurements)
- [ ] Tune landmark height percentages on real data
- [ ] iOS app (Swift/ARKit) client
- [ ] Docker containerization
- [ ] S3 storage + JWT auth
- [ ] SMPL model fitting (future)
