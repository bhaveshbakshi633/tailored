"""
GeomCalc FastAPI application.

Endpoints:
  POST /api/v1/scan/upload     — upload a mesh file (.obj, .ply, .stl)
  POST /api/v1/measurements    — run measurement pipeline on an uploaded scan
  POST /api/v1/sizing          — get size recommendation for a scan
  GET  /health                 — health check
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.api.routes import scan, measurements, sizing
from app.models.schemas import HealthResponse

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)

app = FastAPI(
    title=config.app_name,
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route registration ─────────────────────────────────────────────────

app.include_router(scan.router, prefix="/api/v1/scan", tags=["scan"])
app.include_router(measurements.router, prefix="/api/v1/measurements", tags=["measurements"])
app.include_router(sizing.router, prefix="/api/v1/sizing", tags=["sizing"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version=config.version)


@app.on_event("startup")
async def startup():
    config.storage.upload_dir.mkdir(parents=True, exist_ok=True)
    config.storage.result_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info("GeomCalc %s started", config.version)
