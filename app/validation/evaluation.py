"""
Validation framework: compare predicted measurements against ground truth.

Design
──────
1. **MeasurementLog** records every prediction with metadata (scan_id,
   timestamp, model version, confidence scores).

2. Ground truth is loaded from a CSV / JSON file with columns:
     scan_id, measurement_name, ground_truth_cm

3. **Error metrics computed:**
   • Mean Absolute Error (MAE)
   • Root Mean Squared Error (RMSE)
   • Mean Bias Error (MBE) — directional, indicates systematic over/under
   • 95th-percentile error — worst-case bound
   • Per-measurement breakdown

4. **Statistical evaluation plan:**
   • Bland–Altman analysis: plot (predicted − truth) vs. mean,
     check for proportional bias.
   • Paired t-test per measurement to detect systematic offset.
   • Confidence score calibration: bin predictions by confidence,
     verify that higher confidence correlates with lower error.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.models.schemas import BodyMeasurements, SingleMeasurement

logger = logging.getLogger(__name__)


# ── Measurement logging ────────────────────────────────────────────────

@dataclass
class MeasurementLogEntry:
    scan_id: str
    timestamp: str
    measurement_name: str
    predicted_cm: float
    confidence: float
    model_version: str = "0.1.0"


class MeasurementLogger:
    """Append-only log of all predictions for offline analysis."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, scan_id: str, measurements: BodyMeasurements) -> None:
        """Append all measurements from one scan to the log."""
        now = datetime.now(timezone.utc).isoformat()
        entries = []

        for m in [
            measurements.chest_circumference,
            measurements.waist_circumference,
            measurements.hip_circumference,
            measurements.shoulder_width,
            measurements.inseam,
            measurements.neck_circumference,
        ]:
            entries.append(MeasurementLogEntry(
                scan_id=scan_id,
                timestamp=now,
                measurement_name=m.name,
                predicted_cm=m.value_cm,
                confidence=m.confidence.overall,
            ))

        # Append as JSON lines
        with self.log_path.open("a") as f:
            for entry in entries:
                f.write(json.dumps(entry.__dict__) + "\n")

    def load_all(self) -> list[MeasurementLogEntry]:
        """Load all log entries."""
        entries = []
        if not self.log_path.exists():
            return entries
        with self.log_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(MeasurementLogEntry(**json.loads(line)))
        return entries


# ── Ground truth loading ───────────────────────────────────────────────

@dataclass
class GroundTruthEntry:
    scan_id: str
    measurement_name: str
    ground_truth_cm: float


def load_ground_truth(path: Path) -> list[GroundTruthEntry]:
    """
    Load ground truth from a CSV file with columns:
      scan_id, measurement_name, ground_truth_cm
    """
    entries = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(GroundTruthEntry(
                scan_id=row["scan_id"],
                measurement_name=row["measurement_name"],
                ground_truth_cm=float(row["ground_truth_cm"]),
            ))
    return entries


# ── Error analysis ─────────────────────────────────────────────────────

@dataclass
class ErrorMetrics:
    measurement_name: str
    n_samples: int
    mae_cm: float       # Mean Absolute Error
    rmse_cm: float      # Root Mean Squared Error
    mbe_cm: float       # Mean Bias Error (positive = over-prediction)
    p95_error_cm: float  # 95th percentile absolute error
    std_cm: float       # Standard deviation of errors


@dataclass
class EvaluationReport:
    overall_mae_cm: float
    overall_rmse_cm: float
    per_measurement: list[ErrorMetrics]
    n_total_samples: int
    target_met: bool  # True if overall MAE < 2.0 cm


def evaluate(
    predictions: list[MeasurementLogEntry],
    ground_truth: list[GroundTruthEntry],
) -> EvaluationReport:
    """
    Compare predictions against ground truth and compute error metrics.
    """
    # Build lookup: (scan_id, measurement_name) → ground_truth_cm
    gt_lookup: dict[tuple[str, str], float] = {
        (g.scan_id, g.measurement_name): g.ground_truth_cm
        for g in ground_truth
    }

    # Group errors by measurement name
    errors_by_name: dict[str, list[float]] = {}

    for pred in predictions:
        key = (pred.scan_id, pred.measurement_name)
        if key not in gt_lookup:
            continue
        error = pred.predicted_cm - gt_lookup[key]

        if pred.measurement_name not in errors_by_name:
            errors_by_name[pred.measurement_name] = []
        errors_by_name[pred.measurement_name].append(error)

    # Compute per-measurement metrics
    per_measurement: list[ErrorMetrics] = []
    all_abs_errors: list[float] = []
    all_sq_errors: list[float] = []

    for name, errors in sorted(errors_by_name.items()):
        arr = np.array(errors)
        abs_arr = np.abs(arr)

        metrics = ErrorMetrics(
            measurement_name=name,
            n_samples=len(arr),
            mae_cm=float(np.mean(abs_arr)),
            rmse_cm=float(np.sqrt(np.mean(arr ** 2))),
            mbe_cm=float(np.mean(arr)),
            p95_error_cm=float(np.percentile(abs_arr, 95)),
            std_cm=float(np.std(arr)),
        )
        per_measurement.append(metrics)
        all_abs_errors.extend(abs_arr.tolist())
        all_sq_errors.extend((arr ** 2).tolist())

    n_total = len(all_abs_errors)
    overall_mae = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0
    overall_rmse = float(np.sqrt(np.mean(all_sq_errors))) if all_sq_errors else 0.0

    return EvaluationReport(
        overall_mae_cm=round(overall_mae, 3),
        overall_rmse_cm=round(overall_rmse, 3),
        per_measurement=per_measurement,
        n_total_samples=n_total,
        target_met=overall_mae < 2.0,
    )


# ── Bland–Altman data preparation ─────────────────────────────────────

@dataclass
class BlandAltmanPoint:
    mean_cm: float     # (predicted + truth) / 2
    diff_cm: float     # predicted − truth
    measurement_name: str
    scan_id: str


def bland_altman_data(
    predictions: list[MeasurementLogEntry],
    ground_truth: list[GroundTruthEntry],
) -> list[BlandAltmanPoint]:
    """
    Prepare data for Bland–Altman plots.

    Each point is ( (pred+truth)/2 ,  pred−truth ).
    A well-calibrated system shows points clustered around diff=0
    with no trend (i.e., no proportional bias).
    """
    gt_lookup = {
        (g.scan_id, g.measurement_name): g.ground_truth_cm
        for g in ground_truth
    }

    points = []
    for pred in predictions:
        key = (pred.scan_id, pred.measurement_name)
        if key not in gt_lookup:
            continue
        truth = gt_lookup[key]
        points.append(BlandAltmanPoint(
            mean_cm=(pred.predicted_cm + truth) / 2,
            diff_cm=pred.predicted_cm - truth,
            measurement_name=pred.measurement_name,
            scan_id=pred.scan_id,
        ))

    return points


# ── Confidence calibration check ──────────────────────────────────────

@dataclass
class CalibrationBin:
    confidence_range: tuple[float, float]
    n_samples: int
    mean_abs_error_cm: float


def confidence_calibration(
    predictions: list[MeasurementLogEntry],
    ground_truth: list[GroundTruthEntry],
    n_bins: int = 5,
) -> list[CalibrationBin]:
    """
    Bin predictions by confidence score and compute mean error per bin.

    A well-calibrated confidence model shows monotonically decreasing
    error as confidence increases.
    """
    gt_lookup = {
        (g.scan_id, g.measurement_name): g.ground_truth_cm
        for g in ground_truth
    }

    # Collect (confidence, abs_error) pairs
    pairs: list[tuple[float, float]] = []
    for pred in predictions:
        key = (pred.scan_id, pred.measurement_name)
        if key not in gt_lookup:
            continue
        abs_error = abs(pred.predicted_cm - gt_lookup[key])
        pairs.append((pred.confidence, abs_error))

    if not pairs:
        return []

    # Sort by confidence and bin
    pairs.sort(key=lambda x: x[0])
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_bin = [e for c, e in pairs if lo <= c < hi or (i == n_bins - 1 and c == hi)]
        bins.append(CalibrationBin(
            confidence_range=(lo, hi),
            n_samples=len(in_bin),
            mean_abs_error_cm=float(np.mean(in_bin)) if in_bin else 0.0,
        ))

    return bins
