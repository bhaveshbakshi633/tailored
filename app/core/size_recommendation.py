"""
Clothing size recommendation engine.

Architecture
────────────
A **SizeChart** maps (garment_type, brand) → list of SizeEntry.
Each SizeEntry specifies the valid measurement ranges for that size.

The scoring function computes a per-measurement fit score using a
trapezoidal membership function:

                      1.0  ┌────────────┐
                           │            │
                      0.0 ─┘            └─
                        min-tol  min  max  max+tol

  • Inside [min, max]: score = 1.0  (perfect fit)
  • Within tolerance outside: linear decay to 0.0
  • Beyond tolerance: 0.0

Fit preference offsets the effective body measurement:
  • Tight:   subtract 2 cm  (user wants the garment closer to skin)
  • Regular: no offset
  • Loose:   add 3 cm       (user wants more room)

Fabric stretch factor widens the effective size range:
  • stretch = 1.0: no adjustment
  • stretch = 1.05: ranges expand by 5%  (e.g., jersey knit)
  • stretch = 1.10: ranges expand by 10% (e.g., spandex blend)

The overall score for a size = weighted sum of per-measurement scores,
where weights reflect the importance of each measurement for the garment:
  • Shirts:  chest 0.4, waist 0.2, shoulder 0.25, neck 0.15
  • Pants:   waist 0.4, hip 0.3, inseam 0.3
  • Jackets: chest 0.35, waist 0.2, shoulder 0.30, neck 0.15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config import config
from app.models.schemas import (
    BodyMeasurements,
    FitPreference,
    GarmentType,
    SizeRecommendation,
    SizeScore,
)

logger = logging.getLogger(__name__)

scfg = config.sizing


# ── Size chart data structures ─────────────────────────────────────────

@dataclass
class MeasurementRange:
    """Valid range for one measurement in one size, in cm."""
    min_cm: float
    max_cm: float


@dataclass
class SizeEntry:
    """One size with its measurement ranges."""
    label: str  # e.g. "M", "L", "32"
    ranges: dict[str, MeasurementRange]  # measurement_name → range


@dataclass
class SizeChart:
    """Complete size chart for one garment type and brand."""
    garment_type: GarmentType
    brand: str
    sizes: list[SizeEntry]
    measurement_weights: dict[str, float]  # measurement_name → weight
    tolerance_cm: float = 4.0  # linear decay zone outside the range


# ── Built-in generic size charts ───────────────────────────────────────

def _generic_shirt_chart() -> SizeChart:
    return SizeChart(
        garment_type=GarmentType.shirt,
        brand="generic",
        tolerance_cm=4.0,
        measurement_weights={
            "chest_circumference": 0.40,
            "waist_circumference": 0.20,
            "shoulder_width": 0.25,
            "neck_circumference": 0.15,
        },
        sizes=[
            SizeEntry("XS", {
                "chest_circumference": MeasurementRange(82, 88),
                "waist_circumference": MeasurementRange(68, 74),
                "shoulder_width": MeasurementRange(40, 42),
                "neck_circumference": MeasurementRange(35, 37),
            }),
            SizeEntry("S", {
                "chest_circumference": MeasurementRange(88, 94),
                "waist_circumference": MeasurementRange(74, 80),
                "shoulder_width": MeasurementRange(42, 44),
                "neck_circumference": MeasurementRange(37, 38),
            }),
            SizeEntry("M", {
                "chest_circumference": MeasurementRange(94, 100),
                "waist_circumference": MeasurementRange(80, 86),
                "shoulder_width": MeasurementRange(44, 46),
                "neck_circumference": MeasurementRange(38, 40),
            }),
            SizeEntry("L", {
                "chest_circumference": MeasurementRange(100, 106),
                "waist_circumference": MeasurementRange(86, 92),
                "shoulder_width": MeasurementRange(46, 48),
                "neck_circumference": MeasurementRange(40, 42),
            }),
            SizeEntry("XL", {
                "chest_circumference": MeasurementRange(106, 114),
                "waist_circumference": MeasurementRange(92, 100),
                "shoulder_width": MeasurementRange(48, 50),
                "neck_circumference": MeasurementRange(42, 44),
            }),
            SizeEntry("XXL", {
                "chest_circumference": MeasurementRange(114, 122),
                "waist_circumference": MeasurementRange(100, 108),
                "shoulder_width": MeasurementRange(50, 53),
                "neck_circumference": MeasurementRange(44, 46),
            }),
        ],
    )


def _generic_pants_chart() -> SizeChart:
    return SizeChart(
        garment_type=GarmentType.pants,
        brand="generic",
        tolerance_cm=4.0,
        measurement_weights={
            "waist_circumference": 0.40,
            "hip_circumference": 0.30,
            "inseam": 0.30,
        },
        sizes=[
            SizeEntry("28", {
                "waist_circumference": MeasurementRange(70, 74),
                "hip_circumference": MeasurementRange(86, 90),
                "inseam": MeasurementRange(76, 82),
            }),
            SizeEntry("30", {
                "waist_circumference": MeasurementRange(74, 78),
                "hip_circumference": MeasurementRange(90, 94),
                "inseam": MeasurementRange(76, 82),
            }),
            SizeEntry("32", {
                "waist_circumference": MeasurementRange(78, 82),
                "hip_circumference": MeasurementRange(94, 98),
                "inseam": MeasurementRange(78, 84),
            }),
            SizeEntry("34", {
                "waist_circumference": MeasurementRange(82, 88),
                "hip_circumference": MeasurementRange(98, 102),
                "inseam": MeasurementRange(78, 84),
            }),
            SizeEntry("36", {
                "waist_circumference": MeasurementRange(88, 94),
                "hip_circumference": MeasurementRange(102, 108),
                "inseam": MeasurementRange(80, 86),
            }),
            SizeEntry("38", {
                "waist_circumference": MeasurementRange(94, 100),
                "hip_circumference": MeasurementRange(108, 114),
                "inseam": MeasurementRange(80, 86),
            }),
        ],
    )


def _generic_jacket_chart() -> SizeChart:
    return SizeChart(
        garment_type=GarmentType.jacket,
        brand="generic",
        tolerance_cm=5.0,
        measurement_weights={
            "chest_circumference": 0.35,
            "waist_circumference": 0.20,
            "shoulder_width": 0.30,
            "neck_circumference": 0.15,
        },
        sizes=[
            SizeEntry("S", {
                "chest_circumference": MeasurementRange(88, 94),
                "waist_circumference": MeasurementRange(74, 80),
                "shoulder_width": MeasurementRange(42, 44),
                "neck_circumference": MeasurementRange(37, 38),
            }),
            SizeEntry("M", {
                "chest_circumference": MeasurementRange(94, 100),
                "waist_circumference": MeasurementRange(80, 86),
                "shoulder_width": MeasurementRange(44, 46),
                "neck_circumference": MeasurementRange(38, 40),
            }),
            SizeEntry("L", {
                "chest_circumference": MeasurementRange(100, 108),
                "waist_circumference": MeasurementRange(86, 94),
                "shoulder_width": MeasurementRange(46, 48),
                "neck_circumference": MeasurementRange(40, 42),
            }),
            SizeEntry("XL", {
                "chest_circumference": MeasurementRange(108, 116),
                "waist_circumference": MeasurementRange(94, 102),
                "shoulder_width": MeasurementRange(48, 50),
                "neck_circumference": MeasurementRange(42, 44),
            }),
        ],
    )


# Chart registry
_CHARTS: dict[tuple[GarmentType, str], SizeChart] = {}


def get_size_chart(garment_type: GarmentType, brand: str = "generic") -> SizeChart:
    """Retrieve a size chart; falls back to generic if brand not found."""
    key = (garment_type, brand)
    if key in _CHARTS:
        return _CHARTS[key]

    # Fallback to generic
    generic_key = (garment_type, "generic")
    if generic_key not in _CHARTS:
        # Initialize generic charts on first access
        _CHARTS[(GarmentType.shirt, "generic")] = _generic_shirt_chart()
        _CHARTS[(GarmentType.pants, "generic")] = _generic_pants_chart()
        _CHARTS[(GarmentType.jacket, "generic")] = _generic_jacket_chart()
        # Dress uses shirt chart as fallback
        _CHARTS[(GarmentType.dress, "generic")] = _generic_shirt_chart()

    return _CHARTS.get(generic_key, _generic_shirt_chart())


def register_size_chart(chart: SizeChart) -> None:
    """Register a custom brand size chart."""
    _CHARTS[(chart.garment_type, chart.brand)] = chart


# ── Scoring ────────────────────────────────────────────────────────────

def _trapezoidal_score(
    value: float,
    range_min: float,
    range_max: float,
    tolerance: float,
) -> float:
    """
    Trapezoidal membership function.

        1.0      ┌──────────┐
                 /            \\
        0.0  ───┘              └───
           min-tol  min    max  max+tol
    """
    if range_min <= value <= range_max:
        return 1.0
    elif value < range_min:
        dist = range_min - value
        return float(np.clip(1.0 - dist / tolerance, 0.0, 1.0))
    else:
        dist = value - range_max
        return float(np.clip(1.0 - dist / tolerance, 0.0, 1.0))


def _apply_fit_offset(value_cm: float, fit: FitPreference) -> float:
    """Adjust body measurement for fit preference."""
    if fit == FitPreference.tight:
        return value_cm + scfg.fit_offset_tight_cm  # negative → smaller effective body
    elif fit == FitPreference.loose:
        return value_cm + scfg.fit_offset_loose_cm  # positive → larger effective body
    return value_cm


def _apply_stretch(
    range_min: float, range_max: float, stretch: float,
) -> tuple[float, float]:
    """Widen the size range to account for fabric elasticity."""
    center = (range_min + range_max) / 2
    half = (range_max - range_min) / 2
    return center - half * stretch, center + half * stretch


def _get_measurement_value(measurements: BodyMeasurements, name: str) -> float | None:
    """Extract a named measurement value in cm from BodyMeasurements."""
    mapping = {
        "chest_circumference": measurements.chest_circumference.value_cm,
        "waist_circumference": measurements.waist_circumference.value_cm,
        "hip_circumference": measurements.hip_circumference.value_cm,
        "shoulder_width": measurements.shoulder_width.value_cm,
        "inseam": measurements.inseam.value_cm,
        "neck_circumference": measurements.neck_circumference.value_cm,
    }
    return mapping.get(name)


# ── Main recommendation function ──────────────────────────────────────

def recommend_size(
    measurements: BodyMeasurements,
    garment_type: GarmentType,
    brand: str = "generic",
    fit_preference: FitPreference = FitPreference.regular,
    stretch_factor: float = 1.0,
) -> SizeRecommendation:
    """
    Recommend the best clothing size.

    Algorithm:
      1. Look up the size chart for (garment_type, brand).
      2. For each size, compute a weighted score across all relevant measurements.
      3. The size with the highest score wins.
    """
    chart = get_size_chart(garment_type, brand)
    stretch = stretch_factor if stretch_factor > 0 else scfg.default_stretch_factor

    all_scores: list[SizeScore] = []
    warnings: list[str] = []

    for entry in chart.sizes:
        per_meas: dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for meas_name, weight in chart.measurement_weights.items():
            body_val = _get_measurement_value(measurements, meas_name)
            if body_val is None or body_val <= 0:
                continue

            adjusted_val = _apply_fit_offset(body_val, fit_preference)

            if meas_name in entry.ranges:
                rng = entry.ranges[meas_name]
                r_min, r_max = _apply_stretch(rng.min_cm, rng.max_cm, stretch)
                score = _trapezoidal_score(adjusted_val, r_min, r_max, chart.tolerance_cm)
            else:
                score = 0.5  # no data for this measurement → neutral

            per_meas[meas_name] = round(score, 4)
            weighted_sum += weight * score
            weight_total += weight

        overall = weighted_sum / weight_total if weight_total > 0 else 0.0

        all_scores.append(SizeScore(
            size_label=entry.label,
            score=round(overall, 4),
            per_measurement_scores=per_meas,
        ))

    # Sort by score descending
    all_scores.sort(key=lambda s: s.score, reverse=True)

    if not all_scores:
        warnings.append("No size data available for this garment/brand combination.")
        best = "N/A"
    else:
        best = all_scores[0].size_label
        # Warn if the best score is low
        if all_scores[0].score < 0.5:
            warnings.append(
                f"Best match ({best}) scored only {all_scores[0].score:.2f} — "
                "body measurements may be outside this brand's range."
            )
        # Warn if two sizes are very close
        if len(all_scores) >= 2 and abs(all_scores[0].score - all_scores[1].score) < 0.05:
            warnings.append(
                f"Sizes {all_scores[0].size_label} and {all_scores[1].size_label} "
                "scored very similarly — consider trying both."
            )

    return SizeRecommendation(
        garment_type=garment_type,
        brand=brand,
        recommended_size=best,
        fit_preference=fit_preference,
        stretch_factor=stretch,
        all_scores=all_scores,
        warnings=warnings,
    )
