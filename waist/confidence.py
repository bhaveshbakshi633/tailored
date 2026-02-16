from __future__ import annotations

import numpy as np


def slice_variance_penalty(values_cm: list[float]) -> float:
    if len(values_cm) < 2:
        return 1.0
    arr = np.array(values_cm)
    mu = float(np.mean(arr))
    if mu < 1e-9:
        return 1.0
    return float(np.std(arr) / mu)


def symmetry_penalty(contour: np.ndarray | None) -> float:
    if contour is None or len(contour) < 4:
        return 1.0
    x = contour[:, 0]
    left_ext = float(abs(x.min()))
    right_ext = float(abs(x.max()))
    denom = (left_ext + right_ext) / 2.0
    if denom < 1e-9:
        return 1.0
    return float(abs(left_ext - right_ext) / denom)


def density_penalty(n_segments: int, required: int = 40) -> float:
    if n_segments >= required:
        return 0.0
    return float(max(0.0, 1.0 - n_segments / required))


def compute_confidence(
    values_cm: list[float],
    contour: np.ndarray | None,
    n_segments: int,
) -> float:
    cv = slice_variance_penalty(values_cm)
    asym = symmetry_penalty(contour)
    sparse = density_penalty(n_segments)
    c = 1.0 - (2.0 * cv + 1.5 * asym + 1.0 * sparse)
    return round(max(0.0, c), 4)
