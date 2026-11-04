"""Signal helpers: smoothing, foot-contact/sliding detection, pop detection (numpy only)."""
from __future__ import annotations

import math

import numpy as np


def kernel_radius(sigma: float) -> int:
    return max(1, int(math.ceil(3.0 * sigma)))


def gaussian_kernel(sigma: float) -> np.ndarray:
    r = kernel_radius(sigma)
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_valid(samples: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth ``samples`` (1-D) and return only the part that has full kernel support,
    i.e. ``len(samples) - 2 * radius`` values. Pad the input by ``kernel_radius`` each side."""
    return np.convolve(samples, gaussian_kernel(sigma), mode="valid")


def contact_runs(pos: np.ndarray, up, height_thresh: float, min_len: int = 3):
    """Runs of consecutive frames where the foot is near its lowest height.
    Returns list of (start, end) inclusive frame indices."""
    h = np.asarray(pos, float) @ np.asarray(up, float)
    near = (h - h.min()) < height_thresh
    runs, start = [], None
    for i, v in enumerate(near):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(near) - start >= min_len:
        runs.append((start, len(near) - 1))
    return runs


def sliding_segments(pos: np.ndarray, up, height_thresh: float, drift_tol: float, min_len: int = 3):
    """Contact runs whose foot drifts horizontally more than ``drift_tol``.
    Returns list of dicts: start, end, drift."""
    pos = np.asarray(pos, float)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    out = []
    for a, b in contact_runs(pos, up, height_thresh, min_len):
        seg = pos[a:b + 1]
        horiz = seg - np.outer(seg @ up, up)
        drift = float(np.linalg.norm(horiz - np.median(horiz, axis=0), axis=1).max())
        if drift > drift_tol:
            out.append({"start": a, "end": b, "drift": drift})
    return out


def rotation_outliers(Q: np.ndarray, thresh_rad: float = 0.17, factor: float = 6.0):
    """Frames where a rotation pops: it differs from the midpoint of its neighbours by
    much more than is typical for the curve. ``Q`` is (n, 4) unit quaternions."""
    Q = np.array(Q, float)
    n = len(Q)
    if n < 3:
        return []
    for i in range(1, n):  # sign continuity
        if Q[i] @ Q[i - 1] < 0:
            Q[i] = -Q[i]
    mid = Q[:-2] + Q[2:]
    norm = np.linalg.norm(mid, axis=1, keepdims=True)
    mid = np.where(norm > 1e-9, mid / np.maximum(norm, 1e-9), Q[1:-1])
    dots = np.clip(np.abs(np.sum(mid * Q[1:-1], axis=1)), 0.0, 1.0)
    dev = 2.0 * np.arccos(dots)
    limit = max(thresh_rad, factor * float(np.median(dev)))
    flagged = np.nonzero(dev > limit)[0]
    # a single pop also disturbs its two neighbours' midpoints: report only the peak
    peaks = [i for i in flagged
             if dev[i] >= (dev[i - 1] if i > 0 else 0.0) and dev[i] >= (dev[i + 1] if i + 1 < len(dev) else 0.0)]
    return [int(i) + 1 for i in peaks]


def falloff(d, radius: float):
    """Smooth bump: 1 at distance 0, exactly 0 at ``radius`` and beyond. Works on scalars/arrays."""
    d = np.asarray(d, float)
    if radius <= 0:
        return (d == 0).astype(float) if d.ndim else float(d == 0)
    sigma = radius / 2.0
    edge = math.exp(-0.5 * (radius / sigma) ** 2)
    w = np.where(np.abs(d) < radius, (np.exp(-0.5 * (d / sigma) ** 2) - edge) / (1 - edge), 0.0)
    return w if d.ndim else float(w)


def max_retime_shift(radius: float) -> float:
    """Largest shift for which :func:`warp_time` stays strictly increasing (keys never cross)."""
    return 0.7 * radius


def warp_time(t, center: float, shift: float, radius: float):
    """Move time ``center`` by ``shift`` frames and its neighbours by a falling-off share of it.
    Monotonic as long as ``abs(shift) <= max_retime_shift(radius)``."""
    t = np.asarray(t, float)
    return t + shift * falloff(t - center, radius)
