"""Turn solved joint positions into bone rotations (numpy, no bpy)."""
from __future__ import annotations

import numpy as np

from .skeleton import Skeleton


def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else v


def swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector ``a`` to unit vector ``b``."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: rotate pi about any perpendicular axis
        axis = unit(np.cross(a, [1.0, 0, 0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = unit(np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def fit_rotation(old: np.ndarray, new: np.ndarray, min_sv: float = 1e-4):
    """Best rotation Q with Q @ old[k] ~= new[k] (rows are vectors), or None if the
    vectors do not pin down a rotation (fewer than two independent directions)."""
    old, new = np.asarray(old, float), np.asarray(new, float)
    if len(old) < 2:
        return None
    H = old.T @ new
    U, S, Vt = np.linalg.svd(H)
    if S[1] < min_sv * max(S[0], 1e-12) or S[0] < 1e-12:
        return None
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def solve_rotations(sk: Skeleton, P_old: np.ndarray, P_new: np.ndarray,
                    R_old: dict[str, np.ndarray], eps: float = 1e-9) -> dict[str, np.ndarray]:
    """New armature-space rotation for every edge bone (parents are processed first).

    * A bone carrying several rigidly attached joints (pelvis with both hips, chest with
      both shoulders) is oriented by fitting all of them, so its twist is right too.
    * Any other bone inherits its parent's rotation change (keeps local twist), then
      takes the minimal swing that points it at its solved joint.
    * Each bone is aimed from the head position it will *actually* have after its
      parents are rotated, not from the solver's copy, so small rigid-body fit errors
      never accumulate down a chain (a planted foot stays planted).
    """
    kids: dict[str, list[int]] = {}
    for c in sk.edges:
        if c.parent is not None:
            kids.setdefault(c.parent, []).append(c.a)
    edge = {e.key: e for e in sk.edges}
    R_new: dict[str, np.ndarray] = {}
    head: dict[str, np.ndarray] = {}        # actual head position of every edge bone
    for e in sk.edges:
        if e.parent is None or e.parent not in R_new:
            A = P_new[e.a].copy()
            delta = np.eye(3)
        else:
            pe = edge[e.parent]
            offset = R_old[pe.key].T @ (P_old[e.a] - P_old[pe.a])
            A = head[pe.key] + R_new[pe.key] @ offset
            delta = R_new[pe.key] @ R_old[pe.key].T
        head[e.key] = A

        attached = [e.b] + [a for a in kids.get(e.key, []) if a != e.b]
        if len(attached) >= 2:
            Q = fit_rotation(P_old[attached] - P_old[e.a], P_new[attached] - A)
            if Q is not None:
                R_new[e.key] = Q @ R_old[e.key]
                continue
        d_old = P_old[e.b] - P_old[e.a]
        d_new = P_new[e.b] - A
        if np.linalg.norm(d_old) < eps or np.linalg.norm(d_new) < eps:
            R_new[e.key] = delta @ R_old[e.key]
            continue
        R_new[e.key] = swing(delta @ d_old, d_new) @ delta @ R_old[e.key]
    return R_new
