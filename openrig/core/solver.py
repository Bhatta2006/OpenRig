"""Full-body "pull" solver (position based, numpy, no bpy).

Drag one or more joints (effectors) and the rest of the body follows:

* pinned joints (planted feet) never move,
* bones keep their length (enforced last, so it is exact),
* the pelvis and shoulder girdle stay rigid,
* unpinned limbs are pulled toward "carried along by the torso" so they swing
  with the body instead of trailing behind,
* knees and elbows keep bending the way they were bent (no hyperextension/flip).
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from .prior import PosePrior, PriorGoal
from .rotations import fit_rotation
from .skeleton import Skeleton

# fraction of the way a particle is pulled toward "rigidly carried by the torso" per iteration
_FOLLOW = {
    "hips": 0.30, "spine": 0.25, "neck": 0.30, "head": 0.30, "head_top": 0.30,
    "clavicle": 0.35, "upperarm": 0.35, "thigh": 0.35,
    "forearm": 0.06, "hand": 0.04, "hand_end": 0.04,
    "shin": 0.08, "foot": 0.05, "toe": 0.05, "toe_end": 0.05,
}
# inverse mass: lower = heavier = moves less
_INV_MASS = {"hips": 0.5, "spine": 0.7, "neck": 0.9, "head": 1.0, "head_top": 1.0,
             "clavicle": 0.8, "upperarm": 0.8, "thigh": 0.6}


def _kind(name: str) -> str:
    base = name.split(".")[0]
    return "spine" if base.startswith("spine") else "neck" if base.startswith("neck") else base


@dataclass
class SolveSettings:
    iterations: int = 14
    polish: int = 40
    stiffness: float = 1.0        # multiplier on the follow strengths
    effector_pull: float = 0.85
    polish_effector_weight: float = 1.0   # <1 keeps targets stiffer during the final length polish
    floor: bool = False
    prior: PosePrior | None = None     # learned pose prior (None = off)
    prior_strength: float = 0.6       # 0..1
    floor_height: float = 0.0     # along ``up``
    up: tuple = (0.0, 0.0, 1.0)


def _kabsch(A: np.ndarray, B: np.ndarray):
    """Rigid transform (R, t) minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    R = fit_rotation(A - ca, B - cb)
    if R is None:
        R = np.eye(3)
    return R, cb - R @ ca


class Solver:
    """Built once per drag from the pose at the start of the drag."""

    def __init__(self, sk: Skeleton, P0: np.ndarray, settings: SolveSettings | None = None):
        self.sk, self.s = sk, settings or SolveSettings()
        self.P0 = np.array(P0, float)
        ix = sk.index

        self.dist: list[tuple[int, int, float]] = []
        seen = set()

        def add(i, j):
            key = (min(i, j), max(i, j))
            if key not in seen and i != j:
                seen.add(key)
                self.dist.append((i, j, float(np.linalg.norm(self.P0[i] - self.P0[j]))))

        for e in sk.edges:
            add(e.a, e.b)
        spine_idx = [ix[k] for k in sk.names if k.startswith("spine")]
        pelvis = [ix["hips"], spine_idx[0], ix["thigh.L"], ix["thigh.R"]]
        upper = [spine_idx[-1], ix["upperarm.L"], ix["upperarm.R"]]
        upper += [ix[k] for k in ("neck0", "clavicle.L", "clavicle.R") if k in ix]
        for group in (pelvis, upper):
            for gi, i in enumerate(group):
                for j in group[gi + 1:]:
                    add(i, j)

        axial = [i for i, nm in enumerate(sk.names) if _kind(nm) in ("hips", "spine", "neck", "head", "head_top")]
        girdle = [ix[k] for k in sk.names if _kind(k) in ("clavicle", "upperarm", "thigh")]
        self.torso = axial + girdle

        self.inv_mass = np.array([_INV_MASS.get(_kind(nm), 1.3) for nm in sk.names])
        self.follow = np.array([_FOLLOW.get(_kind(nm), 0.1) for nm in sk.names])

        self._init_frame()
        self.hinges = self._build_hinges()
        self.goal = PriorGoal(sk, self.P0, self.s.prior) if self.s.prior is not None else None
        # (i, j, length, is_max_only): bone lengths/rigid groups are equalities, reach limits are maxima
        self._cons = [(i, j, L, False) for i, j, L in self.dist]
        for ia, _ib, ic, _d, _lk, l1, l2 in self.hinges:
            self._cons.append((ia, ic, l1 + l2, True))
        feet = [ix[k] for k in sk.names if _kind(k) in ("foot", "toe", "toe_end")]
        self.foot_particles = feet

    # ---- body frame -------------------------------------------------------
    def _frame(self, P):
        ix = self.sk.index
        lat = P[ix["thigh.R"]] - P[ix["thigh.L"]]
        lat /= np.linalg.norm(lat) + 1e-12
        up = P[ix["head"]] - P[ix["hips"]]
        up = up - lat * (up @ lat)
        up /= np.linalg.norm(up) + 1e-12
        fwd = np.cross(lat, up) * self._fwd_sign
        return np.stack([lat, up, fwd], axis=1)  # columns

    def _init_frame(self):
        ix = self.sk.index
        self._fwd_sign = 1.0
        B = self._frame(self.P0)
        f = np.zeros(3)
        for s in "LR":
            f += self.P0[ix[f"toe_end.{s}"]] - self.P0[ix[f"foot.{s}"]]
        f -= B[:, 0] * (f @ B[:, 0]) + B[:, 1] * (f @ B[:, 1])
        if np.linalg.norm(f) < 1e-9:
            f = -np.array([0.0, 1.0, 0.0])
            f -= B[:, 0] * (f @ B[:, 0]) + B[:, 1] * (f @ B[:, 1])
        f /= np.linalg.norm(f) + 1e-12
        self._fwd_sign = 1.0 if float(B[:, 2] @ f) >= 0 else -1.0
        self.B0 = self._frame(self.P0)

    def _build_hinges(self):
        ix = self.sk.index
        out = []
        for s in "LR":
            # (root, mid, end, default bend dir in body frame, share of sideways bend kept per pass)
            for a, b, c, default, lat_keep in (
                (f"thigh.{s}", f"shin.{s}", f"foot.{s}", np.array([0, 0, 1.0]), 0.0),
                (f"upperarm.{s}", f"forearm.{s}", f"hand.{s}", np.array([0, 0, -1.0]), 1.0),
            ):
                ia, ib, ic = ix[a], ix[b], ix[c]
                u = self.P0[ic] - self.P0[ia]
                u /= np.linalg.norm(u) + 1e-12
                perp = (self.P0[ib] - self.P0[ia])
                perp = perp - u * (perp @ u)
                length = np.linalg.norm(perp)
                d_body = self.B0.T @ (perp / length) if length > 0.02 * np.linalg.norm(self.P0[ic] - self.P0[ia]) else default
                out.append((ia, ib, ic, d_body, lat_keep,
                        float(np.linalg.norm(self.P0[ib] - self.P0[ia])),
                        float(np.linalg.norm(self.P0[ic] - self.P0[ib]))))
        return out

    def _apply_hinges(self, P, fixed=()):
        """Place knees/elbows exactly (two-bone IK) on the correct side of the limb line.
        Joints in ``fixed`` (pinned) are left alone."""
        B = self._frame(P)
        for ia, ib, ic, d_body, lat_keep, l1, l2 in self.hinges:
            if ib in fixed:
                continue
            ac = P[ic] - P[ia]
            dac = np.linalg.norm(ac)
            if dac < 1e-9:
                continue
            u = ac / dac
            v = P[ib] - P[ia]
            perp = v - u * (v @ u)
            d = B @ d_body
            d = d - u * (d @ u)
            dn = np.linalg.norm(d)
            if dn > 1e-6:
                d /= dn
                comp = perp @ d
                vec = (max(comp, 0.0) + 0.02 * (l1 + l2)) * d + (perp - comp * d) * lat_keep
                vn = np.linalg.norm(vec)
                direction = vec / vn if vn > 1e-6 else d
            else:
                pn = np.linalg.norm(perp)
                if pn < 1e-9:
                    continue
                direction = perp / pn
            reach = min(dac, l1 + l2 - 1e-9)
            x = (l1 * l1 - l2 * l2 + reach * reach) / (2 * reach)
            h = np.sqrt(max(l1 * l1 - x * x, 0.0))
            P[ib] = P[ia] + u * x + direction * h

    # ---- main solve -------------------------------------------------------
    def solve(self, targets: dict[str, np.ndarray], pins: set[str] | None = None,
              frozen: set[str] | None = None, init: np.ndarray | None = None) -> np.ndarray:
        """``targets``: particle name -> desired position. ``pins``: particles that must
        not move. ``frozen``: particles excluded from the follow pull (but free to be
        moved by constraints)."""
        sk, s = self.sk, self.s
        ix = sk.index
        P = (self.P0 if init is None else np.asarray(init, float)).copy()
        pins = pins or set()
        pin_idx = [ix[p] for p in pins if p in ix and p not in targets]
        tgt_idx = [ix[p] for p in targets]
        w_final = self.inv_mass.copy()
        w_final[pin_idx] = 0.0
        w = w_final.copy()
        w[tgt_idx] *= 0.15  # dragged joints are "heavy" while pulling; relaxed in the final polish
        w_final[tgt_idx] *= s.polish_effector_weight
        no_follow = set(pin_idx) | set(tgt_idx) | {ix[p] for p in (frozen or ()) if p in ix}
        follow = self.follow * s.stiffness
        mask = np.array([0.0 if i in no_follow else 1.0 for i in range(len(P))])
        T = np.array([targets[p] for p in targets], float).reshape(-1, 3)
        torso = self.torso
        up = np.asarray(s.up, float)
        floor_set = self.foot_particles if s.floor else []

        for it in range(s.iterations):
            if len(tgt_idx):
                P[tgt_idx] += (T - P[tgt_idx]) * s.effector_pull
            R, t = _kabsch(self.P0[torso], P[torso])
            goal = self.P0 @ R.T + t
            P += (goal - P) * (follow * mask)[:, None]
            if self.goal is not None:
                G = self.goal.goals(P)
                idx = self.goal.pull_idx
                P[idx] += (G[idx] - P[idx]) * (0.4 * s.prior_strength * mask[idx])[:, None]
            self._project(P, w, 2)
            self._apply_hinges(P, pin_idx)
            self._floor(P, floor_set, pin_idx, up)
        for _ in range(s.polish):  # final polish: exact bone lengths, then floor
            self._project(P, w_final, 2)
            self._apply_hinges(P, pin_idx)
            self._floor(P, floor_set, pin_idx, up)
        self._project(P, w_final, 2)   # last: bone lengths exact (floor may be off by a hair)
        return P

    def _floor(self, P, idx, pin_idx, up):
        for i in idx:
            if i in pin_idx:
                continue
            h = P[i] @ up
            if h < self.s.floor_height:
                P[i] += up * (self.s.floor_height - h)

    def _project(self, P, w, sweeps=1):
        """Gauss-Seidel bone-length projection in plain Python floats (much faster than
        tiny numpy ops for ~50 constraints)."""
        X = P.tolist()
        wl = w.tolist()
        sqrt = math.sqrt
        for _ in range(sweeps):
            for i, j, L, kind in self._cons:
                a, b = X[i], X[j]
                dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                ln = sqrt(dx * dx + dy * dy + dz * dz)
                if ln < 1e-12 or (kind and ln <= L):
                    continue
                ws = wl[i] + wl[j]
                if ws == 0.0:
                    continue
                f = (ln - L) / (ln * ws)
                ci, cj = wl[i] * f, wl[j] * f
                a[0] += dx * ci; a[1] += dy * ci; a[2] += dz * ci
                b[0] -= dx * cj; b[1] -= dy * cj; b[2] -= dz * cj
        P[:] = X

    def max_length_error(self, P) -> float:
        return max(abs(np.linalg.norm(P[j] - P[i]) - L) / max(L, 0.05) for i, j, L in self.dist)
