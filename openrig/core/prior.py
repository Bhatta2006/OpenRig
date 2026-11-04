"""Learned pose prior (numpy inference, no bpy).

A small denoising autoencoder trained on motion capture. It looks at the *directions* of the
main bones in a torso frame (hip line + hips-to-chest axis) (so it does not depend on the character's proportions or size)
and pulls an implausible arrangement toward a pose a human could actually take. The solver
uses it as a gentle attractor; hard constraints (pins, targets, bone lengths) still win.

Training code lives in ``tools/``; the weights ship in ``openrig/data/pose_prior.npz``.
"""
from __future__ import annotations

import os

import numpy as np

from .skeleton import Skeleton

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pose_prior.npz")


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class Canon:
    """Which particles of a Skeleton make up the canonical direction set."""

    def __init__(self, sk: Skeleton):
        ix = sk.index
        self.ix = ix
        last_spine = [e for e in sk.edges if e.key.startswith("spine")][-1]
        self.chest = last_spine.b
        self.hips, self.spine0 = ix["hips"], ix["spine0"]
        self.head, self.head_top = ix["head"], ix["head_top"]
        dirs = [(self.chest, self.head), (self.head, self.head_top)]
        self.sides = {}
        for s in "LR":
            toe = ix.get(f"toe.{s}", ix[f"toe_end.{s}"])
            t = dict(thigh=ix[f"thigh.{s}"], shin=ix[f"shin.{s}"], foot=ix[f"foot.{s}"], toe=toe,
                     uarm=ix[f"upperarm.{s}"], farm=ix[f"forearm.{s}"], hand=ix[f"hand.{s}"],
                     hend=ix[f"hand_end.{s}"], clav=ix.get(f"clavicle.{s}"))
            self.sides[s] = t
            dirs += [(t["thigh"], t["shin"]), (t["shin"], t["foot"]), (t["foot"], t["toe"]),
                     (t["uarm"], t["farm"]), (t["farm"], t["hand"]), (t["hand"], t["hend"])]
        dirs.append((self.sides["L"]["uarm"], self.sides["R"]["uarm"]))   # shoulder line, L -> R
        self.dirs = dirs
        self.n_dirs = len(dirs)
        # particles strictly between spine0 and chest / chest and head: interpolated
        self.between_neck = [i for i, n in enumerate(sk.names) if n.startswith("neck") and i not in (self.chest, self.head)]
        self.pull = ([self.head, self.head_top] + self.between_neck
                     + [v for t in self.sides.values() for k, v in t.items()
                                           if k in ("shin", "foot", "toe", "farm", "hand", "hend")])


def forward_sign(canon: Canon, P: np.ndarray) -> float:
    """+1/-1 so that the frame's z axis points where the toes point."""
    B = pelvis_frame(canon, P, 1.0)
    z = B[:, 2]
    toes = sum(P[t["toe"]] - P[t["foot"]] for t in canon.sides.values())
    return 1.0 if float(toes @ z) >= 0 else -1.0


def pelvis_frame(canon: Canon, P: np.ndarray, sign: float) -> np.ndarray:
    """Torso frame. Columns: lateral (left->right), up (mid-hip to chest), forward. It uses only
    the hip line and the chest because skeletons disagree about where the pelvis root sits."""
    lt, rt = P[canon.sides["L"]["thigh"]], P[canon.sides["R"]["thigh"]]
    x = _unit(rt - lt)
    y = P[canon.chest] - (lt + rt) / 2
    y = _unit(y - x * (y @ x))
    return np.stack([x, y, np.cross(x, y) * sign], axis=1)


def features(canon: Canon, P: np.ndarray, sign: float) -> np.ndarray:
    """(n_dirs * 3,) unit directions in the pelvis frame."""
    B = pelvis_frame(canon, P, sign)
    d = np.stack([P[b] - P[a] for a, b in canon.dirs])
    return (_unit(d) @ B).ravel()


class PosePrior:
    def __init__(self, params: dict):
        self.W = [params[f"W{i}"] for i in range(int(params["n_layers"]))]
        self.b = [params[f"b{i}"] for i in range(int(params["n_layers"]))]
        self.mean = params["mean"]

    @classmethod
    def load(cls, path: str = WEIGHTS) -> "PosePrior | None":
        if not os.path.exists(path):
            return None
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def project(self, x: np.ndarray) -> np.ndarray:
        """Denoise a feature vector (or a batch) and renormalise every direction."""
        h = x - self.mean
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        out = h + self.mean
        shp = out.shape
        return _unit(out.reshape(shp[:-1] + (-1, 3))).reshape(shp)


class PriorGoal:
    """Turns a projected feature vector back into positions for one character."""

    def __init__(self, sk: Skeleton, P0: np.ndarray, prior: PosePrior):
        self.canon = c = Canon(sk)
        self.prior = prior
        self.sign = forward_sign(c, P0)
        self.len = np.array([np.linalg.norm(P0[b] - P0[a]) for a, b in c.dirs])
        f0 = features(c, P0, self.sign).reshape(-1, 3)
        self.rest_dirs = f0
        B0 = pelvis_frame(c, P0, self.sign)
        # interpolated particles: fraction along the chord + deviation in the pelvis frame
        self.interp = []
        for group, (a, b) in ((c.between_neck, (c.chest, c.head)),):
            chord = P0[b] - P0[a]
            denom = float(chord @ chord) or 1.0
            for q in group:
                t = float((P0[q] - P0[a]) @ chord) / denom
                self.interp.append((q, a, b, t, B0.T @ (P0[q] - (P0[a] + t * chord))))
        self.pull_idx = np.array(c.pull)

    def goals(self, P: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Goal positions (same shape as P; untouched particles keep their position)."""
        c = self.canon
        f = features(c, P, self.sign)
        proj = self.prior.project(f).reshape(-1, 3)
        if strength < 1.0:  # blend in direction space, then renormalise
            proj = _unit(f.reshape(-1, 3) * (1 - strength) + proj * strength)
        B = pelvis_frame(c, P, self.sign)
        d = proj @ B.T          # world directions
        L = self.len
        G = P.copy()
        G[c.head] = P[c.chest] + L[0] * d[0]
        G[c.head_top] = G[c.head] + L[1] * d[1]
        i = 2
        for s in "LR":
            t = c.sides[s]
            G[t["shin"]] = P[t["thigh"]] + L[i] * d[i]
            G[t["foot"]] = G[t["shin"]] + L[i + 1] * d[i + 1]
            G[t["toe"]] = G[t["foot"]] + L[i + 2] * d[i + 2]
            G[t["farm"]] = P[t["uarm"]] + L[i + 3] * d[i + 3]   # arms hang off the solver's shoulders
            G[t["hand"]] = G[t["farm"]] + L[i + 4] * d[i + 4]
            G[t["hend"]] = G[t["hand"]] + L[i + 5] * d[i + 5]
            i += 6
        for q, a, b, t, dev in self.interp:
            G[q] = G[a] + t * (G[b] - G[a]) + B @ dev
        return G


def mirror_features(F: np.ndarray) -> np.ndarray:
    """Left/right mirror of feature vectors (training augmentation). Slot layout:
    neck, head | 6 left-side dirs | 6 right-side dirs | shoulder line (L->R)."""
    d = np.asarray(F, float).reshape(F.shape[:-1] + (15, 3)).copy()
    m = np.array([-1.0, 1.0, 1.0])
    out = np.empty_like(d)
    out[..., 0:2, :] = d[..., 0:2, :] * m
    out[..., 2:8, :] = d[..., 8:14, :] * m
    out[..., 8:14, :] = d[..., 2:8, :] * m
    out[..., 14, :] = d[..., 14, :] * np.array([1.0, -1.0, -1.0])
    return out.reshape(F.shape)
