"""Minimal BVH reader with forward kinematics (numpy). Training-time only."""
from __future__ import annotations

import numpy as np


class Joint:
    def __init__(self, name, parent):
        self.name, self.parent = name, parent
        self.offset = np.zeros(3)
        self.channels: list[str] = []
        self.children: list["Joint"] = []
        self.end_site: np.ndarray | None = None
        self.col = 0


def _rot(axis: str, a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., :, :] = np.eye(3)
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    R[..., i, i] = c; R[..., i, j] = -s
    R[..., j, i] = s; R[..., j, j] = c
    return R


def load(path, stride: int = 1):
    """Returns (joints in file order, frame_time, motion array (frames, channels))."""
    with open(path) as f:
        text = f.read()
    head, _, body = text.partition("MOTION")
    lines = [t.strip() for t in head.replace("\t", " ").split("\n") if t.strip()]
    joints: list[Joint] = []
    stack: list[Joint] = []
    i = 0
    n_ch = 0
    while i < len(lines):
        w = lines[i].split()
        if w[0] in ("ROOT", "JOINT"):
            j = Joint(w[1], stack[-1] if stack else None)
            if stack:
                stack[-1].children.append(j)
            joints.append(j)
            stack.append(j)
        elif w[0] == "End":
            i += 2  # "{" then OFFSET; the end site's "}" is consumed below
            stack[-1].end_site = np.array([float(x) for x in lines[i].split()[1:4]])
            i += 1
        elif w[0] == "OFFSET":
            stack[-1].offset = np.array([float(x) for x in w[1:4]])
        elif w[0] == "CHANNELS":
            stack[-1].channels = w[2:2 + int(w[1])]
            stack[-1].col = n_ch
            n_ch += int(w[1])
        elif w[0] == "}":
            stack.pop()
        i += 1
    rows = body.strip().split("\n")
    frames = int(rows[0].split()[1])
    ft = float(rows[1].split()[2])
    data = np.fromstring("\n".join(rows[2:2 + frames]), sep=" ").reshape(frames, n_ch)
    return joints, ft, data[::stride]


def fk(joints, data):
    """World position of every joint, (frames, joints, 3), and of each joint's 'tail'
    (first child joint, else its end site), (frames, joints, 3)."""
    F = data.shape[0]
    pos = np.zeros((F, len(joints), 3))
    rot = np.zeros((F, len(joints), 3, 3))
    idx = {id(j): k for k, j in enumerate(joints)}
    for k, j in enumerate(joints):
        t = np.zeros((F, 3))
        R = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
        for ch, c in zip(j.channels, range(j.col, j.col + len(j.channels))):
            v = data[:, c]
            if ch.endswith("position"):
                t[:, "XYZ".index(ch[0])] = v
            else:
                R = R @ _rot(ch[0], np.deg2rad(v))
        if j.parent is None:
            pos[:, k] = j.offset + t
            rot[:, k] = R
        else:
            p = idx[id(j.parent)]
            pos[:, k] = pos[:, p] + np.einsum("fij,fj->fi", rot[:, p], j.offset + t)
            rot[:, k] = rot[:, p] @ R
    tail = np.zeros_like(pos)
    for k, j in enumerate(joints):
        if j.children:
            tail[:, k] = pos[:, idx[id(j.children[0])]]
        elif j.end_site is not None:
            tail[:, k] = pos[:, k] + np.einsum("fij,j->fi", rot[:, k], j.end_site)
        else:
            tail[:, k] = pos[:, k]
    return pos, tail
