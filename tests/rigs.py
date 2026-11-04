"""Synthetic humanoid (Z up, facing -Y) used by the tests."""
import numpy as np

from openrig.core.mapping import map_rig
from openrig.core.skeleton import build_skeleton

V = lambda x, y, z: np.array([x, y, z], float)


def tpose_bones():
    """bone -> (head, tail, parent), Mixamo-style names."""
    b = {}
    def add(name, head, tail, parent):
        b[name] = (V(*head), V(*tail), parent)
    add("Hips", (0, 0, 1.0), (0, 0, 1.1), None)
    add("Spine", (0, 0, 1.1), (0, 0, 1.25), "Hips")
    add("Spine1", (0, 0, 1.25), (0, 0, 1.4), "Spine")
    add("Spine2", (0, 0, 1.4), (0, 0, 1.5), "Spine1")
    add("Neck", (0, 0, 1.5), (0, 0, 1.6), "Spine2")
    add("Head", (0, 0, 1.6), (0, 0, 1.8), "Neck")
    for s, x in (("Left", 1), ("Right", -1)):
        add(f"{s}Shoulder", (0.0, 0, 1.5), (0.12 * x, 0, 1.5), "Spine2")
        add(f"{s}Arm", (0.12 * x, 0, 1.5), (0.40 * x, 0, 1.5), f"{s}Shoulder")
        add(f"{s}ForeArm", (0.40 * x, 0, 1.5), (0.66 * x, 0, 1.5), f"{s}Arm")
        add(f"{s}Hand", (0.66 * x, 0, 1.5), (0.80 * x, 0, 1.5), f"{s}ForeArm")
        add(f"{s}UpLeg", (0.09 * x, 0, 1.0), (0.09 * x, 0.0, 0.55), "Hips")
        add(f"{s}Leg", (0.09 * x, 0.0, 0.55), (0.09 * x, 0, 0.10), f"{s}UpLeg")
        add(f"{s}Foot", (0.09 * x, 0, 0.10), (0.09 * x, -0.12, 0.0), f"{s}Leg")
        add(f"{s}ToeBase", (0.09 * x, -0.12, 0.0), (0.09 * x, -0.22, 0.0), f"{s}Foot")
    return b


def with_prefix(bones, prefix):
    return {prefix + k: (h, t, (prefix + p) if p else None) for k, (h, t, p) in bones.items()}


def particles(sk, bones):
    P = np.zeros((len(sk.names), 3))
    for i, (bone, which) in enumerate(sk.sources):
        h, t, _ = bones[bone]
        P[i] = h if which == "head" else t
    return P


def make(prefix="mixamorig:"):
    bones = with_prefix(tpose_bones(), prefix)
    m = map_rig({k: v[2] for k, v in bones.items()})
    sk = build_skeleton(m)
    return bones, m, sk, particles(sk, bones)
