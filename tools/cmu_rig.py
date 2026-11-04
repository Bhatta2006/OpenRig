"""Map the CMU mocap skeleton onto OpenRig's Skeleton so training uses the exact same code
path (feature extraction) as the add-on does at run time."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from openrig.core.mapping import RigMap
from openrig.core.skeleton import build_skeleton

CMU_MAP = RigMap(hips="Hips", spine=["LowerBack", "Spine", "Spine1"], neck=["Neck", "Neck1"], head="Head")
for _side, _p in (("L", "Left"), ("R", "Right")):
    CMU_MAP.limbs[_side] = {"clavicle": f"{_p}Shoulder", "upperarm": f"{_p}Arm", "forearm": f"{_p}ForeArm",
                            "hand": f"{_p}Hand", "thigh": f"{_p}UpLeg", "shin": f"{_p}Leg",
                            "foot": f"{_p}Foot", "toe": f"{_p}ToeBase"}
SK = build_skeleton(CMU_MAP)


def particles(joints, pos, tail):
    """(frames, n_particles, 3) for the CMU skeleton from bvh.fk output."""
    name_to_k = {j.name: k for k, j in enumerate(joints)}
    out = np.zeros((pos.shape[0], len(SK.names), 3))
    for i, (bone, which) in enumerate(SK.sources):
        k = name_to_k[bone]
        out[:, i] = pos[:, k] if which == "head" else tail[:, k]
    return out
