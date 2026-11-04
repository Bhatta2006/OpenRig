"""Blender glue: read a pose into numpy, write solved rotations back as plain FK."""
from __future__ import annotations

import json

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .core.mapping import RigMap, map_rig
from .core.rotations import solve_rotations
from .core.skeleton import Skeleton, build_skeleton
from .core.solver import Solver, SolveSettings

MAP_KEY = "openrig_map"


# ---- mapping ---------------------------------------------------------------
def detect(obj) -> RigMap:
    parents = {b.name: (b.parent.name if b.parent else None) for b in obj.data.bones}
    m = map_rig(parents)
    obj[MAP_KEY] = json.dumps(m.to_dict())
    return m


def save_map(obj, m: RigMap) -> None:
    obj[MAP_KEY] = json.dumps(m.to_dict())


def stored_map(obj, validate: bool = True) -> RigMap | None:
    raw = obj.get(MAP_KEY)
    if not raw:
        return None
    try:
        m = RigMap.from_dict(json.loads(raw))
    except Exception:  # corrupt or hand-edited custom property
        return None
    if not validate:
        return m
    bones = obj.data.bones
    used = [m.hips, m.head, *m.spine, *m.neck] + [n for s in m.limbs.values() for n in s.values()]
    if any(n and n not in bones for n in used):
        return None  # bones were renamed/deleted since detection
    return m


def constrained_bones(obj, sk: Skeleton) -> list[str]:
    return [e.bone for e in sk.edges if obj.pose.bones[e.bone].constraints]


# ---- rig session -----------------------------------------------------------
class RigSession:
    """Everything needed to read/write the pose of one armature for one tool run."""

    def __init__(self, obj, rigmap: RigMap | None = None):
        self.obj = obj
        self.map = rigmap or stored_map(obj)
        if self.map is None or not self.map.ok:
            raise ValueError("Rig not detected or incomplete. Run Detect Rig first.")
        self.sk = build_skeleton(self.map)
        self._bone_to_key = {e.bone: e.key for e in self.sk.edges}
        bones = obj.data.bones
        self.order = sorted((b.name for b in bones), key=lambda n: self._depth(bones, n))
        self.rest = {b.name: b.matrix_local.copy() for b in bones}
        self.rest_rel = {}
        for b in bones:
            if b.parent:
                self.rest_rel[b.name] = b.parent.matrix_local.inverted_safe() @ b.matrix_local
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in bones}

    @staticmethod
    def _depth(bones, name):
        d, b = 0, bones[name]
        while b.parent:
            b, d = b.parent, d + 1
        return d

    # reading
    def read(self):
        """Armature-space joint positions (n,3) and bone rotations {edge key: 3x3}."""
        pb = self.obj.pose.bones
        P = np.zeros((len(self.sk.names), 3))
        for i, (bone, which) in enumerate(self.sk.sources):
            P[i] = pb[bone].head if which == "head" else pb[bone].tail
        R = {e.key: np.array(pb[e.bone].matrix.to_3x3().normalized()) for e in self.sk.edges}
        return P, R

    def up_local(self) -> np.ndarray:
        inv = self.obj.matrix_world.to_3x3().inverted_safe()
        v = np.array(inv @ Vector((0, 0, 1)))
        return v / (np.linalg.norm(v) + 1e-12)

    def particle_world(self, P, idx):
        return self.obj.matrix_world @ Vector(P[idx])

    def snapshot_rot(self):
        out = {}
        for e in self.sk.edges:
            out[e.bone] = rotation_tuple(self.obj.pose.bones[e.bone])
        h = self.obj.pose.bones[self.map.hips]
        out["__hips_loc"] = tuple(h.location)
        return out

    # writing
    def write(self, R_new, hips_head=None):
        """Apply new armature-space rotations to every mapped bone (a bone whose world
        rotation is unchanged still has to counter-rotate when its parent moved). Float
        noise this writes is far below the tolerance ``changed_bones`` uses for keying."""
        pbs = self.obj.pose.bones
        goal = {e.bone: Matrix(R_new[e.key].tolist()) for e in self.sk.edges}
        hips = self.map.hips
        M: dict[str, Matrix] = {}
        for name in self.order:
            pb = pbs[name]
            par = self.parent[name]
            base = (M[par] @ self.rest_rel[name]) if par else self.rest[name]
            basis = pb.matrix_basis.copy()
            moved_hips = hips_head is not None and name == hips
            if name in goal or moved_hips:
                base_rot = base.to_3x3().normalized()
                loc, rot, scl = basis.decompose()
                if name in goal:
                    rot = (base_rot.inverted_safe() @ goal[name]).to_quaternion()
                    set_rotation(pb, rot)
                if moved_hips:
                    loc = base_rot.inverted_safe() @ (Vector(hips_head) - base.translation)
                    pb.location = loc
                basis = Matrix.LocRotScale(loc, rot, scl)
            M[name] = base @ basis

    def key(self, bones: list[str], frame: int, hips_moved: bool):
        for name in bones:
            pb = self.obj.pose.bones[name]
            pb.keyframe_insert(rotation_path(pb), frame=frame, group=name)
        if hips_moved:
            h = self.obj.pose.bones[self.map.hips]
            h.keyframe_insert("location", frame=frame, group=h.name)

    def changed_bones(self, before: dict, tol=1e-5):
        out = []
        for e in self.sk.edges:
            now = rotation_tuple(self.obj.pose.bones[e.bone])
            if max(abs(a - b) for a, b in zip(now, before[e.bone])) > tol:
                out.append(e.bone)
        h = self.obj.pose.bones[self.map.hips]
        hips_moved = max(abs(a - b) for a, b in zip(h.location, before["__hips_loc"])) > 1e-6
        return out, hips_moved


# ---- keeping the edges of an edit pinned to the original animation ------------
def capture_anchors(context, sess: RigSession, frames) -> dict:
    """Pose of the mapped bones on ``frames`` before an edit changes the curves around them."""
    scene, start = context.scene, context.scene.frame_current
    out = {}
    for f in frames:
        scene.frame_set(f)
        out[f] = sess.snapshot_rot()
    scene.frame_set(start)
    return out


def restore_anchors(context, sess: RigSession, anchors: dict, bones, hips_moved: bool) -> None:
    """Put the captured pose back on the anchor frames and key the edited bones there, so an
    edit never bleeds into the animation outside its frame range."""
    scene, start = context.scene, context.scene.frame_current
    pbs = sess.obj.pose.bones
    for f, snap in anchors.items():
        scene.frame_set(f)
        for b in bones:
            setattr(pbs[b], rotation_path(pbs[b]), snap[b])
        if hips_moved:
            pbs[sess.map.hips].location = snap["__hips_loc"]
        sess.key(list(bones), f, hips_moved)
    scene.frame_set(start)


# ---- rotation property helpers --------------------------------------------
def rotation_path(pb) -> str:
    return {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(
        pb.rotation_mode, "rotation_euler")


def rotation_tuple(pb) -> tuple:
    p = rotation_path(pb)
    return tuple(getattr(pb, p))


def set_rotation(pb, q: Quaternion):
    mode = pb.rotation_mode
    if mode == "QUATERNION":
        if q.dot(pb.rotation_quaternion) < 0:
            q.negate()
        pb.rotation_quaternion = q
    elif mode == "AXIS_ANGLE":
        axis, angle = q.to_axis_angle()
        pb.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
    else:
        pb.rotation_euler = q.to_euler(mode, pb.rotation_euler)


def pose_bone_rotation_quat(pb) -> Quaternion:
    mode = pb.rotation_mode
    if mode == "QUATERNION":
        return pb.rotation_quaternion.copy()
    if mode == "AXIS_ANGLE":
        a = pb.rotation_axis_angle
        return Quaternion(Vector(a[1:]), a[0])
    return pb.rotation_euler.to_quaternion()


# ---- solver wiring ---------------------------------------------------------
def settings_from_props(props, sess: RigSession, P0, live: bool) -> SolveSettings:
    up = sess.up_local()
    foot_h = min(P0[sess.sk.index[k]] @ up for k in ("foot.L", "foot.R", "toe.L", "toe.R", "toe_end.L", "toe_end.R")
                 if k in sess.sk.index)
    s = SolveSettings(stiffness=props.stiffness, floor=props.floor, floor_height=foot_h - 1e-4,
                      up=tuple(up))
    if live:
        s.iterations, s.polish = 14, 30
    else:
        s.iterations, s.polish = 30, 80
    return s


def foot_group(side: str) -> list[str]:
    return [f"foot.{side}", f"toe.{side}", f"toe_end.{side}"]


def handle_targets(sk: Skeleton, P0: np.ndarray, label: str, target: np.ndarray) -> dict:
    """Effector targets for a handle: the grabbed joint plus the rigid parts that ride along."""
    ix = sk.index
    particle = sk.handles[label]
    delta = np.asarray(target) - P0[ix[particle]]
    out = {particle: np.asarray(target, float)}
    riders = []
    if label.startswith("Hand"):
        riders = [f"hand_end.{label[-1]}"]
    elif label.startswith("Foot"):
        riders = [k for k in foot_group(label[-1]) if k != particle]
    for r in riders:
        if r in ix:
            out[r] = P0[ix[r]] + delta
    return out
