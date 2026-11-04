from __future__ import annotations

import math
import re

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

from . import rig
from .core import filters
from .core.rotations import solve_rotations
from .core.solver import Solver

PIN_RADIUS = 18  # px, handle pick distance


# ---- helpers ---------------------------------------------------------------
def active_rig(context):
    obj = context.active_object
    return obj if obj and obj.type == "ARMATURE" else None


def frame_range(context):
    p = context.scene.openrig
    if p.use_scene_range:
        return context.scene.frame_start, context.scene.frame_end
    return p.frame_start, p.frame_end


def fcurves_of(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []  # Blender 5 layered actions
    slot = getattr(ad, "action_slot", None)
    for layer in act.layers:
        for strip in layer.strips:
            cb = strip.channelbag(slot) if slot is not None else None
            if cb:
                out.extend(cb.fcurves)
    return out


class _Base:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = active_rig(context)
        return obj is not None and rig.stored_map(obj) is not None


def _session(op, context):
    try:
        return rig.RigSession(active_rig(context))
    except ValueError as e:
        op.report({"ERROR"}, str(e))
        return None


def _warn_constraints(op, sess):
    bad = rig.constrained_bones(sess.obj, sess.sk)
    if bad:
        op.report({"WARNING"}, f"{len(bad)} mapped bone(s) have constraints (e.g. {bad[0]}); "
                               "results may not match what you see. Mute IK/constraints first.")


# ---- detect ---------------------------------------------------------------
class OPENRIG_OT_detect(bpy.types.Operator):
    bl_idname = "openrig.detect"
    bl_label = "Detect Rig"
    bl_description = "Find the humanoid bones (hips, spine, arms, legs, head) of the active armature"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_rig(context) is not None

    def execute(self, context):
        obj = active_rig(context)
        m = rig.detect(obj)
        p = context.scene.openrig
        if m.ok:
            count = 2 + len(m.spine) + len(m.neck) + sum(len(v) for v in m.limbs.values())
            p.status = f"OK: {count} bones mapped"
            self.report({"INFO"}, p.status)
        else:
            p.status = "Missing: " + ", ".join(m.missing())
            self.report({"WARNING"}, p.status)
        return {"FINISHED"}


# ---- drag handles ----------------------------------------------------------
_STATE = {"running": False, "handler": None, "obj": None, "hover": None, "active": None, "handles": {}}


def _handle_screen_positions(context, obj, sess_sk, region, rv3d):
    pbs = obj.pose.bones
    out = {}
    for label, particle in sess_sk.handles.items():
        bone, which = sess_sk.sources[sess_sk.index[particle]]
        local = pbs[bone].head if which == "head" else pbs[bone].tail
        pos = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.matrix_world @ local)
        if pos is not None:
            out[label] = pos
    return out


def _draw_handles():
    import gpu
    from gpu_extras.presets import draw_circle_2d

    ctx = bpy.context
    obj = _STATE["obj"]
    if not obj or ctx.region is None or ctx.region_data is None:
        return
    try:
        sk = _STATE["sk"]
        pos = _handle_screen_positions(ctx, obj, sk, ctx.region, ctx.region_data)
    except (ReferenceError, KeyError):
        return
    p = ctx.scene.openrig
    gpu.state.blend_set("ALPHA")
    for label, xy in pos.items():
        hot = label in (_STATE["hover"], _STATE["active"])
        pinned = (label == "Foot L" and p.pin_foot_l) or (label == "Foot R" and p.pin_foot_r)
        color = (1.0, 0.75, 0.1, 1.0) if hot else ((0.35, 0.8, 1.0, 0.95) if pinned else (1, 1, 1, 0.9))
        draw_circle_2d(xy, color, 11 if hot else 8, 24)
        if pinned:
            draw_circle_2d(xy, color, 4, 12)
    gpu.state.blend_set("NONE")


class OPENRIG_OT_pose_drag(_Base, bpy.types.Operator):
    bl_idname = "openrig.pose_drag"
    bl_label = "Pose Handles"
    bl_description = ("Drag the hands, feet, hips or head in the viewport; the rest of the body follows "
                      "and pinned feet stay planted. Esc or Enter to leave")

    # -- lifecycle
    def invoke(self, context, event):
        if _STATE["running"]:
            _STATE["running"] = False  # toggle off: the running modal will see this and exit
            return {"CANCELLED"}
        obj = active_rig(context)
        if obj.mode != "POSE":
            bpy.ops.object.mode_set(mode="POSE")
        self.sess = _session(self, context)
        if self.sess is None:
            return {"CANCELLED"}
        _warn_constraints(self, self.sess)
        self.area = context.area
        self.region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        if self.region is None or context.space_data.type != "VIEW_3D":
            self.report({"ERROR"}, "Run this from a 3D viewport")
            return {"CANCELLED"}
        self.rv3d = context.space_data.region_3d
        self.dragging = None
        _STATE.update(running=True, obj=obj, sk=self.sess.sk, hover=None, active=None)
        _STATE["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw_handles, (), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, "Pose Handles on. Drag a circle. Esc / Enter to leave.")
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if _STATE["handler"]:
            bpy.types.SpaceView3D.draw_handler_remove(_STATE["handler"], "WINDOW")
        _STATE.update(running=False, handler=None, obj=None, hover=None, active=None)
        if self.area:
            self.area.tag_redraw()

    def cancel(self, context):
        self._finish(context)

    # -- events
    def _mouse(self, event):
        return Vector((event.mouse_x - self.region.x, event.mouse_y - self.region.y))

    def _inside(self, event):
        r = self.region
        return r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height

    def _pick(self, context, event):
        pos = _handle_screen_positions(context, self.sess.obj, self.sess.sk, self.region, self.rv3d)
        m = self._mouse(event)
        best, best_d = None, PIN_RADIUS
        for label, xy in pos.items():
            d = (xy - m).length
            if d < best_d:
                best, best_d = label, d
        return best

    def modal(self, context, event):
        if not _STATE["running"]:
            self._finish(context)
            return {"CANCELLED"}
        obj = self.sess.obj
        if active_rig(context) is not obj or obj.mode != "POSE":
            self._finish(context)
            return {"CANCELLED"}

        if event.type == "ESC" and event.value == "PRESS":
            if self.dragging:
                self._cancel_drag()
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            self._finish(context)
            return {"FINISHED"}
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS" and not self.dragging:
            self._finish(context)
            return {"FINISHED"}

        if self.dragging:
            if event.type == "RIGHTMOUSE" and event.value == "PRESS":
                self._cancel_drag()
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "MOUSEMOVE":
                self._update_drag(context, event, live=True)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._commit_drag(context, event)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}

        if not self._inside(event):
            if _STATE["hover"]:
                _STATE["hover"] = None
                context.area.tag_redraw()
            return {"PASS_THROUGH"}
        if event.type == "MOUSEMOVE":
            hover = self._pick(context, event)
            if hover != _STATE["hover"]:
                _STATE["hover"] = hover
                context.area.tag_redraw()
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            label = self._pick(context, event)
            if label:
                self._start_drag(context, event, label)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    # -- drag
    def _start_drag(self, context, event, label):
        sess = self.sess
        sess.obj.update_tag()
        self.P0, self.R0 = sess.read()
        self.props = context.scene.openrig
        self.settings_live = rig.settings_from_props(self.props, sess, self.P0, live=True)
        self.solver = Solver(sess.sk, self.P0, self.settings_live)
        self.before = sess.snapshot_rot()
        self.dragging = label
        self.prev = self.P0.copy()
        particle = sess.sk.handles[label]
        h_world = sess.obj.matrix_world @ Vector(self.P0[sess.sk.index[particle]])
        self.depth_pt = h_world
        grab = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, self._mouse(event), h_world)
        self.offset = h_world - grab
        self.pins = set()
        for side, on in (("L", self.props.pin_foot_l), ("R", self.props.pin_foot_r)):
            if on and label != f"Foot {side}":
                self.pins.update(rig.foot_group(side))
        _STATE["active"] = label

    def _solve(self, context, event, solver, init):
        sess = self.sess
        mouse_w = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, self._mouse(event), self.depth_pt)
        target_l = sess.obj.matrix_world.inverted_safe() @ (mouse_w + self.offset)
        targets = rig.handle_targets(sess.sk, self.P0, self.dragging, np.array(target_l))
        return solver.solve(targets, self.pins, init=init)

    def _write(self, P):
        R = solve_rotations(self.sess.sk, self.P0, P, self.R0)
        self.sess.write(R, hips_head=P[self.sess.sk.index["hips"]])

    def _update_drag(self, context, event, live):
        P = self._solve(context, event, self.solver, self.prev)
        self.prev = P
        self._write(P)

    def _commit_drag(self, context, event):
        sess, p = self.sess, self.props
        full = Solver(sess.sk, self.P0, rig.settings_from_props(p, sess, self.P0, live=False))
        P = self._solve(context, event, full, self.prev)
        self._write(P)
        label = self.dragging
        if p.auto_key:
            changed, hips_moved = sess.changed_bones(self.before)
            sess.key(changed, context.scene.frame_current, hips_moved)
        bpy.ops.ed.undo_push(message=f"OpenRig: {label}")
        self.dragging = None
        _STATE["active"] = None

    def _cancel_drag(self):
        sess = self.sess
        for e in sess.sk.edges:
            pb = sess.obj.pose.bones[e.bone]
            setattr(pb, rig.rotation_path(pb), self.before[e.bone])
        sess.obj.pose.bones[sess.map.hips].location = self.before["__hips_loc"]
        self.dragging = None
        _STATE["active"] = None


# ---- keying ------------------------------------------------------------------
class OPENRIG_OT_key_pose(_Base, bpy.types.Operator):
    bl_idname = "openrig.key_pose"
    bl_label = "Key Pose"
    bl_description = "Insert rotation keys on all detected bones (and hips location) at the current frame"

    def execute(self, context):
        sess = _session(self, context)
        if not sess:
            return {"CANCELLED"}
        sess.key([e.bone for e in sess.sk.edges], context.scene.frame_current, True)
        return {"FINISHED"}


# ---- smooth ------------------------------------------------------------------
_PATH = re.compile(r'^pose\.bones\["(.+)"\]\.(rotation_quaternion|rotation_euler|rotation_axis_angle|location)$')


class OPENRIG_OT_smooth(_Base, bpy.types.Operator):
    bl_idname = "openrig.smooth"
    bl_label = "Smooth"
    bl_description = ("Remove jitter from the motion of the selected bones (or all detected bones if none "
                      "are selected) over the frame range. Keyframes keep their frames; only values change")

    def execute(self, context):
        sess = _session(self, context)
        if not sess:
            return {"CANCELLED"}
        p = context.scene.openrig
        a, b = frame_range(context)
        sel = {pb.name for pb in (context.selected_pose_bones or [])}
        mapped = {e.bone for e in sess.sk.edges}
        names = (sel & mapped) if sel & mapped else mapped
        count = 0
        for fc in fcurves_of(sess.obj):
            m = _PATH.match(fc.data_path)
            if not m or m.group(1) not in names:
                continue
            if m.group(2) == "location" and m.group(1) != sess.map.hips:
                continue
            count += self._smooth_curve(fc, a, b, p.smooth_sigma)
        self.report({"INFO"}, f"Smoothed {count} keys between frames {a} and {b}")
        return {"FINISHED"}

    @staticmethod
    def _smooth_curve(fc, a, b, sigma):
        r = filters.kernel_radius(sigma)
        frames = range(a - r, b + r + 1)
        ys = np.array([fc.evaluate(f) for f in frames])
        sm = filters.smooth_valid(ys, sigma)  # len == b - a + 1
        n = 0
        for kp in fc.keyframe_points:
            t = kp.co.x
            if not (a <= t <= b):
                continue
            i = t - a
            i0 = int(math.floor(i))
            i1 = min(i0 + 1, len(sm) - 1)
            new = sm[i0] * (1 - (i - i0)) + sm[i1] * (i - i0)
            d = new - kp.co.y
            kp.co.y = new
            kp.handle_left.y += d
            kp.handle_right.y += d
            n += 1
        fc.update()
        return n


# ---- lock in place ----------------------------------------------------------
def lock_foot(context, sess, props, side, f0, f1, plant_frame):
    """Hold a foot where it is at ``plant_frame`` for frames f0..f1 by re-solving the leg."""
    scene, obj, sk = context.scene, sess.obj, sess.sk
    ix = sk.index
    start = scene.frame_current
    scene.frame_set(plant_frame)
    P, _ = sess.read()
    group = [k for k in rig.foot_group(side) if k in ix]
    plant_world = {k: obj.matrix_world @ Vector(P[ix[k]]) for k in group}
    pins = set(sk.names) - {f"shin.{side}", *group}
    by_key = sk.edge_by_key
    leg_bones = [by_key[k].bone for k in (f"thigh.{side}", f"shin.{side}", f"foot.{side}", f"toe.{side}") if k in by_key]
    keyed = 0
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        before = sess.snapshot_rot()
        for _ in range(5):  # closed loop: re-read the real pose and correct what is left
            P, R = sess.read()
            inv = obj.matrix_world.inverted_safe()
            targets = {k: np.array(inv @ plant_world[k]) for k in group}
            err = max(np.linalg.norm(P[ix[k]] - targets[k]) for k in group)
            if err < 5e-4 * max(1.0, obj.matrix_world.median_scale):
                break
            settings = rig.settings_from_props(props, sess, P, live=False)
            settings.floor = False
            settings.polish_effector_weight = 0.1  # the planted foot is a hard goal
            Pn = Solver(sk, P, settings).solve(targets, pins)
            sess.write(solve_rotations(sk, P, Pn, R))
            context.view_layer.update()
        changed, _ = sess.changed_bones(before)
        # always key the whole leg: a bone with no key on a frame would otherwise fall back to
        # whatever its neighbouring keys say, which would un-lock the first/last frames
        to_key = list(dict.fromkeys(leg_bones + changed))
        sess.key(to_key, f, False)
        keyed += len(to_key)
    scene.frame_set(start)
    return keyed


class OPENRIG_OT_lock_foot(_Base, bpy.types.Operator):
    bl_idname = "openrig.lock_foot"
    bl_label = "Lock in Place"
    bl_description = ("Hold the chosen foot where it is on the first frame of the range so it stops "
                      "sliding, and re-solve the leg (keys the changed leg bones on every frame)")

    def execute(self, context):
        sess = _session(self, context)
        if not sess:
            return {"CANCELLED"}
        p = context.scene.openrig
        a, b = frame_range(context)
        n = lock_foot(context, sess, p, p.lock_side, a, b, a)
        self.report({"INFO"}, f"Locked {p.lock_side} foot over frames {a}-{b} ({n} bone keys)")
        return {"FINISHED"}


# ---- motion check ------------------------------------------------------------
_MARK = "OR "


def run_check(context, sess):
    """Sample the range and return (slides, pops). Restores the current frame."""
    scene, obj, sk = context.scene, sess.obj, sess.sk
    ix = sk.index
    a, b = frame_range(context)
    start = scene.frame_current
    foot = {s: [] for s in "LR"}
    quats = {e.bone: [] for e in sk.edges}
    up = None
    leg = 1.0
    for f in range(a, b + 1):
        scene.frame_set(f)
        P, _ = sess.read()
        if up is None:
            up = np.array(obj.matrix_world.to_3x3() @ Vector((0, 0, 1)))
            leg = float(np.linalg.norm(P[ix["foot.L"]] - P[ix["thigh.L"]])) * obj.matrix_world.median_scale
        for s in "LR":
            foot[s].append(np.array(obj.matrix_world @ Vector(P[ix[f"foot.{s}"]])))
        for e in sk.edges:
            quats[e.bone].append(tuple(rig.pose_bone_rotation_quat(obj.pose.bones[e.bone])))
    scene.frame_set(start)

    slides = []
    for s in "LR":
        for seg in filters.sliding_segments(np.array(foot[s]), (0, 0, 1), 0.05 * leg, 0.02 * leg):
            slides.append({"side": s, "start": a + seg["start"], "end": a + seg["end"], "drift": seg["drift"]})
    pops = {}
    for bone, q in quats.items():
        idx = filters.rotation_outliers(np.array(q))
        if idx:
            pops[bone] = [a + i for i in idx]
    return slides, pops


class OPENRIG_OT_motion_check(_Base, bpy.types.Operator):
    bl_idname = "openrig.motion_check"
    bl_label = "Motion Check"
    bl_description = ("Look for sliding feet and popping bones in the frame range. Adds timeline "
                      "markers named 'OR ...' where problems are")

    def execute(self, context):
        sess = _session(self, context)
        if not sess:
            return {"CANCELLED"}
        scene = context.scene
        for mk in [m for m in scene.timeline_markers if m.name.startswith(_MARK)]:
            scene.timeline_markers.remove(mk)
        slides, pops = run_check(context, sess)
        lines = []
        for s in slides:
            scene.timeline_markers.new(f"{_MARK}slide {s['side']}", frame=s["start"])
            lines.append(f"Foot {s['side']} slides {s['drift'] * 100:.1f} cm, frames {s['start']}-{s['end']}")
        for bone, frames in sorted(pops.items(), key=lambda kv: -len(kv[1]))[:8]:
            for f in frames[:3]:
                scene.timeline_markers.new(f"{_MARK}pop", frame=f)
            lines.append(f"{bone}: pops at frames {', '.join(map(str, frames[:5]))}")
        context.scene.openrig.report_text = "\n".join(lines) if lines else "No problems found."
        self.report({"INFO"}, f"{len(slides)} sliding segment(s), {len(pops)} bone(s) with pops")
        return {"FINISHED"}


class OPENRIG_OT_fix_sliding(_Base, bpy.types.Operator):
    bl_idname = "openrig.fix_sliding"
    bl_label = "Fix Sliding Feet"
    bl_description = "Run Motion Check, then lock every sliding foot segment in place"

    def execute(self, context):
        sess = _session(self, context)
        if not sess:
            return {"CANCELLED"}
        slides, _ = run_check(context, sess)
        p = context.scene.openrig
        for s in slides:
            lock_foot(context, sess, p, s["side"], s["start"], s["end"], s["start"])
        self.report({"INFO"}, f"Fixed {len(slides)} sliding segment(s)")
        return {"FINISHED"}


CLASSES = (
    OPENRIG_OT_detect, OPENRIG_OT_pose_drag, OPENRIG_OT_key_pose, OPENRIG_OT_smooth,
    OPENRIG_OT_lock_foot, OPENRIG_OT_motion_check, OPENRIG_OT_fix_sliding,
)
