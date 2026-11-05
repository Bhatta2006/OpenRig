"""Motion Trail: see the path of a handle over time and drag a point on it to reshape the
motion. The pose on every affected frame is re-solved, so the whole body follows the edit."""
from __future__ import annotations

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

from . import rig
from .core import filters
from .core.rotations import solve_rotations
from .core.solver import Solver

HANDLE_ITEMS = [("Hand L", "Left Hand", ""), ("Hand R", "Right Hand", ""), ("Head", "Head", ""),
                ("Hips", "Hips", ""), ("Foot L", "Left Foot", ""), ("Foot R", "Right Foot", "")]
MAX_FRAMES = 400
PICK_PX = 12


def handle_source(sess, label):
    sk = sess.sk
    return sk.sources[sk.index[sk.handles[label]]]


def sample_trail(context, sess, label, f0, f1):
    """World position of a handle on every frame in f0..f1 (restores the current frame)."""
    scene, obj = context.scene, sess.obj
    bone, which = handle_source(sess, label)
    pb = obj.pose.bones[bone]
    start = scene.frame_current
    pts = []
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        pts.append(obj.matrix_world @ (pb.head if which == "head" else pb.tail))
    scene.frame_set(start)
    return pts


def falloff(frame, center, radius):
    return float(filters.falloff(frame - center, radius))


def apply_trail_edit(context, sess, props, label, frame, delta_world, radius, f0, f1):
    """Move ``label`` by ``delta_world`` on ``frame`` and by a falling-off share on the frames
    around it, re-solving the whole body each frame and keying what changed."""
    scene, obj, sk = context.scene, sess.obj, sess.sk
    start = scene.frame_current
    lo, hi = max(f0, frame - radius), min(f1, frame + radius)
    anchors = rig.capture_anchors(context, sess, (lo - 1, hi + 1))
    # Read the ORIGINAL pose of every frame first: keys inserted while editing would otherwise
    # change what later frames evaluate to, and the edit would stack on itself.
    originals = {}
    for fr in range(lo, hi + 1):
        scene.frame_set(fr)
        P, R = sess.read()
        # Object rotation/scale can be animated: the same world-space drag has a
        # different armature-space direction and size on each sampled frame.
        delta_local = np.array(obj.matrix_world.to_3x3().inverted_safe() @ Vector(delta_world))
        originals[fr] = (P, R, sess.snapshot_rot(), delta_local)
    touched, hips_moved = [], False
    for fr in range(lo, hi + 1):
        w = falloff(fr, frame, radius)
        P, R, before, delta_local = originals[fr]
        scene.frame_set(fr)
        target = P[sk.index[sk.handles[label]]] + delta_local * w
        pins = set()
        for side, on in (("L", props.pin_foot_l), ("R", props.pin_foot_r)):
            if on and label != f"Foot {side}":
                pins.update(rig.foot_group(side))
        settings = rig.settings_from_props(props, sess, P, live=False)
        solver = Solver(sk, P, settings)
        Pn = solver.solve(rig.handle_targets(sk, P, label, target), pins)
        sess.write(solve_rotations(sk, P, Pn, R), hips_head=Pn[sk.index["hips"]])
        changed, moved = sess.changed_bones(before)
        sess.key(changed, fr, moved)
        touched = list(dict.fromkeys(touched + changed))
        hips_moved = hips_moved or moved
    # second pass: every touched bone gets a key on every edited frame so none falls back to
    # interpolation between distant old keys
    for fr in range(lo, hi + 1):
        scene.frame_set(fr)
        sess.key(touched, fr, hips_moved)
    rig.restore_anchors(context, sess, anchors, touched, hips_moved)
    scene.frame_set(start)
    return touched


def retime_keys(sess, frame, shift, radius):
    """Shift the timing around ``frame`` by up to ``shift`` frames (with falloff) so that moment
    happens earlier or later. Every curve in the armature's action moves together, otherwise
    fingers, props and controls would drift out of sync with the body."""
    from .operators import fcurves_of
    shift = max(-filters.max_retime_shift(radius), min(filters.max_retime_shift(radius), shift))
    moved = 0
    for fc in fcurves_of(sess.obj):
        for kp in fc.keyframe_points:
            new = float(filters.warp_time(kp.co.x, frame, shift, radius))
            d = new - kp.co.x
            if abs(d) > 1e-9:
                kp.co.x = new
                kp.handle_left.x += d
                kp.handle_right.x += d
                moved += 1
        fc.update()
    return moved


# ---- viewport ---------------------------------------------------------------
_T = {"running": False, "handler": None, "gen": 0, "sess": None, "pts": [], "f0": 0, "grab": None,
      "dragging": None, "hover": None, "preview": None, "radius": 6}


def _teardown():
    if _T["handler"]:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_T["handler"], "WINDOW")
        except (ValueError, RuntimeError):
            pass
    _T.update(running=False, handler=None, sess=None, pts=[], dragging=None, hover=None, preview=None)


def _to_2d(region, rv3d, p):
    return view3d_utils.location_3d_to_region_2d(region, rv3d, p)


def _draw():
    import gpu
    from gpu_extras.batch import batch_for_shader
    from gpu_extras.presets import draw_circle_2d

    ctx = bpy.context
    pts = _T["pts"]
    if not _T["running"] or not pts or ctx.region is None or ctx.region_data is None:
        return
    pv = _T["preview"]
    cur = ctx.scene.frame_current
    xy = []
    for i, p in enumerate(pts):
        q = _to_2d(ctx.region, ctx.region_data, pv[i] if pv else p)
        xy.append(q)
    line = [tuple(q) for q in xy if q is not None]
    if len(line) > 1:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.line_width_set(2.0)
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": line})
        shader.uniform_float("color", (0.25, 0.8, 1.0, 0.8))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
        for i, q in enumerate(xy):
            if q is None:
                continue
            frame = _T["f0"] + i
            hot = i == _T["hover"] or i == _T["dragging"]
            color = (1.0, 0.75, 0.1, 1.0) if hot else ((1, 1, 1, 1) if frame == cur else (0.25, 0.8, 1.0, 0.9))
            draw_circle_2d(q, color, 6 if (hot or frame == cur) else 3, 12)
        gpu.state.blend_set("NONE")


class OPENRIG_OT_trail_edit(bpy.types.Operator):
    bl_idname = "openrig.trail_edit"
    bl_label = "Motion Trail"
    bl_description = ("Show the path of the chosen handle over the frame range. Drag a point to reshape "
                      "the motion; neighbouring frames follow with a falloff and the body is re-solved. "
                      "Esc / Enter to leave")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE" and rig.stored_map(obj) is not None

    def invoke(self, context, event):
        if _T["running"]:
            _T["gen"] += 1
            _teardown()
            context.area.tag_redraw()
            return {"CANCELLED"}
        if context.space_data.type != "VIEW_3D":
            self.report({"ERROR"}, "Run this from a 3D viewport")
            return {"CANCELLED"}
        try:
            self.sess = rig.RigSession(context.active_object)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if context.active_object.mode != "POSE":
            bpy.ops.object.mode_set(mode="POSE")
        from .operators import frame_range
        self.f0, self.f1 = frame_range(context)
        if self.f1 - self.f0 + 1 > MAX_FRAMES:
            self.f1 = self.f0 + MAX_FRAMES - 1
            self.report({"WARNING"}, f"Trail limited to {MAX_FRAMES} frames")
        self.area = context.area
        self.region = next(r for r in context.area.regions if r.type == "WINDOW")
        self.rv3d = context.space_data.region_3d
        self.label = context.scene.openrig.trail_handle
        self._resample(context)
        _T["gen"] += 1
        self.gen = _T["gen"]
        _T.update(running=True, sess=self.sess, f0=self.f0, hover=None, dragging=None, preview=None,
                  radius=context.scene.openrig.trail_radius)
        _T["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _resample(self, context):
        _T["pts"] = sample_trail(context, self.sess, self.label, self.f0, self.f1)

    def _finish(self, context):
        if _T["gen"] == self.gen:
            _teardown()
        try:
            self.area.tag_redraw()
        except ReferenceError:
            pass

    def cancel(self, context):
        self._finish(context)

    def _mouse(self, event):
        return Vector((event.mouse_x - self.region.x, event.mouse_y - self.region.y))

    def _inside(self, event):
        for r in self.area.regions:
            if r.type != "WINDOW" and r.width > 1 and r.height > 1 and \
                    r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height:
                return False
        r = self.region
        return r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height

    def _pick(self, event):
        m = self._mouse(event)
        best, best_d = None, PICK_PX
        for i, p in enumerate(_T["pts"]):
            q = _to_2d(self.region, self.rv3d, p)
            if q is not None and (q - m).length < best_d:
                best, best_d = i, (q - m).length
        return best

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except ReferenceError:
            self._finish(context)
            return {"CANCELLED"}

    def _modal(self, context, event):
        if self.gen != _T["gen"] or not _T["running"]:
            return {"CANCELLED"}
        cur = context.active_object
        if cur is None or cur.name != self.sess.obj.name:
            self._finish(context)
            return {"CANCELLED"}
        a = self.area
        over = a.x <= event.mouse_x < a.x + a.width and a.y <= event.mouse_y < a.y + a.height
        dragging = _T["dragging"]
        if event.type == "ESC" and event.value == "PRESS" and (dragging is not None or over):
            if dragging is not None:
                _T.update(dragging=None, preview=None, retime=False)
                context.workspace.status_text_set(None)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}
            self._finish(context)
            return {"FINISHED"}
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS" and dragging is None and over:
            self._finish(context)
            return {"FINISHED"}

        if dragging is not None:
            if event.type == "MOUSEMOVE":
                self._preview(event)
                context.area.tag_redraw()
            elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._commit(context, event)
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if not self._inside(event):
            return {"PASS_THROUGH"}
        if event.type == "MOUSEMOVE":
            hover = self._pick(event)
            if hover != _T["hover"]:
                _T["hover"] = hover
                context.area.tag_redraw()
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            i = self._pick(event)
            if i is not None:
                p = _T["pts"][i]
                grab = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, self._mouse(event), p)
                _T.update(dragging=i, grab=(p, p - grab), preview=list(_T["pts"]),
                          retime=event.alt, x0=event.mouse_x)
                if event.alt:
                    context.workspace.status_text_set("Retime: drag left/right to move this moment earlier or later")
                return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _delta(self, event):
        p, offset = _T["grab"]
        now = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, self._mouse(event), p)
        return (now + offset) - p

    RETIME_PX = 12  # mouse pixels per frame while retiming

    def _shift(self, event):
        r = _T["radius"]
        shift = (event.mouse_x - _T["x0"]) / self.RETIME_PX
        lim = filters.max_retime_shift(r)
        return max(-lim, min(lim, shift))

    def _preview(self, event):
        if _T.get("retime"):
            bpy.context.workspace.status_text_set(f"Retime: {self._shift(event):+.1f} frames")
            return
        d = self._delta(event)
        i = _T["dragging"]
        r = _T["radius"]
        _T["preview"] = [p + d * falloff(self.f0 + k, self.f0 + i, r) for k, p in enumerate(_T["pts"])]

    def _commit(self, context, event):
        i = _T["dragging"]
        retime = _T.get("retime")
        shift = self._shift(event) if retime else 0.0
        delta = Vector((0, 0, 0)) if retime else self._delta(event)
        _T.update(dragging=None, preview=None, retime=False)
        if retime:
            context.workspace.status_text_set(None)
            if abs(shift) < 0.05:
                return
            retime_keys(self.sess, self.f0 + i, shift, _T["radius"])
            self._resample(context)
            bpy.ops.ed.undo_push(message=f"OpenRig: retime {self.label}")
            return
        if delta.length < 1e-6:
            return
        props = context.scene.openrig
        apply_trail_edit(context, self.sess, props, self.label, self.f0 + i, delta, _T["radius"],
                         self.f0, self.f1)
        self._resample(context)
        bpy.ops.ed.undo_push(message=f"OpenRig: trail {self.label}")


CLASSES = (OPENRIG_OT_trail_edit,)
