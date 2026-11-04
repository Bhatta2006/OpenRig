"""Ghost poses: stick figures of the frames around a chosen frame, drawn in the viewport.
Optionally spread out side by side so a short span of motion reads like a strip of frames."""
from __future__ import annotations

import bpy
from mathutils import Vector

from . import rig

_G = {"handler": None, "frames": [], "center": None, "obj": None, "step": 1, "count": 1}   # frames: [(frame, [(a, b), ...])]


def sample_ghosts(context, sess, center, count, step):
    """Return [(frame, [(world a, world b), ...])] for the frames around ``center``."""
    scene, obj, sk = context.scene, sess.obj, sess.sk
    start = scene.frame_current
    frames = []
    for k in range(-count, count + 1):
        if k == 0:
            continue
        f = center + k * step
        scene.frame_set(f)
        P, _ = sess.read()
        w = [obj.matrix_world @ Vector(p) for p in P]
        frames.append((f, [(w[e.a], w[e.b]) for e in sk.edges]))
    scene.frame_set(start)
    return frames


def _draw():
    import gpu
    from gpu_extras.batch import batch_for_shader

    ctx = bpy.context
    if not _G["frames"] or ctx.region_data is None:
        return
    p = ctx.scene.openrig
    right = ctx.region_data.view_rotation @ Vector((1, 0, 0))
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(2.0)
    center = _G["center"]
    n = max(1, _G["count"])
    for f, segs in _G["frames"]:
        k = (f - center) / max(1, _G["step"])
        shift = right * (k * p.ghost_spread)
        pos = []
        for a, b in segs:
            pos += [tuple(a + shift), tuple(b + shift)]
        fade = 0.85 - 0.5 * (abs(k) - 1) / n
        color = (0.3, 0.55, 1.0, fade) if k < 0 else (1.0, 0.55, 0.2, fade)
        shader.uniform_float("color", color)
        batch_for_shader(shader, "LINES", {"pos": pos}).draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def clear():
    if _G["handler"]:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_G["handler"], "WINDOW")
        except (ValueError, RuntimeError):
            pass
    _G.update(handler=None, frames=[], center=None, obj=None)


def active() -> bool:
    return _G["handler"] is not None


class OPENRIG_OT_ghosts(bpy.types.Operator):
    bl_idname = "openrig.ghosts"
    bl_label = "Ghost Frames"
    bl_description = ("Draw stick-figure ghosts of the frames before (blue) and after (orange) the current "
                      "frame. Press again to refresh around the new current frame")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE" and rig.stored_map(obj) is not None

    def execute(self, context):
        try:
            sess = rig.RigSession(context.active_object)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        p = context.scene.openrig
        center = context.scene.frame_current
        _G["frames"] = sample_ghosts(context, sess, center, p.ghost_count, p.ghost_step)
        _G.update(center=center, obj=sess.obj.name, step=p.ghost_step, count=p.ghost_count)
        if not _G["handler"]:
            _G["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


class OPENRIG_OT_ghosts_clear(bpy.types.Operator):
    bl_idname = "openrig.ghosts_clear"
    bl_label = "Clear Ghosts"
    bl_description = "Remove the ghost frames from the viewport"
    bl_options = {"REGISTER"}

    def execute(self, context):
        clear()
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


class OPENRIG_OT_goto_frame(bpy.types.Operator):
    bl_idname = "openrig.goto_frame"
    bl_label = "Go To Ghost Frame"
    bl_description = "Jump to this frame so you can pose it with the handles"
    bl_options = {"REGISTER"}
    frame: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {"FINISHED"}


CLASSES = (OPENRIG_OT_ghosts, OPENRIG_OT_ghosts_clear, OPENRIG_OT_goto_frame)
