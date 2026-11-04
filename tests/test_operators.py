"""Operator-level tests in headless Blender (no UI, so the modal drag is exercised via stubs)."""
import types

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
from mathutils import Vector  # noqa: E402

import openrig  # noqa: E402
from openrig import operators, rig  # noqa: E402
from test_blender import make_armature, rig_fcurves  # noqa: E402


@pytest.fixture
def scene_rig():
    obj = make_armature()
    try:
        openrig.register()
    except Exception:
        openrig.unregister(); openrig.register()
    bpy.ops.openrig.detect()
    yield obj
    openrig.unregister()


def foot_world(obj, frame, side="L"):
    bpy.context.scene.frame_set(frame)
    return np.array(obj.matrix_world @ obj.pose.bones[f"mixamorig:{'Left' if side == 'L' else 'Right'}Foot"].head)


def key_hips_slide(obj, frames=20, dx=0.1):
    """Bend the knees (feet stay planted), then slide the whole character sideways over time so
    both feet skate on the ground."""
    from test_blender import run_drag
    run_drag(obj, "Hips", (0, 0, -0.15))
    for f in range(1, frames + 1):
        obj.location.x = dx * (f - 1) / (frames - 1)
        obj.keyframe_insert("location", frame=f, index=0)
    bpy.context.scene.frame_set(1)


def test_detect_operator_sets_status(scene_rig):
    p = bpy.context.scene.openrig
    assert p.status.startswith("OK")
    assert rig.stored_map(scene_rig).ok


def test_key_pose_operator(scene_rig):
    bpy.context.scene.frame_set(3)
    bpy.ops.openrig.key_pose()
    paths = {fc.data_path for fc in rig_fcurves(scene_rig.animation_data.action)}
    assert 'pose.bones["mixamorig:Head"].rotation_quaternion' in paths
    assert 'pose.bones["mixamorig:Hips"].location' in paths


def test_motion_check_finds_sliding_and_lock_fixes_it(scene_rig):
    key_hips_slide(scene_rig)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 20
    bpy.ops.openrig.motion_check()
    p = bpy.context.scene.openrig
    assert "slides" in p.report_text, p.report_text
    assert any(m.name.startswith("OpenRig: slide") for m in bpy.context.scene.timeline_markers)

    plant = foot_world(scene_rig, 1)
    bpy.context.scene.openrig.lock_side = "L"
    bpy.ops.openrig.lock_foot()
    for f in (1, 5, 10, 15, 20):
        assert np.linalg.norm(foot_world(scene_rig, f) - plant) < 1.5e-3, f
    # the other foot is untouched and still slides
    assert np.linalg.norm(foot_world(scene_rig, 20, "R") - foot_world(scene_rig, 1, "R")) > 0.08


def test_fix_sliding_clears_report(scene_rig):
    key_hips_slide(scene_rig)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 20
    bpy.ops.openrig.fix_sliding()
    bpy.ops.openrig.motion_check()
    assert bpy.context.scene.openrig.report_text == "No problems found.", bpy.context.scene.openrig.report_text


def test_smooth_reduces_jitter_but_keeps_key_frames(scene_rig):
    rng = np.random.default_rng(1)
    arm = scene_rig.pose.bones["mixamorig:LeftArm"]
    for f in range(1, 41):
        arm.rotation_quaternion = (1, 0.002 * f + rng.normal(0, 0.02), 0, 0)
        arm.keyframe_insert("rotation_quaternion", frame=f)
    fc = next(c for c in rig_fcurves(scene_rig.animation_data.action)
              if c.data_path.endswith('LeftArm"].rotation_quaternion') and c.array_index == 1)
    before = np.array([k.co.y for k in fc.keyframe_points])
    frames_before = [k.co.x for k in fc.keyframe_points]
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 40
    bpy.context.scene.openrig.smooth_sigma = 3
    bpy.ops.openrig.smooth()
    after = np.array([k.co.y for k in fc.keyframe_points])
    assert [k.co.x for k in fc.keyframe_points] == frames_before
    assert np.abs(np.diff(after, 2)).mean() < 0.4 * np.abs(np.diff(before, 2)).mean()


def test_pop_detected(scene_rig):
    arm = scene_rig.pose.bones["mixamorig:LeftForeArm"]
    for f in range(1, 31):
        arm.rotation_quaternion = (1, 0.01 * f, 0, 0)
        if f == 15:
            arm.rotation_quaternion = (0.6, 0.8, 0, 0)
        arm.keyframe_insert("rotation_quaternion", frame=f)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 30
    bpy.ops.openrig.motion_check()
    assert "LeftForeArm: pops at frames 15" in bpy.context.scene.openrig.report_text


def test_drag_code_path_with_stubbed_viewport(scene_rig, monkeypatch):
    """Exercise start/update/commit of the modal operator without a real 3D view."""
    obj = scene_rig
    cls = operators.OPENRIG_OT_pose_drag
    sess = rig.RigSession(obj)
    stub = types.SimpleNamespace(sess=sess, region=None, rv3d=None, dragging=None)
    stub._mouse = lambda ev: Vector((0, 0))
    stub._solve = lambda *a, **k: cls._solve(stub, *a, **k)
    stub._write = lambda P: cls._write(stub, P)
    stub._update_drag = lambda c, e, live: cls._update_drag(stub, c, e, live)
    stub._cancel_drag = lambda: cls._cancel_drag(stub)

    where = {"p": None}
    monkeypatch.setattr(operators.view3d_utils, "region_2d_to_location_3d",
                        lambda r, v, xy, depth: where["p"])
    hand = obj.matrix_world @ obj.pose.bones["mixamorig:LeftHand"].head
    where["p"] = hand.copy()
    ctx = bpy.context
    cls._start_drag(stub, ctx, None, "Hand L")
    assert stub.dragging == "Hand L" and "foot.L" in "".join(stub.pins)

    where["p"] = hand + Vector((-0.2, -0.3, 0.25))
    cls._update_drag(stub, ctx, None, True)
    bpy.context.view_layer.update()
    moved = obj.matrix_world @ obj.pose.bones["mixamorig:LeftHand"].head
    assert (moved - where["p"]).length < 0.05

    cls._commit_drag(stub, ctx, None)
    assert stub.dragging is None
    paths = {fc.data_path for fc in rig_fcurves(obj.animation_data.action)}
    assert 'pose.bones["mixamorig:LeftArm"].rotation_quaternion' in paths

    # cancel restores the pose exactly
    stub.before = sess.snapshot_rot()
    stub.P0, stub.R0 = sess.read()
    stub.props = ctx.scene.openrig
    stub.settings_live = rig.settings_from_props(stub.props, sess, stub.P0, live=True)
    stub.solver = operators.Solver(sess.sk, stub.P0, stub.settings_live)
    stub.prev = stub.P0.copy(); stub.pins = set(); stub.depth_pt = hand
    stub.offset = Vector((0, 0, 0)); stub.dragging = "Hips"
    where["p"] = obj.matrix_world @ obj.pose.bones["mixamorig:Hips"].head + Vector((0, 0, -0.3))
    cls._update_drag(stub, ctx, None, True)
    cls._cancel_drag(stub)
    after = sess.snapshot_rot()
    for k, v in stub.before.items():
        assert np.allclose(after[k], v, atol=1e-9), k


def test_lock_only_keys_the_locked_leg(scene_rig):
    key_hips_slide(scene_rig)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 20
    bpy.context.scene.openrig.lock_side = "L"
    bpy.ops.openrig.lock_foot()
    import re
    bones = {re.match(r'pose\.bones\["(.+)"\]', fc.data_path).group(1)
             for fc in rig_fcurves(scene_rig.animation_data.action) if fc.data_path.startswith("pose.bones")}
    assert bones == {"mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase"}


def test_motion_markers_do_not_delete_user_markers(scene_rig):
    sc = bpy.context.scene
    sc.timeline_markers.new("OR gate", frame=5)
    key_hips_slide(scene_rig)
    sc.frame_start, sc.frame_end = 1, 20
    bpy.ops.openrig.motion_check()
    bpy.ops.openrig.motion_check()
    assert "OR gate" in [m.name for m in sc.timeline_markers]


def test_motion_check_rejects_tiny_range(scene_rig):
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 5, 5
    with pytest.raises(RuntimeError):
        bpy.ops.openrig.motion_check()


def test_corrupt_map_property_does_not_break_polls(scene_rig):
    scene_rig["openrig_map"] = "[1, 2, 3]"
    assert rig.stored_map(scene_rig) is None
    scene_rig["openrig_map"] = "not json"
    assert rig.stored_map(scene_rig) is None


def _modal_stub(obj, gen_offset=0):
    cls = operators.OPENRIG_OT_pose_drag
    area = types.SimpleNamespace(regions=[], x=0, y=0, width=10, height=10, tag_redraw=lambda: None)
    stub = types.SimpleNamespace(sess=types.SimpleNamespace(obj=obj), area=area,
                                 region=types.SimpleNamespace(x=0, y=0, width=10, height=10),
                                 dragging=None, gen=operators._STATE["gen"] + gen_offset)
    stub._over_area = lambda e: cls._over_area(stub, e)
    stub._inside = lambda e: cls._inside(stub, e)
    stub._finish = lambda c: cls._finish(stub, c)
    return cls, stub


def test_modal_does_not_cancel_on_object_wrapper_identity(scene_rig):
    cls, stub = _modal_stub(scene_rig)
    operators._STATE.update(running=True, obj=scene_rig)
    # a fresh wrapper of the same armature, like bpy hands out on every access
    stub.sess.obj = bpy.data.objects[scene_rig.name]
    ev = types.SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_x=500, mouse_y=500)
    assert scene_rig.mode == "POSE"
    assert cls._modal(stub, bpy.context, ev) == {"PASS_THROUGH"}
    operators._teardown()


def test_modal_exits_when_replaced_or_toggled_off(scene_rig):
    cls, stub = _modal_stub(scene_rig, gen_offset=-1)   # an older session
    operators._STATE.update(running=True, obj=scene_rig)
    ev = types.SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_x=1, mouse_y=1)
    assert cls._modal(stub, bpy.context, ev) == {"CANCELLED"}
    assert operators._STATE["running"], "an old session must not tear down the new one"
    operators._teardown()
    assert not operators._STATE["running"]


def test_modal_survives_freed_object(scene_rig):
    cls, stub = _modal_stub(scene_rig)
    class Freed:
        @property
        def name(self):
            raise ReferenceError("StructRNA has been removed")
    stub.sess.obj = Freed()
    operators._STATE.update(running=True, obj=scene_rig)
    ev = types.SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_x=1, mouse_y=1)
    stub._modal = lambda c, e: cls._modal(stub, c, e)
    assert cls.modal(stub, bpy.context, ev) == {"CANCELLED"}
    assert not operators._STATE["running"]
