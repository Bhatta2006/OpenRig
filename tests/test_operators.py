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


# ---------------- motion trail, ghosts, bone map -----------------------------
from openrig import ghosts, trail  # noqa: E402


def animate_hand(obj, frames=12):
    """Key a simple reach so the left hand travels along a path."""
    from test_blender import PINS
    from openrig.core.rotations import solve_rotations
    from openrig.core.solver import Solver
    sess = rig.RigSession(obj)
    P0, R0 = sess.read()
    ix = sess.sk.index
    for f in range(1, frames + 1):
        bpy.context.scene.frame_set(1)
        # restore rest pose before each solve so every frame starts from the same pose
        for e in sess.sk.edges:
            pb = obj.pose.bones[e.bone]
            pb.rotation_quaternion = (1, 0, 0, 0)
        obj.pose.bones[sess.map.hips].location = (0, 0, 0)
        bpy.context.view_layer.update()
        P, R = sess.read()
        t = P[ix["hand.L"]] + np.array([-0.1, -0.05 * (f - 1), 0.0])
        Pn = Solver(sess.sk, P).solve(rig.handle_targets(sess.sk, P, "Hand L", t), PINS)
        sess.write(solve_rotations(sess.sk, P, Pn, R), hips_head=Pn[ix["hips"]])
        sess.key([e.bone for e in sess.sk.edges], f, True)
    bpy.context.scene.frame_set(1)
    return sess


def hand_world(obj, f):
    bpy.context.scene.frame_set(f)
    return np.array(obj.matrix_world @ obj.pose.bones["mixamorig:LeftHand"].head)


def test_trail_samples_the_path(scene_rig):
    sess = animate_hand(scene_rig)
    pts = trail.sample_trail(bpy.context, sess, "Hand L", 1, 12)
    assert len(pts) == 12
    assert np.linalg.norm(np.array(pts[0]) - hand_world(scene_rig, 1)) < 1e-4
    assert abs(np.array(pts[-1])[1] - np.array(pts[0])[1]) > 0.3


def test_trail_edit_reshapes_with_falloff_and_leaves_outside_untouched(scene_rig):
    sess = animate_hand(scene_rig, frames=14)
    props = bpy.context.scene.openrig
    before = {f: hand_world(scene_rig, f) for f in range(1, 15)}
    delta = (0.0, 0.0, 0.12)
    trail.apply_trail_edit(bpy.context, sess, props, "Hand L", 7, delta, 3, 1, 14)
    after = {f: hand_world(scene_rig, f) for f in range(1, 15)}
    # the dragged frame moved by about the requested amount, upward
    moved = after[7] - before[7]
    assert moved[2] > 0.09 and np.linalg.norm(moved - np.array(delta)) < 0.04, moved
    # neighbours move less, in the same direction
    assert 0 < (after[6] - before[6])[2] < moved[2]
    # frames outside the falloff are unchanged (anchored)
    for f in (1, 2, 3, 11, 12, 13, 14):
        assert np.linalg.norm(after[f] - before[f]) < 6e-3, f
    # planted feet stayed planted on edited frames
    for f in (5, 7):
        bpy.context.scene.frame_set(f)
        for s in ("Left", "Right"):
            assert obj_foot_height(scene_rig, s) < 0.12


def obj_foot_height(obj, side):
    return (obj.matrix_world @ obj.pose.bones[f"mixamorig:{side}Foot"].head).z


def test_trail_falloff_shape():
    assert trail.falloff(5, 5, 4) == 1.0
    assert trail.falloff(7, 5, 4) < trail.falloff(6, 5, 4) < 1.0
    assert trail.falloff(6, 5, 0) == 0.0


def test_ghost_sampling_restores_frame_and_returns_neighbours(scene_rig):
    sess = animate_hand(scene_rig)
    bpy.context.scene.frame_set(6)
    g = ghosts.sample_ghosts(bpy.context, sess, 6, 2, 2)
    assert [f for f, _ in g] == [2, 4, 8, 10]
    assert bpy.context.scene.frame_current == 6
    assert len(g[0][1]) == len(sess.sk.edges)


def test_ghost_operator_toggles_handler(scene_rig):
    animate_hand(scene_rig)
    bpy.ops.openrig.ghosts()
    assert ghosts.active() and len(ghosts._G["frames"]) == 6
    bpy.ops.openrig.ghosts_clear()
    assert not ghosts.active()


def test_assign_bone_operator_fixes_a_missing_role(scene_rig):
    m = rig.stored_map(scene_rig)
    m.assign("toe.L", [])
    rig.save_map(scene_rig, m)
    bpy.ops.pose.select_all(action="DESELECT")
    scene_rig.data.bones.active = scene_rig.data.bones["mixamorig:LeftToeBase"]
    bpy.ops.openrig.assign_bone(role="toe.L")
    assert rig.stored_map(scene_rig).limbs["L"]["toe"] == "mixamorig:LeftToeBase"
    bpy.ops.openrig.assign_bone(role="toe.L", clear=True)
    assert "toe" not in rig.stored_map(scene_rig).limbs["L"]


def test_assign_bone_requires_selection(scene_rig):
    bpy.ops.pose.select_all(action="DESELECT")
    scene_rig.data.bones.active = None
    with pytest.raises(RuntimeError):
        bpy.ops.openrig.assign_bone(role="hips")


def test_lock_does_not_disturb_frames_outside_range(scene_rig):
    key_hips_slide(scene_rig)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, 20
    # give the left leg some animation outside the lock range first
    sess = rig.RigSession(scene_rig)
    sc.frame_set(25)
    leg = scene_rig.pose.bones["mixamorig:LeftLeg"]
    leg.rotation_quaternion = (0.95, 0.31, 0, 0)
    leg.keyframe_insert("rotation_quaternion", frame=25)
    sc.frame_set(21)
    outside_before = np.array(scene_rig.pose.bones["mixamorig:LeftLeg"].rotation_quaternion)
    sc.openrig.lock_side = "L"
    bpy.ops.openrig.lock_foot()
    sc.frame_set(21)
    outside_after = np.array(scene_rig.pose.bones["mixamorig:LeftLeg"].rotation_quaternion)
    assert np.allclose(outside_before, outside_after, atol=1e-4)


def test_retime_moves_keys_with_falloff_and_never_crosses(scene_rig):
    sess = animate_hand(scene_rig, frames=14)
    fc = next(c for c in rig_fcurves(scene_rig.animation_data.action)
              if c.data_path.endswith('LeftArm"].rotation_quaternion') and c.array_index == 1)
    before = [k.co.x for k in fc.keyframe_points]
    assert before == [float(i) for i in range(1, 15)]
    n = trail.retime_keys(sess, 7, 2.0, 4)
    after = [k.co.x for k in fc.keyframe_points]
    assert n > 0
    assert abs(after[6] - 9.0) < 1e-6                      # the dragged moment moved 2 frames later
    assert all(b > a for a, b in zip(after, after[1:])), after  # order preserved
    assert after[0] == before[0] and after[-1] == before[-1]    # far keys untouched
    # an excessive shift is clamped so keys cannot cross
    trail.retime_keys(sess, 7, 99.0, 4)
    after2 = [k.co.x for k in fc.keyframe_points]
    assert all(b > a for a, b in zip(after2, after2[1:])), after2


def test_trail_edit_is_not_applied_on_top_of_itself_with_sparse_keys(scene_rig):
    """Keys only every 6 frames: later frames must still be solved from the original animation."""
    sess = animate_hand(scene_rig, frames=19)
    for fc in rig_fcurves(scene_rig.animation_data.action):
        for kp in list(fc.keyframe_points):
            if (int(kp.co.x) - 1) % 6 != 0:
                fc.keyframe_points.remove(kp)
        fc.update()
    props = bpy.context.scene.openrig
    before = {f: hand_world(scene_rig, f) for f in range(1, 20)}
    trail.apply_trail_edit(bpy.context, sess, props, "Hand L", 10, (0, 0, 0.10), 5, 1, 19)
    for f in range(1, 20):
        want = 0.10 * trail.falloff(f, 10, 5)
        got = (hand_world(scene_rig, f) - before[f])[2]
        assert abs(got - want) < 0.03, (f, got, want)


def test_trail_edit_ends_do_not_pop(scene_rig):
    sess = animate_hand(scene_rig, frames=16)
    props = bpy.context.scene.openrig
    before = {f: hand_world(scene_rig, f) for f in range(1, 17)}
    trail.apply_trail_edit(bpy.context, sess, props, "Hand L", 8, (0, 0, 0.12), 4, 1, 16)
    for f in (4, 12):      # exactly radius away: zero edit, no step against the anchors
        assert np.linalg.norm(hand_world(scene_rig, f) - before[f]) < 6e-3, f


def test_retime_moves_every_curve(scene_rig):
    sess = animate_hand(scene_rig, frames=14)
    scene_rig.location.x = 0.0
    scene_rig.keyframe_insert("location", frame=7, index=0)
    trail.retime_keys(sess, 7, 2.0, 4)
    obj_fc = next(fc for fc in rig_fcurves(scene_rig.animation_data.action) if fc.data_path == "location")
    assert abs(obj_fc.keyframe_points[0].co.x - 9.0) < 1e-6


def test_ghosts_remember_sampling_settings(scene_rig):
    animate_hand(scene_rig)
    p = bpy.context.scene.openrig
    p.ghost_step, p.ghost_count = 4, 2
    bpy.context.scene.frame_set(6)
    bpy.ops.openrig.ghosts()
    p.ghost_step = 9
    assert ghosts._G["step"] == 4 and ghosts._G["count"] == 2
    bpy.ops.openrig.ghosts_clear()


def test_goto_ghost_frame_operator(scene_rig):
    bpy.context.scene.frame_set(1)
    bpy.ops.openrig.goto_frame(frame=17)
    assert bpy.context.scene.frame_current == 17


def test_trail_edit_uses_each_frames_object_transform(scene_rig):
    sess = animate_hand(scene_rig, frames=14)
    scene = bpy.context.scene
    scene.openrig.use_prior = False
    for frame, scale in ((1, 1.0), (14, 4.0)):
        scene_rig.scale = (scale,) * 3
        scene_rig.keyframe_insert("scale", frame=frame)
    before = {frame: hand_world(scene_rig, frame) for frame in range(1, 15)}
    scene.frame_set(14)
    delta = np.array((0.0, 0.0, 0.18))
    trail.apply_trail_edit(bpy.context, sess, scene.openrig, "Hand L", 7, delta, 3, 1, 14)
    assert scene.frame_current == 14
    moved = hand_world(scene_rig, 7) - before[7]
    assert np.linalg.norm(moved - delta) < 0.025, moved


def test_motion_check_detects_pop_with_nonunit_quaternion_keys(scene_rig):
    arm = scene_rig.pose.bones["mixamorig:LeftForeArm"]
    for frame in range(1, 31):
        angle = 0.02 * frame + (0.6 if frame == 15 else 0.0)
        arm.rotation_quaternion = (2 * np.cos(angle / 2), 2 * np.sin(angle / 2), 0, 0)
        arm.keyframe_insert("rotation_quaternion", frame=frame)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 30
    bpy.ops.openrig.motion_check()
    assert "LeftForeArm: pops at frames 15" in bpy.context.scene.openrig.report_text
