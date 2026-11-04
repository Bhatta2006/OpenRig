"""Integration tests: run inside real (headless) Blender via the bpy wheel."""
import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
from mathutils import Vector  # noqa: E402

from openrig import rig  # noqa: E402
from openrig.core.rotations import solve_rotations  # noqa: E402
from openrig.core.solver import Solver  # noqa: E402
from rigs import tpose_bones, with_prefix  # noqa: E402


def make_armature(prefix="mixamorig:", rotation_mode="QUATERNION", roll_hack=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = bpy.data.armatures.new("Arm")
    obj = bpy.data.objects.new("Rig", arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bones = with_prefix(tpose_bones(), prefix)
    made = {}
    # parents first
    for name in sorted(bones, key=lambda n: _depth(bones, n)):
        h, t, par = bones[name]
        eb = arm.edit_bones.new(name)
        eb.head, eb.tail = Vector(h), Vector(t)
        if roll_hack:
            eb.roll = 0.7  # arbitrary roll to test non-trivial rest orientations
        if par:
            eb.parent = made[par]
        made[name] = eb
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.rotation_mode = rotation_mode
    return obj


def _depth(bones, n):
    d = 0
    while bones[n][2]:
        n, d = bones[n][2], d + 1
    return d


PINS = {"foot.L", "foot.R", "toe.L", "toe.R", "toe_end.L", "toe_end.R"}


def run_drag(obj, label, delta):
    m = rig.detect(obj)
    assert m.ok, m.missing()
    sess = rig.RigSession(obj)
    P0, R0 = sess.read()
    sol = Solver(sess.sk, P0)
    target = P0[sess.sk.index[sess.sk.handles[label]]] + np.array(delta)
    P1 = sol.solve(rig.handle_targets(sess.sk, P0, label, target), PINS)
    R1 = solve_rotations(sess.sk, P0, P1, R0)
    sess.write(R1, hips_head=P1[sess.sk.index["hips"]])
    bpy.context.view_layer.update()
    return sess, P0, P1, R0, R1


@pytest.mark.parametrize("mode", ["QUATERNION", "XYZ", "YXZ"])
@pytest.mark.parametrize("roll", [False, True])
@pytest.mark.parametrize("label,delta", [
    ("Hand L", (-0.2, -0.4, 0.3)),
    ("Hips", (0.0, 0.05, -0.2)),
    ("Head", (0.05, -0.2, -0.1)),
    ("Foot R", (0.1, -0.3, 0.15)),
])
def test_written_pose_matches_solution(mode, roll, label, delta):
    obj = make_armature(rotation_mode=mode, roll_hack=roll)
    sess, P0, P1, R0, R1 = run_drag(obj, label, delta)
    P2, R2 = sess.read()
    # joint positions of every bone head must equal the solved ones (rigid chain)
    # rotations must land exactly as computed (this is the Blender write path)
    for e in sess.sk.edges:
        assert np.allclose(R2[e.key], R1[e.key], atol=1e-4), e.key
    # joints end up within a centimetre of the solver's (rigid-body fit residual) ...
    assert np.linalg.norm(P2 - P1, axis=1).max() < 1.2e-2
    # ... and planted feet really stay planted (the dragged foot is allowed to move)
    ix = sess.sk.index
    for k in ("foot.L", "foot.R", "toe_end.L", "toe_end.R"):
        if label == "Foot R" and k.endswith("R"):
            continue
        assert np.linalg.norm(P2[ix[k]] - P0[ix[k]]) < 3e-3, k


def test_zero_drag_leaves_pose_untouched():
    obj = make_armature()
    sess, P0, P1, R0, R1 = run_drag(obj, "Hand L", (0, 0, 0))
    P2, _ = sess.read()
    assert np.allclose(P2, P0, atol=1e-3)


def test_works_with_object_transform():
    obj = make_armature()
    obj.location = (3, 2, 1)
    obj.rotation_euler = (0, 0, 1.2)
    obj.scale = (1.5, 1.5, 1.5)
    bpy.context.view_layer.update()
    sess, P0, P1, R0, R1 = run_drag(obj, "Hand R", (0.1, -0.3, 0.2))
    P2, _ = sess.read()
    assert np.linalg.norm(P2 - P1, axis=1).max() < 1.2e-2


def test_keys_only_changed_bones():
    obj = make_armature()
    rig.detect(obj)
    sess = rig.RigSession(obj)
    # a no-op drag changes nothing -> nothing to key
    before = sess.snapshot_rot()
    P0, R0 = sess.read()
    sess.write(solve_rotations(sess.sk, P0, P0, R0), hips_head=P0[sess.sk.index["hips"]])
    changed, hips_moved = sess.changed_bones(before)
    assert changed == [] and not hips_moved
    # a real drag changes the dragged arm and gets keyed on the current frame
    before = sess.snapshot_rot()
    P0, R0 = sess.read()
    target = P0[sess.sk.index["hand.L"]] + np.array([-0.1, -0.3, 0.2])
    P1 = Solver(sess.sk, P0).solve(rig.handle_targets(sess.sk, P0, "Hand L", target), PINS)
    sess.write(solve_rotations(sess.sk, P0, P1, R0), hips_head=P1[sess.sk.index["hips"]])
    changed, hips_moved = sess.changed_bones(before)
    assert "mixamorig:LeftArm" in changed and "mixamorig:LeftForeArm" in changed
    sess.key(changed, 5, hips_moved)
    act = obj.animation_data.action
    paths = {fc.data_path for fc in rig_fcurves(act)}
    assert 'pose.bones["mixamorig:LeftArm"].rotation_quaternion' in paths


def rig_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []  # Blender 5 layered actions
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


def test_detect_reports_incomplete_rig():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = bpy.data.armatures.new("A")
    obj = bpy.data.objects.new("O", arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones.new("Hips"); eb.head, eb.tail = (0, 0, 0), (0, 0, 1)
    bpy.ops.object.mode_set(mode="POSE")
    m = rig.detect(obj)
    assert not m.ok
    with pytest.raises(ValueError):
        rig.RigSession(obj)
