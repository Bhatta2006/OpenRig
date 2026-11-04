# OpenRig — plan

Goal: a Blender add-on that lets you pose and fix animation on a humanoid rig by
dragging a hand, foot, hips or head. The rest of the body follows plausibly and
planted feet hold the floor. No generation: you set poses, the add-on solves the
rest. Offline, CPU only, writes plain FK keys on the user's own rig.

This is an independent implementation. It does not copy any other product's code,
models, assets or branding.

## Architecture

```
openrig/                  <- the Blender add-on (zip this folder)
  core/                   <- pure Python + numpy, no bpy, unit-testable
    mapping.py            bone-name -> humanoid role mapping (Mixamo, UE5, Rigify,
                          Biped, VRoid, Daz/CC, generic L/R names)
    skeleton.py           roles -> particles/edges/constraint groups
    solver.py             full-body "pull" solver (position-based, see below)
    rotations.py          swing rotations, propagating parent deltas
    filters.py            smoothing + foot-contact detection
  rig.py                  bpy glue: read pose, write FK keys, armature-space maths
  operators.py            Setup, Drag Handles (modal), Key, Pins, Lock, Smooth, Check
  ui.py                   N-panel + viewport handle drawing
tests/                    pytest on core (no Blender) + bpy integration test
build.py                  produces dist/openrig.zip (installable add-on)
```

## Solver (phase 1, no ML)

Joints are particles, bones are edges. Per drag event:

1. Effector (dragged joint) is pulled toward the cursor target.
2. Pinned joints (feet by default, plus anything the user pins) are fixed.
3. Rigid groups (pelvis, shoulder girdle) keep their shape.
4. Everything else is pulled toward "follow the torso" with per-joint stiffness,
   so unpinned limbs swing with the body instead of trailing behind.
5. Bone lengths are enforced last, so bones never stretch.
6. Knee/elbow hinge rule stops hyperextension and flipping.
7. Particle positions are turned into bone rotations by swing rotations that
   preserve each bone's twist, then written as FK keys.

Phase 2 can replace steps 3-4 with a learned pose prior behind the same interface.

## Milestones

| # | Deliverable | Status |
|---|---|---|
| 1 | Bone mapping for common humanoid rigs | this PR |
| 2 | Pull solver + tests | this PR |
| 3 | Modal drag handles, foot pins, keyframing, undo | this PR |
| 4 | Smooth (filter selected curves), Lock in Place, Motion Check | this PR |
| 5 | Motion Trail (edit paths, retime) | later |
| 6 | Zoetrope (poseable ghost frames) | later |
| 7 | Learned pose prior (needs mocap with a clear license) | later |

## Testing

- `core/` is tested with pytest and synthetic skeletons (bone lengths preserved, feet
  stay put, body follows, no knee hyperextension, rotation round-trips).
- Integration test builds a rig in headless `bpy`, runs the solver, checks the pose.
- I cannot test interactive viewport behaviour here; that needs a real Blender session.
