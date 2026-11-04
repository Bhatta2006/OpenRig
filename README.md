# OpenRig

A Blender add-on for posing humanoid characters by dragging one handle. Move a hand, foot,
hips or head and the rest of the body follows while planted feet stay on the floor. It
writes plain FK keyframes on **your own rig** (nothing is added to it) and runs offline on
the CPU.

You still set the poses. OpenRig solves the rest of the body so you don't have to pose every
bone, and gives you tools to clean up the result.

## Install

Build the zips, then install one in Blender:

```
python3 build.py
```

| File | Where |
|---|---|
| `dist/openrig-0.1.0.zip` | Blender 3.6 to 5.x: *Preferences > Add-ons > Install...* |
| `dist/openrig-0.1.0-extension.zip` | Blender 4.2+: *Preferences > Get Extensions > Install from Disk* |

Then enable **OpenRig** and open the sidebar (`N`) > **OpenRig** tab.

## Use

1. Select your armature, click **Detect Rig**. It finds the hips, spine, head, arms and legs
   by name (Mixamo, Unreal mannequin, 3ds Max Biped, VRoid, Character Creator, Daz, and generic
   `.L/.R` rigs). Finger and twist/IK/helper bones are ignored.
2. Click **Pose Handles**. Circles appear on the hands, feet, hips and head.
   Drag one. Blue circles are pinned feet. `Esc` / `Enter` leaves the mode.
   `Esc` or right click during a drag cancels it.
3. Dragging ends with keyframes on the bones that changed (turn **Auto-Key** off to skip).

Settings: **Pin L/R Foot**, **Follow** (how stiffly the torso carries the limbs),
**Floor** (stops free feet sinking below where they started).

### Polish tools (operate on the scene frame range or a custom range)

* **Smooth**: removes jitter from selected bones (or all detected bones) without moving
  keyframes in time.
* **Lock in Place**: holds the chosen foot where it is on the first frame of the range and
  re-solves the leg on every frame.
* **Motion Check**: finds sliding feet and popping bones and drops timeline markers named
  `OR ...`. **Fix Sliding Feet** runs the check and locks every sliding segment.

## How it works

`openrig/core` is pure Python + numpy with no Blender dependency, so it is unit tested directly.

* Joints are particles, bones are fixed-length edges.
* A position-based solver pulls the dragged joint to the cursor, keeps pinned joints fixed,
  keeps the pelvis and shoulder girdle rigid, carries unpinned limbs along with the torso,
  and places knees/elbows with exact two-bone IK on the side they were already bending.
* Solved joint positions become bone rotations that keep each bone's twist, so only normal
  rotation channels are written.

This is a hand-built solver, not a trained model, so very large or unusual poses can look
mechanical. A learned pose prior could replace the follow step later behind the same interface.

## Known limits (v0.1)

* Humanoid rigs only. Rigify rigs are not recognised yet. Rename or map manually if needed.
* Bones with constraints (IK etc.) are not understood; mute them while using OpenRig. A warning is shown.
* The viewport interaction (the modal drag and handle drawing) has **not** been run in a real
  Blender UI session in development; everything else is covered by headless tests.
* Not implemented yet: Motion Trail and Zoetrope style tools, manual bone-map editing UI.

## Develop

```
pip install numpy pytest bpy      # bpy wheel lets the integration tests run headless
cd tests && python -m pytest
```

`PLAN.md` has the architecture and roadmap.

## License

`blender_manifest.toml` declares `SPDX:GPL-3.0-or-later`, which Blender requires for add-ons
that use `bpy`. Confirm this is the licence you want and add the LICENSE file before publishing.
