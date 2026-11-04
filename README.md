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
| `dist/openrig-0.3.0.zip` | Blender 3.6 to 5.x: *Preferences > Add-ons > Install...* |
| `dist/openrig-0.3.0-extension.zip` | Blender 4.2+: *Preferences > Get Extensions > Install from Disk* |

Then enable **OpenRig** and open the sidebar (`N`) > **OpenRig** tab.

## Use

1. Select your armature, click **Detect Rig**. It finds the hips, spine, head, arms and legs
   by name (Mixamo, Unreal mannequin, 3ds Max Biped, VRoid, Character Creator, Daz, Rigify, and
   generic `.L/.R` rigs). Finger and twist/IK/helper bones are ignored. If a bone is wrong or
   missing, open the **Bone Map** panel, select the bone(s) in Pose Mode and use
   *Assign Selected Bone(s) To...*.
2. Click **Pose Handles**. Circles appear on the hands, feet, hips and head.
   Drag one. Blue circles are pinned feet. `Esc` / `Enter` leaves the mode.
   `Esc` or right click during a drag cancels it.
3. Dragging ends with keyframes on the bones that changed (turn **Auto-Key** off to skip).

Settings: **Pin L/R Foot**, **Follow** (how stiffly the torso carries the limbs), **Pose Prior**
(the learned model below, with its strength),
**Floor** (stops free feet sinking below where they started).

### Polish tools (operate on the scene frame range or a custom range)

* **Smooth**: removes jitter from selected bones (or all detected bones) without moving
  keyframes in time.
* **Lock in Place**: holds the chosen foot where it is on the first frame of the range and
  re-solves the leg on every frame.
* **Motion Check**: finds sliding feet and popping bones and drops timeline markers named
  `OpenRig: ...`. **Fix Sliding Feet** runs the check and locks every sliding segment.

### Motion Trail

Pick a handle in the **Motion Trail** panel and click **Edit Trail**. The path of that handle
over the frame range is drawn in the viewport.

* **Drag a dot** to reshape the motion. The frames around it follow with a falloff (the
  *Falloff* setting) and the whole body is re-solved on each of them, with keys written.
* **Alt+drag a dot sideways** to retime: that moment of the motion happens earlier or later.
  Keys never cross each other.

Frames outside the edited span keep their original animation.

### Ghost Frames

**Show Ghosts** draws stick-figure ghosts of the frames before (blue) and after (orange) the
current frame. Raise *Spread* to pull them apart sideways like a strip of frames. Press the
frame numbers to jump to a ghost's frame and pose it with the handles. Ghosts are a snapshot:
press **Refresh** after moving the playhead or editing.

## How it works

`openrig/core` is pure Python + numpy with no Blender dependency, so it is unit tested directly.

* Joints are particles, bones are fixed-length edges.
* A position-based solver pulls the dragged joint to the cursor, keeps pinned joints fixed,
  keeps the pelvis and shoulder girdle rigid, carries unpinned limbs along with the torso,
  and places knees/elbows with exact two-bone IK on the side they were already bending.
* Solved joint positions become bone rotations that keep each bone's twist, so only normal
  rotation channels are written.

### Learned pose prior

On top of the solver sits a small neural network (a denoising autoencoder, about 45k weights,
numpy only, ships in `openrig/data/pose_prior.npz`). It looks at the directions of the main
bones in a torso frame, so it does not depend on the character's size or proportions, and
nudges the arms, legs and neck toward arrangements a person could actually take. Pins, targets
and bone lengths always win over it.

It was trained on the CMU Graphics Lab Motion Capture Database (2,548 clips, 113 subjects,
about 356k poses, mirrored for left/right balance). Honest numbers, measured on three subjects
the network never saw: given only where the hands, feet, hips and head are, how far are the
*other* joints from the real captured pose (as % of torso length, 200 poses, lower is better)?

| | geometry only | prior 0.6 | prior 1.0 |
|---|---|---|---|
| elbows | 15.5 | 14.1 | 13.7 |
| knees | 24.0 | 23.2 | 23.0 |
| shoulders | 13.5 | 13.0 | 12.8 |
| chest | 8.1 | 8.3 | 8.2 |
| head | 8.6 | 9.0 | 8.9 |
| mid-spine | 4.7 | 4.7 | 4.7 |
| **average** | **12.4** | **12.0** | **11.9** |

So it is a real but modest improvement (at full strength elbows about 12% closer, overall about
4%), not a transformation. Head accuracy is marginally worse, chest is unchanged, and knees
remain the weakest joint. The default strength is 0.8. Lock in
Place does not use the prior. Reproduce with `tools/build_dataset.py`, `tools/train_prior.py`
and `tools/eval_prior.py` (the dataset itself is not in this repo).

> Motion data used for training was obtained from mocap.cs.cmu.edu. The database was created
> with funding from NSF EIA-0196217.

## Known limits (v0.1)

* Humanoid rigs only.
* Bones with constraints (IK etc.) are not understood; mute them while using OpenRig. A warning is shown.
* The viewport interaction (Pose Handles, Motion Trail, ghost drawing) has **not** been run in
  a real Blender UI session in development. The solving, keying, trail editing, retiming and
  ghost sampling behind them are covered by headless tests; the mouse handling and drawing
  are not.
* Ghost frames are a drawn snapshot, not live, and are not themselves draggable.
* The learned prior is modest (see the table above) and was trained on one dataset of mostly
  everyday and sports motion, so unusual poses get little help from it.

## Develop

```
pip install numpy pytest bpy      # bpy wheel lets the integration tests run headless
cd tests && python -m pytest
```

`PLAN.md` has the architecture and roadmap.

## License

GPL-3.0-or-later (see `LICENSE`). Blender requires a GPL-compatible licence for add-ons that
use `bpy`. Change it if you prefer another compatible licence.
