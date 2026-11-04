import bpy
from .trail import HANDLE_ITEMS
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty


class OpenRigProps(bpy.types.PropertyGroup):
    pin_foot_l: BoolProperty(name="Pin Left Foot", default=True,
                             description="Keep the left foot planted while dragging other handles")
    pin_foot_r: BoolProperty(name="Pin Right Foot", default=True,
                             description="Keep the right foot planted while dragging other handles")
    stiffness: FloatProperty(name="Follow", default=1.0, min=0.1, max=3.0,
                             description="How strongly limbs are carried along by the torso. "
                                         "Lower = looser, floppier; higher = stiffer")
    floor: BoolProperty(name="Floor", default=False,
                        description="Stop free feet from sinking below the height they started at")
    use_prior: BoolProperty(name="Learned Pose Prior", default=True,
                            description="Nudge the body toward poses a real person could take, learned from "
                                        "motion capture. Off = geometry-only solver")
    prior_strength: FloatProperty(name="Prior Strength", default=0.8, min=0.0, max=1.0)
    auto_key: BoolProperty(name="Auto-Key", default=True,
                           description="Insert keyframes on the bones that changed when a drag ends")

    use_scene_range: BoolProperty(name="Scene Range", default=True)
    frame_start: IntProperty(name="Start", default=1)
    frame_end: IntProperty(name="End", default=100)
    smooth_sigma: FloatProperty(name="Strength", default=2.0, min=0.5, max=12.0,
                                description="Smoothing width in frames")
    lock_side: EnumProperty(name="Foot", items=[("L", "Left", ""), ("R", "Right", "")], default="L")

    trail_handle: EnumProperty(name="Handle", items=HANDLE_ITEMS, default="Hand L")
    trail_radius: IntProperty(name="Falloff", default=6, min=0, max=40,
                              description="Frames on each side of the dragged point that move with it")
    ghost_count: IntProperty(name="Count", default=3, min=1, max=8, description="Ghosts before and after")
    ghost_step: IntProperty(name="Step", default=4, min=1, max=60, description="Frames between ghosts")
    ghost_spread: FloatProperty(name="Spread", default=0.0, min=0.0, max=5.0, unit="LENGTH",
                                description="Pull ghosts apart sideways, like a strip of frames "
                                            "(0 = onion skin in place)")

    status: StringProperty(name="Status", default="")
    report_text: StringProperty(name="Report", default="")
