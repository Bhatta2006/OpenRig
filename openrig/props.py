import bpy
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
    auto_key: BoolProperty(name="Auto-Key", default=True,
                           description="Insert keyframes on the bones that changed when a drag ends")

    use_scene_range: BoolProperty(name="Scene Range", default=True)
    frame_start: IntProperty(name="Start", default=1)
    frame_end: IntProperty(name="End", default=100)
    smooth_sigma: FloatProperty(name="Strength", default=2.0, min=0.5, max=12.0,
                                description="Smoothing width in frames")
    lock_side: EnumProperty(name="Foot", items=[("L", "Left", ""), ("R", "Right", "")], default="L")

    status: StringProperty(name="Status", default="")
    report_text: StringProperty(name="Report", default="")
