import bpy

from . import rig
from . import ghosts, trail
from .operators import _STATE


class OPENRIG_PT_main(bpy.types.Panel):
    bl_label = "OpenRig"
    bl_idname = "OPENRIG_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRig"

    def draw(self, context):
        col = self.layout.column()
        obj = context.active_object
        p = context.scene.openrig
        if not obj or obj.type != "ARMATURE":
            col.label(text="Select a humanoid armature", icon="INFO")
            return
        box = col.box()
        mapped = rig.stored_map(obj)
        if mapped and mapped.ok:
            box.label(text=p.status or "Rig detected", icon="CHECKMARK")
        else:
            box.label(text=p.status or "Rig not detected", icon="ERROR")
        box.operator("openrig.detect", icon="ARMATURE_DATA")


class OPENRIG_PT_pose(bpy.types.Panel):
    bl_label = "Pose"
    bl_idname = "OPENRIG_PT_pose"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRig"
    bl_parent_id = "OPENRIG_PT_main"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == "ARMATURE" and rig.stored_map(obj) is not None

    def draw(self, context):
        col = self.layout.column(align=True)
        p = context.scene.openrig
        on = _STATE["running"]
        col.operator("openrig.pose_drag", text="Stop Handles" if on else "Pose Handles",
                     icon="HANDLETYPE_ALIGNED_VEC" if on else "TRANSFORM_ORIGINS", depress=on)
        row = col.row(align=True)
        row.prop(p, "pin_foot_l", text="Pin L Foot", toggle=True, icon="PINNED")
        row.prop(p, "pin_foot_r", text="Pin R Foot", toggle=True, icon="PINNED")
        col.prop(p, "stiffness")
        row = col.row(align=True)
        row.prop(p, "use_prior", text="Pose Prior", toggle=True, icon="OUTLINER_OB_ARMATURE")
        sub = row.row(align=True)
        sub.active = p.use_prior
        sub.prop(p, "prior_strength", text="")
        row = col.row(align=True)
        row.prop(p, "floor", toggle=True)
        row.prop(p, "auto_key", toggle=True)
        col.operator("openrig.key_pose", icon="KEYFRAME")


class OPENRIG_PT_fix(bpy.types.Panel):
    bl_label = "Polish"
    bl_idname = "OPENRIG_PT_fix"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRig"
    bl_parent_id = "OPENRIG_PT_main"
    poll = OPENRIG_PT_pose.poll

    def draw(self, context):
        lay = self.layout
        p = context.scene.openrig
        col = lay.column(align=True)
        col.prop(p, "use_scene_range")
        if not p.use_scene_range:
            row = col.row(align=True)
            row.prop(p, "frame_start")
            row.prop(p, "frame_end")
        lay.separator()
        col = lay.column(align=True)
        col.prop(p, "smooth_sigma")
        col.operator("openrig.smooth", icon="MOD_SMOOTH")
        lay.separator()
        col = lay.column(align=True)
        row = col.row(align=True)
        row.prop(p, "lock_side", expand=True)
        col.operator("openrig.lock_foot", icon="PINNED")
        lay.separator()
        col = lay.column(align=True)
        col.operator("openrig.motion_check", icon="VIEWZOOM")
        col.operator("openrig.fix_sliding", icon="MODIFIER")
        if p.report_text:
            box = lay.box()
            for line in p.report_text.split("\n")[:10]:
                box.label(text=line)


class OPENRIG_PT_trail(bpy.types.Panel):
    bl_label = "Motion Trail"
    bl_idname = "OPENRIG_PT_trail"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRig"
    bl_parent_id = "OPENRIG_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    poll = OPENRIG_PT_pose.poll

    def draw(self, context):
        col = self.layout.column(align=True)
        p = context.scene.openrig
        on = trail._T["running"]
        col.prop(p, "trail_handle", text="")
        col.prop(p, "trail_radius")
        col.operator("openrig.trail_edit", text="Stop Trail" if on else "Edit Trail", icon="CURVE_PATH", depress=on)
        col.label(text="Drag a dot to reshape. Alt+drag to retime.")


class OPENRIG_PT_ghosts(bpy.types.Panel):
    bl_label = "Ghost Frames"
    bl_idname = "OPENRIG_PT_ghosts"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRig"
    bl_parent_id = "OPENRIG_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    poll = OPENRIG_PT_pose.poll

    def draw(self, context):
        lay = self.layout
        p = context.scene.openrig
        col = lay.column(align=True)
        col.prop(p, "ghost_count")
        col.prop(p, "ghost_step")
        col.prop(p, "ghost_spread")
        row = col.row(align=True)
        row.operator("openrig.ghosts", text="Refresh" if ghosts.active() else "Show Ghosts", icon="ONIONSKIN_ON")
        row.operator("openrig.ghosts_clear", text="", icon="X")
        if ghosts.active() and ghosts._G["frames"]:
            grid = lay.grid_flow(columns=4, align=True)
            for f, _ in ghosts._G["frames"]:
                grid.operator("openrig.goto_frame", text=str(f)).frame = f


class OPENRIG_PT_map(bpy.types.Panel):
    bl_label = "Bone Map"
    bl_idname = "OPENRIG_PT_map"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRig"
    bl_parent_id = "OPENRIG_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def draw(self, context):
        lay = self.layout
        m = rig.stored_map(context.active_object, validate=False)
        if m is None:
            lay.label(text="Run Detect Rig first, or assign bones by hand.")
        else:
            for label, name in m.rows():
                row = lay.row()
                row.label(text=label)
                row.label(text=name, icon="BONE_DATA" if name != "-" else "BLANK1")
        lay.operator_menu_enum("openrig.assign_bone", "role", text="Assign Selected Bone(s) To...",
                               icon="BONE_DATA")


CLASSES = (OPENRIG_PT_main, OPENRIG_PT_pose, OPENRIG_PT_fix, OPENRIG_PT_trail, OPENRIG_PT_ghosts, OPENRIG_PT_map)
