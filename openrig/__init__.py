bl_info = {
    "name": "OpenRig",
    "author": "OpenRig contributors",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > OpenRig",
    "description": "Drag a hand, foot, hips or head and the whole humanoid body follows. "
                   "Smooth, lock feet and check motion. Offline, writes plain FK keys.",
    "category": "Animation",
}

try:
    import bpy  # noqa: F401
except ImportError:  # core/ is importable (and tested) without Blender
    bpy = None

if bpy is not None:
    from . import operators, props, ui

    _classes = (props.OpenRigProps, *operators.CLASSES, *ui.CLASSES)

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        bpy.types.Scene.openrig = bpy.props.PointerProperty(type=props.OpenRigProps)

    def unregister():
        operators._STATE["running"] = False
        if operators._STATE.get("handler"):
            bpy.types.SpaceView3D.draw_handler_remove(operators._STATE["handler"], "WINDOW")
            operators._STATE["handler"] = None
        del bpy.types.Scene.openrig
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)
