"""Initialize an independent GUI camera from the actual overview sensor."""

VIEW_CAMERA_PATH = "/World/GUIView"


def initialize_overview_view(stage, overview_path):
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Usd, UsdGeom

    source = UsdGeom.Camera(stage.GetPrimAtPath(overview_path))
    if not source:
        raise ValueError(f"Overview is not a camera: {overview_path}")
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("GUI started without an active viewport")
    # Copy values, not a USD reference: mouse navigation must never modify the
    # calibrated sensor or its robot/camera mount. Sample its world transform
    # after the camera rig and environment mount overrides have been applied.
    viewer = UsdGeom.Camera.Define(stage, VIEW_CAMERA_PATH)
    viewer.ClearXformOpOrder()
    viewer.AddTransformOp().Set(
        UsdGeom.XformCache().GetLocalToWorldTransform(source.GetPrim()))
    viewer.SetResetXformStack(True)
    for attribute in source.GetPrim().GetAttributes():
        if attribute.GetName() in UsdGeom.Camera.GetSchemaAttributeNames(False):
            value = attribute.Get(Usd.TimeCode.Default())
            if value is not None:
                viewer.GetPrim().GetAttribute(attribute.GetName()).Set(value)
    viewport.camera_path = VIEW_CAMERA_PATH
    return {"source_camera": overview_path, "viewer_camera": VIEW_CAMERA_PATH,
            "initialization": "copy_world_pose_and_lens_once", "sensor_linked": False}
