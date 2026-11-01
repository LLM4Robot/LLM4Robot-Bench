"""RGB and optional aligned depth from the same camera render products."""

import numpy as np

from ..robots.transforms import pose_transform, transform_pose
from ..types import DepthFrame, RGBFrame
from .profile import CameraRigProfile
from . import RGBNotReady
from .geometry import calibration_metadata


def _set_pose(prim, pose):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(pose[3], Gf.Vec3d(*pose[4:])))


def validate_existing_camera(stage, spec, parent):
    """Reject stale calibration or a wrist camera attached to the wrong link."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(spec.existing_prim_path)
    if not prim or not prim.IsA(UsdGeom.Camera):
        raise ValueError(f"{spec.name}: missing USD camera {spec.existing_prim_path}")
    if not spec.existing_prim_path.startswith(parent + "/"):
        raise ValueError(f"{spec.name}: camera must descend from {parent}")
    camera = UsdGeom.Camera(prim)
    if camera.GetProjectionAttr().Get() != "perspective" or any(spec.distortion):
        raise ValueError(f"{spec.name}: existing camera must use an undistorted perspective lens")
    w, h = spec.resolution
    focal = camera.GetFocalLengthAttr().Get()
    ha, va = camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()
    if not np.isclose(ha / va, w / h, rtol=1e-4):
        raise ValueError(f"{spec.name}: render resolution must preserve the USD camera aspect ratio")
    # USD aperture offsets shift the frustum. Pixel y points down while USD
    # camera y points up; keep this sign when exporting OpenCV intrinsics.
    ox, oy = camera.GetHorizontalApertureOffsetAttr().Get(), camera.GetVerticalApertureOffsetAttr().Get()
    intrinsic = (w * focal / ha, h * focal / va, w * (.5-ox/ha), h * (.5+oy/va))
    if not np.allclose(intrinsic, spec.intrinsics, rtol=1e-6, atol=1e-4):
        raise ValueError(f"{spec.name}: camera profile intrinsics differ from the supplied USD")
    cache = UsdGeom.XformCache()
    world = np.array(cache.GetLocalToWorldTransform(prim)).T
    base = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))).T
    if not np.allclose(np.linalg.inv(base) @ world, pose_transform(spec.optical_pose_wxyz), atol=1e-6):
        raise ValueError(f"{spec.name}: camera mount differs from the supplied USD")
    if not np.allclose(camera.GetClippingRangeAttr().Get(), spec.clipping_m, atol=1e-6):
        raise ValueError(f"{spec.name}: clipping range differs from the supplied USD")
    return str(prim.GetPath())


class IsaacCameraRig:
    def __init__(self, stage, rep, profile: CameraRigProfile, names: tuple[str, ...], *, include_depth=False):
        self.stage, self.rep, self.profile = stage, rep, profile
        self.specs = profile.select(names)
        self.paths, self.products, self.annotators, self.sensors = {}, {}, {}, {}
        self.include_depth = include_depth
        self.depth_annotators = {}

    def create(self, ee_paths: dict[str, str]) -> dict:
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom
        report = {"profile": str(self.profile.source), "calibration_status": self.profile.calibration_status,
                  "hardware_calibrated": False, "notes": self.profile.notes, "cameras": {}}
        for spec in self.specs:
            parent = "/World" if spec.parent == "world" else ee_paths[spec.parent]
            if spec.existing_prim_path:
                self.paths[spec.name] = validate_existing_camera(self.stage, spec, parent)
            else:
                root_path = f"{parent}/Sensors/{spec.name}"
                root = UsdGeom.Xform.Define(self.stage, root_path)
                _set_pose(root.GetPrim(), spec.optical_pose_wxyz)
                housing = UsdGeom.Xform.Define(self.stage, root_path + "/Housing")
                housing.GetPrim().GetReferences().AddReference(str(spec.asset))
                _set_pose(housing.GetPrim(), spec.asset_pose_wxyz)
                camera = UsdGeom.Camera.Define(self.stage, root_path + "/RGB")
                camera.CreateClippingRangeAttr(spec.clipping_m)
                camera.CreateFStopAttr(0.0)
                # USD camera looks down -Z. The optical root carries all mount transforms.
                sensor = Camera(str(camera.GetPath()), name=spec.name, resolution=spec.resolution)
                fx, fy, cx, cy = spec.intrinsics
                sensor.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=list(spec.distortion))
                self.sensors[spec.name] = sensor
                self.paths[spec.name] = str(camera.GetPath())
            product = self.rep.create.render_product(self.paths[spec.name], spec.resolution)
            self.products[spec.name] = product
            annotator = self.rep.AnnotatorRegistry.get_annotator("rgb")
            self.annotators[spec.name] = annotator
            annotator.attach(product)
            if self.include_depth:
                depth = self.rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                self.depth_annotators[spec.name] = depth
                depth.attach(product)
            report["cameras"][spec.name] = {**spec.metadata(), "parent_prim": parent,
                                           "camera_prim": self.paths[spec.name],
                                           "binding": "existing" if spec.existing_prim_path else "created",
                                           "depth_enabled": self.include_depth,
                                           "depth_source": "ideal_rendered_optical_axis_m" if self.include_depth else None,
                                           "hardware_depth_emulated": False}
        return report

    def read(self, timestamp: float, capture_id: int = 0) -> tuple[RGBFrame, ...]:
        frames = []
        poses = self.world_poses()
        for spec in self.specs:
            data = np.asarray(self.annotators[spec.name].get_data())
            if data.size == 0:
                raise RGBNotReady(f"{spec.name}: RGB buffer is not ready")
            width, height = spec.resolution
            if data.ndim != 3 or data.shape[:2] != (height, width) or data.shape[2] not in {3, 4}:
                raise RuntimeError(f"{spec.name}: invalid RGB dimensions {data.shape}")
            rgb = np.ascontiguousarray(data[..., :3], dtype=np.uint8)
            metadata = calibration_metadata(spec, poses[spec.name], timestamp, capture_id)
            depth_frame = None
            if self.include_depth:
                data = np.asarray(self.depth_annotators[spec.name].get_data())
                if data.size == 0:
                    raise RGBNotReady(f"{spec.name}: depth buffer is not ready")
                if data.shape == (height, width, 1):
                    data = data[..., 0]
                if data.shape != (height, width):
                    raise RuntimeError(f"{spec.name}: invalid depth dimensions {data.shape}")
                depth = np.array(data, dtype="<f4", order="C", copy=True)
                # Renderer background may be inf/0. Never turn it into a surface.
                depth[~np.isfinite(depth) | (depth <= 0) | (depth < spec.clipping_m[0])
                      | (depth > spec.clipping_m[1])] = np.nan
                depth_frame = DepthFrame(spec.name, timestamp, width, height, depth.tobytes(), capture_id)
            frames.append(RGBFrame(spec.name, timestamp, width, height, rgb.tobytes(), metadata, depth_frame))
        return tuple(frames)

    def world_poses(self) -> dict[str, tuple[float, ...]]:
        from pxr import Usd, UsdGeom
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return {name: transform_pose(np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))).T)
                for name, path in self.paths.items()}

    def close(self) -> None:
        for name, annotator in self.annotators.items():
            annotator.detach(self.products[name])
        for name, annotator in self.depth_annotators.items():
            annotator.detach(self.products[name])
        for product in self.products.values():
            product.destroy()
        for sensor in self.sensors.values():
            sensor.destroy()
        self.annotators.clear()
        self.depth_annotators.clear()
        self.products.clear()
        self.sensors.clear()
        self.paths.clear()
