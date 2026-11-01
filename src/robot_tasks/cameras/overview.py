"""Shared overview pose and optics; wrist cameras remain in rig profiles."""

from dataclasses import replace

import numpy as np

from ..robots.transforms import transform_pose


def resolve_view(settings, config):
    position = np.asarray(settings.camera_position, dtype=float)
    target = np.asarray(settings.camera_target, dtype=float)
    if np.linalg.norm(target - position) < 1e-8:
        raise ValueError("Overview camera position and target must differ")
    if settings.camera_reference_frame == "tuojing_lab" and settings.environment != "tuojing_lab_v4":
        from ..backends.scene_geometry import bench_geometry

        bench = bench_geometry(config)
        # Preserve offset from the table's front edge and the robot centreline.
        # Lab +Y points into the table; legacy -X points into the table.
        source = np.array([0.1956512155, 0.7376504171222011, 0.78])
        destination = np.array([bench.centre[0] + bench.size[0] / 2,
                                bench.centre[1], bench.top])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        position = destination + rotation @ (position - source)
        target = destination + rotation @ (target - source)
    return tuple(position), tuple(target)


def override_overview(profile, settings):
    eye, target = np.asarray(settings.camera_position), np.asarray(settings.camera_target)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("Overview look-at direction must not be parallel to world up")
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, np.cross(right, forward), -forward))
    matrix[:3, 3] = eye
    source = (f"Shared simulation.camera_position/target ({settings.camera_reference_frame} reference); "
              f"effective world eye={eye.tolist()}, target={target.tolist()}. Not hardware calibrated.")
    cameras = []
    for camera in profile.cameras:
        if camera.name == "overview":
            if camera.parent != "world":
                raise ValueError("Shared overview must be attached to world")
            camera = replace(camera, optical_pose_wxyz=transform_pose(matrix), pose_source=source)
            if settings.camera_resolution is not None:
                camera = replace(
                    camera,
                    resolution=settings.camera_resolution,
                    intrinsics=settings.camera_intrinsics,
                    distortion=(0.0,) * 5,
                    intrinsics_source="Shared simulation.camera_resolution/intrinsics; video-matched pinhole estimate, not hardware calibrated.",
                )
        cameras.append(camera)
    if not any(c.name == "overview" for c in cameras):
        raise ValueError("Camera rig has no overview camera")
    optics = "Overview optics follow the shared configuration." if settings.camera_resolution is not None else "Overview optics follow the rig profile."
    return replace(profile, cameras=tuple(cameras), notes=source + " " + optics + " Wrists follow the rig profile.")
