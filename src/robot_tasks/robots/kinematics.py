"""URDF-derived R5 forward kinematics and coordinate transforms; no inverse solver."""

import numpy as np

from .profile import RobotProfile
from .transforms import (axis_rotation, finite_vector, origin_transform,
                         pose_transform, transform_pose)
from .urdf import read_urdf


class R5Kinematics:
    position_tolerance_m = 1e-4
    orientation_tolerance_rad = 1e-3

    def __init__(self, profile: RobotProfile):
        if profile.tcp_pose_wxyz is None or not profile.tcp_source:
            raise ValueError("Cartesian control requires an explicitly defined TCP and source")
        self.profile = profile
        description = read_urdf(profile.asset_path, dict(profile.package_roots))
        if description.chain(profile.base_link, profile.ee_link) != profile.joint_names:
            raise ValueError("IK joint order must match the URDF chain")
        by_child = {joint.child: joint for joint in description.joints}
        chain, tip = [], profile.ee_link
        while tip != profile.base_link:
            joint = by_child[tip]
            if joint.kind not in {"fixed", "revolute", "continuous"} or joint.mimic is not None:
                raise ValueError("R5 IK supports independent rotational arm joints and fixed links")
            chain.append(joint)
            tip = joint.parent
        self.chain = tuple(reversed(chain))
        self.origins = tuple(origin_transform(j.origin_xyz, j.origin_rpy) for j in self.chain)
        self.tcp = pose_transform(profile.tcp_pose_wxyz)
        self.bases = {mount.name: pose_transform(mount.base_pose_wxyz) for mount in profile.mounts
                      if mount.base_pose_wxyz is not None}
        limits = {bound.joint: bound for bound in profile.operating_limits}
        if not set(profile.joint_names).issubset(limits):
            raise ValueError("IK requires configured arm operating limits")
        self.lower = np.array([limits[name].lower for name in profile.joint_names])
        self.upper = np.array([limits[name].upper for name in profile.joint_names])
        self.reach_bound_m = sum(np.linalg.norm(t[:3, 3]) for t in self.origins) + np.linalg.norm(self.tcp[:3, 3])

    def _forward_transform(self, positions: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        index = 0
        for joint, origin in zip(self.chain, self.origins):
            transform = transform @ origin
            if joint.kind != "fixed":
                axis = np.asarray(joint.axis) / np.linalg.norm(joint.axis)
                motion = np.eye(4)
                motion[:3, :3] = axis_rotation(axis, positions[index])
                transform = transform @ motion
                index += 1
        return transform @ self.tcp

    def forward(self, positions, arm: str, frame: str = "world") -> tuple[float, ...]:
        q = finite_vector(positions, len(self.profile.joint_names), "Joint positions")
        transform = self._forward_transform(q)
        self._check_frame(arm, frame)
        return transform_pose(self.bases[arm] @ transform if frame == "world" else transform)

    def _check_frame(self, arm: str, frame: str) -> None:
        if frame not in {"world", "robot_base"}:
            raise ValueError("Cartesian frame must be world or robot_base")
        if arm not in self.bases:
            raise ValueError(f"No configured base pose for arm: {arm}")

    def world_pose(self, pose, arm: str, frame: str) -> tuple[float, ...]:
        self._check_frame(arm, frame)
        target = pose_transform(pose)
        return transform_pose(self.bases[arm] @ target if frame == "robot_base" else target)
