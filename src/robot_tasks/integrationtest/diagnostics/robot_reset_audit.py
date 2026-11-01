"""Read actual PhysX base/joint/TCP state after a passive reset, without motion."""

from dataclasses import asdict
import numpy as np

from ...robots.transforms import pose_transform


def pose_error(actual, expected):
    a, b = pose_transform(actual), pose_transform(expected)
    angle = np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))
    return {"translation_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            "rotation_rad": float(angle)}


def capture(backend, robot):
    from isaacsim.core.prims import SingleRigidPrim

    rows = {}
    states = robot.state()
    for mount in robot.config.robot.mounts:
        base = SingleRigidPrim(mount.prim_path + "/base_link", reset_xform_properties=False)
        base.initialize()
        position, rotation = base.get_world_pose()
        actual_base = tuple(float(x) for x in (*position, *rotation))
        state = states[mount.name]
        expected_tcp = robot.kinematics.forward(state.joint_positions_rad, mount.name)
        target = robot.initial_targets[mount.name]
        joint_error = np.asarray(state.joint_positions_rad) - np.asarray(target[:6])
        finger_error = np.asarray(state.finger_positions_m) - np.asarray(target[6:])
        base_error = pose_error(actual_base, mount.base_pose_wxyz)
        tcp_error = pose_error(state.tcp_pose_world_wxyz, expected_tcp)
        rows[mount.name] = {
            "state": asdict(state), "actual_base_pose_wxyz": actual_base,
            "configured_base_pose_wxyz": mount.base_pose_wxyz,
            "commanded_initial_joints_rad": target[:6], "commanded_initial_fingers_m": target[6:],
            "joint_error_rad": joint_error.tolist(), "finger_error_m": finger_error.tolist(),
            "base_error": base_error, "tcp_fk_error": tcp_error,
            "passed": bool(max(abs(joint_error)) < .01 and max(abs(finger_error)) < .001
                           and base_error['translation_m'] < .0001 and base_error['rotation_rad'] < .001
                           and tcp_error['translation_m'] < .001 and tcp_error['rotation_rad'] < .005),
        }
    return {"simulation_time_s": backend.time_s(), "passed": all(r['passed'] for r in rows.values()),
            "scope": "PhysX reset versus configured simulation profile and independent URDF FK; not measured hardware joint alignment",
            "tolerances": {"joints_rad": .01, "fingers_m": .001, "base_m": .0001,
                           "base_rad": .001, "tcp_fk_m": .001, "tcp_fk_rad": .005},
            "arms": rows}
