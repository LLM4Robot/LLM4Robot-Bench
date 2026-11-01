"""Engineering motion fixture for L2-03; never imported by a task/policy.

Uses measured cup-to-hand geometry to qualify physical reach, grip, pouring and
release through the production planner. It does NOT test a model's perception
or decisions. No cup pose writes, grasp attachments, disabled collisions or
scripted benchmark results. l2_execution_probe owns lifecycle and recording.
"""

import math

import numpy as np

from ...robots.transforms import axis_rotation, pose_transform, transform_pose
from ...tasks.bench.native_geometry import cup_cavity
from ...types import Action, ArmCommand


def finish_pour(backend, robot, task, config, execute, capture, assess):
    arm = config.arms[0]
    rows = task.scene.measure()
    cup = pose_transform(rows["source_cup"]["pose"])
    tcp = pose_transform(robot.state()[arm].tcp_pose_world_wxyz)
    cup_to_tcp = np.linalg.inv(cup) @ tcp
    upright_rotation = cup[:3, :3].copy()
    receiver = np.asarray(rows["receiver_cup"]["pose"][:3])
    initial = np.asarray(task.scene.initial["source_cup"][:3])
    direction = receiver - initial
    direction[2] = 0
    direction /= np.linalg.norm(direction)
    tilt_axis = np.cross([0, 0, 1], direction)
    _, _, rim = cup_cavity()

    def wait(steps):
        for _ in range(steps):
            backend.step(config.control_dt_s)
            assess()

    def move_cup(target, label):
        execute(Action((ArmCommand(arm, "cartesian_pose",
                                   transform_pose(target @ cup_to_tcp), frame="world"),),
                       config.control_dt_s))
        capture(label)

    # Establish generous clearance before approaching the fixed receiver.
    raised = cup.copy()
    raised[2, 3] = receiver[2] + .23
    move_cup(raised, "raised_clear")

    def tilted(degrees):
        target = np.eye(4)
        target[:3, :3] = axis_rotation(tilt_axis, math.radians(degrees)) @ upright_rotation
        # Move the mouth over the opening as it tilts; never rotate at a low,
        # fixed TCP. Heights are diagnostic choices, not production waypoints.
        # Forward bead velocity needs an upstream mouth, not centre-over-centre.
        mouth = receiver + (-.15 + .075 * min(degrees / 90, 1)) * direction + [0, 0, .24]
        target[:3, 3] = mouth - target[:3, :3] @ [0, 0, rim]
        return target

    for degrees in (0, 20, 40, 60, 75, 90, 105, 120):
        move_cup(tilted(degrees), f"pour_{degrees:03}")
        wait(6)
    wait(24)
    capture("drained")

    for degrees in (90, 60, 30, 0):
        move_cup(tilted(degrees), f"restore_{degrees:03}")
    returned = tilted(0)
    returned[:2, 3] = initial[:2]
    move_cup(returned, "return_above_table")
    returned[2, 3] = initial[2] + .003
    move_cup(returned, "lower_to_table")
    execute(Action((ArmCommand(arm, "gripper", (.070,)),), config.control_dt_s))
    capture("released")
    # Side withdrawal avoids catching the wider upper wall with a 70 mm jaw.
    retreat = list(robot.state()[arm].tcp_pose_world_wxyz)
    retreat[1] += .10
    execute(Action((ArmCommand(arm, "cartesian_pose", tuple(retreat), frame="world"),),
                   config.control_dt_s))
    wait(12)
    capture("final")
