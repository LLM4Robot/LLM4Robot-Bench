from ..config import ExecutionSettings, TaskConfig
from ..interfaces import Backend
from ..types import Action, ArmState, Feedback, InvalidSimulationState
from .audit import require_joint_sim_ready
from .control import (
    ARM_JOINT_TRACKING_TOLERANCE_RAD,
    bounded_measured_joints,
    empty_close_tolerance,
    joints_to_width,
    plan_targets,
    width_to_joints,
)
from .settling import SettlingWindow


class RobotAdapter:
    def initialize(self, backend: Backend, config: TaskConfig) -> None:
        raise NotImplementedError(
            "Load robot asset, joints, limits, gripper and controllers"
        )

    def reset(self, seed: int) -> None:
        # Initial-state assignment is allowed during reset only.
        raise NotImplementedError("Robot initial configuration")

    def apply(self, action: Action) -> None:
        # TODO: validate limits, resolve frames; use IK for Cartesian targets.
        # Apply drive targets/forces, NEVER teleport objects to manufacture success.
        # For two arms, apply both targets before stepping the shared world.
        raise NotImplementedError("Action -> joint/Cartesian/gripper controller")

    def state(self) -> dict[str, ArmState]:
        raise NotImplementedError("Read robot state")

    def feedback(self) -> Feedback:
        raise NotImplementedError("Execution completion/error feedback")


class R5Adapter(RobotAdapter):
    """Dual-arm R5 SIMULATION adapter; never connects to the real robot SDK.

    The manufacturer description and names are configured in config.robot.
    Cartesian goals use the shared cuRobo planner followed by bounded drive tracking.
    Gripper completion describes execution, never evidence of task success.
    """

    def __init__(self) -> None:
        self.backend = self.config = None
        self.instances = {}
        self.indices = {}
        self.targets = {}
        self.initial_targets = {}
        self.drive_targets = {}
        self.kinematics = None
        self.cartesian_targets = {}
        self.rejection = None
        self.pending_grippers = {}
        self.motion_arms = set()
        self.motion_stable_s = 0.0
        self.execution = ExecutionSettings()
        self.planner = None
        self.paths = {}
        self.path_time_s = 0.0
        self.planning_diagnostics = {}
        self.last_plan = None
        self.grip_elapsed = {}
        self.grip_durations = {}
        self.grip_closing = {}
        self.grip_arms = set()
        self.stable_grips = {}
        self.settling = SettlingWindow()
        self.tracking_time_s = 0.0
        self.collision_stop = None
        self.gravity_compensation = False
        self.cartesian_completion_tolerance_m = 0.001

    def initialize(self, backend: Backend, config: TaskConfig) -> None:
        from ..backends.isaacsim import IsaacSimBackend

        require_joint_sim_ready(config.robot)
        if not isinstance(backend, IsaacSimBackend):
            raise TypeError(
                "R5Adapter requires IsaacSimBackend; it never connects to real hardware"
            )
        self.config, self.backend = config, backend
        self.cartesian_completion_tolerance_m = backend.settings.cartesian_completion_tolerance_m
        self.execution = ExecutionSettings.from_config(config)
        from .kinematics import R5Kinematics
        from ..planning.client import PlannerClient

        self.kinematics = R5Kinematics(config.robot)
        self.instances = backend.add_r5(config.robot)
        self.planner = PlannerClient(
            config.robot,
            backend.settings.asset_directory / "curobo",
            log_directory=getattr(backend, "run_directory", None),
            attachments=backend.wrist_geometry,
        )
        backend.scene_report["ik_solver"] = "curobo"
        backend.scene_report["motion_planner"] = self.planner.metadata
        backend.before_physics_step = self.before_physics_step

    def reset(self, seed: int) -> None:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        profile = self.config.robot
        self.stable_grips.clear()
        self.settling = SettlingWindow()
        self.tracking_time_s = 0.0
        self.collision_stop = None
        self.paths.clear()
        self.drive_targets.clear()
        self.grip_elapsed.clear()
        self.grip_durations.clear()
        self.grip_closing.clear()
        self.grip_arms.clear()
        self.path_time_s = 0.0
        self.planning_diagnostics = {}
        self.last_plan = None
        self.cartesian_targets.clear()
        self.rejection = None
        self.pending_grippers.clear()
        self.motion_arms.clear()
        self.motion_stable_s = 0.0
        self.gravity_compensation = self.backend.settings.robot_gravity_compensation
        self.backend.scene_report["arm_tracking"] = {
            "gravity_compensation": ("live generalized gravity forces on six arm joints only"
                                     if self.gravity_compensation else "disabled by simulation profile"),
            "cartesian_completion_tolerance_m": self.backend.settings.cartesian_completion_tolerance_m,
            "gravity_retained_for_robot_and_objects": True,
        }
        ordered = profile.joint_names + profile.gripper.joint_names
        limits = {limit.joint: limit for limit in profile.operating_limits}
        from ..video_delivery import initial_gripper_width

        self.backend.scene_report["gripper_execution"] = {
            "protocol": "rate_limited_drive_reference_v1",
            "joint_reference_speed_m_s": {n: limits[n].velocity for n in profile.gripper.joint_names},
            "timeout": "hold_last_applied_reference",
            "scope": "Limits commanded reference speed; contact impulses can cause measured velocity transients.",
        }
        for mount in profile.mounts:
            width = initial_gripper_width(self.config, mount.name,
                                          sum(profile.gripper.width_limits_m) / 2)
            arm = self.instances[mount.name]
            actual = tuple(arm.dof_names)
            if len(actual) != len(ordered) or set(actual) != set(ordered):
                raise ValueError(
                    f"{mount.name}: USD joint names do not match profile: {actual}"
                )
            indices = np.array([actual.index(name) for name in ordered], dtype=np.int32)
            self.indices[mount.name] = indices
            target = tuple(mount.home_positions_rad) + width_to_joints(profile, width)
            self.targets[mount.name] = self.initial_targets[mount.name] = target
            self.drive_targets[mount.name] = target
            settings = self.backend.settings
            controller = arm.get_articulation_controller()
            stiffness, damping = np.empty(8), np.empty(8)
            stiffness[indices] = [settings.arm_stiffness] * 6 + [
                settings.finger_stiffness
            ] * 2
            damping[indices] = [settings.arm_damping] * 6 + [
                settings.finger_damping
            ] * 2
            controller.set_gains(kps=stiffness, kds=damping)
            controller.set_max_efforts(
                np.array([min(limits[name].effort, settings.finger_effort_limit)
                          if settings.stable_gripper_drive and name in profile.gripper.joint_names
                          else limits[name].effort for name in ordered]),
                joint_indices=indices,
            )
            arm.set_joint_positions(np.array(target), joint_indices=indices)
            arm.set_joint_velocities(np.zeros(8), joint_indices=indices)
            arm.apply_action(
                ArticulationAction(
                    joint_positions=np.array(target), joint_indices=indices
                )
            )
            self.backend.scene_report["arms"][mount.name]["dof_names"] = actual
            if settings.stable_gripper_drive:
                # Read the physics handles, not just the authored TOML/USD values.
                properties = arm.dof_properties
                actual_fingers = []
                for joint_name in profile.gripper.joint_names:
                    row = properties[actual.index(joint_name)]
                    limit = limits[joint_name]
                    actual_fingers.append({"joint": joint_name, "lower": float(row["lower"]),
                        "upper": float(row["upper"]), "max_velocity_m_s": float(row["maxVelocity"]),
                        "max_effort_n": float(row["maxEffort"])})
                    if not (np.isclose(row["maxVelocity"], limit.velocity)
                            and np.isclose(row["lower"], limit.lower)
                            and np.isclose(row["upper"], limit.upper)
                            and np.isclose(row["maxEffort"], min(limit.effort, settings.finger_effort_limit))):
                        raise RuntimeError("Physical gripper limits did not match configured limits: " + joint_name)
                self.backend.scene_report["arms"][mount.name]["actual_finger_limits"] = actual_fingers
                from .gripper_drive import SymmetricGripDrive
                finger_limits = [limits[n] for n in profile.gripper.joint_names]
                self.stable_grips[mount.name] = SymmetricGripDrive(
                    sum(target[6:]) / 2, min(v.velocity for v in finger_limits),
                    settings.finger_max_acceleration, settings.finger_closing_lead,
                    max(v.lower for v in finger_limits), min(v.upper for v in finger_limits))
                self.backend.scene_report["gripper_execution"].update(
                    protocol="symmetric_bounded_lead_v2", synchronization="software common reference; no mechanical coupling claim",
                    max_acceleration_m_s2=settings.finger_max_acceleration,
                    closing_lead_m=settings.finger_closing_lead, effort_limit_n=settings.finger_effort_limit,
                    timeout="hold current bounded reference; release decisions remain with policy")

    def apply(self, action: Action) -> None:
        if not self.targets:
            raise RuntimeError("Reset the R5 robot before applying commands")
        try:
            if not isinstance(action, Action):
                raise ValueError("Expected an Action")
            # Validate executable fields before solving and changing drive targets.
            from ..runner import validate_action

            validate_action(action, self.config, require_control_duration=False)
            if not action.commands:
                plan_targets(
                    self.config.robot, self.config.command_arms, action, self.targets
                )
                # This new wait was accepted. Report current execution feedback,
                # not the rejection of a previous decision or its old plan.
                self.rejection = None
                self.last_plan = None
                return
            states = self.state()
            # Validate all non-Cartesian fields before calling the planner.
            from dataclasses import replace

            validated = replace(
                action,
                commands=tuple(
                    c for c in action.commands if c.kind != "cartesian_pose"
                ),
            )
            planned = plan_targets(
                self.config.robot, self.config.command_arms, validated, self.targets
            )
            motion = tuple(c for c in action.commands if c.kind != "gripper")
            paths = {}
            diagnostics = {}
            if motion:
                from ..planning.scene import snapshot
                from ..planning.trajectory import JointPath

                reply = self.planner.plan(action, states, snapshot(self.backend))
                for arm, data in reply["paths"].items():
                    limits = [
                        next(
                            x
                            for x in self.config.robot.operating_limits
                            if x.joint == name
                        )
                        for name in self.config.robot.joint_names
                    ]
                    paths[arm] = JointPath.from_timed(
                        data["positions"], data["times_s"], states[arm].joint_positions_rad, limits
                    )
                    # Preserve the exact time law already checked for BOTH arms.
                    planned[arm] = paths[arm].positions[-1] + planned[arm][6:]
                if set(paths) != {c.arm for c in motion}:
                    raise ValueError("Planner did not return every requested arm")
                if (
                    max(p.duration_s for p in paths.values())
                    > (
                        self.execution.max_action_steps
                        - self.execution.completion_samples
                    )
                    * self.config.control_dt_s
                    - 1.0
                ):
                    raise ValueError(
                        "Planned motion exceeds per-action budget; use a closer target"
                    )
                diagnostics = reply["diagnostics"]
            cartesian_targets = dict(self.cartesian_targets)
            for command in action.commands:
                if command.kind == "cartesian_pose":
                    cartesian_targets[command.arm] = self.kinematics.world_pose(
                        command.values, command.arm, command.frame
                    )
                elif command.kind == "joint_position":
                    cartesian_targets.pop(command.arm, None)
        except ValueError as exc:
            # Retain previous targets for BOTH arms, expose feedback, and propagate the error.
            metrics = {
                "reason_code": "invalid_target",
                "requested_commands": [
                    {"arm": c.arm, "kind": c.kind, "values": c.values, "frame": c.frame}
                    for c in getattr(action, "commands", ())
                ],
            }
            from ..planning.failures import PlanningFailure

            if isinstance(exc, PlanningFailure):
                metrics["reason_code"] = exc.diagnostic["code"]
                metrics["planning_failure"] = exc.diagnostic
            self.rejection = Feedback("rejected", str(exc), metrics)
            raise
        self.collision_stop = None
        monitor = getattr(self.backend, "collision_monitor", None)
        release_arm = None
        if (monitor is not None and getattr(monitor, "allow_actor_recovery", False)
                and monitor.blocked and len(action.commands) == 1):
            command = action.commands[0]
            measured = states[command.arm].finger_positions_m
            goal = planned[command.arm][6:]
            if (command.kind == 'gripper' and monitor.blocked.get('arm') == command.arm
                    and all(target >= actual - 1e-5 for target, actual in zip(goal, measured))
                    and sum(goal) > sum(measured) + .001):
                release_arm = command.arm
                # A release exemption never carries a stale Cartesian/joint
                # demand along with it. Both arm chains hold measured joints.
                limits = [next(v for v in self.config.robot.operating_limits if v.joint == joint)
                          for joint in self.config.robot.joint_names]
                planned = {name: bounded_measured_joints(states[name].joint_positions_rad, limits) + target[6:]
                           for name, target in planned.items()}
                cartesian_targets.clear()
        if monitor is not None:
            monitor.begin_action(allow_retreat=bool(paths), release_arm=release_arm)
        self.pending_grippers = {}
        self.paths = paths
        self.path_time_s = 0.0
        self.planning_diagnostics = diagnostics
        self.last_plan = {
            "paths": {
                name: {"positions": path.positions, "times_s": path.times_s}
                for name, path in paths.items()
            },
            "diagnostics": diagnostics,
        }
        self.motion_arms = {c.arm for c in action.commands if c.kind != "gripper"}
        self.grip_arms = {c.arm for c in action.commands if c.kind == "gripper"}
        initial = dict(planned)
        for name, path in paths.items():
            initial[name] = path.positions[0] + initial[name][6:]
        for c in action.commands:
            if c.kind == "gripper":
                if c.arm in self.motion_arms:
                    self.pending_grippers[c.arm] = planned[c.arm][6:]
                    initial[c.arm] = initial[c.arm][:6] + self.targets[c.arm][6:]
                else:
                    self._start_grip(c.arm, planned[c.arm][6:], states[c.arm])
        self.motion_stable_s = 0.0
        self._send_targets(initial)
        self.cartesian_targets = cartesian_targets
        self.rejection = None

    def _start_grip(self, arm, target, state):
        travel = max(abs(a - b) for a, b in zip(target, state.finger_positions_m))
        speed = min(
            x.velocity
            for x in self.config.robot.operating_limits
            if x.joint in self.config.robot.gripper.joint_names
        )
        self.grip_elapsed[arm] = 0.0
        self.grip_durations[arm] = max(0.5, travel / speed + 0.5)
        self.grip_closing[arm] = sum(target) < sum(state.finger_positions_m) - 0.001

    def close(self):
        if self.planner is not None:
            self.planner.close()
            self.planner = None

    def _send_targets(self, planned):
        # Keep requested goals separate from the rate-limited drive reference.
        # Timeout recovery may reuse drive references sourced from NumPy.
        # Comparisons against those must not leak np.bool_ into logged feedback.
        self.targets = {name: tuple(float(v) for v in target)
                        for name, target in planned.items()}
        self._advance_drive_targets(0.0)

    def _advance_drive_targets(self, dt_s):
        if not self.instances:
            return
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        limits = {limit.joint: limit.velocity
                  for limit in self.config.robot.operating_limits}
        speeds = tuple(limits[name] for name in self.config.robot.gripper.joint_names)
        for name, goal in self.targets.items():
            previous = self.drive_targets[name]
            fingers = tuple(
                old + max(-speed * dt_s, min(speed * dt_s, desired - old))
                for old, desired, speed in zip(previous[6:], goal[6:], speeds)
            )
            if name in self.stable_grips:
                measured = self.instances[name].get_joint_positions(joint_indices=self.indices[name])[6:]
                fingers = self.stable_grips[name].advance(goal[6:], measured, dt_s)
            reference = goal[:6] + fingers
            efforts = None
            if self.gravity_compensation:
                # Compensate the arm's static sag without disabling gravity or
                # changing finger contact forces. PhysX supplies torques for
                # the measured pose, not task/object goals.
                gravity = self.instances[name]._articulation_view.get_generalized_gravity_forces(
                    joint_indices=self.indices[name][:6])
                values = np.asarray(gravity, dtype=float).reshape(-1)
                if values.shape != (6,) or not np.isfinite(values).all():
                    raise InvalidSimulationState(f"{name}: invalid gravity compensation")
                effort_limits = np.array([next(x.effort for x in self.config.robot.operating_limits
                                              if x.joint == joint) for joint in self.config.robot.joint_names])
                efforts = np.r_[np.clip(values, -effort_limits, effort_limits), 0., 0.]
            if reference != previous or efforts is not None:
                self.instances[name].apply_action(ArticulationAction(
                    joint_positions=np.array(reference), joint_efforts=efforts,
                    joint_indices=self.indices[name]))
                self.drive_targets[name] = reference

    def before_physics_step(self, dt_s: float) -> None:
        self._stop_on_collision()
        self.tracking_time_s += dt_s
        for name, arm in self.instances.items():
            self.settling.add(self.tracking_time_s, name,
                             arm.get_joint_positions(joint_indices=self.indices[name]))
        for arm in self.grip_elapsed:
            self.grip_elapsed[arm] += dt_s
        if self.paths:
            measured = self.state()
            tracking_lag = max(
                abs(actual - target)
                for name in self.paths
                for actual, target in zip(
                    measured[name].joint_positions_rad, self.targets[name][:6]
                )
            )
            # A lagging drive must catch up before the reference advances.
            # Persistent obstruction remains bounded by the per-action timeout.
            if tracking_lag <= 0.10:
                self.path_time_s += dt_s
            targets = dict(self.targets)
            finished = []
            for arm, path in self.paths.items():
                targets[arm] = path.sample(self.path_time_s) + targets[arm][6:]
                if self.path_time_s >= path.duration_s:
                    finished.append(arm)
            self._send_targets(targets)
            for arm in finished:
                del self.paths[arm]
        if self.pending_grippers and not self.paths:
            feedback = self._tracking_feedback()
            motion_ready = all(
                feedback.metrics["arms"][name]["motion_reached"]
                for name in self.motion_arms
            )
            self.motion_stable_s = self.motion_stable_s + dt_s if motion_ready else 0.0
            if (
                self.motion_stable_s
                >= self.execution.completion_samples * self.config.control_dt_s
            ):
                targets = dict(self.targets)
                states = self.state()
                for name, fingers in self.pending_grippers.items():
                    targets[name] = targets[name][:6] + fingers
                    self._start_grip(name, fingers, states[name])
                self.pending_grippers.clear()
                self._send_targets(targets)

        self._advance_drive_targets(dt_s)

    def _stop_on_collision(self):
        monitor = getattr(getattr(self, "backend", None), "collision_monitor", None)
        if monitor is not None and monitor.blocked and self.collision_stop is None:
            self.collision_stop = dict(monitor.blocked)
            self.on_action_timeout()
            self.motion_arms.clear()
            self.cartesian_targets.clear()
            self.grip_arms.clear()

    def on_action_timeout(self) -> None:
        """Cancel queued motion/closing; preserve the current bounded grip drive."""
        self.pending_grippers.clear()
        self.paths.clear()
        # The timed-out Cartesian goal is cancelled, not an outstanding demand
        # that can poison completion feedback for a later gripper-only action.
        for name in self.motion_arms:
            self.cartesian_targets.pop(name, None)
        states = self.state()
        limits_by_name = {x.joint: x for x in self.config.robot.operating_limits}
        limits = [limits_by_name[n] for n in self.config.robot.joint_names]
        self._send_targets(
            {
                name: bounded_measured_joints(states[name].joint_positions_rad, limits)
                + self.drive_targets.get(name, target)[6:]
                for name, target in self.targets.items()
            }
        )

    def state(self) -> dict[str, ArmState]:
        import math

        states = {}
        for name, arm in self.instances.items():
            indices = self.indices[name]
            positions = tuple(
                float(v) for v in arm.get_joint_positions(joint_indices=indices)
            )
            velocities = tuple(
                float(v) for v in arm.get_joint_velocities(joint_indices=indices)
            )
            if not all(math.isfinite(value) for value in positions + velocities):
                raise InvalidSimulationState(f"{name}: nonfinite R5 joint state")
            tcp_pose = None
            if self.kinematics is not None:
                from .transforms import pose_transform, transform_pose

                tcp_pose = transform_pose(
                    pose_transform(self.backend.read_ee_pose(name))
                    @ self.kinematics.tcp
                )
            states[name] = ArmState(
                joint_positions_rad=positions[:6],
                joint_velocities_rad_s=velocities[:6],
                gripper_width_m=joints_to_width(self.config.robot, positions[6:]),
                finger_positions_m=positions[6:],
                finger_velocities_m_s=velocities[6:],
                tcp_pose_world_wxyz=tcp_pose,
            )
        return states

    def feedback(self) -> Feedback:
        if not self.targets:
            return Feedback("not_initialized", "R5 state is unavailable before reset")
        if self.rejection is not None:
            return self.rejection
        self._stop_on_collision()
        if self.collision_stop is not None:
            row = self.collision_stop
            return Feedback("collision_blocked",
                            f"Motion stopped: {row['arm']} {row['component']} contacted an external shape. "
                            "Inspect the images and choose a clear retreat or a different approach; "
                            "no automatic recovery was performed. This is not a task-failure verdict.",
                            {"collision": {k: row[k] for k in ("arm", "component")},
                             "source": "simulated rigid contact flag; no force magnitude or object pose supplied",
                             "phase": "held_after_collision"})
        return self._tracking_feedback()

    def _tracking_feedback(self) -> Feedback:
        """Controller feedback independent of the last policy-request rejection."""
        errors = {}
        tcp_errors = {}
        details = {}
        settled = True
        for name, state in self.state().items():
            target = self.targets[name]
            arm_error = max(
                abs(q - desired)
                for q, desired in zip(state.joint_positions_rad, target[:6])
            )
            finger_error = max(
                abs(q - desired)
                for q, desired in zip(state.finger_positions_m, target[6:])
            )
            errors[name] = (arm_error, finger_error)
            if name in self.cartesian_targets:
                from .transforms import pose_errors

                if state.tcp_pose_world_wxyz is None:
                    return Feedback(
                        "error", f"{name}: measured TCP pose is unavailable"
                    )
                position_error, orientation_error = pose_errors(
                    state.tcp_pose_world_wxyz, self.cartesian_targets[name]
                )
                tcp_errors[name] = {
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "requested_pose_world_wxyz": self.cartesian_targets[name],
                    "measured_pose_world_wxyz": state.tcp_pose_world_wxyz,
                    "remaining_xyz_m": tuple(
                        t - q
                        for t, q in zip(
                            self.cartesian_targets[name][:3],
                            state.tcp_pose_world_wxyz[:3],
                        )
                    ),
                }
            window_arm, window_fingers = self.settling.settled(
                self.tracking_time_s, name, state.joint_positions_rad + state.finger_positions_m)
            arm_settled = all(abs(v) < 0.02 for v in state.joint_velocities_rad_s) or window_arm
            cartesian_reached = name not in tcp_errors or (
                position_error < self.cartesian_completion_tolerance_m and orientation_error < 0.02
            )
            desired_width = joints_to_width(
                self.config.robot, self.pending_grippers.get(name, target[6:])
            )
            gripper_reached = (
                finger_error < 0.0015 and name not in self.pending_grippers
                and self.drive_targets.get(name, target)[6:] == target[6:]
            )
            fingers_still = all(abs(v) < 0.003 for v in state.finger_velocities_m_s) or window_fingers
            closing_obstructed = (
                self.grip_closing.get(name, False)
                and not gripper_reached
                and name not in self.pending_grippers
                and fingers_still
                and self.grip_elapsed.get(name, 0.0)
                >= self.grip_durations.get(name, float("inf"))
                and state.gripper_width_m is not None
                and state.gripper_width_m > desired_width + 0.0015
            )
            motion_reached = (
                arm_error < ARM_JOINT_TRACKING_TOLERANCE_RAD
                and arm_settled
                and cartesian_reached
                and name not in self.paths
            )
            gripper_complete = fingers_still and (gripper_reached or closing_obstructed)
            details[name] = {
                "position_window_settled": {"arm": window_arm, "fingers": window_fingers, "duration_s": .25},
                "motion_reached": motion_reached,
                "joint_error_max_rad": arm_error,
                "finger_error_max_m": finger_error,
                "requested_gripper_width_m": desired_width,
                "measured_gripper_width_m": state.gripper_width_m,
                "gripper_width_error_m": (
                    None
                    if state.gripper_width_m is None
                    else desired_width - state.gripper_width_m
                ),
                "gripper_reached": gripper_reached,
                "gripper_execution_complete": gripper_complete,
                "gripper_state": (
                    "pending_motion"
                    if name in self.pending_grippers
                    else "closing_obstructed_or_stalled"
                    if closing_obstructed
                    else "target_reached"
                    if gripper_reached
                    else "moving_or_not_reached"
                ),
                "grasp_confirmed": None,
                "drive_reference_fingers_m": self.drive_targets.get(name, target)[6:],
                "finger_center_error_m": (state.finger_positions_m[0] - state.finger_positions_m[1]) / 2,
                "gripper_drive_phase": self.stable_grips[name].phase if name in self.stable_grips else "position",
                "empty_grasp_suspected": bool(
                    self.grip_closing.get(name, False)
                    and gripper_complete and state.gripper_width_m is not None
                    and state.gripper_width_m <= self.config.robot.gripper.width_limits_m[0]
                    + empty_close_tolerance(self.config.robot.gripper.width_limits_m[0])
                ),
                "grasp_evidence": (
                    "Not determined by controller; verify contact and object response using images "
                    "and a small task-appropriate movement along its allowed axis; do not lift a constrained part"
                    if self.config.grasp_check_mode == "contact_inspection"
                    else "Not determined by controller; verify object following a short lift in images"
                ),
                "gripper_pending": name in self.pending_grippers,
            }
            settled = settled and motion_reached and gripper_complete
        complete = settled and not self.pending_grippers and not self.paths
        return Feedback(
            "completed" if complete else "executing",
            f"Joint tracking errors (rad, m): {errors}",
            {
                "tcp": tcp_errors,
                "arms": details,
                "planning": self.planning_diagnostics,
                "phase": "trajectory_tracking"
                if self.paths
                else "motion_before_gripper"
                if self.pending_grippers
                else "target_tracking",
            },
        )
