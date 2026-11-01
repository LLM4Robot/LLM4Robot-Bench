"""Shared observation-only grasp checkpoint; never reads evaluator state.

The model still chooses all motions. Pending grasps permit observation moves,
opening, and a small vertical test lift, but cannot proceed to transport merely
because a decision note says 'grasped'. Evidence is deliberately conservative.
"""

from dataclasses import dataclass, replace

import numpy as np

from .backends.scene_geometry import bench_geometry
from .cameras.grasp_evidence import (
    Surface,
    changed_surface_support,
    match_surfaces,
    visible_surfaces,
)
from .robots.transforms import pose_errors, pose_transform, transform_pose
from .robots.control import empty_close_tolerance
from .cameras.release_evidence import ReleaseEvidence
from .types import Action, Feedback, GraspReference, Observation

MIN_LIFT_M = 0.015
MAX_TEST_LIFT_M = 0.040
MAX_TEST_XY_M = 0.005
MAX_TEST_ROTATION_RAD = 0.10


@dataclass
class Attempt:
    reference: GraspReference
    surfaces: tuple[Surface, ...]
    status: str = "awaiting_lift"
    reason: str = (
        "Close, then test-lift vertically 0.02-0.03 m and inspect the paired views."
    )
    opening_requested: bool = False
    verified_once: bool = False
    evidence: dict | None = None
    tracking_surfaces: tuple[Surface, ...] = ()
    last_verified_pose: tuple | None = None
    last_verified_width: float | None = None
    uncertainty_pose: tuple | None = None


class GraspGuard:
    def __init__(self, config):
        self.config = config
        self.table_z_m = bench_geometry(config).top
        self.attempts: dict[str, Attempt] = {}
        self.releases = {}
        self.rejections = 0
        self.verifications = 0
        self._last_observation_key = None

    def reset(self):
        # Run totals remain, but no evidence survives a physical reset.
        self.attempts.clear()
        self.releases.clear()
        self._last_observation_key = None

    def _world_target(self, command):
        if command.frame == "world":
            return command.values
        mount = next(m for m in self.config.robot.mounts if m.name == command.arm)
        return transform_pose(
            pose_transform(mount.base_pose_wxyz) @ pose_transform(command.values)
        )

    def reject_reason(self, action: Action, observation: Observation) -> str | None:
        """Validate the entire action before planner/robot side effects."""
        closing = {
            c.arm
            for c in action.commands
            if c.kind == "gripper"
            and c.arm in self.config.arms
            and observation.robot[c.arm].gripper_width_m is not None
            and c.values[0] < observation.robot[c.arm].gripper_width_m - 0.001
        }
        for command in action.commands:
            arm = command.arm
            if arm not in self.config.arms:
                continue
            state = observation.robot[arm]
            attempt = self.attempts.get(arm)
            if command.kind == "gripper":
                opening = (state.gripper_width_m is None
                           or command.values[0] > state.gripper_width_m + 0.001)
                elevated = (state.tcp_pose_world_wxyz is None or (attempt is not None
                    and state.tcp_pose_world_wxyz[2]
                    > attempt.reference.robot[arm].tcp_pose_world_wxyz[2] + MIN_LIFT_M))
                if (attempt and attempt.status not in ("verified", "empty_suspected")
                        and opening and (attempt.verified_once or elevated)):
                    return (
                        f"{arm}: release blocked while elevated holding is uncertain. "
                        "Do not drop the object in midair. Retain closure; inspect or use the "
                        "bounded recovery described in grasp_check to regain visual verification, "
                        "then choose a supported placement before opening."
                    )
                continue
            if command.kind == "joint_position":
                if arm in closing or (attempt and attempt.status != "verified"):
                    return f"{arm}: use Cartesian test lift while grasp evidence is pending; no joint-motion bypass."
                continue
            target = self._world_target(command)
            if state.tcp_pose_world_wxyz is None:
                return f"{arm}: measured TCP is missing; cannot verify grasp motion."
            position, rotation = pose_errors(state.tcp_pose_world_wxyz, target)
            if arm in closing and (position > 0.002 or rotation > 0.03):
                return f"{arm}: approach first, then close in a separate decision so the actual grasp pose is observed."
            if position <= 1e-6 and rotation <= 1e-5:
                continue  # Holding the measured pose must never prevent opening/recovery.
            if not attempt or attempt.status == "verified":
                continue
            if attempt.verified_once and attempt.last_verified_pose is not None:
                # Tracking loss is not a new pickup. Allow a bounded return to
                # the last observed held pose, or a vertical recovery descent.
                # Normal planner/payload collision checks still apply.
                anchor = attempt.uncertainty_pose or state.tcp_pose_world_wxyz
                delta = np.array(target[:3]) - state.tcp_pose_world_wxyz[:3]
                _, anchor_turn = pose_errors(anchor, target)
                return_pos, return_turn = pose_errors(attempt.last_verified_pose, target)
                retreat = (return_pos <= 0.002 and return_turn <= 0.03
                           and position <= 0.05 and rotation <= np.deg2rad(35))
                descent = (
                    np.linalg.norm(np.array(target[:2]) - anchor[:2]) <= 0.005
                    and -0.02 <= delta[2] <= 0.0001
                    and target[2] >= attempt.reference.robot[arm].tcp_pose_world_wxyz[2]
                    and anchor_turn <= 0.03
                )
                if retreat or descent:
                    continue
                return (
                    f"{arm}: holding is temporarily unverified after a verified lift. "
                    "Transport/insertion is blocked. Keep closure and inspect; or return to "
                    "grasp_check.last_verified_pose_world_wxyz (within 5 cm / 35 degrees per recovery), "
                    "or descend vertically at most 2 cm per decision with unchanged orientation "
                    "and cumulative XY drift <= 5 mm from the uncertainty pose. Choose a clear "
                    "recovery path from images; normal collision checks apply. Reinspect each step. "
                    "Opening remains blocked until visual verification is restored or empty closure is measured."
                )
            origin = attempt.reference.robot[arm].tcp_pose_world_wxyz
            delta = np.array(target[:3]) - origin[:3]
            _, turn = pose_errors(origin, target)
            if (
                np.linalg.norm(delta[:2]) > MAX_TEST_XY_M
                or delta[2] < -0.002
                or delta[2] > MAX_TEST_LIFT_M
                or turn > MAX_TEST_ROTATION_RAD
            ):
                return (
                    f"{arm}: grasp is {attempt.status}, not verified. {attempt.reason} "
                    "Transport/insertion is blocked. Hold this arm and inspect with an observation arm, "
                    "If the hand is visibly empty, open separately before repositioning/regrasping. "
                    "Do not release an elevated object merely because evidence is unknown. "
                    "A pending grasp permits only a vertical test lift within 0.04 m of its closing pose "
                    "(at most 0.005 m horizontal drift); repeated small steps cannot bypass this bound."
                )
        return None

    def accepted(
        self, action: Action, observation: Observation
    ) -> tuple[GraspReference, ...]:
        """Capture a baseline only AFTER atomic robot.apply accepted the action."""
        references = []
        for command in action.commands:
            arm = command.arm
            if command.kind != "gripper" or arm not in self.config.arms:
                continue
            state = observation.robot[arm]
            width = state.gripper_width_m
            if width is None or state.tcp_pose_world_wxyz is None:
                continue
            if command.values[0] > width + 0.001:
                if arm not in self.releases:
                    self.releases[arm] = (ReleaseEvidence.capture(
                        observation, arm, self.table_z_m, self.config.robot), command.values[0])
                if arm in self.attempts:
                    self.attempts[arm].opening_requested = True
                    self.attempts[arm].status = "release_pending"
                continue
            if command.values[0] >= width - 0.001:
                continue  # Reissuing close must not move the reference origin.
            self.releases.pop(arm, None)
            if arm in self.attempts:
                self.attempts[arm].opening_requested = False
                self.attempts[arm].status = "awaiting_lift"
                continue
            reference = GraspReference(
                arm, observation.time_s, observation.rgb, observation.robot
            )
            surfaces = visible_surfaces(
                observation.rgb, state.tcp_pose_world_wxyz, width, self.table_z_m,
                narrow_grasp=command.values[0] < 0.006,
                robot_profile=self.config.robot,
            )
            self.attempts[arm] = Attempt(reference, surfaces)
            references.append(reference)
        self._last_observation_key = None
        return tuple(references)

    def enrich(self, observation: Observation) -> Observation:
        if not self.attempts and not self.releases:
            return observation
        key = (
            observation.time_s,
            tuple((f.camera, f.calibration.get("capture_id")) for f in observation.rgb),
        )
        if key != self._last_observation_key:
            self._update(observation)
            self._last_observation_key = key
        details = {
            arm: {
                "status": attempt.status,
                "reason": attempt.reason,
                "reference_time_s": attempt.reference.time_s,
                "source": "visible_rgbd_and_measured_robot_only",
                **(attempt.evidence or {}),
            }
            for arm, attempt in self.attempts.items()
        }
        release_details = {}
        for arm, (evidence, target) in self.releases.items():
            state = observation.robot[arm]
            if state.tcp_pose_world_wxyz is None or state.gripper_width_m is None:
                release_details[arm] = {"status": "release_unknown", "advisory": True,
                                       "reason": "Measured TCP/opening unavailable."}
                continue
            release_details[arm] = evidence.update(observation, arm, self.table_z_m,
                self.config.robot, state.gripper_width_m >= target - .0015)
        feedback = replace(
            observation.feedback,
            metrics={**observation.feedback.metrics, "grasp_check": details, "release_check": release_details},
        )
        references = tuple(
            a.reference for a in self.attempts.values() if a.status != "verified"
        )
        return replace(observation, feedback=feedback, grasp_references=references)

    def _update(self, observation):
        minimum, maximum = self.config.robot.gripper.width_limits_m
        for arm, attempt in list(self.attempts.items()):
            state = observation.robot[arm]
            pose, width = state.tcp_pose_world_wxyz, state.gripper_width_m
            if pose is None or width is None:
                attempt.status, attempt.reason = (
                    "unknown",
                    "Measured TCP/opening unavailable; retain closure and inspect.",
                )
                continue
            before = attempt.reference.robot[arm]
            # Release only when opening is actually measured, not just commanded.
            previous_width = (attempt.evidence or {}).get("measured_width_m", width)
            if attempt.opening_requested and (
                width > previous_width + 0.002 or width >= maximum - 0.002
            ):
                del self.attempts[arm]
                continue
            rise = pose[2] - before.tcp_pose_world_wxyz[2]
            attempt.evidence = {
                "measured_width_m": width,
                "tcp_rise_m": rise,
                "matching_points": 0,
                "supporting_cameras": [],
            }
            if width <= minimum + empty_close_tolerance(minimum):
                attempt.verified_once = False
                attempt.status = "empty_suspected"
                attempt.reason = (
                    "Fingers reached the empty-close limit; this is not grasp evidence. "
                    "For an object wider than this measured gap, inspect its location and regrasp."
                )
                continue
            current = visible_surfaces(observation.rgb, pose, width, self.table_z_m,
                                       robot_profile=self.config.robot)
            evidence = match_surfaces(attempt.surfaces, current, width)
            if attempt.verified_once:
                recent = match_surfaces(attempt.tracking_surfaces, current, width)
                if recent["supported"]:
                    evidence = recent
                # Width is only a contradiction check, never positive grasp evidence.
                if abs(width - attempt.last_verified_width) > 0.003:
                    evidence["supported"] = False
                    evidence["width_changed_since_verified"] = True
            if rise >= MIN_LIFT_M and not attempt.verified_once:
                evidence.update(
                    changed_surface_support(
                        attempt.surfaces,
                        current,
                        attempt.reference.rgb,
                        observation.rgb,
                        before.tcp_pose_world_wxyz,
                        pose,
                    )
                )
                evidence["supported"] &= evidence["visible_motion_supported"]
            attempt.evidence.update(evidence)
            if attempt.verified_once:
                if evidence["supported"] and not attempt.opening_requested:
                    attempt.status, attempt.reason = (
                        "verified",
                        "Visible surfaces remain consistent with the lifted object near the fingers.",
                    )
                else:
                    if attempt.uncertainty_pose is None:
                        attempt.uncertainty_pose = tuple(pose)
                    attempt.status, attempt.reason = (
                        "tracking_uncertain",
                        "Previously verified holding is temporarily unconfirmed; this is not proof of loss. "
                        "Retain closure, inspect or return to the last verified pose. Bounded vertical "
                        "recovery descent is permitted; transport/insertion and opening are blocked.",
                    )
            elif rise < MIN_LIFT_M:
                attempt.status, attempt.reason = (
                    "awaiting_lift",
                    "Not yet verified: test-lift vertically 0.02-0.03 m, then inspect.",
                )
            elif evidence["supported"] and not attempt.opening_requested:
                attempt.status, attempt.reason = (
                    "verified",
                    "RGB-D surface geometry/appearance near the fingers follows the measured test lift.",
                )
                attempt.verified_once = True
                self.verifications += 1
            else:
                attempt.status, attempt.reason = (
                    "unknown",
                    "No sufficient co-moving surface evidence after lift; inspect the paired views. "
                    "Reopen only if visibly empty or supported; never drop an elevated object.",
                )
            if attempt.status == "verified":
                # Refresh only from a positively matched capture, never from an
                # occluded frame. Original pickup evidence remains immutable.
                attempt.tracking_surfaces = current
                attempt.last_verified_pose = tuple(pose)
                attempt.last_verified_width = width
                attempt.uncertainty_pose = None
            if attempt.verified_once:
                attempt.evidence.update(
                    previously_verified=True,
                    last_verified_pose_world_wxyz=attempt.last_verified_pose,
                    uncertainty_pose_world_wxyz=attempt.uncertainty_pose,
                )

    def rejection(self, reason: str, observation: Observation) -> Observation:
        self.rejections += 1
        feedback = Feedback(
            "grasp_check_rejected",
            reason,
            {**observation.feedback.metrics, "reason_code": "grasp_not_verified"},
        )
        return replace(observation, feedback=feedback)
