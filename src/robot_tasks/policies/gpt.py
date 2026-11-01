"""Shared GPT control workflow: RGB by default, with opt-in aligned depth."""

import base64
from collections import deque
from dataclasses import asdict, replace
from io import BytesIO
import json
import math
import hashlib

from PIL import Image

from ..config import TaskConfig
from ..types import Action, PolicyInput, RGBFrame, RetryablePolicyError
from .model_config import GPTSettings
from .responses import ModelError, ResponsesClient
from .reasoning import DecisionProgress
from ..workflow import validate_gpt_workflow, workflow_metadata
from .sparse_action import sparse_schema, decode_sparse_action, orientation_degrees
from ..cameras.geometry import resized_intrinsics
from ..robot_roles import ARM_IDENTITY_PROTOCOL, arm_identities


def prepare_frame(frame: RGBFrame, max_edge: int, *, robot=None, bench=None):
    if min(frame.width, frame.height) <= 0 or len(frame.data) != frame.width * frame.height * 3:
        raise ValueError("Invalid packed RGB8 frame")
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
    if frame.calibration is None:
        raise ValueError("RGB has no camera metadata; the GPT workflow requires calibration")
    from ..cameras.visual_aids import annotate
    image = annotate(image, frame.calibration, robot or {}, bench=bench)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    calibration = None
    if frame.calibration is not None:
        calibration = {**frame.calibration, "resolution": list(image.size),
                       "intrinsics_fx_fy_cx_cy": resized_intrinsics(
                           frame.calibration["intrinsics_fx_fy_cx_cy"], (frame.width, frame.height), image.size)}
    return buffer.getvalue(), image.size, calibration


class GPTPolicy:
    def __init__(self, config: TaskConfig, settings: GPTSettings, *, client=None):
        self.config, self.settings = config, settings
        self.arm_identities = arm_identities(config)
        from ..backends.scene_geometry import bench_geometry
        self.bench = bench_geometry(config)
        self.client = client if client is not None else ResponsesClient(settings)
        validate_gpt_workflow(config)
        joint_targets = config.raw.get("completion", {}).get("return_home", False)
        self.schema = sparse_schema(config.command_arms, joint_targets=joint_targets)
        if config.include_depth:
            from .depth_query import with_depth_queries
            self.schema = with_depth_queries(self.schema, config.cameras)
        from .decision_note import with_decision_note
        self.schema = with_decision_note(self.schema)
        self.history = deque()
        self.calls = 0
        self.last_exchange: dict = {}
        self.instructions = (
            "Control the ARX R5 using RGB, task instruction, measured state and execution feedback. "
            "The simulator pauses while you decide. Return only JSON matching the schema. "
            "Commands use absolute world TCP targets x_m,y_m,z_m and orientation angles in degrees. "
            "Every nullable field is required; null holds the LATEST MEASURED value, not the last command. "
            "roll_deg,pitch_deg,yaw_deg describe Rz(yaw)*Ry(pitch)*Rx(roll)*R_reference. "
            "At all zero angles TCP +X points world -Z, jaw closing +Y points world +Y. "
            "If all three angles are null, preserve the measured quaternion exactly. "
            "gripper is normalized: 0 closes toward the configured minimum with bounded drive effort; 1 opens fully; null retains the previous drive target. Contact may prevent reaching 0. "
            "At most one row per commandable arm; omit arms that should remain unchanged. commands=[] waits. request_stop=true requires commands=[] "
            "and only requests termination, never declares measured success. An unsuccessful stop request is rejected, consumes one call, and you must continue using the retained scene/history until measured success or the total call budget. "
            "Plan the task yourself. Reobserve near contact; never infer nonexistent force readings. "
            "For planning rejections, inspect feedback.metrics.planning_failure when present. "
            "A start_collision means the current state is rejected by the planner's collision model, "
            "not that the requested goal is unreachable. A reported self-collision cannot be cleared "
            "just by changing the camera view or repeating a target from the unchanged state. "
            "goal_ik_not_found and path_not_found describe finite search failures, not proven global unreachability. "
            f"Commandable arms: {config.command_arms}. Physical width limits in metres: {config.robot.gripper.width_limits_m}. "
            f"Schema: {json.dumps(self.schema)}"
        )
        from ..config import ExecutionSettings
        if joint_targets:
            self.instructions += (
                " joint_positions_rad is an optional six-joint absolute target in radians. "
                "For ordinary Cartesian or gripper commands set it to null. To return home, "
                "set it to the prescribed home joints and set all six TCP fields to null. "
                "Joint and Cartesian targets cannot coexist in one command. Joint targets use "
                "the same trajectory planner, limits and execution feedback; they are not teleports."
            )
        self.instructions += f" Combined motion/gripper timing: {ExecutionSettings.from_config(config).gripper_timing}."
        self.instructions += f" Arm roles: {json.dumps(config.arm_roles)}. Only manipulation arms may manipulate task objects. Fixed arms must remain still."
        if config.observation_arms:
            observation_advice = (
                "If the available views do not clearly show the grasp, working end, target opening or "
                "their relative alignment, actively reposition an observation arm before continuing "
                "manipulation. This includes uncertainty about whether an object is still held after "
                "lifting, or why an approach or insertion failed. You choose whether and where to inspect. "
            )
            if (config.task_id == "L1-02"
                    and config.raw.get("task", {}).get("prompt_protocol") != "feishu_english_20261015_v1"):
                observation_advice = (
                    "For L1-02, distinguish uncertain holding from an occluded blade end. If holding "
                    "is unverified, actively reposition an observation arm or use the permitted grasp "
                    "recovery; if automatic grasp feedback is enabled, its motion bounds still apply. If holding is verified and "
                    "current images support a safe clearance move, seeing both blades is not a "
                    "prerequisite for lifting clear or a small free-space move toward an insertion-ready pose. "
                    "Choose the grasp from final insertion clearance: both fingers and palm must permit "
                    "blade entry and housing seating. Prefer an observation-arm view when needed and feasible; "
                    "do not sacrifice a useful manipulation pose merely to show the underside. "
                    "Follow the task's two-attempt observation limit; exhaustion does not require regrasp. "
                    "Before a recovery placement, identify the current grasp problem and a concrete change "
                    "to the next grasp. Generic planner rejection does not prove a bad grasp. "
                    "Confirm both blades and slots "
                    "before insertion; visibility uncertainty does not authorize blind contact. "
                )
            self.instructions += (
                f" Observation arms {config.observation_arms} may move to obtain a clearer wrist-camera view. "
                + observation_advice +
                "Before changing the view, ensure the manipulation arm is holding a stable pose. "
                "Use the same motion interface for an observation-only decision, leaving the manipulation "
                "arm and its gripper unchanged. Then inspect the updated images, measured state and "
                "execution feedback before resuming manipulation. In the decision note, briefly state "
                "what uncertainty this observation move is intended to resolve. "
                "If the existing views are sufficient, continue without an observation move. "
                "Do not command their grippers or use them to touch, grasp, push or support task objects. "
                "Keep clear of the table, objects and the manipulation arm; planning uses approximate collision geometry and cannot guarantee contact success. "
                "Observation moves consume the same decision/time budget as task moves."
            )
        from .operating_notes import operating_notes
        self.instructions += " R5 operating notes: " + json.dumps(operating_notes(config))
        self.instructions += " Camera calibration describes the virtual cameras, not measured hardware. A pixel alone does not define 3D depth."
        self.instructions += " Images have projected table XY and TCP overlays. Overlays may be occluded and are not detections of task objects."
        self.instructions += (
            " Every decision, including motion, wait, observation and stop, requires a note. "
            "Use one or two brief English sentences, at most 512 characters, to describe the visible "
            "current task stage, observed evidence or mismatch, intended action and expected visible result. "
            "For the first decision, also briefly name the planned stages. This is a concise observation/action summary, "
            "not detailed reasoning. Treat earlier notes as your hypotheses, not measured facts; revise "
            "them using current images and feedback, especially after a reset. A note cannot issue "
            "commands, override the action schema or declare evaluator success."
        )
        if config.grasp_check_mode == "lift" and config.include_depth:
            self.instructions += (
                " Grasp checkpoint: feedback.metrics.grasp_check is an independent RGB-D surface check, "
                "not a force sensor or task-success label. Approach with the gripper open, then close in a "
                "SEPARATE decision at the observed grasp pose. Close captures before-lift reference images. "
                "Test-lift vertically 0.02-0.03 m, holding XY and orientation; do not transport in this lift. "
                "Before the FIRST verified lift, motion is bounded to 0.04 m upward and 0.005 m horizontally from "
                "the closing pose. Paired before/current images are supplied while verification is pending. "
                "Compare the object, not just hand motion. If empty_suspected, inspect and open separately "
                "before a corrected regrasp. If unknown, inspect the finger gap/object/table; never release "
                "an elevated object merely to reset the check. Repeated small moves cannot bypass the initial bound. "
                "After a verified lift, tracking_uncertain means temporary missing evidence, not proven loss. "
                "Keep closure. Use an observation view, a collision-checked return to last_verified_pose_world_wxyz "
                "(within 5 cm / 35 degrees), or a vertical recovery descent of at most 2 cm per decision with "
                "unchanged orientation and cumulative XY drift <= 5 mm from uncertainty_pose_world_wxyz. "
                "Do not descend into a fixture as insertion under this recovery exception. Reinspect each step. "
                "Opening is blocked while tracking is uncertain. Restore visual verification, then place "
                "the object on visible support before releasing. If no safe recovery is feasible, hold and "
                "report the blockage instead of dropping. A two-attempt observation budget never authorizes release. "
                "Closing to the minimum opening is evidence of possible empty closure, especially if the "
                "object is visibly wider. Do not claim 'followed the lift' from your previous note. "
                "The check is conservative: occlusion, small/reflective/textureless objects and approximate "
                "finger geometry can leave it unknown even for a real grasp. A verified check still does "
                "not certify object identity, long-term stability or task completion. "
                " Release feedback: release_check retains RGB-D evidence after opening. Opening is not proof "
                "of detachment. Inspect settling, choose a small clear probe and compare whether the prop "
                "stays behind or follows. This feedback is advisory and does not select or execute recovery. "
                "Unknown or occluded evidence must not be treated as successful release. "
            )
        elif config.grasp_check_mode == "contact_inspection":
            self.instructions += (
                " Constrained contact task: the free-object vertical test-lift guard is NOT applicable. "
                "Do not pull or bend an engaged assembly to verify a grip. Approach and close in separate "
                "decisions; inspect finger contact and actual object motion in the current images, then use small "
                "increments consistent with the task's axis and contact constraints. Follow the task "
                "instructions for any initial free pickup and later constrained regrasp. Reobserve after every increment. "
                "Blocked closure alone is not proof of holding; do not continue with an empty hand."
            )
        else:
            self.instructions += (
                " Visual grasp verification: approach, close, and inspect a short task-appropriate lift. "
                "Compare object motion with the fingers and table using current RGB views and measured robot state. "
                "A completed command or blocked closure is not proof of holding. Choose an observation-arm view "
                "when permitted and needed; confirm holding before transport and detachment after opening. "
                "There is no automatic grasp_check/release_check or verified status to wait for in RGB mode. "
                "You choose the next motion from visual evidence; no depth-based grasp gate is applied."
            )
        if not config.include_depth:
            self.instructions += (
                " Observation mode: RGB only, with camera calibration, measured robot state and execution feedback. "
                "No depth images, pixel-distance queries or depth-derived grasp/release feedback are available. "
                "Estimate spatial relationships from complementary RGB views, known camera/robot geometry and "
                "observed motion; calibration alone does not measure an object's distance. Do not request "
                "unavailable tools or wait for automatic visual verification. Task references to verification "
                "mean your own observation unless automatic feedback is explicitly enabled."
            )
        if config.include_depth:
            self.instructions += (
                " Each RGB view is followed by aligned ideal simulation depth, not a calibrated Orbbec sensor emulation. "
                "Depth is optical-axis Z in metres, not Euclidean range. The coloured preview has a fixed clipped "
                "0.02-2.00 m scale (blue near, red far, black invalid); its bottom legend is outside the image. "
                "Do not estimate precise metric depth from colours. For precision, select pixels in the RGB image "
                "you receive; each depth_queries entry specifies camera, u (integer column) and v (integer row). "
                "Coordinates start at the top-left pixel (0,0), u right and v down; use the supplied rgb_image_size. "
                "Up to 8 queries per decision. A query requires commands=[] and request_stop=false; "
                "it does not advance physics or move the robot, and results arrive in the next observation feedback. "
                "Queries consume the SAME model-call budget as motion decisions. "
                "For motion, wait or stop, set depth_queries=[]. Results include depth_z_m and point_world_m "
                "of the nearest source pixel's VISIBLE SURFACE, not an object's centre, grasp pose or hidden hole bottom. "
                "no_depth means unknown; never treat it as zero distance. Use current captures after movement or reset."
            )

    @property
    def run_metadata(self) -> dict:
        client_metadata = getattr(self.client, "provider_metadata", {})
        if not isinstance(client_metadata, dict):
            client_metadata = {}
        return {"provider": {**self.settings.public_dict(),
                             **client_metadata}, "instructions": self.instructions,
                "action_schema": self.schema, "workflow": workflow_metadata(self.config),
                "arm_roles": self.config.arm_roles,
                "arm_identity_protocol": ARM_IDENTITY_PROTOCOL if self.arm_identities else None,
                "arm_identities": self.arm_identities,
                "task_initial_state": self.config.initial_state_description,
                "grasp_check_mode": self.config.grasp_check_mode,
                "observation_modalities": workflow_metadata(self.config)["observation_modalities"],
                "observation_format": ("RGB PNG" + (" + aligned depth preview + metric pixel queries" if self.config.include_depth else "")
                                       + "; full text history, bounded image history")}

    def reset(self, seed: int) -> None:
        reset = getattr(self.client, "reset_cancellation", None)
        if reset is not None:
            reset()
        # Seed belongs to the environment. Do not claim deterministic API output.
        self.history.clear()
        self.calls = 0
        self.last_exchange = {}

    def observation_message(self, context: PolicyInput):
        """Encode disclosed sensor input without calling the provider or changing history."""
        if not context.observation.rgb:
            raise ValueError("GPT robot policy requires at least one real RGB frame")
        observation = context.observation
        if tuple(f.camera for f in observation.rgb) != self.config.cameras:
            raise ValueError("RGB observation must include every configured camera in order")
        if any(not math.isclose(f.timestamp_s, observation.time_s, rel_tol=0, abs_tol=1e-8) for f in observation.rgb):
            raise ValueError("RGB and robot observation simulation timestamps disagree")
        text = {"instruction": context.instruction, "step": context.step, "simulation_time_s": observation.time_s,
                "robot": {name: asdict(state) for name, state in observation.robot.items()},
                "feedback": asdict(observation.feedback)}
        from ..robots.control import width_to_normalized
        for name, state in observation.robot.items():
            text["robot"][name]["gripper_normalized_measured"] = (None if state.gripper_width_m is None else
                width_to_normalized(self.config.robot, state.gripper_width_m))
        text["tcp_orientation_reference_deg"] = {
            name: dict(zip(("roll_deg", "pitch_deg", "yaw_deg"), orientation_degrees(state.tcp_pose_world_wxyz)))
            for name, state in observation.robot.items() if state.tcp_pose_world_wxyz is not None}
        content = [{"type": "input_text", "text": json.dumps(text, allow_nan=False)}]
        from .grasp_comparison import comparison_images
        comparisons = []
        for label, png in comparison_images(observation, self.settings.image_max_edge):
            content.extend([{"type": "input_text", "text": json.dumps(label)},
                            {"type": "input_image", "detail": "high", "image_url":
                             "data:image/png;base64," + base64.b64encode(png).decode("ascii")}])
            comparisons.append({**label, "sha256": hashlib.sha256(png).hexdigest()})
        frame_info = []
        image_sizes = {}
        for frame in observation.rgb:
            encoded, size, calibration = prepare_frame(frame, self.settings.image_max_edge,
                                                       robot=observation.robot, bench=self.bench)
            label = {"camera": frame.camera, "timestamp_s": frame.timestamp_s,
                     "modality": "rgb", "rgb_image_size": list(size)}
            owner = next((row for row in self.arm_identities.values()
                          if row["wrist_camera"] == frame.camera), None)
            if owner is not None:
                label["owner_arm"] = owner["command_arm"]
                label["owner_role"] = owner["role"]
                label["real_wrist_camera_key"] = owner["real_wrist_camera_key"]
            elif self.arm_identities and frame.camera == "overview":
                label["view_reference"] = "Fixed overview; left/right refer to robot bases, not current hands"
            image_sizes[frame.camera] = size
            if calibration is None:
                raise ValueError("Calibration disclosure requested but RGB has no camera metadata")
            label["calibration"] = calibration
            content.append({"type": "input_text", "text": json.dumps(label)})
            content.append({"type": "input_image", "detail": "high",
                            "image_url": "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")})
            frame_info.append({**label, "original_size": [frame.width, frame.height], "encoded_size": list(size),
                               "png_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
                               "visual_aids": True})
            if self.config.include_depth:
                from ..cameras.depth import depth_metadata, encode_depth_preview
                depth_png, preview = encode_depth_preview(frame, size)
                depth_info = {"camera": frame.camera, "modality": "depth", **depth_metadata(frame), **preview}
                if owner is not None:
                    depth_info["owner_arm"] = owner["command_arm"]
                    depth_info["owner_role"] = owner["role"]
                content.append({"type": "input_text", "text": json.dumps(depth_info, allow_nan=False)})
                content.append({"type": "input_image", "detail": "high", "image_url":
                                "data:image/png;base64," + base64.b64encode(depth_png).decode("ascii")})
                frame_info[-1]["depth"] = {**depth_info, "png_bytes": len(depth_png),
                                           "sha256": hashlib.sha256(depth_png).hexdigest()}
        message = {"role": "user", "content": content}
        return message, frame_info, comparisons, image_sizes

    def remember(self, message, response_text):
        self.history.append((message, {"role": "assistant", "content": response_text}))
        # Release old image bytes but preserve accepted action/observation text.
        cutoff = len(self.history) - self.settings.history_steps
        for index, (user, assistant) in enumerate(self.history):
            if index < cutoff:
                user["content"] = [item for item in user["content"] if item["type"] != "input_image"]

    def cancel_pending(self):
        cancel = getattr(self.client, "cancel_pending", None)
        if cancel is not None:
            cancel()

    def act(self, context: PolicyInput) -> Action:
        self.last_exchange = {}  # Input validation errors must not re-log an earlier API call.
        if self.settings.max_calls > 0 and self.calls - getattr(self, "generation_failure_calls", 0) >= self.settings.max_calls:
            raise ModelError("Episode model-call budget exhausted")
        message, frame_info, comparisons, image_sizes = self.observation_message(context)
        observation = context.observation
        messages = [item for pair in self.history for item in pair] + [message]
        self.calls += 1
        self.last_exchange = {"call": self.calls, "step": context.step, "status": "requesting",
                              "requested_model": self.settings.model, "frames": frame_info,
                              "grasp_comparisons": comparisons,
                              "history_steps": len(self.history)}
        try:
            with DecisionProgress(context.step, self.last_exchange) as progress:
                reply = self.client.generate(instructions=self.instructions, messages=messages, schema=self.schema,
                                             on_reasoning=progress.summary)
            self.last_exchange.update(reply.metadata(), response_text=reply.text)
            action_text, queries, note = reply.text, (), ""
            from .decision_note import split_decision_note
            action_text, note = split_decision_note(action_text)
            self.last_exchange["note"] = note
            if self.config.include_depth:
                from .depth_query import split_depth_reply
                action_text, queries = split_depth_reply(action_text, image_sizes)
            action = decode_sparse_action(action_text, self.config, observation)
            action = replace(action, depth_queries=queries, note=note)
        except (ModelError, ValueError) as exc:
            self.last_exchange.update(status="rejected", error=str(exc),
                                      retryable=isinstance(exc, RetryablePolicyError))
            diagnostic = getattr(self.client, "last_diagnostic", None)
            if isinstance(diagnostic, dict):
                self.last_exchange["provider_diagnostic"] = diagnostic
            raise
        self.remember(message, reply.text)
        self.last_exchange.update(status="accepted", action=asdict(action))
        return action
