"""Common episode lifecycle. It does not know any task-specific success rule."""

import math
import os
from .policies.responses import ModelError
import json
from collections.abc import Callable
import time
from dataclasses import asdict, replace
from pathlib import Path

from .config import ExecutionSettings, TaskConfig
from .interfaces import Backend, Policy, Robot, Task
from .policy_worker import PolicyRequestCleanupError, request_action
from .recording import EpisodeRecorder
from .robot_roles import FixedArmMonitor, validate_role_command
from .types import Action, Assessment, EvaluationState, Feedback, InvalidPolicyAction, InvalidSimulationState, Observation, PolicyInput, RetryablePolicyError


GENERATION_RETRY_DELAYS_S = (2., 4., 8.)  # Repeat the last delay within episode budgets.


def validate_action(action: Action, config: TaskConfig, *, require_control_duration: bool = True) -> None:
    if not isinstance(action, Action):
        raise ValueError("Policy must return Action")
    if not math.isfinite(action.duration_s) or action.duration_s <= 0:
        raise ValueError("Action duration must be positive and finite")
    if require_control_duration and not math.isclose(action.duration_s, config.control_dt_s, abs_tol=1e-9):
        raise ValueError("This scaffold uses a fixed control_dt_s per action")
    if action.request_stop and action.commands:
        raise ValueError("A stop request cannot contain commands")
    if action.depth_queries:
        from .cameras.depth import validate_depth_queries
        if not config.include_depth or not config.include_feedback or action.commands or action.request_stop:
            raise ValueError("Depth queries require RGB-D/feedback and cannot accompany motion or stop")
        validate_depth_queries(action.depth_queries, config.cameras)
    seen = set()
    for command in action.commands:
        validate_role_command(config, command.arm, command.kind)
        if command.kind not in {"joint_position", "cartesian_pose", "gripper"}:
            raise ValueError(f"Unsupported action kind: {command.kind}")
        channel = "gripper" if command.kind == "gripper" else "arm_motion"
        key = (command.arm, channel)
        if key in seen:
            raise ValueError("Conflicting targets for the same arm/channel")
        seen.add(key)
        if not command.values or not all(math.isfinite(v) for v in command.values):
            raise ValueError("Action values must be nonempty and finite")
        if command.kind == "cartesian_pose":
            if len(command.values) != 7 or command.frame not in {"world", "robot_base"}:
                raise ValueError("Pose needs xyz + quaternion(wxyz), and an explicit supported frame")
            norm = math.sqrt(sum(v * v for v in command.values[3:]))
            if not math.isclose(norm, 1.0, abs_tol=1e-3):
                raise ValueError("Pose quaternion must be normalized")
        if command.kind == "gripper" and (len(command.values) != 1 or command.values[0] < 0):
            raise ValueError("Gripper command is one nonnegative total opening width")
    # Joint count/limits, reachable workspace and force limits depend on the
    # actual robot. They MUST also be validated by Robot.apply before execution.


class EpisodeLimit(Exception):
    """A recorded experimental limit, not a simulator/service failure."""


def run_episode(config: TaskConfig, backend: Backend, robot: Robot, task: Task,
                policy: Policy, output_root: Path, *, dry_run: bool = False,
                step_limit: int | None = None, run_directory: Path | None = None,
                record_video: bool = False,
                recovery=None,
                preparation: Action | None = None,
                summarize: Callable[[dict], dict] | None = None) -> tuple[Path, dict]:
    """One fresh run. Task truth is evaluated separately from policy inputs.

    max_steps counts control ticks, including settling and final observation.
    A target is sent once, then held until completion or its execution budget.
    Terminal task failure ends the run; only model transport/generation may retry.
    """
    limit = config.max_steps if step_limit is None else step_limit
    if (type(limit) is not int or limit < 0 or (step_limit is not None and limit == 0)
            or (config.max_steps and limit > config.max_steps)):
        raise ValueError("step_limit must be in [1, max_steps]")
    if dry_run and not limit:
        raise ValueError("Dry-run requires a finite step_limit")
    options = ExecutionSettings() if dry_run else ExecutionSettings.from_config(config)
    legacy_persistent = (not dry_run and (config.task_id.startswith("L3-") or config.task_id == "L2-07")
                         and os.environ.get("GPT_BENCHMARK_PERSISTENT_MODEL_RETRY") == "1")
    completed_budget = options.model_budget_basis == "completed_generations" or legacy_persistent
    retry_model_errors = options.retry_model_errors or legacy_persistent
    from .grasp_guard import GraspGuard
    grasp_guard = None if dry_run or not config.include_depth or config.grasp_check_mode == "contact_inspection" else GraspGuard(config)
    role_monitor = FixedArmMonitor(config, options.fixed_arm_tolerance_rad)
    mode = "dry_run" if dry_run else "simulation"
    recorder = EpisodeRecorder(output_root, config, mode, limit, run_directory=run_directory)
    backend.run_directory = recorder.directory
    start = time.monotonic() - (recovery.used_wall if recovery is not None else 0.0)
    # Batch deadline includes simulator startup, before this runner was entered.
    external_deadline = float(os.environ.get("GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC", "inf"))
    assessment = Assessment("not_evaluated", "Episode has not been evaluated")
    ticks = decisions = executed = timeouts = rejected = invalid_model_actions = 0
    rejected_stops = 0
    consecutive_generation_failures = generation_failures = model_retry_attempts = 0
    status, reason, error = "error", "startup", None
    videos = {}
    feedback = Feedback("not_initialized")
    initial = final = {}
    stopped = ready = False
    final_time = 0.0
    model_thread_cleanup_complete = True

    def budget():
        now = time.monotonic()
        if now >= external_deadline:
            raise EpisodeLimit("wall_time_limit")
        if config.max_wall_time_s > 0 and now - start >= config.max_wall_time_s:
            raise EpisodeLimit("wall_time_limit")
        if limit and ticks >= limit:
            raise EpisodeLimit("max_steps")

    def observe() -> Observation:
        observation = Observation(backend.time_s(), backend.capture_rgb(),
                                  robot.state() if config.include_robot_state else {},
                                  feedback if config.include_feedback else Feedback("not_provided"))
        return grasp_guard.enrich(observation) if grasp_guard is not None else observation

    def evaluate() -> None:
        nonlocal assessment
        state = EvaluationState({**backend.evaluation_state().values,
                                 "simulation_time_s": backend.time_s(), "robot": robot.state()})
        assessment = task.evaluate(state, robot.feedback())
        if not dry_run:
            assessment = role_monitor.update(state.values["robot"], assessment)
        if dry_run and assessment.status != "not_evaluated":
            assessment = Assessment("not_evaluated", "Rejected task score during dry-run")
            raise ValueError("Dry-run must never produce a task score")

    def tick(*, phase="execution", score=True):
        nonlocal ticks, feedback
        budget()
        backend.step(config.control_dt_s)
        ticks += 1
        feedback = robot.feedback()
        # Rejection describes the last requested action. Existing drive targets
        # may still be observed during an explicit wait or final settling.
        if feedback.status == "error":
            raise RuntimeError(f"Robot execution failed: {feedback.message}")
        if score:
            evaluate()
        if videos:
            for frame in backend.capture_rgb():
                videos[frame.camera].add(frame)
        if not dry_run:
            recorder.event({"event": "physics", "tick": ticks, "phase": phase,
                            "time_s": backend.time_s(), "feedback": asdict(feedback),
                            "robot": {k: asdict(v) for k, v in robot.state().items()},
                            "evaluation": asdict(assessment)})

    def prepare_robot() -> None:
        if dry_run or preparation is None:
            return
        budget()
        validate_action(preparation, config)
        recorder.event({"event": "preparation_start", "action": asdict(preparation),
                        "model_generated": False})
        robot.apply(preparation)
        stable = 0
        for _ in range(options.max_action_steps):
            tick(phase="preparation", score=False)
            stable = stable + 1 if feedback.status == "completed" else 0
            if stable >= options.completion_samples:
                break
        if stable < options.completion_samples:
            cancel = getattr(robot, "on_action_timeout", None)
            if cancel is not None:
                cancel()
            raise RuntimeError("Fixed startup preparation did not reach its pose; GPT was not called")
        recorder.event({"event": "preparation_complete", "time_s": backend.time_s(),
                        "feedback": asdict(feedback),
                        "robot": {k: asdict(v) for k, v in robot.state().items()}})

    def initialize_task() -> None:
        hook = getattr(task, "initialize_episode", None)
        if hook is not None and not dry_run:
            recorder.event({"event": "task_initialization_start", "model_generated": False})
            hook(backend, robot, tick)
            recorder.event({"event": "task_initialization_complete", "model_generated": False})

    try:
        policy_metadata = getattr(policy, "run_metadata", None)
        if policy_metadata is not None:
            recorder.event({"event": "policy_configuration", "configuration": policy_metadata})
        backend.open(config)
        robot.initialize(backend, config)
        task.build_scene(backend, config)
        backend.reset(config.seed)
        task.reset(backend, config.seed)
        robot.reset(config.seed)
        policy.reset(config.seed)
        policy.generation_failure_calls = 0
        feedback = robot.feedback()
        if recovery is not None:
            if dry_run:
                raise ValueError("Physical recovery is not a dry run")
            feedback = recovery.restore(backend, robot, task, policy, role_monitor)
            decisions = recovery.used
            ticks = recovery.snapshot.get("tick", 0)
            generation_failures = recovery.failures
            recovery.record(recorder)
        if record_video and not dry_run:
            from .backends.settings import SimulationSettings
            from .video import MotionVideo
            settings = SimulationSettings.from_config(config)
            if not math.isclose(config.control_dt_s * settings.fps, 1.0, abs_tol=1e-9):
                raise ValueError("Video requires one frame per control tick")
            for frame in backend.capture_rgb():
                directory = recorder.directory / "video" / frame.camera
                directory.mkdir(parents=True)
                videos[frame.camera] = MotionVideo(directory, replace(settings, width=frame.width, height=frame.height))
        if recovery is None:
            for _ in range(options.settle_steps):
                tick(phase="settle", score=False)
            prepare_robot()
            initialize_task()
        initial = {k: asdict(v) for k, v in robot.state().items()}
        if not dry_run:
            evaluate()  # Establish evaluator reference state after settling, before any model action.
        ready = True
        scene_report = getattr(backend, "scene_report", None)
        if scene_report:
            from .recording import write_json
            write_json(recorder.directory / "scene.json", scene_report)
            write_json(recorder.directory / "import.json", getattr(backend, "import_report", {}))
            backend.save_scene(recorder.directory / "scene.usda")
        observation = observe()
        recorder.event({"event": "reset", "observation": recorder.observation(observation, 0)})
        status, reason = "finished", "model_call_limit"
        if recovery is not None and recovery.probe:
            raise EpisodeLimit("recovery_probe_complete")
        policy_limit = getattr(getattr(policy, "settings", None), "max_calls", config.max_steps or 150)
        limits = [n for n in (options.max_decisions, policy_limit) if n]
        decision_limit = min(limits) if limits else None
        while decision_limit is None or decisions < decision_limit:
            step = decisions + 1
            if assessment.status in {"succeeded", "failed", "invalid"}:
                reason = "task_terminal"
                break
            budget()
            decisions = step
            t0 = time.monotonic()
            if consecutive_generation_failures:
                model_retry_attempts += 1
            if not dry_run:
                print(f"R5_POLICY_REQUEST {step}", flush=True)
            try:
                try:
                    context = PolicyInput(config.instruction, observation, step)
                    if dry_run:
                        action = policy.act(context)
                    else:
                        try:
                            action = request_action(policy, context,
                                idle=getattr(backend, "pump_events", None), check_budget=budget)
                        except PolicyRequestCleanupError:
                            model_thread_cleanup_complete = False
                            raise
                finally:
                    # Read mutable policy history only after its worker has joined.
                    exchange = (getattr(policy, "last_exchange", None)
                                if model_thread_cleanup_complete else None)
                    if exchange:
                        recorder.event({"event": "model_exchange", "step": step, "exchange": exchange})
            except (RetryablePolicyError, ModelError) as exc:
                if not retry_model_errors and not isinstance(exc, RetryablePolicyError):
                    raise
                generation_failures += 1
                consecutive_generation_failures += 1
                can_retry = completed_budget or decision_limit is None or decisions < decision_limit
                recorder.event({"event": "model_retry" if can_retry else "model_retry_exhausted",
                                "step": step, "error": str(exc),
                                "failure_number": consecutive_generation_failures,
                                "retry_limit": None,
                                "remaining_decisions": (None if decision_limit is None else decision_limit - decisions),
                                "policy_time_s": time.monotonic() - t0,
                                "simulation_time_s": observation.time_s,
                                "retry_exhaustion": None if can_retry else "model_call_limit"})
                if not can_retry:
                    raise
                if completed_budget:
                    policy.generation_failure_calls += 1
                    decisions -= 1  # Failed generations do not consume the completed-decision budget.
                    delay = (2., 4., 8., 16., 30.)[min(consecutive_generation_failures - 1, 4)]
                else:
                    delay = GENERATION_RETRY_DELAYS_S[min(consecutive_generation_failures - 1, 2)]
                if config.max_wall_time_s > 0:
                    delay = min(delay, max(0., start + config.max_wall_time_s - time.monotonic()))
                delay = min(delay, max(0., external_deadline - time.monotonic()))
                print(f"R5_MODEL_RETRY {consecutive_generation_failures}/unlimited in {delay:g}s: {exc}", flush=True)
                budget()
                deadline = time.monotonic() + delay
                pump = getattr(backend, "pump_events", None)
                if pump is None or config.raw.get("simulation", {}).get("headless", True):
                    time.sleep(delay)
                else:
                    while time.monotonic() < deadline:
                        budget()
                        pump()
                        time.sleep(min(.05, max(0., deadline - time.monotonic())))
                budget()
                continue
            except InvalidPolicyAction as exc:
                consecutive_generation_failures = 0  # Generation completed; action schema was invalid.
                invalid_model_actions += 1
                feedback = Feedback("rejected", "Your previous response was not a valid action. "
                                    "No commands were applied and simulation time has not advanced. "
                                    "Return a new JSON object matching the action schema. " + str(exc),
                                    {"reason_code": "invalid_model_action"})
                observation = replace(observation, feedback=(feedback if config.include_feedback
                                                            else Feedback("not_provided")))
                recorder.event({"event": "policy_action_rejected", "step": step, "error": str(exc),
                                "policy_time_s": time.monotonic() - t0,
                                "observation": recorder.observation(observation, step)})
                # One failed response consumes one decision/call. Retry from the
                # exact same state without applying a wait action or resetting.
                budget()
                continue
            consecutive_generation_failures = 0
            policy_time_s = time.monotonic() - t0
            validate_action(action, config)
            if not dry_run:
                recorder.event({"event": "decision", "step": step, "action": asdict(action),
                                "policy_time_s": policy_time_s})
            budget()  # Never execute a response that arrived after the wall-time deadline.
            if action.depth_queries:
                from .cameras.depth import resolve_depth_queries
                results = resolve_depth_queries(observation.rgb, action.depth_queries)
                feedback = Feedback("observed", "Pixel depth queries completed. No commands applied; "
                                    "simulation time has not advanced. Results describe visible surfaces in these images.",
                                    {"depth_queries": results, "execution_status": robot.feedback().status})
                # Reuse the exact queried capture. Do not rerender, tick, reset,
                # invoke Robot.apply, or start another uncounted model call.
                observation = replace(observation, feedback=feedback)
                recorder.event({"event": "depth_query", "step": step, "action": asdict(action),
                                "observation": recorder.observation(observation, step)})
                continue
            if action.request_stop:
                stopped = True
                for _ in range(options.stop_steps):
                    tick(phase="final_observation")
                    if assessment.status in {"succeeded", "failed", "invalid"}:
                        break
                if dry_run or assessment.status == "succeeded":
                    reason = "policy_requested_stop"
                    break
                if assessment.status in {"failed", "invalid"}:
                    recorder.event({"event": "transition", "step": step,
                                    "action": asdict(action), "policy_time_s": policy_time_s,
                                    "observation": recorder.observation(observe(), step),
                                    "evaluation": asdict(assessment)})
                    reason = "task_terminal"
                    break
                rejected_stops += 1
                feedback = Feedback("stop_rejected", "The independent task evaluator has not confirmed completion. "
                    "Your stop request was rejected and consumed one decision. The scene and history are retained "
                    "for this single episode. Inspect the current images and measured state, "
                    "then continue or correct the task. No object pose or detailed score is supplied.",
                    {"task_complete": False, "remaining_decisions": (None if decision_limit is None else decision_limit - decisions)})
                observation = observe()
                recorder.event({"event": "stop_rejected", "step": step, "action": asdict(action),
                                "observation": recorder.observation(observation, step),
                                "evaluation": asdict(assessment)})
                continue
            if grasp_guard is not None:
                rejection = grasp_guard.reject_reason(action, observation)
                if rejection:
                    observation = grasp_guard.rejection(rejection, observation)
                    feedback = observation.feedback
                    rejected += 1
                    recorder.event({"event": "grasp_check_rejected", "step": step,
                                    "action": asdict(action),
                                    "observation": recorder.observation(observation, step)})
                    continue
            try:
                robot.apply(action)
            except ValueError:
                feedback = robot.feedback()
                if feedback.status != "rejected":
                    raise
                rejected += 1
                observation = observe()
                recorder.event({"event": "action_rejected", "step": step, "action": asdict(action),
                                "observation": recorder.observation(observation, step)})
                # The adapter rejects the whole action atomically. Do not clip,
                # repair, move, reset, or score an unchanged physics timestamp.
                continue
            if grasp_guard is not None:
                for reference in grasp_guard.accepted(action, observation):
                    recorder.event({"event": "grasp_reference", "step": step,
                                    "arm": reference.arm,
                                    "observation": recorder.observation(
                                        Observation(reference.time_s, reference.rgb, reference.robot,
                                                    Feedback("grasp_reference")), step,
                                        reference_arm=reference.arm)})
            plan_record = getattr(robot, "last_plan", None)
            if isinstance(plan_record, dict) and plan_record.get("paths"):
                recorder.event({"event": "motion_plan", "step": step, **plan_record})
            executed += bool(action.commands)
            stable = 0
            completed = False
            action_ticks = options.max_action_steps if action.commands else options.wait_steps
            for _ in range(action_ticks):
                tick()
                if assessment.status in {"succeeded", "failed", "invalid"}:
                    break
                if feedback.status == "collision_blocked":
                    recorder.event({"event": "collision_blocked", "step": step,
                                    "collision": getattr(robot, "collision_stop", None)})
                    break  # The controller held the measured pose; ask the model for recovery.
                stable = stable + 1 if feedback.status == "completed" else 0
                if action.commands and stable >= options.completion_samples:
                    completed = True
                    break
            if (action.commands and not completed and feedback.status == "executing"
                    and assessment.status not in {"succeeded", "failed", "invalid"}):
                timeouts += 1
                cancel_pending = getattr(robot, "on_action_timeout", None)
                if cancel_pending is not None:
                    cancel_pending()
                feedback = Feedback("action_timeout", "Target did not settle within the per-action budget. "
                                    "Inspect measured state and images before choosing the next action.",
                                    {**feedback.metrics, "last_status": feedback.status})
            observation = observe()
            recorder.event({"event": "transition", "step": step,
                            "action": asdict(action), "policy_time_s": policy_time_s,
                            "observation": recorder.observation(observation, step),
                            "evaluation": asdict(assessment)})
            if assessment.status in {"succeeded", "failed", "invalid"}:
                reason = "task_terminal"
                break
            if limit and ticks >= limit:
                reason = "max_steps"
                break
        if not dry_run:
            recorder.event({"event": "final_observation",
                            "observation": recorder.observation(observe(), decisions + 1)})
        if dry_run and status == "finished":
            status = "dry_run_complete"
    except EpisodeLimit as exc:
        status, reason = ("dry_run_complete" if dry_run else "finished"), str(exc)
    except NotImplementedError as exc:
        status, reason, error = "implementation_missing", "stub_reached", str(exc)
    except KeyboardInterrupt:
        status, reason, error = "cancelled", "user_interrupt", None
    except InvalidSimulationState as exc:
        status, reason, error = "invalid", "simulation_invalid", str(exc)
        assessment = Assessment("invalid", str(exc), {
            **assessment.metrics, "invalid_reason": "simulation_invalid",
            "metrics_snapshot": "last valid evaluation, before numerical/readback failure",
        })
        recorder.event({"event": "simulation_invalid", "error": str(exc),
                        "last_assessment": asdict(assessment)})
    except PolicyRequestCleanupError as exc:
        status, reason, error = "error", "model_thread_cleanup_error", str(exc)
    except Exception as exc:
        status, reason, error = "error", "exception", f"{type(exc).__name__}: {exc}"
    finally:
        try:
            final_time = backend.time_s()
            if ready:
                final = {k: asdict(v) for k, v in robot.state().items()}
        except Exception:
            pass
        for video in videos.values():
            try:
                video.close()
            except Exception as exc:
                status, reason, error = "error", "recording_error", f"{type(exc).__name__}: {exc}"
        if assessment.status == "invalid" and status == "finished":
            status = "invalid"
            reason = assessment.metrics.get("invalid_reason", "simulation_invalid")
        evaluated = (not dry_run and not (recovery is not None and recovery.probe)
                     and ready and status == "finished"
                     and assessment.status in {"running", "succeeded", "failed"})
        if evaluated and assessment.status == "running":
            assessment = Assessment("failed", f"Episode ended before success: {reason}", assessment.metrics)
        result = {
            "task_id": config.experiment_id, "mode": mode, "status": status,
            **({"base_task_id": config.task_id} if config.experiment_id != config.task_id else {}),
            "termination_reason": reason, "completed_steps": ticks, "decisions": decisions,
            "model_calls": getattr(policy, "calls", 0), "executed_actions": executed,
            "model_thread_cleanup_complete": model_thread_cleanup_complete,
            "action_timeouts": timeouts, "rejected_actions": rejected, "policy_requested_stop": stopped,
            "invalid_model_actions": invalid_model_actions,
            "generation_failures": generation_failures,
            "model_retry_attempts": model_retry_attempts,
            "generation_retry_limit": None,
            "persistent_model_retry": True,
            "model_budget_basis": "completed_generations" if completed_budget else "all_requests",
            "retry_model_errors": retry_model_errors,
            "rejected_stop_requests": rejected_stops,
            "grasp_check_rejections": grasp_guard.rejections if grasp_guard is not None else 0,
            "grasp_visual_verifications": grasp_guard.verifications if grasp_guard is not None else 0,
            "grasp_check_protocol": "rgbd_surface_comotion_v3_tracking_recovery" if grasp_guard is not None else (None if dry_run else "contact_inspection" if config.grasp_check_mode == "contact_inspection" else "model_visual_inspection"),
            "release_check_protocol": "rgbd_stationary_vs_following_advisory_v1" if grasp_guard is not None else None,
            "stop_protocol": "evaluator_gated_continue_within_total_call_budget",
            "false_completion": bool(stopped and evaluated and assessment.status != "succeeded"),
            "wall_time_s": time.monotonic() - start,
            "simulation_time_s": final_time,
            "evaluation_protocol": "single_episode",
            "task_evaluated": evaluated,
            "evaluation_validity": ("invalid" if status == "invalid" else "valid" if evaluated else "not_evaluated"),
            "failure_category": assessment.metrics.get("failure_category"),
            "success": (assessment.status == "succeeded") if evaluated else None,
            "assessment": asdict(assessment), "error": error,
            "initial_robot": initial, "final_robot": final,
            "video_status": "recorded" if videos else "not_requested",
            "videos": {name: {"path": f"video/{name}/motion.mp4", "frames": video.count}
                       for name, video in videos.items()},
        }
        if recovery is not None:
            recovery.record(recorder)
            result["recovery"] = recovery.report
            result["segment_wall_time_s"] = max(0., result["wall_time_s"] - recovery.used_wall)
            result["evaluation_protocol"] = "interrupted_episode_measured_state_continuation"
        if summarize is not None:
            result = summarize(result)
        result = json.loads(json.dumps(result, allow_nan=False))
        recorder.finish(result)
        if not dry_run:
            print("R5_EPISODE_RESULT " + json.dumps({"output": str(recorder.directory), **result}), flush=True)
        try:
            try:
                close_robot = getattr(robot, "close", None)
                if close_robot is not None:
                    close_robot()
            finally:
                backend.close()
        except Exception as exc:
            result.update(status="error", termination_reason="cleanup_error", success=None, task_evaluated=False,
                          evaluation_validity="not_evaluated",
                          error=f"{error or ''} close: {type(exc).__name__}: {exc}")
            if "passed" in result:
                result.update(passed=False, task_success=None)
            recorder.finish(result)
    return recorder.directory, result
