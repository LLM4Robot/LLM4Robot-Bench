"""Configuration-derived robot facts. Never include task-object ground truth."""

from ..config import ExecutionSettings
from ..robots.kinematics import R5Kinematics
from ..backends.scene_geometry import bench_geometry
from ..robot_roles import ARM_IDENTITY_PROTOCOL, REAL_SKILL_REVISION, arm_identities


def task_execution_guidance(*, include_depth=False) -> str:
    """Shared observation-driven planning protocol; no task script or hidden state."""
    return (
        "On receiving the task, identify its required final state, process constraints and arm roles. "
        "Break it into a short sequence of observable stages appropriate to this task; do not assume "
        "every task involves pickup, insertion or release. Briefly name the stages in your first note, "
        "alongside the first useful action or observation. A separate planning-only call is not required. "
        "Before each decision, compare the last action's expected result with the latest images, "
        "measured robot state and execution feedback. Advance a stage only when its prerequisite is "
        "observed; a sent command, old note or execution-completed flag is not proof of object progress. "
        "Choose an action with a concrete observable outcome, then check that outcome on the next turn. "
        "Use the retained text history to track confirmed progress and failed attempts, but revalidate "
        "anything that may have changed. Use current observations to reassess previous grasp, alignment and "
        "completion assumptions and plan from the latest observation. "
        "When an outcome differs from expectation, identify the observable mismatch and revise the "
        "relevant stage before continuing. After repeated planning rejections, stop extending speculative "
        "detours: reobserve the target, working end, held-object offset and clearance, using "
        + ("depth queries or " if include_depth else "") +
        "an allowed observation-arm move as needed. A no-valid-trajectory response is a planning "
        "failure, not proof that a particular obstacle or corridor caused it. Select a revised target "
        "from new evidence rather than repeating a rejected goal or moving progressively farther away. "
        "Keep each note within the existing 512-character limit: current stage, observed evidence or "
        "mismatch, next action and its expected visible result. Carry forward a brief relevant progress "
        "or failure reminder when needed; do not produce lengthy reasoning or recite the full plan every turn. "
        "Before requesting stop, recheck all required final conditions from current observations, "
        "including support, release and stability when the task requires them. Independent evaluation "
        "alone determines task success."
    )


def manipulation_guidance(config) -> str:
    """Shared operating advice, independent of the task and object layout."""
    opening = config.robot.gripper.width_limits_m[1]
    grasp_advice = (
        "For an engaged part, verify contact through images and small task-appropriate movements along or about its axis; do not bend it with a vertical test lift. "
        if config.grasp_check_mode == "contact_inspection" else
        "Verify a grasp from object motion during a short lift and from current images and feedback. "
    )
    depth_advice = (
        "An empty hole pixel may show the background or a distant surface; query visible rim or object "
        "surfaces and account for their geometry instead of treating one depth point as an object's centre. "
        "A depth sample can hit the tabletop, background or gripper instead of the intended part. "
        "Check that the returned surface and height match the visible part; do not keep an incorrect "
        "sample's XY and merely replace its Z. "
        if config.include_depth else
        "Use complementary views and observed motion to judge alignment and clearance; "
        "a single image projection does not establish distance or contact. "
    )
    return (
        "Use the latest measured state after preparation, execution, timeout or reset. "
        "Read all available views before moving near uncertain geometry. "
        f"{depth_advice}Cross-check complementary views before closing. "
        "After a missed grasp, re-localize and correct the full pose, not only its height. "
        "Use an observation arm when needed, following its role restrictions. "
        "Both arms start at their configured joint home, without a scripted downward or viewing pose. "
        "Choose reachable approach and viewing poses from the measured state; do not assume a ready grasp pose.\n\n"
        "Choose grips that leave functional ends and the later approach path clear. Check clearance "
        "for the complete fingers and carried object, not only the TCP. A held object's centre, axis "
        "and working tip may be offset from the TCP; estimate that relationship from the actual grasp "
        "and use it when choosing subsequent TCP targets. No object-to-hand attachment is provided.\n\n"
        f"When an open grasp approach is needed, set gripper=1 (physical opening {opening:.3f} m) "
        "in a separate action BEFORE approaching: a combined command moves the arm first. "
        "When the task permits full closure, use gripper=0 to close around a rigid object with the configured effort cap. "
        "For deformation-sensitive objects, use intermediate normalized openings and inspect deformation; full closure is not mandatory. "
        "The fingers stop where physical contact permits; the actual measured opening can remain larger than the empty closing target. "
        "Do not widen merely to eliminate this expected width error, but revise the grasp if observations show slip or excessive deformation. "
        f"{grasp_advice}"
        "A completed gripper command or a blocked closing target alone is not grasp evidence. "
        "Recheck that the object is still held before every transport or insertion decision. "
        "If it slips, falls or is no longer visible between the fingers, do not continue the old "
        "insertion plan or assume its previous offset from the TCP is valid. Reobserve and "
        "re-establish the grasp if the task has not ended. Terminal failure ends the episode; "
        "there is no automatic scene reset or another attempt.\n\n"
        "Move efficiently in clear free space; near contact use millimetre-scale corrections and "
        "reobserve. Before transporting, ensure the lowest part of the payload clears intervening "
        "surfaces. The planner checks approximate robot/table/fixture geometry and measured held-object clearance. This is planner-only privileged geometry, not task reasoning or proof of a grasp; moving unheld objects are excluded to allow task contact. A planned path can still deviate under physical contact. Large simultaneous "
        "translation and rotation requests can fail; intermediate reachable targets or separated "
        "orientation changes may help. Never assume the commanded target was achieved.\n\n"
        "If progress stops or unexpected tilt/contact appears, inspect the views and feedback, "
        "withdraw to a clear pose when appropriate, and revise alignment or grasp. Do not repeatedly "
        "send the same blocked target or invent force measurements or unsupported force commands. "
        "After release, withdraw without dragging the object and verify the task's required stable "
        "support and duration using simulation timestamps. Wait actions hold existing drive targets; "
        "ensure those targets are appropriate before waiting. Stop only according to the task goal "
        "and the declared termination contract."
    )



def operating_notes(config) -> dict:
    profile = config.robot
    kinematics = R5Kinematics(profile)
    bench = bench_geometry(config)
    notes = {
        "units": "metres, radians unless explicitly labelled degrees; quaternion wxyz",
        "world_axes": bench.world_axes,
        "base_poses_world_wxyz": {m.name: m.base_pose_wxyz for m in profile.mounts},
        "bench": {"centre_m": bench.centre, "size_m": bench.size, "surface_world_z_m": bench.top},
        "tcp": {"pose_in_link6_wxyz": profile.tcp_pose_wxyz, "source": profile.tcp_source,
                "meaning": "Control point between distal fingers. Command this point directly; do not add a tool offset.",
                "axes": "+X forward along fingers; +Y jaw closing; +Z transverse"},
        "gripper": {"total_width_limits_m": profile.gripper.width_limits_m,
                    "command": "gripper in [0,1]: 0 closes, 1 opens; null retains drive",
                    "calibration_source": profile.gripper.calibration_source},
        "conservative_reach_bound_m": kinematics.reach_bound_m,
        "reach_warning": "Bound is necessary, not sufficient. A local IK failure does not prove global unreachability.",
        "gripper_timing": ExecutionSettings.from_config(config).gripper_timing,
        "contact": "Use measured TCP and RGB to reobserve. A blocked closing target is not proof of a grasp. No force sensor is provided.",
        "information_boundary": "Robot configuration only; no object pose, grasp point, detailed task score, or successful trajectory. A rejected stop discloses only that task completion was not confirmed.",
        "manipulation_guidance": manipulation_guidance(config),
        "task_execution_guidance": task_execution_guidance(include_depth=config.include_depth),
    }
    if config.raw.get("simulation", {}).get("environment") == "tuojing_lab_v4":
        notes["rigid_collision_feedback"] = (
            "collision_blocked means the current action was stopped after measured contact "
            "by a non-finger arm part/camera/mount, or contact with the other robot. "
            "Normal finger-object contact is allowed. Use RGB and measured pose to decide "
            "a clear retreat or another approach; there is no automatic recovery. This binary "
            "simulation contact flag is not a calibrated force sensor or a task-success verdict."
        )
        notes["arm_naming"] = {
            "protocol": ARM_IDENTITY_PROTOCOL,
            "real_skill_revision": REAL_SKILL_REVISION,
            "reference": "Fixed overview camera, facing the two robots. Arm IDs refer to base locations, not current TCP positions.",
            "simulation_left": "Robot on overview-image LEFT; corresponds to the real setup's RIGHT arm.",
            "simulation_right": "Robot on overview-image RIGHT; corresponds to the real setup's LEFT arm.",
            "commands_and_cameras": "Use simulation IDs in commands. left_wrist belongs to simulation left; right_wrist belongs to simulation right. Wrist image orientation does not rename arms. Task instructions already use simulation IDs; do not swap them again.",
            "by_command_arm": arm_identities(config),
            "coordinate_rule": "Converting real arm names does not mirror object positions or motion coordinates. Both simulation arms use the same absolute world frame; the real robot's two base frames and historical CAN labels are not command frames here. An arm crossing the image centre retains its ID, role and wrist camera.",
        }
        from ..video_delivery import enabled, PROMPT_FRAME
        if enabled(config):
            notes["arm_naming"]["commands_and_cameras"] = PROMPT_FRAME
        notes["real_skill_observation_guidance"] = (
            "Use the overview to identify robot bases and the entire approach corridor, "
            "then wrist views to inspect fingers and contact. Include the camera housing, "
            "mount and held object when checking clearance. To resolve uncertain object motion, "
            "compare it with a stationary support or background in a view held still across the move; "
            "a camera moving with the hand can hide slip or make a stationary object seem to move. "
            "Reframing by rotating a wrist does not itself add a translational viewing baseline. "
            "Keep the working arm holding while an allowed observation arm repositions, inspect "
            "the new images, then resume work. A support/grasp role requires manipulation permission. "
            "Only the current task defines required release, return-home and success conditions."
        )
    camera_path = config.raw.get("simulation", {}).get("camera_profile")
    if camera_path:
        from ..cameras.profile import load_camera_rig
        notes["camera_roles"] = {c.name: {"model": c.model, "parent": c.parent}
                                 for c in load_camera_rig(config.source.parent / camera_path).select(config.cameras)}
    notes["motion_backend"] = "curobo_trajectory"
    notes["startup_preparation"] = (
        "Before your first observation the arms settle at their configured joint home. "
        "No scripted downward movement or observation-arm aiming has occurred. Use the measured TCP state. "
        "You may move an observation arm for a clearer view using the same motion interface, "
        "within its declared role restrictions."
    )
    if config.initial_state_description:
        notes["startup_preparation"] += " " + config.initial_state_description
    return notes
