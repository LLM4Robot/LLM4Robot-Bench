"""Native, pre-loosened nut removal. No scripted manipulation or thread joint."""

import math
from dataclasses import dataclass, replace

import numpy as np

from ...robots.transforms import pose_transform, transform_pose
from ...types import Assessment, InvalidSimulationState
from ..bench.geometry import design
from ..bench.spec import BenchSpec
from ..l1.specs import Article, pose
from ..bench.native_l2_specs import BOX_TOP, BOX_CENTER, support_box


def nut_spec():
    # The assembled 148 mm span is centred between the two R5 bases.
    # Native +Z points along world +X: body image-left, nut image-right.
    base = pose(BOX_CENTER[0] - 0.074, BOX_CENTER[1], BOX_TOP + 0.0145, pitch=math.pi / 2)
    lift = 0.014  # Seven whole RH turns; 5 mm / 2.5 turns remain below the tip.
    nut_pose = transform_pose(
        pose_transform(base) @ pose_transform(pose(0, 0, 0.121 + lift))
    )
    articles = (
        Article(
            "anchor",
            (),
            base,
            base,
            0.1856324,
            asset="tuojing:anchor_removable_nut",
            asset_bounds=((-0.0095, -0.0095, 0), (0.0095, 0.0095, 0.14)),
        ),
        Article(
            "nut",
            (),
            nut_pose,
            nut_pose,
            0.0320911,
            asset="tuojing:anchor_removable_nut",
            asset_bounds=((-0.012, -0.0138564, 0), (0.012, 0.0138564, 0.013)),
        ),
    )
    return BenchSpec(
        replace(
            design("L3-02", "双臂旋下预松螺母", articles),
            version="l3-05-native-stud-right-nut-v4",
            release=True,
            provenance="2026-10-11 Feishu independent conical stud/nut with SDF thread contact",
        ),
        arms=("left", "right"),
        mechanism="native_bimanual_sdf_nut_removal",
        limitations="2026-10-11 native 140 mm conical stud, 24 mm hex nut, 2 mm RH pitch. Native SDF contacts, estimated mass/friction. Nut pre-loosened 7 turns/14 mm leaving 5 mm engagement; not real preload calibration. No thread joint or scripted motion.",
        facts="The 140 mm silver stud lies horizontally on the green box centre between both arms, body toward image-left and 24 mm hex nut toward image-right. LEFT picks and holds the 14 mm smooth shaft; RIGHT pinches the nut flats and unscrews counterclockwise viewed from the exposed threaded end. Lift clear of the box before bringing the second hand in. Nut thickness 13 mm, 2 mm pitch, approximately 2.5 turns remain. Use actual relative movement, not whole-assembly rotation, to check progress. Once fully detached, place both parts safely on the box or table and release.",
        parameters={
            "table_z": 0.78, "support_z": BOX_TOP,
            "thread_pitch_m": 0.002,
            "shaft_tip_z": 0.14,
            "initial_nut_bottom_z": 0.135,
            "nut_height_m": 0.013,
            "body_arm": "left",
            "nut_arm": "right",
            "hold_s": 0.0,
            "native_instances": [support_box(),
                {
                    "name": "anchor_assembly",
                    "asset": "anchor_removable_nut",
                    "pose": base,
                    "tracked": {"anchor": "AnchorBody", "nut": "Nut"},
                    "initial_body_transforms": {
                        "Nut": pose(0, 0, lift, yaw=14 * math.pi)
                    },
                }
            ],
        },
    )


def gripped(row, arm):
    return min(row["fingers"][arm]) > 0.005


@dataclass
class NutProgress:
    previous: tuple | None = None
    net_turn: float = 0.0
    net_rise: float = 0.0

    def update(self, yaw, z, controlled):
        if self.previous is not None:
            angle, height, was_controlled = self.previous
            turn = math.atan2(math.sin(yaw - angle), math.cos(yaw - angle))
            # Only measured relative motion while BOTH parts are gripped is evidence.
            # Translation/rotation of the whole assembly gives zero relative progress.
            if controlled and was_controlled and abs(turn) < math.pi / 2:
                self.net_turn += turn
                self.net_rise += z - height
        self.previous = (yaw, z, controlled)


class NutEvaluator:
    def __init__(self, spec):
        self.spec = spec
        self.progress = NutProgress()
        self.last_time = None
        self.pickup_seen = False
        self.removal_seen = False

    def update(self, now, rows):
        if not math.isfinite(now) or (
            self.last_time is not None and now <= self.last_time
        ):
            raise InvalidSimulationState(
                "Nut measurements require increasing finite time"
            )
        self.last_time = now
        body, nut = rows["anchor"], rows["nut"]
        for row in rows.values():
            values = [
                *row["pose"],
                *row["linear_velocity"],
                *row["angular_velocity"],
                row["bench_force_n"],
                *row["pair_forces"].values(),
                *[v for pair in row["fingers"].values() for v in pair],
                *np.asarray(row["corners"]).ravel(),
            ]
            if not np.isfinite(values).all():
                raise InvalidSimulationState("Nonfinite native nut measurement")
        bt, nt = pose_transform(body["pose"]), pose_transform(nut["pose"])
        relative = np.linalg.inv(bt) @ nt
        local = (np.asarray(nut["corners"]) - bt[:3, 3]) @ bt[:3, :3]
        body_held, nut_held = gripped(body, self.spec.parameters["body_arm"]), gripped(nut, self.spec.parameters["nut_arm"])
        if (
            body_held
            and min(c[2] for c in body["corners"])
            > self.spec.parameters["support_z"] + 0.005
        ):
            self.pickup_seen = True
        # Nut's entire bound must clear the shaft end, in the CURRENT body frame.
        gap = float(local[:, 2].min() - self.spec.parameters["shaft_tip_z"])
        contact = float(nut["pair_forces"].get("anchor", 0.0))
        # After removal, either part can be rotated or placed beside the other.
        # Do not require the final placement to retain the extraction axis.
        center = local.mean(axis=0)
        radius = np.linalg.norm(local[:, :2] - center[:2], axis=1).max()
        beside = np.linalg.norm(center[:2]) - radius > 0.0095 + 0.0003
        detached = bool((gap > 0.0003 or beside) and contact < 0.005)
        yaw = math.atan2(relative[1, 0], relative[0, 0])
        self.progress.update(
            yaw, float(relative[2, 3]), body_held and nut_held and self.pickup_seen
        )
        # Noise floor for a rotation/rise witness, not a prescribed number of turns.
        turned = (
            self.progress.net_turn > math.radians(2)
            and self.progress.net_rise > 0.00005
        )
        if gap > 0.0003 and detached and body_held and nut_held and self.pickup_seen and turned:
            self.removal_seen = True
        from ..bench.evaluation import released
        placed = all(released(r) and r["bench_force_n"] + r["fixture_force_n"] > 0.02
                     for r in (body, nut))
        release_required = self.spec.parameters.get("release_required", True)
        complete = detached and self.removal_seen and (placed or not release_required)
        fallen = [
            n
            for n, r in rows.items()
            if min(c[2] for c in r["corners"]) < self.spec.parameters["table_z"] - 0.08
        ]
        metrics = {
            "task_version": self.spec.design.version,
            "mechanism": self.spec.mechanism,
            "limitations": self.spec.limitations,
            "objects": rows,
            "criteria": {
                "body_picked_up": self.pickup_seen,
                "body_held": body_held,
                "nut_held": nut_held,
                "relative_turn_rad": self.progress.net_turn,
                "relative_rise_m": self.progress.net_rise,
                "rotation_witness": turned,
                "tip_clearance_m": gap,
                "thread_contact_n": contact,
                "nut_detached": detached,
                "bimanual_removal_observed": self.removal_seen,
                "release_required": release_required,
                "parts_released_supported": placed,
                "timed_hold_s": 0,
            },
            "goal_complete": bool(complete),
            "process_constraints_satisfied": not fallen,
        }
        if fallen:
            return Assessment(
                "failed", "Object fell below the table: " + ", ".join(fallen), metrics
            )
        return Assessment(
            "succeeded" if complete else "running",
            (
                ("Nut removed and both parts released onto a support" if release_required
                 else "Nut fully unscrewed and detached")
                if complete
                else "Waiting for bimanual pickup and nut removal"
            ),
            metrics,
        )


class NutTask:
    @staticmethod
    def prepare_config(config):
        from ...video_delivery import enabled, robot_profile
        if enabled(config):
            config = replace(config, robot=robot_profile(config.robot, config.scene_id))
        if config.arms != ("left", "right") or config.observation_arms:
            raise ValueError(
                "L3-02 nut removal requires both manipulation arms and no observation-only arm"
            )
        if config.physics_dt_s > 1 / 1920 + 1e-12:
            raise ValueError("Native fine threads require physics_dt_s <= 1/1920")
        if config.robot.gripper.width_limits_m[0] >= 0.014:
            raise ValueError("Native 14 mm stud needs a smaller minimum opening")
        if config.raw.get("simulation", {}).get("environment") != "tuojing_lab_v4":
            raise ValueError("Native nut task requires the Tuojing laboratory scene")
        from ..completion import completion_instruction

        from ..bench.english_reference import uses_reference
        facts = "" if uses_reference(config.raw["task"]) else nut_spec().facts
        return replace(
            config,
            instruction=config.instruction + (
                "" if facts in config.instruction else "\n" + facts
            ) + completion_instruction(config),
            grasp_check_mode="contact_inspection",
            initial_state_description="Both hands start empty; the assembly rests on the green support box, with the nut pre-loosened but still threaded on the body. No robot attachments.",
        )

    def build_scene(self, backend, config):
        from ..bench.native_l2 import NativeL2Scene

        self.config = config
        from ..bench.english_reference import apply_reference
        from ...video_delivery import enabled
        if enabled(config):
            from ..bench.delivered_scene import DeliveredScene
            self.scene = DeliveredScene(backend, config, apply_reference(nut_spec(), config.raw["task"]))
        else:
            self.scene = NativeL2Scene(backend, config, apply_reference(nut_spec(), config.raw["task"]))

    def reset(self, backend, seed):
        self.scene.reset(seed)
        self.evaluator = NutEvaluator(self.scene.spec)
        self._last_time = self._assessment = None

    def evaluate(self, state, feedback):
        now = state.values["simulation_time_s"]
        if now != self._last_time:
            from ..completion import require_home

            self._assessment = require_home(self.evaluator.update(now, self.scene.measure()),
                                            state.values["robot"], self.config)
            self._last_time = now
        return self._assessment
