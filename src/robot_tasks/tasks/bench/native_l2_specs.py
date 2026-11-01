"""L20 task layouts and goals in the delivered laboratory's metre/world frame."""

import math
from dataclasses import replace

from ..l1.specs import Article, box, pose
from .geometry import design
from .spec import BenchSpec

TABLE_Z = 0.78
BOX_CENTER = (0.19565, 1.30)
BOX_TOP = TABLE_Z + 0.0935  # Native recessed centre; rim is 100 mm high.


def support_box(x=BOX_CENTER[0], y=BOX_CENTER[1]):
    return {"name": "support_box", "asset": "green_support_box",
            "pose": pose(x, y, TABLE_Z), "static": [""], "tracked": {}}



def marker():
    initial = pose(BOX_CENTER[0] - 0.0755, BOX_CENTER[1], BOX_TOP + 0.0098, pitch=math.pi / 2)
    body = Article(
        "body",
        (),
        initial,
        initial,
        0.017,
        asset="tuojing:pullable_marker",
        asset_bounds=((-0.0082, -0.0082, 0), (0.0082, 0.0082, 0.128)),
    )
    cap = Article(
        "cap",
        (),
        initial,
        initial,
        0.005,
        asset="tuojing:pullable_marker",
        asset_bounds=((-0.0087, -0.0087, 0.105), (0.0087, 0.0087, 0.151)),
    )
    return BenchSpec(
        replace(
            design("L2-01", "双臂拔马克笔帽", [body, cap]),
            version="l2-01-detach-contact-v5",
            release=False,
            provenance="Pinned L20 marker body/cap with four passive radial spring shoes",
        ),
        arms=("left", "right"),
        mechanism="native_radial_spring_marker_fit",
        limitations="Native free body/cap and four passive radial springs. Uncalibrated detent/friction surrogate. Five solid support colliders omit decorative grooves; a cap envelope is used only against this box, preserving native barrel/cap and finger contacts. Prior Sharpa fixed-body test is not R5 bimanual validation. No rigid attachment, material damage or active cap force assistance.",
        facts="The capped marker is horizontal at the centre of the recessed green box top. Barrel points image-left; cap image-right. LEFT picks up and holds the barrel; RIGHT pulls the cap axially toward image-right. Native length 151 mm, barrel 16.4 mm, cap 17.6 mm. Both arms manipulate. Lift clear of the 10 cm box before bringing the cap hand in. Complete the task as soon as the cap is fully separated from the barrel. Keep holding the two parts; placement, release and return home are not required. No prescribed final yaw or minimum force.",
        parameters={
            "table_z": TABLE_Z,
            "hold_s": 0.0,
            "support_z": BOX_TOP,
            "body_arm": "left", "cap_arm": "right",
            "body_tip_z": 0.128,
            "cap_mouth_z": 0.105,
            "cap_end_z": 0.151,
            "body_mass_kg": 0.017,
            "cap_mass_kg": 0.005,
            "native_instances": [
                support_box(),
                {
                    "name": "marker",
                    "asset": "pullable_marker",
                    "pose": initial,
                    "tracked": {"body": "Body", "cap": "Cap"},
                }
            ],
        },
    )


def books():
    # Photo layout: box centred and forward; books offset right and slightly
    # towards the arms, leaving exposed box top on image-left and at the front.
    box_x, box_y = BOX_CENTER[0], 1.20
    x, y = box_x + 0.035, box_y + 0.020
    # Books span the raised box rim (100 mm), not its recessed centre.
    zs = [TABLE_Z + 0.100 + 0.00979 + i * 0.0198 for i in range(3)]
    poses = [pose(x, y, zs[0]), pose(x + 0.040, y, zs[1]), pose(x, y, zs[2])]
    names = ("lower_book", "middle_book", "upper_book")
    articles = [Article(n, (), q, q, 0.3789, asset="tuojing:closed_book",
                        asset_bounds=((-0.07, -0.11, -0.00979), (0.07, 0.11, 0.00979)))
                for n, q in zip(names, poses)]
    instances = [support_box(box_x, box_y)] + [
        {"name": a.name, "asset": "closed_book", "pose": a.initial,
         "tracked": {a.name: ""}} for a in articles]
    return BenchSpec(
        replace(design("L2-02", "双臂支撑并抽取中层书", articles),
                version="l2-03-photo-offset-right-pull-v7", release=True),
        arms=("left", "right"), mechanism="native_book_free_neighbours",
        limitations="Native rigid closed books, 378.9 g each; no page flex. All books are dynamic. Support box anchored as a passive fixture. Friction uncalibrated.",
        facts="Three 220 x 140 x 19.6 mm books lie flat on a green box centred between both arms, farther forward on the table. The stack is offset 35 mm toward overview-image RIGHT and 20 mm toward the arms relative to the box centre, exposing box top on the left and front. The middle book protrudes another 40 mm to image-right (world +X). All books can move. RIGHT pinches the exposed middle book across its thickness, LEFT steadies the upper book. Pull the middle book visibly farther toward image-right; 20 mm of measured displacement operationalizes the real task's relaxed pull-out-a-little goal. Complete extraction is not required. Slight neighbouring-book movement is allowed, but they must remain supported on the stack/box. Release the books after the pull; do not let them drop off the support.",
        parameters={"table_z": TABLE_Z, "hold_s": 0.0,
                    "outward_axis_world": (1.0, 0.0, 0.0),
                    "manipulation_arm": "right", "support_arm": "left",
                    "minimum_travel_m": 0.02, "support_z": TABLE_Z + 0.10,
                    "native_instances": instances})


def orange():
    x, y = BOX_CENTER
    initial = pose(x, y - 0.00895, BOX_TOP + 0.002)
    article = Article("orange", (), initial, initial, 0.0192,
        asset="tuojing:orange_segment",
        asset_bounds=((-0.025, -0.00740, 0), (0.025, 0.02530, 0.02530)))
    return BenchSpec(
        replace(design("L2-04", "绿盒橘子抬升并放回", [article]),
                version="l2-06-overview-right-lift-return-v11", release=True),
        arms=("right",),
        mechanism="native_fem_orange_lift",
        limitations="Native 19.2 g GPU FEM orange, 50 x 32.7 x 25.3 mm. Material overrides are engineering estimates. No tissue damage claim. Support and release are measured geometrically for FEM; no grasp attachment.",
        facts="The rounded orange lies flat, long axis across the image (world X), centred on the recessed green box top. RIGHT grasps its fuller middle and lifts the ORANGE 5-20 cm relative to its original rest. Firm squeezing is allowed. Then lower it back onto the SAME box, release and withdraw. No lateral transfer destination or exact original point is required; anywhere safely on the original box is acceptable. LEFT observes only.",
        parameters={"table_z": TABLE_Z, "scoring_protocol": "soft_box_lift_return_v7",
                    "soft_asset": "orange_segment", "lift_m": 0.05,
                    "max_lift_m": 0.20, "support_z": BOX_TOP,
                    "pedestal_asset": "green_support_box",
                    "pedestal_pose": pose(x, y, TABLE_Z),
                    "support_half_extents_m": (0.087, 0.092),
                    "finger_proximity_m": 0.003})


def drawer():
    x, y = 0.55, 1.12  # Photo match: far image-right; keep the 100 mm front clearance.
    yaw = -2 * math.pi / 3  # Front/rail points world -X/+Y, toward the arms.
    cabinet = pose(x, y, TABLE_Z, yaw=yaw)
    from ...robots.transforms import pose_transform, transform_pose
    import numpy as np
    t = pose_transform(cabinet)
    opening = 0.030
    closed = transform_pose(t @ pose_transform(pose(-0.0833, 0, 0.053)))
    outward = t[:3, :3] @ np.array([0.0, -1.0, 0.0])
    initial = transform_pose(t @ pose_transform(pose(-0.0833, -opening, 0.053)))
    a = Article("drawer", (), initial, initial, 0.080, asset="tuojing:desk_organizer",
                asset_bounds=((-0.0545, -0.0735, 0), (0.0545, 0.0695, 0.040)))
    offsets = {"Drawer1": (0, -opening, 0)}
    offsets.update({f"Contents/Cube{i:02}": (0, -opening, 0) for i in range(1, 6)})
    return BenchSpec(
        replace(design("L2-05", "预开抽屉拉出并推回", [a]),
                version="l2-07-overview-right-pull-return-v10", release=True),
        arms=("right",), mechanism="native_prismatic_drawer_pull_return",
        limitations="Delivered cabinet fixed, lower drawer fixed; original upper rail and passive resistance retained. Five dynamic 20 mm cubes remain loose in the upper drawer. Friction and cabinet dimensions uncalibrated.",
        facts="The white organizer is at the far RIGHT of the overview image, diagonal to the camera as in the real setup. Its centre is 70 mm farther right than the previous layout and retains the 100 mm table-front clearance shift. Part of the cabinet may be outside the overview frame; use wrist views when necessary. Its upper drawer starts 30 mm open, original grip edge facing diagonally toward the robots (world -X/+Y), away from the overview camera. Pulling moves the drawer toward the arms; pushing returns it away from the arms. The cabinet and lower drawer are fixed. RIGHT grips the existing edge/handle, pulls the upper drawer about 50 mm FURTHER along its rail, then pushes it back to its INITIAL 30 mm opening and releases. It should not finish fully closed. LEFT observes. Five blue cubes are loose inside; no object-removal task and no added handle.",
        parameters={"table_z": TABLE_Z, "closed_pose": closed,
                    "outward_axis_world": tuple(outward), "manipulation_arm": "right",
                    "initial_open_m": opening, "required_open_m": opening + 0.05,
                    "minimum_grasped_pull_m": 0.005, "return_tolerance_m": 0.008,
                    "scoring_protocol": "drawer_pull_return_v5", "max_travel_m": 0.09,
                    "max_speed_m_s": 0.25, "hold_s": 0.0,
                    "native_instances": [{"name": "organizer", "asset": "desk_organizer",
                        "pose": cabinet, "static": ["Body", "Drawer0"],
                        "disabled_joints": ["Joints/Drawer0"],
                        "initial_body_offsets": offsets, "tracked": {"drawer": "Drawer1"}}]})
