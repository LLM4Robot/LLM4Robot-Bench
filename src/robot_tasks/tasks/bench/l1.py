"""L1 props in the delivered laboratory."""

import math
from dataclasses import replace

from ..l1.specs import BLUE, GREY, Article, box, pose
from .geometry import cup_parts, design
from .spec import BenchSpec


def marker():
    from .native_geometry import cup_cavity

    # Native delivery coordinates are bottom-centred, metres, Z-up. The cup is
    # the fixed pen receptacle; no hidden joint is used to keep the marker up.
    # Looking from the front of the table (+Y): mug on image left, marker on
    # image right, at the same depth. Photo-proportional, not a measured calibration.
    table_z, holder_xy = 0.78, (0.04, 1.22)
    _, cup_floor, cup_rim = cup_cavity('black_ceramic_mug')
    a = Article(
        "marker", (), pose(0.34, 1.22, table_z + 0.001), pose(*holder_xy, table_z + cup_floor), 0.022,
        asset="tuojing:whiteboard_marker",
        asset_bounds=((-0.00898, -0.00898, 0.0), (0.00898, 0.00898, 0.1512)),
    )
    d = replace(design("L1-01", "白板笔入固定黑色陶瓷杯", [a], (), friction=0.52),
                fixture_asset="tuojing:black_ceramic_mug",
                fixture_pose=pose(*holder_xy, table_z, yaw=-math.pi / 2),
                version="l1-01-tuojing-marker-ceramic-mug-v8-nearest-arm",
                provenance="Tuojing locked 2026-10-05 marker and 2026-10-09 ceramic mug; photo-proportional front-view layout, handle toward table front")
    return BenchSpec(
        d,
        facts="The capped black whiteboard marker is 0.151 m tall; barrel diameter is 0.0162 m and maximum visible width about 0.018 m. The fixed black ceramic mug with white interior serves as the pen holder: height 0.090 m, inside mouth diameter about 0.085 m, floor 0.007 m above its bottom. It has a bulging body and an open handle; the target is the main cup cavity, not the handle hole. Both are rigid. Keep the cap on. L1 success is genuine entry of either marker end into the main cavity, without significant wall or floor penetration. Holding the marker, moving and leaning are allowed. Releasing above the opening and letting the marker fall into the cavity is also valid; release alone is not success, actual entry must occur. Release, cup support, bottom seating and a stable waiting period are not required. There is no prescribed angle or 40 mm insertion depth. Merely aligning above the mouth or resting across its rim is insufficient. Returning to or toppling on the tabletop permits recovery in the same scene; falling below the table still ends the episode. There is no automatic reset. No absolute prop coordinates are provided.",
        limitations="Photo-estimated native dimensions and initial layout; rigid fixed ceramic mug retains delivered static/dynamic contact friction. The reference real mug is free-standing, whereas this task retains its fixed-receptacle protocol. Flat marker barrel collider replaces a rounded-bottom capsule for upright reset; 0.2 mm contact margin and configured velocity solver iterations. No material deformation or hardware calibration.",
        parameters={
            "scoring_protocol": "l1_entry_only_v1",
            "hold_s": 0.0,
            "table_z": table_z,
            "holder_xy": holder_xy,
            "rim": table_z + cup_rim,
            "floor": table_z + cup_floor,
            "radius": 0.029,
            "mouth_radius": 0.0425,
            "marker_visual_radius": 0.00898,  # Visual label/ridges only; never a scoring envelope.
            "marker_height": 0.1512,
            "min_support_n": 0.022 * 9.81 * 0.5,
        },
    )


def plug():
    table_z, socket_xy = 0.78, (0.02, 1.20)
    source_xy = (0.35, 1.20)
    strip_yaw = math.pi / 2
    # These payload samples follow the three native convex colliders. A single
    # enclosing box would fill the gap between the two blades in the planner.
    parts = (
        box("housing", (0.049, 0.027, 0.053), (0, 0, 0.0435), BLUE),
        box("blade_left", (0.0015, 0.0063, 0.018), (-0.00685, 0, 0.009), GREY),
        box("blade_right", (0.0015, 0.0063, 0.018), (0.00685, 0, 0.009), GREY),
    )
    a = Article(
        "plug", parts, pose(*source_xy, table_z + 0.0045),
        pose(*socket_xy, table_z + 0.01315, yaw=strip_yaw), 0.055,
        asset="tuojing:usb_charger",
        asset_bounds=((-0.0245, -0.01355, 0), (0.0245, 0.0135, 0.07075)),
    )
    # The native 20.8-g holder is free-standing, not attached to the charger or
    # table. Its recessed floor supports the housing; both blades enter the
    # 22-mm through-hole. Start 1.5 mm above nominal shoulder seating and settle.
    base_pose = pose(*source_xy, table_z + 0.001)
    base = Article("plug_storage_base", (), base_pose, base_pose, 0.0208,
                   asset="tuojing:plug_storage_base",
                   asset_bounds=((-0.0507, -0.0507, 0), (0.0507, 0.0507, 0.055)))
    d = replace(design("L1-02", "USB 充电器插入固定插线板", [a, base], (), friction=0),
                fixture_asset="tuojing:power_strip", fixture_pose=pose(*socket_xy, table_z, yaw=strip_yaw),
                version="l1-03-tuojing-charger-holder-v4-entry-only",
                release=False,
                provenance="Tuojing locked charger, strip and 2026-10-09 free-standing plug holder; photo-proportional strip-left/charger-right layout, not calibrated")
    return BenchSpec(
        d,
        facts="The white USB charger starts upright, blades downward, naturally supported in a black recessed storage base on the right side of the table. The free-standing base is about 101 x 101 x 55 mm and 20.8 g, with a 22-mm central through-hole; it is passive storage, not the target socket. Grip the exposed upper plastic housing (whole housing 49 x 27 x 53 mm), not the base or metal blades. Lift vertically clear of the recess before translation or rotation; the base should remain on the table. The fixed unpowered white strip lies horizontally on the left, with its coiled cable behind it. Insert both blades downward into the MIDDLE blue two-slot socket. Blade spacing is 13.7 mm; each blade is 1.5 x 6.3 mm. Each target slot is 2.6 x 8.2 mm. Native side clearance is only 0.55 mm across blade thickness. There is no latch or powered circuit. L1 success requires both blades genuinely entering their corresponding middle slots without significant wall, floor or housing penetration. Holding, motion and partial insertion are allowed. No 15-mm insertion depth, housing seating, release, support force, grasp history or stable waiting period is required. Alignment above the face, rim contact and only one blade entering are insufficient. Entry must exceed the existing 0.15-mm numerical contact tolerance; this is not a task-depth target.",
        limitations="Photo-estimated layout and rigid native props, not hardware calibrated. Free-standing 20.8-g storage base retains its hollow native colliders; shared task object friction is 0.8. Fixed strip keeps native slots, 0.55 mm minimum side clearance and zero fixture friction. No electrical contacts, spring retention or flexible cable.",
        parameters={"table_z": table_z, "socket_xy": socket_xy,
                    "scoring_protocol": "l1_plug_entry_only_v1", "hold_s": 0.0,
                    "slot_centres": ((-0.00685, 0), (0.00685, 0)),
                    "opening": (0.0026, 0.0082), "tongue": (0.0015, 0.0063),
                    "floor": table_z + 0.01315, "rim": table_z + 0.03015,
                    "seat_gap_m": 0.001, "contact_tolerance_m": 0.00015},
    )


def hat():
    from ...robots.transforms import pose_transform, transform_pose
    from .native_geometry import hook_geometry
    import numpy as np

    # Present Hook0 toward the left arm (+Y), with a shorter approach from its
    # base. Keep native asset dimensions and the selected support peg intact.
    rack_yaw = 2 * math.pi / 3
    rack_pose = pose(-0.04, 1.09, 0.78, yaw=rack_yaw)
    source_xy = (-0.23, 1.33)
    start, end, radius = hook_geometry(rack_pose)
    direction = (end-start) / np.linalg.norm(end-start)
    up = np.array((0.,0.,1.))
    band_y = up - direction * direction[2]
    band_y /= np.linalg.norm(band_y)
    rotation = np.column_stack((np.cross(band_y,direction),band_y,direction))
    goal = np.eye(4); goal[:3,:3] = rotation
    # The upper rear inner band bears on the shaft, with its round tip beyond
    # the opening. This reference pose is evaluator/planner data, never a policy target.
    goal[:3,3] = start + .70*(end-start) - rotation @ np.array((0.,.098-radius,.035))
    # Turn the brim away from the rack/source-side legs into the open +Y side.
    article = Article('hat', (), pose(*source_xy,.806,yaw=math.pi), transform_pose(goal), .065,
                      asset='tuojing:yellow_visor',
                      asset_bounds=((-0.115833,-0.195,0.0025),(0.115833,0.100,0.087)))
    # A 45-mm passive cradle exposes the underside of the brim for a grasp
    # spanning brim-to-band. The 2-mm band alone cannot be pinched by this gripper.
    rack_transform = pose_transform(rack_pose)
    supports = tuple(replace(
        box('hat_source_pad_'+str(i),(.025,.035,.045),
            tuple((np.array((source_xy[0]+sx*.050,source_xy[1]+sy*.083,.8025))-rack_transform[:3,3])
                  @ rack_transform[:3,:3]), GREY),
        rotation=(0.,0.,-rack_yaw))
        for i,(sx,sy) in enumerate(((-1,-1),(1,-1),(-1,1),(1,1))))
    return BenchSpec(
        replace(design('L1-03','黄色空顶帽挂上木质挂架',[article],supports),
                fixture_asset='tuojing:wooden_rack', fixture_pose=rack_pose,
                version='l1-07-tuojing-native-visor-v4-left-facing-layout',
                provenance='Locked native yellow visor and wooden rack; no scaling, cloth approximation remains rigid'),
        facts="The yellow open-top visor is ONE rigid native prop, about 232 x 295 x 85 mm and 65 g. Its oval head opening is about 176 x 196 mm. The gripper has a nominal 0-70 mm opening. Four 45-mm passive source pads expose the brim underside: pinch the band upper edge from above with vertical distal fingers (one inside, one outside), or pinch an exposed brim edge. Keep the fingers clear of supports and the peg insertion region. Use gentle closure and verify a short lift. Lift the whole visor. Hang its HEAD OPENING on the LOWEST sloping 20-mm peg of the fixed 82-cm wooden rack, letting the rear inner band bear on that peg. Keep the wide brim and both fingers clear of the trunk and other pegs. The round tip must pass beyond the loop, not merely support a rim-balanced hat. Release and inspect actual support on the selected peg. Passive sway is allowed; no stability waiting period.",
        limitations="Photo-estimated native stiff visor represented as one rigid body with 208 separate convex band/brim pieces; no cloth deformation. Native 82-cm rack fixed to bench; 45-mm passive pickup pads only, no hidden attachment or enlarged aperture.",
        parameters={'table_z':.78,'hold_s':0.,'hook_start':tuple(start),'hook_end':tuple(end),
                    'hook_radius':radius,'band_plane_z':.035,
                    'target_hook_suffix':'/Colliders/Hook0',
                    'min_hook_support_n':.065*9.81*.5},
    )


def cups():
    # Both cups retain the same native dimensions, hollow walls and 14 g mass.
    # Photo-proportional layout, not calibrated: viewed from the table front,
    # the source is on the left and the receiver is 30 cm right, 2 cm forward.
    source_xy, receiver_xy, table_z = (0.04, 1.20), (0.34, 1.18), 0.78
    bounds = ((-0.0426, -0.0426, 0), (0.0426, 0.0426, 0.107594535))
    # Hollow payload samples preserve the opening instead of filling it with an AABB.
    parts = cup_parts(bottom_radius=0.025, top_radius=0.0393, height=0.1075,
                      wall=0.003, segments=48)
    lower = Article("lower_cup", parts, pose(*receiver_xy, table_z + 0.001), pose(*receiver_xy, table_z),
                    0.014, asset="tuojing:ripple_paper_cup", asset_bounds=bounds)
    upper = Article("upper_cup", parts, pose(*source_xy, table_z + 0.001), pose(*receiver_xy, table_z + 0.025),
                    0.014, asset="tuojing:ripple_paper_cup", asset_bounds=bounds)
    return BenchSpec(
        replace(design("L1-04", "两个波纹纸杯套叠", [lower, upper]),
                version="l1-08-tuojing-identical-cups-v4-photo-layout",
                provenance="Two native hollow ripple paper cups; photo-proportional left-source/right-receiver layout, receiver slightly toward the table front; not calibrated; no resizing"),
        facts="Two identical empty red-brown ripple cups start upright. Viewed from the table front toward the robots, move the initially LEFT cup into the initially RIGHT cup, which starts slightly toward the table front. Both are movable. Each is 107.5 mm tall, outer mouth about 85 mm wide, lower-body diameter about 56-67 mm within the bottom 40 mm. Gripper opening is nominally 0-70 mm: the full mouth is too wide. A shallow pinch across one rigid wall near the rim is possible from above with vertical fingers, one distal pad inside and one outside. Alternatively grip opposite outer sides of the narrower lower body, with both finger tips clear of the table. Lift and verify support, carry upright, and centre its bottom over the other cup's mouth. A recommended release point is with the cup bottom 5-8 mm INSIDE the mouth and before any finger reaches the rim. Open separately and inspect settling. Opening does not prove detachment; choose a small clear withdrawal probe and inspect whether the cup follows before a full retreat. If release_check feedback is enabled, it is advisory; the model chooses recovery. Gravity may finish settling the entered cup onto the taper. A 20-35 mm rim separation is typical, not a hard scoring band. Completion uses geometric cavity fit and stable support, without a 3 mm centre cutoff or fixed small-angle gate. No yaw target is required. Temporary tilt or a missed placement can be recovered in the same scene without reset.",
        limitations="Native photo-estimated rigid paper cups, identical taper and size; no paper crushing. Rigid native cup walls; nominal 0-70 mm fingers permit a rim-wall pinch or a lower-body grasp.",
        parameters={"table_z": 0.78, "cup_height": 0.1075, "entry_depth": 0.008,
                    "cup_outer_bottom_radius": 0.028, "cup_outer_top_radius": 0.042,
                    "cup_inner_bottom_radius": 0.025, "cup_inner_top_radius": 0.0393,
                    "cup_floor": 0.004, "cup_contact_tolerance": 0.0015,
                    "min_pair_support_n": 0.014 * 9.81 * 0.5,
                    "min_bench_support_n": 0.028 * 9.81 * 0.5},
    )


def flower():
    # Physics, scoring and carried-object planning share the native bent stem
    # and bud colliders. Do not widen the stem at its tabletop pickup contact.
    table_z, holder_xy = 0.78, (0.105, 1.115)
    rose = Article(
        # Native collider/COM resting pose: both the tip and head start near
        # support, with 0.01 mm clearance, rather than dropping a horizontal rose.
        # Long axis remains along +X. Free dynamics still settle this reset pose.
        "rose", (), (-0.28, 1.22, table_z + 0.0009854593730216,
                     0.2951003515573617, -0.6265583937983512,
                     0.27773423465629116, -0.6657357257737007),
        pose(*holder_xy, table_z + 0.0041), 0.008,
        asset="tuojing:artificial_rose",
        asset_bounds=((-0.0178, -0.0178, -0.000014), (0.0179, 0.0179, 0.3018)),
    )
    d = replace(
        design("L1-05", "刚体玫瑰插花", [rose], ()),
        fixture_asset="tuojing:ripple_paper_cup", fixture_pose=pose(*holder_xy, table_z),
        version="l1-09-tuojing-rigid-rose-v4-settled-reset",
        provenance="Native locked Tuojing rigid artificial rose and fixed ripple paper cup; no scaling or bending joints",
    )
    return BenchSpec(
        d,
        mechanism="rigid_rose_in_fixed_cup",
        facts="The red artificial rose is ONE rigid object about 302 mm long and 8 g, with a thin green stem about 2.1 mm thick and a bud about 34-36 mm wide. It starts lying on the table. The thin stem is difficult to capture reliably; prefer opposite sides of the wider bud or calyx without blocking the free stem tip. The source asset's stem grasp marker is not suitable for this gripper. The entire rose moves rigidly and does not bend. Place the free stem end into the fixed red-brown ripple paper cup (107.5 mm tall, inside mouth about 78.6 mm). Keep the bud above the rim, place the free stem end genuinely inside the cavity, lower gently and release for cup support. No fixed tilt angle or 25 mm insertion depth is required. Natural leaning against the inner wall is allowed. No soft-stem, petal-deformation or fracture behaviour is evaluated.",
        limitations="Photo-estimated native rigid artificial rose and rigid fixed paper cup. Stem, calyx and petals move together; no bending or damage model. Carried-object planning samples the same four bent stem capsules and bud colliders used by native physics and scoring.",
        parameters={"table_z": table_z, "vase_xy": holder_xy,
                    "floor": table_z + 0.004, "rim": table_z + 0.1075,
                    "radius": 0.025, "mouth_radius": 0.0393,
                    "stem_radius": 0.00105,
                    "min_support_n": 0.008 * 9.81 * 0.5},
    )


BUILDERS = {
    "L1-01": marker,
    "L1-02": plug,
    "L1-03": hat,
    "L1-04": cups,
    "L1-05": flower,
}
