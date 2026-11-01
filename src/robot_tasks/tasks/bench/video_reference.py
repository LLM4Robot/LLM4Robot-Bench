"""Bind goals/scoring to the current complete delivery, in its native frame."""

from dataclasses import replace
import numpy as np

from ...robots.transforms import pose_transform, transform_pose
from ...video_delivery import DELIVERY, scenes
from ..l1.specs import Article


def apply_delivery(spec, task):
    if task.get("scene_delivery") != DELIVERY:
        return spec
    tid = task["id"]
    baseline = scenes()[tid]
    if tid.startswith("L1-"):
        from .initial_placement import scene_record
        row = scene_record(task)
    elif tid in ('L3-01', 'L3-03'):
        from ...initial_layout import scene_record
        row = scene_record(task)
    else:
        row = scenes()[task.get("scene_variant", tid)]
    from .measured_assets import measured_record, marker_axial
    row = measured_record(row, tid)
    p = {**spec.parameters, "table_z": 0.88, "scene_delivery": DELIVERY,
         "geometry_prefix": "video:", "hold_s": 0.0}
    p.pop("fixed_articles", None)
    d = spec.design
    articles = []
    old = {a.name: a for a in d.articles}
    for name, record in row["tracked"].items():
        values = dict(initial=tuple(record["pose"]), goal=tuple(record["pose"]),
                      mass=record["mass"] or 0.01,
                      asset="delivery:" + name, asset_bounds=tuple(map(tuple, record["bounds"])),
                      parts=())
        articles.append(replace(old[name], **values) if name in old else Article(name=name, **values))
    if tid == "L2-04":
        articles = [replace(d.articles[0], initial=tuple(row["orange"]["pose"]),
                            goal=tuple(row["orange"]["pose"]), asset="delivery:orange")]
    fixture_pose = tuple(row["fixtures"][0]["pose"]) if row["fixtures"] else d.fixture_pose
    d = replace(d, articles=tuple(articles), fixtures=(), fixture_pose=fixture_pose,
                version=d.version + "-complete-video-scene", provenance="Complete 2026-10-15 task scene with declared physical-binding and video-orientation repairs")
    if tid == "L1-01":
        from .native_geometry import cup_cavity
        _, floor, rim = cup_cavity("video:black_ceramic_mug")
        d = replace(d, fixture_asset="tuojing:video:black_ceramic_mug")
        p.update(rim=fixture_pose[2]+rim, floor=fixture_pose[2]+floor)
    elif tid == "L1-02":
        p.update(slot_centres=row["slot_centres"], rim=fixture_pose[2]+row["socket_frame_z"],
                 floor=fixture_pose[2]+0.00515,
                 opening=(0.0082, 0.0026), target_socket="Socket0 nearest robot-left arm")
    elif tid == "L1-03":
        from .native_geometry import hook_geometry
        a, b, radius = hook_geometry(fixture_pose, "video:wooden_rack")
        p.update(hook_start=tuple(a), hook_end=tuple(b), hook_radius=radius)
        from .native_geometry import shapes, transformed_poly
        band = [i for i,s in enumerate(shapes("video:yellow_visor")) if s["name"].startswith("Band_")]
        vertices = np.concatenate([transformed_poly("video:yellow_visor", i, np.eye(4))[0] for i in band])
        p["band_plane_z"] = float((vertices[:,2].min()+vertices[:,2].max())/2)
    elif tid == "L1-04":
        from .native_geometry import cup_cavity
        _, floor, rim = cup_cavity("video:ripple_paper_cup")
        # This task's centre-placement goal stays at the same table location;
        # perturb only the two starting cups, not the virtual goal region.
        centre = np.mean([v["pose"][:2] for v in baseline["tracked"].values()], axis=0)
        p.update(centre_xy=tuple(centre), cup_height=rim, cup_floor=floor,
                 lower_cup_arm="right", upper_cup_arm="left")
    elif tid == "L1-05":
        from .native_geometry import cup_cavity
        _, floor, rim = cup_cavity("video:ripple_paper_cup")
        target = row["tracked"]["target_cup"]["pose"]
        p.update(vase_xy=tuple(target[:2]), floor=target[2]+floor, rim=target[2]+rim,
                 dynamic_target_cup=True, rose_source_cup=True,
                 fixed_articles=tuple(n for n in ("source_cup", "target_cup")
                                      if row["tracked"][n].get("kinematic")))
        if row.get('derived_contents'):
            p.update(target_filler_names=tuple(row['derived_contents']['added']),filler_radius_m=.008)
    elif tid in {"L2-01", "L2-02"}:
        support = row['l2_support_measurement']
        fixture = next(r for r in row['fixtures'] if r['path'] == support['fixture'])
        # The marker rests inside the recess; the longer books span the rim.
        p["support_z"] = fixture['pose'][2] + (support['inset_platform_z_m'] if tid == 'L2-01'
                                               else support['outer_dimensions_m'][2])
        if tid == "L2-02":
            p["minimum_travel_m"] = .05
            p["outward_axis_world"] = tuple(pose_transform(row["tracked"]["middle_book"]["pose"])[:3, 0])
        else:
            p.update(cap_mouth_z=marker_axial(.10625), cap_end_z=marker_axial(.14875),
                     body_tip_z=marker_axial(p['body_tip_z']))
    elif tid == "L2-03":
        count = sum(n.startswith('load_') for n in row['tracked'])
        # User acceptance: empty source and at least half received. The English
        # task prompt stays unchanged; record this scoring revision explicitly.
        d = replace(d, version=d.version + "-half-received-v1")
        p.update(count=count, required_received=(count + 1) // 2, minimum_loaded_lift_count=count,
                 require_source_empty=True, pouring_success_rule="empty_source_half_received_v1",
                 scoring_protocol="empty_source_half_received_v1",
                 ball_radius_m=.008, cup_geometry="video:ripple_paper_cup",
                 ball_mass_kg=row['tracked']['load_0']['mass'],
                 loaded_source_mass_kg=row['tracked']['source_cup']['mass']+
                     sum(r['mass'] for n,r in row['tracked'].items() if n.startswith('load_')))
    elif tid == "L2-04":
        pedestal = row["fixtures"][0]
        support = row['l2_support_measurement']
        p.update(pedestal_pose=tuple(pedestal["pose"]),
                 support_z=pedestal['pose'][2] + support['inset_platform_z_m'],
                 support_rim_z=pedestal['pose'][2] + support['outer_dimensions_m'][2],
                 support_half_extents_m=tuple(support['inset_half_extents_m']))
    elif tid == "L2-05":
        body = row["fixtures"][0]
        outward = pose_transform(body["pose"])[:3, :3] @ np.array([0., -1., 0.])
        initial = row["tracked"]["drawer"]["pose"]
        closed = np.array(initial)
        opening = .03 * body["inherited_scale"][1]
        closed[:3] -= opening * outward
        p.update(outward_axis_world=tuple(outward), closed_pose=tuple(closed), initial_open_m=opening)
    elif tid == "L3-01":
        p.update(neck_top_z=.1565, cap_bottom_z=.1442, cap_top_z=.1600,
                 body_fixed=True, body_arm="right", cap_arm="left")
        repair = row['physical_repairs']['bottle_closure']
        p.update(initial_cap_lift_m=repair['initial_cap_lift_m'],
                 closure_contact_revision=repair['revision'])
    elif tid == "L3-02":
        body = pose_transform(row["tracked"]["anchor"]["pose"])
        relative = np.linalg.inv(body) @ pose_transform(row["tracked"]["nut"]["pose"])
        p.update(shaft_tip_z=.120, initial_nut_bottom_z=float(relative[2, 3]),
                 nut_height_m=.0105, support_z=.955, thread_pitch_m=.0015)
    elif tid == "L2-07":
        p.update(board_pose=fixture_pose, pad_z=0., pad_half_extents=(.059, .029))
    elif tid == "L3-03":
        centre = row["target_centre_xy"]
        # Native outline is rotated. Derive its frame and sizes at scene bind.
        articles = [replace(a, goal=tuple(centre)+(0.88, 1., 0., 0., 0.)) for a in d.articles]
        d = replace(d, articles=tuple(articles))
        p["target_vertices"] = tuple(tuple(np.asarray(v)-centre) for v in row["target_vertices_world"])
        from itertools import product
        if "lower_book" in row["tracked"]:
            names = ("lower_book", "upper_book")
            p.pop("native_instances", None)
            p.pop("block_vertices", None)
            p.update(start_centre_xy=tuple(row['start_centre_xy']),
                     start_vertices=tuple(tuple(np.asarray(v)-row['start_centre_xy']) for v in row['start_vertices_world']),
                     required_stack_turn_rad=2*np.pi, joint_push_distance_m=.005,
                     constraint_debounce_s=.25, scoring_protocol='l314_bimanual_stack_full_turn_20261020',
                     boundary_reference=task.get("boundary_reference", "both_books"))
            p.update(stacked_book_names=names,
                     book_vertices={n: tuple(product(*zip(*row["tracked"][n]["bounds"]))) for n in names})
            d = replace(d, name="双臂推叠书旋转360度移框", version=d.version+"-stacked-books-full-turn-20261020",
                        provenance="Two independent book bodies reused from L2-02; placement fitted to the 2026-10-19 supplied real video")
            spec = replace(spec, mechanism="independent_stacked_book_pushing")
        else:
            lo, hi = row["tracked"]["eraser"]["bounds"]
            p["block_vertices"] = tuple(product(*zip(lo,hi)))
    elif tid == 'L3-04':
        count = sum(n.startswith('block_') for n in row['tracked'])
        p.update(count=count, required=count, require_brush_release=False)
        d = replace(d, name="持刷扫三块入固定簸箕", release=False)
        p.update(pan_floor_vertices=row['pan_floor_vertices'],
                 pan_floor_height_range=row['pan_floor_height_range'])
    return replace(spec, design=d, parameters=p,
                   limitations="Latest complete 2026-10-15 source assets with declared runtime binding/orientation repairs. L2-01 uses the latest measured exterior with the supplied hollow cap and passive retention mechanism fitted axially. Geometry, lighting and materials are video/photo estimates; unobservable physical parameters are not measured. No pose writes, object attachments or forced success during manipulation.")
