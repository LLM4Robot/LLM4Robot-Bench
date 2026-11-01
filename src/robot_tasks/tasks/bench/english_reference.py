"""Versioned scene/goal changes for the reviewed Feishu English instructions.

Source text is preserved separately from the one-time real-to-simulation arm
conversion. Legacy specs remain available for historical diagnostics. These
overrides change scene setup and measured scoring, never issue robot actions.
"""

from dataclasses import replace
from functools import lru_cache
import json
import math

import numpy as np

from ...paths import PROJECT_ROOT
from ..l1.specs import Article, box, pose

PROTOCOL = "feishu_english_20261015_v1"


@lru_cache(maxsize=1)
def reference_tasks():
    data = json.loads((PROJECT_ROOT / "configs/task_references/feishu_20261015.json").read_text(encoding="utf-8"))
    return {row["task_id"]: row for row in data["tasks"]}


def uses_reference(task):
    return task.get("prompt_protocol") == PROTOCOL


def apply_reference(spec, task):
    if not uses_reference(task):
        return spec
    tid = task["id"]
    source = reference_tasks().get(task.get("variant_id"), reference_tasks()[tid])
    p = {**spec.parameters, "scoring_protocol": PROTOCOL, "hold_s": 0.0}
    d = replace(spec.design, version=spec.design.version + "-feishu-en-20261015",
                instruction=source["simulation_prompt_en"])
    limitations = spec.limitations
    mechanism = spec.mechanism

    if tid == "L1-01":
        p["require_marker_release"] = True
    elif tid == "L1-02":
        # Only the native middle blue socket has a parallel, dimension-matched
        # pair for these blades. Other five-hole groups are not equivalent slots.
        # This explicit engineering disambiguation and near-face bound are not
        # claimed to be thresholds supplied by the source document.
        p.update(alignment_only=True, alignment_max_gap_m=0.03)
        d = replace(d, name="USB 充电器插脚对齐")
        limitations += " English alignment proxy: native compatible middle two-slot pair; maximum blade-tip gap 30 mm. The source does not specify a numbered socket or metric alignment tolerance."
    elif tid == "L1-04":
        centre = (0.19565, 1.20)
        articles = []
        for a in d.articles:
            xy = (0.04, 1.20) if a.name == "lower_cup" else (0.34, 1.18)
            z = 0.78 if a.name == "lower_cup" else 0.805
            articles.append(replace(a, initial=pose(*xy, 0.781), goal=pose(*centre, z)))
        d = replace(d, articles=tuple(articles))
        p.update(centre_xy=centre, centre_tolerance_m=0.04,
                 lower_cup_arm="right", upper_cup_arm="left")
        limitations += " Centre placement uses a declared 40 mm engineering region, not a source-specified precision target."
    elif tid == "L1-05":
        from .geometry import cup_parts

        source_xy, target_xy = (0.04, 1.20), (0.34, 1.20)
        rose = replace(d.articles[0], initial=pose(*source_xy, 0.7841),
                       goal=pose(*target_xy, 0.7841))
        cup = Article("source_cup", cup_parts(bottom_radius=0.025, top_radius=0.0393,
                      height=0.1075, wall=0.003, segments=48),
                      pose(*source_xy, 0.78), pose(*source_xy, 0.78), 0.014,
                      asset="tuojing:ripple_paper_cup",
                      asset_bounds=((-0.0426, -0.0426, 0), (0.0426, 0.0426, 0.1076)))
        d = replace(d, articles=(rose, cup), fixture_pose=pose(*target_xy, 0.78))
        p.update(vase_xy=target_xy, rose_source_cup=True, fixed_articles=("source_cup",))
        limitations += " Both source and target cups are fixed passive receptacles. Source anchoring prevents the unballasted light cup from toppling; this differs from the real filled cup and is an explicit setup proxy."
    elif tid == "L2-02":
        p.update(release_required=False, require_top_book_hold=True)
    elif tid == "L2-03":
        from .pearl_pouring import PEARL_MASS, PEARL_RADIUS, packing

        source_pose, receiver_pose = pose(0.10, 1.18, 0.781), pose(0.30, 1.18, 0.78)
        old = {a.name: a for a in d.articles}
        articles = [replace(old[n], initial=v, goal=v)
                    for n, v in (("source_cup", source_pose), ("receiver_cup", receiver_pose))]
        instances = [{"name": n, "asset": "ripple_paper_cup", "pose": v,
                      "tracked": {n: ""}, **({"static": [""]} if n == "receiver_cup" else {})}
                     for n, v in (("source_cup", source_pose), ("receiver_cup", receiver_pose))]
        points = packing(10)
        for i, point in enumerate(points):
            n = f"load_{i}"
            initial = pose(*(np.asarray(source_pose[:3]) + point))
            goal = pose(*(np.asarray(receiver_pose[:3]) + point))
            articles.append(replace(old[n], initial=initial, goal=goal))
            instances.append({"name": n, "asset": "pearl_8mm", "pose": initial, "tracked": {n: ""}})
        d = replace(d, articles=tuple(articles))
        p.update(native_instances=instances, count=10, required_received=10,
                 require_source_empty=True, minimum_loaded_lift_count=8,
                 initial_packing_top_m=max(v[2] for v in points) + PEARL_RADIUS,
                 loaded_source_mass_kg=0.014 + 10 * PEARL_MASS)
    elif tid == "L2-04":
        limitations += " Physical tissue damage has no calibrated detector; it remains an unverified source constraint, not a claimed measured capability."
    elif tid == "L2-05":
        p["release_required"] = False
    elif tid == "L3-01":
        mechanism = "native_fixed_bottle_cap_contact"
        p.update(body_fixed=True, body_arm="right", cap_arm="left",
                 cap_lift_min_m=0.03, cap_lift_max_m=0.05)
        p["native_instances"] = [{**i, "static": ["Body"]} for i in p["native_instances"]]
        d = replace(d, name="固定瓶身旋下瓶盖并抬离瓶口")
        limitations += " User clarification: the bottle body is fixed to the table; the other hand need not hold it. The retained pre-loosened threads do not reproduce unmeasured real opening torque."
    elif tid == "L3-02":
        p["release_required"] = False
        d = replace(d, release=False)
    elif tid == "L2-07":
        p.update(whiteboard_surface=True, require_all_writing=True)
        d = replace(d, name="抓板擦擦除白板字迹")
        limitations += " Runtime whiteboard appearance uses the existing board geometry and removable ink mask; it is a cleaning proxy, not a newly calibrated whiteboard asset."
    elif tid == "L3-03":
        target = ((-0.16, -0.14), (0.16, -0.14), (0.16, 0.14), (-0.16, 0.14))
        a = d.articles[0]
        outline = []
        for i, (u, v) in enumerate(zip(target, (*target[1:], target[0]))):
            dx, dy = v[0] - u[0], v[1] - u[1]
            outline.append(box(f"target_outline_{i}", (math.hypot(dx, dy), 0.002, 0.0004),
                               (a.goal[0] + (u[0] + v[0]) / 2,
                                a.goal[1] + (u[1] + v[1]) / 2, 0.7802),
                               (0.7, 0.1, 0.08), (0, 0, math.atan2(dy, dx)), collision=False))
        d = replace(d, fixtures=tuple(outline), name="夹爪推板擦入红色矩形区")
        p["target_vertices"] = target

    spec = replace(spec, design=d, arms=tuple(source["active_arms"]), facts="", mechanism=mechanism,
                   limitations=limitations, parameters=p)
    from .video_reference import apply_delivery
    return apply_delivery(spec, task)


def initial_state_issue(spec, rows):
    """Reject an invalid source-cup setup; never reposition or attach the flower."""
    if not spec.parameters.get("rose_source_cup"):
        return None
    from ...robots.transforms import pose_transform
    from .native_geometry import cavity_fit, cup_cavity, shapes, transformed_poly

    source = pose_transform(rows["source_cup"]["pose"])
    relative = np.linalg.inv(source) @ pose_transform(rows["rose"]["pose"])
    prefix = spec.parameters.get("geometry_prefix", "")
    stems = [i for i, shape in enumerate(shapes(prefix + "artificial_rose"))
             if "stem" in shape["name"].lower()]
    inside, _ = cavity_fit(prefix + "artificial_rose", relative, .001, indices=stems,
                          receptacle=prefix + "ripple_paper_cup")
    tip = transformed_poly(prefix + "artificial_rose", stems[0], relative)[0]
    _, floor, rim = cup_cavity(prefix + "ripple_paper_cup")
    if (source[2, 2] < .95 or not inside or tip[:, 2].min() >= rim - .001
            or tip[:, 2].max() <= floor - .001):
        return "L1-05 initial flower must remain inside an upright source cup"
    return None
