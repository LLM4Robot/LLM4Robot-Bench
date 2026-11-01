"""Freeze independently sampled, reproducible L1 placement variants; no GPT or USD writes.

Usage: PYTHONPATH=src python tools/generate_l1_position_variants.py
Position-only URDF IK is a reachability screen, not a collision-free trajectory
or a demonstration of manipulation success. Passive Isaac checks are separate.
"""
import argparse
from copy import deepcopy
import hashlib
from itertools import product
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task
from robot_tasks.robots.kinematics import R5Kinematics
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.backends.scene_geometry import bench_geometry
from robot_tasks.tasks.bench.initial_placement import yaw_about_pivot
from robot_tasks.tasks.bench.native_geometry import hook_geometry
from robot_tasks.video_delivery import scenes, effective_cameras

PROFILE_ID = 'l1_xy_random_p03_20261018'
OUTPUT = PROJECT_ROOT / 'configs/task_references/l1_position_perturbations_p03_20261018.json'
REJECTIONS = OUTPUT.with_name('l1_position_perturbations_p03_rejections_20261018.json')
TASKS = ('L1-01', 'L1-02', 'L1-03', 'L1-04', 'L1-05')
RANGES = {
    'L1-01': ((.03, .07), (.02, .05)),
    'L1-02': ((.03, .07), (.02, .05)),
    'L1-03': ((.03, .07), (0., .03)),
    'L1-04': ((.03, .06), (.03, .06)),
    'L1-05': ((.03, .07), (.03, .07)),
}


def sample_delta(rng, limits):
    # Uniform by area, rather than over-weighting the centre of the annulus.
    low, high = limits
    radius = np.sqrt(rng.uniform(low * low, high * high))
    angle = rng.uniform(-np.pi, np.pi)
    return [round(float(radius * np.cos(angle)), 8),
            round(float(radius * np.sin(angle)), 8), 0.]


def candidate_poses(baseline, candidate):
    records = [*baseline['tracked'].values(), *baseline['fixtures']]
    poses = {r['path']: pose_transform(r['pose']) for r in records}
    for g in candidate['groups']:
        for path in g['paths']:
            if path in poses:
                poses[path][:3, 3] += g['delta_m']
    for r in candidate.get('rotations', []):
        poses[r['path']] = yaw_about_pivot(r['yaw_deg'], poses[r['pivot_path']][:3, 3]) @ poses[r['path']]
    return poses


@lru_cache(maxsize=1)
def rack_collider_vertices():
    """Actual separate rack segments, avoiding treating empty leg gaps as solid."""
    from pxr import Usd, UsdGeom, UsdPhysics
    baseline = scenes()['L1-03']
    stage = Usd.Stage.Open(str(PROJECT_ROOT / 'assets/tuojing' / baseline['runtime_path']))
    cache = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Household/Fixture')):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        result.append(points @ transform[:3, :3].T + transform[:3, 3])
    return result


def polygons_separated(a, b, margin=.005):
    a, b = (p[ConvexHull(p).vertices] for p in (a, b))
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for dx, dy in edges:
            axis = np.array([-dy, dx]) / np.hypot(dx, dy)
            pa, pb = a @ axis, b @ axis
            if max(pb.min() - pa.max(), pa.min() - pb.max()) >= margin:
                return True
    return False


def geometry_screen(tid, config, baseline, candidate):
    poses = candidate_poses(baseline, candidate)
    bench = bench_geometry(config)
    lower = np.array(bench.centre[:2]) - np.array(bench.size[:2]) / 2 + .01
    upper = np.array(bench.centre[:2]) + np.array(bench.size[:2]) / 2 - .01
    records = {r['path']: r for r in [*baseline['tracked'].values(), *baseline['fixtures']]}
    camera = effective_cameras(tid)['overview']
    inverse_camera = np.linalg.inv(pose_transform(camera['pose']))
    fx, fy, cx, cy = camera['intrinsics']
    boxes, group_points = [], []
    for group in candidate['groups']:
        points = []
        for path in group['paths']:
            if path not in records:
                continue  # independent reset-stack leaves move with their parent group
            local = np.array(list(product(*zip(*records[path]['bounds']))))
            t = poses[path]
            world = local @ t[:3, :3].T + t[:3, 3]
            # The rack top is cropped in the original framing. Require all other
            # moved props (not merely their centres) to fit the unchanged view.
            if not (tid == 'L1-03' and path.endswith('/Fixture')):
                view = world @ inverse_camera[:3, :3].T + inverse_camera[:3, 3]
                uv = np.column_stack((cx + fx * view[:, 0] / -view[:, 2],
                                      cy - fy * view[:, 1] / -view[:, 2]))
                if (np.any(view[:, 2] >= 0) or np.any(uv < 8)
                        or np.any(uv > np.array(camera['resolution']) - 8)):
                    return 'prop_bounds_outside_overview_margin'
            points.extend(world[:, :2])
        points = np.array(points)
        lo, hi = points.min(axis=0), points.max(axis=0)
        if np.any(lo < lower) or np.any(hi > upper):
            return 'outside_table_margin'
        boxes.append((lo, hi))
        group_points.append(points)
    # Deliberately conservative XY AABBs prevent separate groups starting in contact.
    if tid == 'L1-03':
        top = max((poses[path] @ np.r_[point, 1.])[2]
                  for path in candidate['groups'][0]['paths']
                  for point in product(*zip(*records[path]['bounds'])))
        rack_shift = np.asarray(candidate['groups'][1]['delta_m'])
        for points in rack_collider_vertices():
            moved = points + rack_shift
            if moved[:, 2].min() > top + .005:
                continue
            if not polygons_separated(group_points[0], moved[:, :2]):
                return 'hat_box_intersects_actual_rack_segment_footprint'
    elif all(boxes[0][0] - .005 < boxes[1][1]) and all(boxes[1][0] - .005 < boxes[0][1]):
        return 'independent_group_footprints_overlap'
    shifts = [np.array(g['delta_m']) for g in candidate['groups']]
    if np.linalg.norm(shifts[0] - shifts[1]) < .015:
        return 'relative_offset_change_under_15mm'
    if tid == 'L1-04':
        right, left = (poses['/World/Household/' + n][:3, 3] for n in ('lower_cup', 'upper_cup'))
        if right[0] - left[0] < .09:
            return 'cups_overlap_or_exchange_sides'
        if not .01 <= abs(right[1] - left[1]) <= .05:
            return 'cup_front_back_difference_outside_1_to_5cm'
    if tid == 'L1-05':
        if poses['/World/Household/source_cup'][0, 3] <= poses['/World/Household/target_cup'][0, 3] + .09:
            return 'flower_must_stay_on_overview_right'
    return None


def reach_targets(tid, baseline, candidate):
    poses = candidate_poses(baseline, candidate)
    def at(name, local):
        t = poses['/World/Household/' + name]
        return t[:3, :3] @ local + t[:3, 3]
    if tid == 'L1-01':
        return [('marker_grasp', 'right', at('marker', [0, 0, .125])),
                ('mug_approach', 'right', at('Fixture', [0, 0, .13]))]
    if tid == 'L1-02':
        return [('plug_grasp', 'right', at('plug', [0, 0, .048])),
                ('socket_approach', 'right', at('Fixture', [0, 0, .10]))]
    if tid == 'L1-03':
        from robot_tasks.robots.transforms import transform_pose
        a, b, _ = hook_geometry(transform_pose(poses['/World/Household/Fixture']), 'video:wooden_rack')
        return [('hat_band_region', 'right', at('hat', [.085, 0, .04])),
                ('lowest_hook_region', 'right', (a + b) / 2)]
    if tid == 'L1-04':
        centre = np.mean([r['pose'][:2] for r in baseline['tracked'].values()], axis=0)
        return [('right_cup_grasp', 'right', at('lower_cup', [0, 0, .055])),
                ('left_cup_grasp', 'left', at('upper_cup', [0, 0, .055])),
                ('centre_right', 'right', np.r_[centre, 1.02]),
                ('centre_left', 'left', np.r_[centre, 1.02])]
    point = baseline['flower_collision_repair']['head_grasp_local_m']
    return [('flower_head_grasp', 'right', at('rose', point)),
            ('target_cup_above', 'right', at('target_cup', [0, .05, .25]))]


def reach_screen(tid, config, baseline, candidate):
    kin = R5Kinematics(config.robot)
    rng = np.random.default_rng(174)
    checks = []
    for label, arm, target in reach_targets(tid, baseline, candidate):
        start = np.asarray(next(m.home_positions_rad for m in config.robot.mounts if m.name == arm))
        best = None
        def error(q):
            return np.asarray(kin.forward(q, arm)[:3]) - target
        for guess in [start, *rng.uniform(kin.lower, kin.upper, (5, 6))]:
            fit = least_squares(error, np.clip(guess, kin.lower, kin.upper),
                                bounds=(kin.lower, kin.upper), max_nfev=100,
                                ftol=1e-7, xtol=1e-7, gtol=1e-7)
            residual = float(np.linalg.norm(fit.fun))
            if best is None or residual < best['position_error_m']:
                best = {'region': label, 'arm_tool_id': arm, 'target_world_m': list(target),
                        'position_error_m': residual, 'joint_positions_rad': fit.x.tolist()}
            if residual < .001:
                break
        checks.append(best)
        if best['position_error_m'] > .003:
            return None
    return {'scope': 'URDF position-only IK for key interaction regions; no orientation/path/contact-success guarantee',
            'passed': True, 'checks': checks}


def generate(master_seed=20261018):
    from robot_tasks.task_numbering import to_legacy
    previous = json.loads((OUTPUT.parent / 'l1_position_perturbations_p02_20261018.json').read_text(encoding='utf-8'))
    ledger = json.loads(REJECTIONS.read_text(encoding='utf-8')) if REJECTIONS.exists() else {'master_seed': master_seed, 'rejections': []}
    rejected = {r['candidate_sha256'] for r in ledger['rejections']} if ledger['master_seed'] == master_seed else set()
    result = {
        'id': PROFILE_ID, 'description': 'P03: five independently sampled constrained random initial layouts per L1 task, frozen for reproducibility.',
        'coordinate_frame': previous['coordinate_frame'], 'max_translation_m': .07, 'max_yaw_deg': 25.,
        'master_seed': master_seed, 'variant_count': 5,
        'sampling': {'translation': 'uniform area on each group annulus; independent direction and radius',
                     'hat_yaw': 'equal probability of each sign; magnitude uniform in [10,25] degrees',
                     'rejection': ['10mm tabletop margin', '5mm separation of independent group XY bounds; L1-03 uses actual separate rack segment polygons',
                                   'relative translation change >=15mm', 'preserve cup sides',
                                   'L1-04 cup front/back difference 10..50mm', 'whole moved props fit overview with 8px margin (rack top exempt)',
                                   'position-only URDF IK <=3mm', 'recorded passive-physics rejections'],
                     'runtime_resampling': False, 'reset_policy': 'reuse selected frozen variant; select 1..5 explicitly'},
        'physical_rejections': ledger['rejections'] if ledger['master_seed'] == master_seed else [],
        'tasks': {},
    }
    for tid in TASKS:
        _, config = resolve_task(tid)
        baseline = scenes()[tid]
        old = previous['tasks'][tid]
        row = {'baseline_runtime_sha256': baseline['runtime_sha256'],
               'translation_ranges_m': [list(v) for v in RANGES[tid]], 'variants': []}
        for number in range(1, 6):
            # Preserve the frozen sampling identity across catalogue renumbering.
            # Changing this label would generate different physical placements.
            seed_id = to_legacy(tid)
            seed = int.from_bytes(hashlib.sha256(f'{master_seed}:{seed_id}:{number}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            rejections = {}
            for attempt in range(1, 2001):
                groups = deepcopy(old['groups'])
                for group, limits in zip(groups, RANGES[tid]):
                    group['delta_m'] = sample_delta(rng, limits)
                candidate = {'variant': number, 'seed': seed, 'sampling_attempt': attempt, 'groups': groups}
                if tid == 'L1-03':
                    rotations = deepcopy(old['rotations'])
                    rotations[0]['yaw_deg'] = round(float(rng.choice([-1, 1]) * rng.uniform(10., 25.)), 4)
                    candidate['rotations'] = rotations
                reason = geometry_screen(tid, config, baseline, candidate)
                fingerprint = hashlib.sha256(json.dumps({'groups': candidate['groups'],
                    'rotations': candidate.get('rotations', [])}, sort_keys=True).encode()).hexdigest()
                if reason is None and fingerprint in rejected:
                    reason = 'recorded_passive_physics_rejection'
                if reason:
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue
                reach = reach_screen(tid, config, baseline, candidate)
                if reach is None:
                    rejections['position_ik_failed'] = rejections.get('position_ik_failed', 0) + 1
                    continue
                candidate.update(rejections_before_acceptance=rejections, reachability_screen=reach)
                row['variants'].append(candidate)
                print(tid, number, 'seed', seed, 'attempt', attempt,
                      'offsets_cm', [[round(x * 100, 2) for x in g['delta_m'][:2]] for g in groups],
                      'yaw', candidate.get('rotations', [{}])[0].get('yaw_deg'), flush=True)
                break
            else:
                raise RuntimeError(f'Cannot produce a valid placement: {tid} variant {number}')
        result['tasks'][tid] = row
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = generate(args.seed)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
