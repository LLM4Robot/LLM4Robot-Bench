"""Reproducible pre-physics XY/yaw variants of immutable delivered scenes."""

from copy import deepcopy
import hashlib
import json

import numpy as np

from ...paths import PROJECT_ROOT
from ...video_delivery import scenes
from ...robots.transforms import pose_transform, transform_pose

PROFILE = PROJECT_ROOT / 'configs/task_references/l1_position_perturbations_20261018.json'
PROFILES = {
    'l1_xy_small_p01_20261018': PROFILE,
    'l1_xy_5cm_hat15_p02_20261018': PROFILE.with_name('l1_position_perturbations_p02_20261018.json'),
    'l1_xy_random_p03_20261018': PROFILE.with_name('l1_position_perturbations_p03_20261018.json'),
}
_LAYER_KEY = 'robot_tasks_initial_placement'


def yaw_about_pivot(degrees, pivot):
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    rotation = np.eye(4)
    rotation[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    rotation[:3, 3] = pivot - rotation[:3, :3] @ pivot
    return rotation


def scene_record(task):
    """Derive active poses without modifying the baseline manifest or geometry."""
    baseline = scenes()[task['id']]
    selected = task.get('initial_placement_profile', 'baseline')
    if selected == 'baseline':
        return baseline
    if selected not in PROFILES:
        raise ValueError('Unknown initial placement profile for ' + task['id'] + ': ' + str(selected))
    contents = PROFILES[selected].read_bytes()
    profile = json.loads(contents)
    if selected != profile['id'] or task['id'] not in profile['tasks']:
        raise ValueError('Unknown initial placement profile for ' + task['id'] + ': ' + str(selected))
    variant = profile['tasks'][task['id']]
    selection = {}
    if 'variants' in variant:
        number = task.get('initial_placement_variant', 1)
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError('Initial placement variant must be an integer')
        choices = [v for v in variant['variants'] if v['variant'] == number]
        if len(choices) != 1:
            raise ValueError('Unknown initial placement variant: ' + str(number))
        variant = {**variant, **choices[0]}
        selection = {k: variant[k] for k in ('variant', 'seed', 'sampling_attempt')}
    if baseline['runtime_sha256'] != variant['baseline_runtime_sha256']:
        raise ValueError('Initial placement profile requires its reviewed baseline scene: ' + task['id'])
    row = deepcopy(baseline)
    records = [*row['tracked'].values(), *row['fixtures']]
    originals = {record['path']: list(record['pose']) for record in records}
    by_path = {record['path']: record for record in records}
    paths = set()
    for group in variant['groups']:
        delta = np.asarray(group['delta_m'], dtype=float)
        if (delta.shape != (3,) or not np.isfinite(delta).all() or delta[2] != 0.
                or np.linalg.norm(delta) > profile['max_translation_m'] + 1e-12):
            raise ValueError('Initial placement must be a finite bounded horizontal translation')
        for path in group['paths']:
            if path in paths or not path.startswith('/World/Household/'):
                raise ValueError('Duplicate or invalid placement path: ' + path)
            paths.add(path)
            for record in records:
                if record['path'] != path:
                    continue
                before = list(record['pose'])
                record['pose'][:3] = (np.asarray(before[:3]) + delta).tolist()
                world = np.array(record['native_world_matrix'], dtype=float)
                world[:3, 3] += delta
                record['native_world_matrix'] = world.tolist()
    rotations = deepcopy(variant.get('rotations', []))
    rotated_paths = set()
    for rotation in rotations:
        degrees = rotation['yaw_deg']
        path, pivot_path = rotation['path'], rotation['pivot_path']
        if (path not in by_path or pivot_path not in by_path or path in rotated_paths
                or isinstance(degrees, bool) or not np.isfinite(degrees)
                or abs(degrees) > profile.get('max_yaw_deg', 0.)):
            raise ValueError('Invalid initial placement yaw or pivot')
        rotated_paths.add(path)
        record = by_path[path]
        pivot = np.asarray(by_path[pivot_path]['pose'][:3])
        transform = yaw_about_pivot(degrees, pivot)
        record['pose'] = list(transform_pose(transform @ pose_transform(record['pose'])))
        record['native_world_matrix'] = (transform @ np.asarray(record['native_world_matrix'])).tolist()
    changed = [{'path': record['path'], 'before_pose': originals[record['path']],
                'after_pose': list(record['pose'])} for record in records
               if record['path'] in paths | rotated_paths]
    row['initial_placement_perturbation'] = {
        'profile': selected, 'profile_sha256': hashlib.sha256(contents).hexdigest(),
        'baseline_runtime_sha256': baseline['runtime_sha256'],
        'coordinate_frame': profile['coordinate_frame'],
        'groups': deepcopy(variant['groups']), 'changed_records': changed,
        'rotations': rotations,
        'applied_before_physics': True, 'random_resampling': False,
        'z_geometry_materials_and_body_types_unchanged': True,
        'orientations_unchanged': not rotations,
        **selection,
    }
    return row


def apply_to_stage(stage, row):
    """Apply full-group XY moves, then declared world-yaw rotations once."""
    plan = row.get('initial_placement_perturbation')
    if plan is None:
        return
    from pxr import Gf, Usd, UsdGeom

    session = stage.GetSessionLayer()
    identity = plan['profile'] + ':' + plan['profile_sha256'] + ':' + plan['baseline_runtime_sha256']
    if 'variant' in plan:
        identity += ':variant=' + str(plan['variant'])
    previous = session.customLayerData.get(_LAYER_KEY)
    if previous:
        if previous != identity:
            raise ValueError('Another initial placement has already been applied to this stage')
        return
    cache = UsdGeom.XformCache()
    targets = {}
    for group in plan['groups']:
        delta = np.asarray(group['delta_m'])
        for path in group['paths']:
            prim = stage.GetPrimAtPath(path)
            if not prim or not UsdGeom.Xformable(prim):
                raise ValueError('Missing initial placement prim: ' + path)
            world = np.asarray(cache.GetLocalToWorldTransform(prim)).T.copy()
            world[:3, 3] += delta
            targets[path] = world
    for rotation in plan.get('rotations', []):
        path, pivot_path = rotation['path'], rotation['pivot_path']
        prim = stage.GetPrimAtPath(path)
        pivot_prim = stage.GetPrimAtPath(pivot_path)
        if not prim or not pivot_prim:
            raise ValueError('Missing initial placement rotation or pivot prim')
        pivot_world = targets.get(pivot_path)
        if pivot_world is None:
            pivot_world = np.asarray(cache.GetLocalToWorldTransform(pivot_prim)).T
        world = targets.get(path)
        if world is None:
            world = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        targets[path] = yaw_about_pivot(rotation['yaw_deg'], pivot_world[:3, 3]) @ world
    # Record all targets before translating parents. Reset-stack leaves use
    # independent world transforms and must move with the stem's support group.
    with Usd.EditContext(stage, session):
        for path in sorted(targets, key=lambda p: (p.count('/'), p)):
            prim = stage.GetPrimAtPath(path)
            xf = UsdGeom.Xformable(prim)
            reset = xf.GetResetXformStack()
            parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
            local = targets[path] if reset else np.linalg.inv(parent) @ targets[path]
            xf.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
            xf.SetResetXformStack(reset)
    session.customLayerData = {**session.customLayerData, _LAYER_KEY: identity}
