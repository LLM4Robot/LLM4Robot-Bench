"""Video-guided L2 book, cup, orange and drawer placement at reset only."""

import json
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from robot_tasks.robots.transforms import pose_transform, transform_pose
from video_scene_metadata import record

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'configs/task_references/l2_layout_refinement_20261015.json'
STAMP = 'gptBenchmark:l2VideoPlacement'


def world(stage, path):
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T


def set_world(stage, path, matrix):
    prim = stage.GetPrimAtPath(path)
    parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent) @ matrix).T.tolist()))


def book_incline(stage, row, reference):
    if reference.get('revision') in ('l2-book-rim-shape-video-fit-v2', 'l2-book-rim-shape-video-fit-v3', 'l2-book-straight-rim-slope-v4'):
        from l2_book_rim_fit import apply as fit_book_rim
        return fit_book_rim(stage, row, reference)
    fixture = row['l2_support_measurement']['fixture']
    box = world(stage, fixture)
    inverse = np.linalg.inv(box)
    lower = world(stage, row['tracked']['lower_book']['path'])
    relative = inverse @ lower
    yaw = np.arctan2(relative[1, 0], relative[0, 0])
    rotation = Rotation.from_euler('z', yaw).as_matrix()
    xy = np.array(reference['lower_book_centre_box_xy_m'])
    bounds = row['tracked']['lower_book']['bounds']
    bottom = np.array([[x, y, bounds[0][2]] for x in (bounds[0][0], bounds[1][0])
                       for y in (bounds[0][1], bounds[1][1])])
    vertices = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(fixture)):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            transform = inverse @ world(stage, str(prim.GetPath()))
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            vertices.extend(points @ transform[:3, :3].T + transform[:3, 3])
    vertices = np.asarray(vertices)

    def candidate(angle):
        transform = np.eye(4)
        transform[:3, :3] = rotation @ Rotation.from_euler('x', angle).as_matrix()
        transform[:2, 3] = xy
        transform[2, 3] = .060 + reference['floor_clearance_m'] - (bottom @ transform[:3, :3].T)[:, 2].min()
        local = (vertices - transform[:3, 3]) @ transform[:3, :3]
        inside = np.all((local[:, :2] >= np.asarray(bounds[0][:2])) &
                        (local[:, :2] <= np.asarray(bounds[1][:2])), axis=1)
        gap = bounds[0][2] - local[inside, 2].max()
        return transform, gap

    angle = brentq(lambda a: candidate(a)[1] - .00020, 0., np.deg2rad(10.))
    placed, clearance = candidate(angle)
    delta = box @ placed @ np.linalg.inv(lower)
    old = {name: world(stage, r['path']) for name, r in row['tracked'].items()}
    for name, r in row['tracked'].items():
        set_world(stage, r['path'], delta @ old[name])
    return {'tilt_deg': float(np.rad2deg(angle)), 'lower_book_box_pose': list(transform_pose(placed)),
            'minimum_sampled_collider_clearance_m': float(clearance),
            'all_three_books_remain_dynamic': True, 'new_hidden_supports_or_constraints': False,
            'method': reference['fit_method'], 'limitation': reference['limitation']}


def apply(stage, tid, row):
    if tid not in ('L2-02', 'L2-03', 'L2-04', 'L2-05'):
        return None
    config = json.loads(REFERENCE.read_text(encoding='utf-8'))
    reference = config['tasks'][tid]
    revision = reference.get('revision', config['revision'])
    root = stage.GetPrimAtPath('/World/L2')
    stamp = root.GetAttribute(STAMP)
    if stamp and stamp.Get():
        result = json.loads(stamp.Get())
        if result['revision'] != revision:
            raise ValueError('Rebuild from immutable source before changing the L2 placement revision')
        return result
    if tid == 'L2-02':
        result = book_incline(stage, row, reference)
    elif tid == 'L2-04':
        path = row['orange']['path']
        matrix = world(stage, path)
        # Rotate the full FEM hierarchy about its support-plane origin. Keep
        # inherited scaling, height, material and rest mesh data unchanged.
        rotation = Rotation.from_euler('z', reference['yaw_delta_deg'], degrees=True).as_matrix()
        matrix[:3, :3] = rotation @ matrix[:3, :3]
        set_world(stage, path, matrix)
        result = {'method': reference['placement_method'],
                  'yaw_delta_deg': reference['yaw_delta_deg'],
                  'shared_camera_changed': False,
                  'fem_geometry_and_physics_preserved': True}
    elif tid == 'L2-05':
        delta = np.asarray(reference['translation_world_m'], dtype=float)
        # Translate the entire assembly, including loose contents and both
        # joint frames. The drawer opening and support contact stay unchanged.
        for path in ('/World/L2/organizer', row['l2_support_measurement']['fixture']):
            matrix = world(stage, path)
            matrix[:3, 3] += delta
            set_world(stage, path, matrix)
        result = {'method': reference['placement_method'],
                  'translation_world_m': delta.tolist(),
                  'shared_camera_changed': False,
                  'drawer_relative_pose_and_joint_parameters_preserved': True}
    else:
        source_path = row['tracked']['source_cup']['path']
        old_source = world(stage, source_path)
        new_source = pose_transform(reference['cup_poses']['source_cup'])
        delta = new_source @ np.linalg.inv(old_source)
        for name, item in row['tracked'].items():
            if name.startswith('load_'):
                set_world(stage, item['path'], delta @ world(stage, item['path']))
        for name, pose in reference['cup_poses'].items():
            set_world(stage, row['tracked'][name]['path'], pose_transform(pose))
        result = {'method': reference['placement_method'],
                  'cup_poses': reference['cup_poses'],
                  'cup_base_centres_target_px': reference['cup_base_centres_target_px'],
                  'annotation_uncertainty_px': reference['annotation_uncertainty_px'],
                  'shared_camera_changed': False}
    result['revision'] = revision
    root.CreateAttribute(STAMP, Sdf.ValueTypeNames.String, custom=True).Set(json.dumps(result, sort_keys=True))
    return result


def refresh(stage, row):
    row['tracked'] = {name: record(stage, item['path']) for name, item in row['tracked'].items()}
    row['fixtures'] = [record(stage, item['path']) for item in row['fixtures']]
    if 'orange' in row:
        row['orange'] = record(stage, row['orange']['path'])


def update_runtime_files():
    import hashlib
    import os
    manifest_path = ROOT / 'assets/tuojing/video_20261015.json'
    setup_path = ROOT / 'configs/task_references/video_visual_setup_20261015.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    setup_bytes = setup_path.read_bytes()
    setup = json.loads(setup_bytes)
    history = setup_path.parent / 'video_visual_setup_history' / (hashlib.sha256(setup_bytes).hexdigest() + '.json')
    history.write_bytes(setup_bytes)
    for tid in ('L2-02', 'L2-03', 'L2-04', 'L2-05'):
        row = manifest['tasks'][tid]
        path = ROOT / 'assets/tuojing' / row['runtime_path']
        stage = Usd.Stage.Open(str(path))
        row['video_placement_refinement'] = apply(stage, tid, row)
        refresh(stage, row)
        if tid == 'L2-03':
            for name in list(row.get('derived_contents', {}).get('added', {})):
                stage.RemovePrim(row['tracked'].pop(name)['path'])
            from video_scene_contents import apply as add_contents
            row['derived_contents'] = add_contents(stage, tid, manifest['tasks'], ROOT / 'assets/tuojing')
            row['tracked'].update(row['derived_contents']['added'])
            layer = stage.Flatten()
            def portable(asset):
                basename = asset.rsplit('/', 1)[-1].rstrip(']')
                if basename in {'OmniPBR.mdl', 'OmniGlass.mdl', 'OmniSurface.mdl'}:
                    return basename
                if '.usdz[' in asset:
                    archive, internal = asset.split('.usdz[', 1)
                    archive = Path(archive + '.usdz')
                    if archive.is_absolute():
                        return os.path.relpath(archive, path.parent) + '[' + internal
                if Path(asset).is_absolute() and Path(asset).is_relative_to(ROOT):
                    return os.path.relpath(asset, path.parent)
                return asset
            UsdUtils.ModifyAssetPaths(layer, portable)
            layer.Export(str(path))
        else:
            stage.GetRootLayer().Save()
        row['runtime_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        row['runtime_bytes'] = path.stat().st_size
        setup['tasks'][tid]['runtime_scene_sha256'] = row['runtime_sha256']
        print(tid, row['video_placement_refinement'], flush=True)
    setup['revision'] = json.loads(REFERENCE.read_text())['visual_setup_revision']
    for path, data in ((manifest_path, manifest), (setup_path, setup)):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temp.replace(path)


if __name__ == '__main__':
    update_runtime_files()
