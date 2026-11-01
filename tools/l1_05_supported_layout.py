"""Naturally settled L1-05 cup/charger support; authored only at scene build."""
from pathlib import Path
import hashlib
import json
import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
# Measured after passive settling, then verified by a fresh 30 s free-rest run.
# No runtime pins, springs, gravity changes, velocity writes or hidden supports.
POSES = {'rose': [0.39192846417427063, 1.1448321342468262, 0.882034420967102, 0.5910032987594604, -0.08147041499614716, -0.06131855398416519, 0.8001986145973206], 'source_cup': [0.40515777468681335, 1.145155906677246, 0.8799996972084045, 0.9999663829803467, 1.4533288776874542e-06, 2.2351741790771484e-06, 0.00821627490222454], 'source_charger': [0.4034442603588104, 1.1453951597213745, 0.9499719738960266, 3.894599558407208e-06, 0.7604058980941772, 0.6494481563568115, 3.4105796657968313e-06]}


def apply(stage, row):
    for name, pose in POSES.items():
        record = row['tracked'][name]
        transform = np.eye(4)
        transform[:3,:3] = Rotation.from_quat([*pose[4:],pose[3]]).as_matrix()
        transform[:3,3] = pose[:3]
        prim = stage.GetPrimAtPath(record['path'])
        parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent)@transform).T.tolist()))
        record['pose'] = list(pose)
        record['native_world_matrix'] = transform.tolist()
    return {'revision':'passive_cup_charger_support_20261016_v1',
            'poses':POSES, 'source':'passive physics settled pose, independently reset and rested for 30 s',
            'flower_tilt_deg':11.705, 'physics_and_geometry_unchanged':True,
            'note':'Cup-contained position adjustments only; charger prongs remain up; no runtime constraints.'}


if __name__ == '__main__':
    manifest_path=ROOT/'assets/tuojing/video_20261015.json'
    manifest=json.loads(manifest_path.read_text());row=manifest['tasks']['L1-05']
    path=ROOT/'assets/tuojing'/row['runtime_path']
    stage=Usd.Stage.Open(str(path));row['flower_initial_support']=apply(stage,row)
    stage.GetRootLayer().Save()
    row['runtime_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    row['runtime_bytes']=path.stat().st_size
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    setup_path=ROOT/'configs/task_references/video_visual_setup_20261015.json'
    contents=setup_path.read_bytes();history=setup_path.parent/'video_visual_setup_history'
    history.mkdir(exist_ok=True);(history/(hashlib.sha256(contents).hexdigest()+'.json')).write_bytes(contents)
    setup=json.loads(contents);setup['tasks']['L1-05']['runtime_scene_sha256']=row['runtime_sha256']
    setup_path.write_text(json.dumps(setup,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(row['flower_initial_support'],indent=2))
