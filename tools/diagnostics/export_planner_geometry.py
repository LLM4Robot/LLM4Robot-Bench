#!/usr/bin/env python3
"""Export the fitted model used by a known local run, with input checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from robot_tasks.registry import resolve_task
from robot_tasks.planning.reference_assets import geometry_digest
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--attachments',type=Path,required=True);a=p.parse_args()
_,config=resolve_task('L2-01');profile=config.robot
attachments=json.loads(a.attachments.read_text());robot=json.loads((a.cache/'robot.json').read_text())
identity=hashlib.sha256(profile.source.read_bytes()+profile.asset_lock.read_bytes()+(ROOT/'src/robot_tasks/planning/assets.py').read_bytes()+json.dumps(attachments,sort_keys=True).encode()).hexdigest()[:16]
if identity!=robot['metadata']['asset_identity']:raise SystemExit('Cache does not match supplied attachments and current code/profile')
root=ET.parse(a.cache/'arm.urdf').getroot()
for mesh in root.iter('mesh'):
 path=Path(mesh.get('filename'))
 for package,base in profile.package_roots:
  if path.is_relative_to(base):mesh.set('filename','package://'+package+'/'+path.relative_to(base).as_posix());break
 else:raise SystemExit('Mesh outside declared package roots')
cfg=robot['config']['robot_cfg']['kinematics'];cfg['urdf_path']='arm.urdf';cfg['asset_root_path']='.'
result={'schema':1,'source_asset_identity':identity,'inputs':{'profile_sha256':hashlib.sha256(profile.source.read_bytes()).hexdigest(),'asset_lock_sha256':hashlib.sha256(profile.asset_lock.read_bytes()).hexdigest(),'attachments_sha256_10um':geometry_digest(attachments)},'arm_urdf':ET.tostring(root,encoding='unicode'),'robot':robot}
out=ROOT/'assets/r5/planner_reference/geometry.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(out)
