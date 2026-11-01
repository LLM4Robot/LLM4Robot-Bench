"""Regenerate locked L1 collider metadata; requires USD Python and scipy.

Run from the repo root with an environment providing pxr and scipy. This tool
only reads the hash-locked delivery; it never edits source USDs. Mesh hulls are
separate per collider, preserving cup and visor openings. No GPT calls.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics
result={'delivery':'2026-10-05 + 2026-10-09 mug and plug holder','source':'authored enabled convex colliders, native metres; marker flat-barrel runtime override recorded explicitly','props':{}}
for name in ['whiteboard_marker','ripple_paper_cup','black_ceramic_mug','usb_charger','plug_storage_base','artificial_rose','yellow_visor','wooden_rack']:
 from robot_tasks.asset_delivery import verified_prop
 path,_=verified_prop(name);s=Usd.Stage.Open(str(path)); cache=UsdGeom.XformCache(); shapes=[]
 for p in s.Traverse():
  if not p.HasAPI(UsdPhysics.CollisionAPI):continue
  if name=='wooden_rack' and p.GetName()!='Hook0':continue
  d={'name':p.GetName(),'kind':p.GetTypeName().lower(),'transform':np.array(cache.GetLocalToWorldTransform(p)).T.tolist()}
  if p.IsA(UsdGeom.Mesh):
   points=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float); hull=ConvexHull(points); chosen=sorted(hull.vertices);mapping={v:i for i,v in enumerate(chosen)}
   edges=sorted({tuple(sorted((mapping[a],mapping[b]))) for tri in hull.simplices for a,b in zip(tri,np.roll(tri,-1))})
   d.update(points=points[chosen].tolist(),edges=edges,faces=[[mapping[v] for v in tri] for tri in hull.simplices])
  elif p.IsA(UsdGeom.Capsule) or p.IsA(UsdGeom.Cylinder):
   shape=UsdGeom.Capsule(p) if p.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(p)
   d.update(radius=shape.GetRadiusAttr().Get(),height=shape.GetHeightAttr().Get(),axis=shape.GetAxisAttr().Get())
  else:raise ValueError(p.GetTypeName())
  if name=='whiteboard_marker' and p.GetName()=='Barrel':d.update(kind='cylinder',height=.111)
  shapes.append(d)
 result['props'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shapes':shapes}
Path('assets/tuojing/l1_scoring_geometry.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print({k:len(v['shapes']) for k,v in result['props'].items()})
