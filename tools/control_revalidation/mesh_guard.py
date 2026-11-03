"""Narrow-phase replacement for finger-only coarse start overlaps.

No collision exemption is granted. Actual finger collision meshes replace only
the finger spheres, for bounded paths independently checked before actuation.
"""
import json, math, time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import fcl
import trimesh
from robot_tasks.planning.worker import link_transforms
from robot_tasks.robots.transforms import pose_transform,origin_transform
from robot_tasks.robots.importer import prepare_urdf

MIN_GAP_M=0.000025
MAX_CONFIGURATIONS=4096

def mesh_shape(mesh):
    m=fcl.BVHModel();m.beginModel(len(mesh.vertices),len(mesh.faces));m.addSubModel(np.asarray(mesh.vertices,dtype=float),np.asarray(mesh.faces,dtype=np.int32));m.endModel();return m

def convex_shape(hull):
    faces=np.c_[np.full(len(hull.faces),3,dtype=np.int32),np.asarray(hull.faces,dtype=np.int32)].reshape(-1)
    return fcl.Convex(np.asarray(hull.vertices,dtype=float),len(hull.faces),faces)

class FingerMeshGuard:
    def __init__(self,worker,arm,request,directory):
        self.w=worker;self.arm=arm;self.request=request;self.directory=Path(directory)
        resolved=prepare_urdf(worker.profile,self.directory/'finger-source.urdf')
        root=ET.parse(resolved).getroot();self.fingers=[]
        for finger_index,name in enumerate(worker.assets['finger_geometry']):
            node=root.find(f"link[@name='{name}']/collision");geom=node.find('geometry/mesh');mesh=trimesh.load(geom.get('filename'),force='mesh')
            mesh.apply_scale([float(v) for v in geom.get('scale','1 1 1').split()]);o=node.find('origin')
            if o is not None:mesh.apply_transform(origin_transform(tuple(map(float,o.get('xyz','0 0 0').split())),tuple(map(float,o.get('rpy','0 0 0').split()))))
            # Delivered STL surfaces are not closed. A convex outer hull covers
            # every source vertex and the physical convex decomposition; do not
            # pretend an open triangle soup certifies solid clearance.
            for component_index,component in enumerate(mesh.split(only_watertight=False)):
                hull=component.convex_hull
                if not hull.is_watertight:raise ValueError('Finger component hull is not closed: '+name)
                self.fingers.append((name+f'/component{component_index}',convex_shape(hull),np.array(hull.vertices),worker.assets['finger_geometry'][name],finger_index))
        self.static=[]
        for name,box in request['scene']['obstacles'].get('cuboid',{}).items():
            T=pose_transform(box['pose']);s=fcl.Box(*box['dims']);obj=fcl.CollisionObject(s,fcl.Transform(T[:3,:3],T[:3,3]));v=trimesh.creation.box(extents=box['dims'],transform=T).vertices;self.static.append((name,obj,np.min(v,axis=0),np.max(v,axis=0)))
        for name,row in request['scene']['obstacles'].get('mesh',{}).items():
            m=trimesh.Trimesh(vertices=row['vertices'],faces=row['faces'],process=False);m.apply_transform(pose_transform(row['pose']));obj=fcl.CollisionObject(mesh_shape(m));self.static.append((name,obj,*m.bounds))
        for other,state in request['states'].items():
            if other==arm:continue
            for i,(x,y,z,r) in enumerate(worker.world_spheres(other,state['joint_positions_rad'],worker.spheres(state,other))):
                c=np.array([x,y,z]);self.static.append((f'other_{other}_{i}',fcl.CollisionObject(fcl.Box(2*r,2*r,2*r),fcl.Transform(c)),c-r,c+r))
        self.calls=0;self.cache={};self.minimum_gap=float('inf');self.closest=None
        # Original aggregate link6 (including fingers) was checked against these
        # same nonadjacent own-link spheres. Do not relax its self-collision scope.
        cfg=worker.assets['config']['robot_cfg']['kinematics'];ee=worker.profile.ee_link;ignores=cfg['self_collision_ignore'];self.own_links=[]
        for link in cfg['collision_link_names']:
            if link==ee or ee in ignores.get(link,[]) or link in ignores.get(ee,[]):continue
            self.own_links.append(link)
        self.geometry=worker.original_spheres(request['states'][arm],arm)
    @staticmethod
    def gap(obj,other):
        collision=fcl.CollisionResult()
        if fcl.collide(obj,other,fcl.CollisionRequest(num_max_contacts=1),collision):return -1.
        return float(fcl.distance(obj,other,fcl.DistanceRequest(enable_nearest_points=False),fcl.DistanceResult()))
    def distance(self,q):
        key=tuple(float(v) for v in q)
        if key in self.cache:return self.cache[key]
        self.calls+=1
        if self.calls>MAX_CONFIGURATIONS:raise ValueError('Exact-finger validation configuration budget exceeded')
        w=self.w;T6=w.kin.bases[self.arm]@link_transforms(w.kin,q)[w.profile.ee_link];fpositions=self.request['states'][self.arm]['finger_positions_m'];best=float('inf');pair=None
        own={l:self.geometry[l] for l in self.own_links};others=list(self.static)
        for i,(x,y,z,r) in enumerate(w.world_spheres(self.arm,q,own)):
            c=np.array([x,y,z]);others.append((f'own_nonadjacent_{i}',fcl.CollisionObject(fcl.Sphere(r),fcl.Transform(c)),c-r,c+r))
        for name,shape,vertices,geo,finger_index in self.fingers:
            opening=fpositions[finger_index]
            t=np.array(geo['origin']);t[:3,3]+=t[:3,:3]@(np.asarray(geo['axis'])*opening);T=T6@t;obj=fcl.CollisionObject(shape,fcl.Transform(T[:3,:3],T[:3,3]));v=vertices@T[:3,:3].T+T[:3,3];low,high=v.min(axis=0),v.max(axis=0)
            ordered=sorted(others,key=lambda a:np.linalg.norm(np.maximum(np.maximum(a[2]-high,low-a[3]),0.)))
            for label,other,lo,hi in ordered:
                bound=float(np.linalg.norm(np.maximum(np.maximum(lo-high,low-hi),0.)))
                if bound>best:continue
                g=self.gap(obj,other)
                if not math.isfinite(g):raise ValueError('Nonfinite exact mesh separation')
                if g<best:best=g;pair=(name,label)
                if best<=MIN_GAP_M:break
            if best<=MIN_GAP_M:break
        if best<self.minimum_gap:self.minimum_gap=best;self.closest=pair
        self.cache[key]=best;return best
    def validate(self,positions):
        t=time.monotonic();positions=np.asarray(positions,dtype=float)
        if len(positions)<2:raise ValueError('No path to validate')
        # A deliberately loose global bound on motion of any robot point under
        # joint interpolation. Factor2 also bounds two moving own-link points.
        reach=float(self.w.kin.reach_bound_m)+.5
        def segment(a,b,depth=0):
            ga,gb=self.distance(a),self.distance(b)
            if min(ga,gb)<=MIN_GAP_M:raise ValueError('Finger mesh collision/insufficient clearance')
            displacement=2*reach*float(np.sum(np.abs(b-a)))
            if min(ga,gb)>displacement+MIN_GAP_M:return
            if depth>=20:raise ValueError('Exact-finger swept clearance could not be certified')
            mid=(a+b)/2;segment(a,mid,depth+1);segment(mid,b,depth+1)
        for a,b in zip(positions[:-1],positions[1:]):segment(a,b)
        return {'protocol':'bounded_finger_mesh_narrow_phase_v1','configurations':self.calls,'minimum_mesh_gap_m':self.minimum_gap,'clearance_margin_m':MIN_GAP_M,'closest_pair':self.closest,'wall_s':time.monotonic()-t,'continuous_between_samples':'Conservative joint-displacement bound with adaptive subdivision.','original_other_link_self_world_payload_guards_retained':True,'geometry_limitation':'Conservative closed convex hull per connected component of original URDF finger collision mesh; no components removed. Physical cooking remains independently monitored. Not a hardware collision guarantee.'}
