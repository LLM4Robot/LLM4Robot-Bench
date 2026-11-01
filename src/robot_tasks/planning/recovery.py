"""Bounded escape from shallow planner-only scene overlap; no collision bypass.

An opt-in simulation retreat must end feasible and reduce every existing
sphere/obstacle overlap without creating another. Ordinary plans are unchanged.
"""
import numpy as np
from ..robots.transforms import pose_transform, pose_errors


def check_overlap_trace(depths, *, max_initial_m=.005, epsilon_m=.00002):
    """Depths are metres, per waypoint AND per sphere/obstacle pair."""
    d = np.asarray(depths, float)
    if d.ndim != 2 or len(d) < 2 or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('Invalid retreat collision trace')
    if d[0].max(initial=0) > max_initial_m:
        raise ValueError('Start overlap exceeds the 5 mm retreat bound')
    if not np.any(d[0] > epsilon_m):
        raise ValueError('No measurable shallow scene overlap to recover')
    if np.any(d[:, d[0] <= epsilon_m] > epsilon_m):
        raise ValueError('Retreat creates a new sphere/obstacle overlap')
    if np.any(np.diff(d, axis=0) > epsilon_m) or np.any(d > d[0] + epsilon_m):
        raise ValueError('Retreat deepens an existing overlap')
    if d[-1].max(initial=0) > epsilon_m:
        raise ValueError('Retreat endpoint remains in collision')


class SceneOverlap:
    def __init__(self, worker, arm, request):
        import trimesh
        self.worker, self.arm = worker, arm
        active = worker.assets['config']['robot_cfg']['kinematics']['collision_link_names']
        self.spheres = {k:v for k,v in worker.spheres(request['states'][arm], arm).items() if k in active}
        self.labels = [k for k,v in self.spheres.items() for s in v if s['radius'] > 0]
        obstacles = request['scene']['obstacles']
        self.meshes = []
        for name, data in obstacles.get('mesh', {}).items():
            mesh = trimesh.Trimesh(vertices=data['vertices'], faces=data['faces'], process=False)
            mesh.apply_transform(pose_transform(data['pose']))
            self.meshes.append((name, mesh))
        for name, data in obstacles.get('cuboid', {}).items():
            self.meshes.append((name, trimesh.creation.box(extents=data['dims'], transform=pose_transform(data['pose']))))
        for other, state in request['states'].items():
            if other == arm:
                continue
            for i, (x,y,z,r) in enumerate(worker.world_spheres(other, state['joint_positions_rad'], worker.spheres(state,other))):
                transform=np.eye(4);transform[:3,3]=[x,y,z]
                self.meshes.append((f'other_{other}_{i}',trimesh.creation.box(extents=[2*r]*3,transform=transform)))

    def world(self, q):
        return self.worker.world_spheres(self.arm,q,self.spheres)

    def depths(self, positions):
        import trimesh
        spheres=np.asarray([self.world(q) for q in positions])
        centers=spheres[:,:,:3].reshape(-1,3);radii=spheres[:,:,3].reshape(-1)
        values=np.zeros((len(positions),len(self.labels),len(self.meshes)))
        flat=values.reshape(-1,len(self.meshes))
        for j,(_,mesh) in enumerate(self.meshes):
            lo,hi=mesh.bounds
            near=np.all((centers+radii[:,None]>=lo)&(centers-radii[:,None]<=hi),axis=1)
            if near.any():
                flat[near,j]=np.maximum(0.,trimesh.proximity.signed_distance(mesh,centers[near])+radii[near])
        return values

    def describe(self, q):
        depths=self.depths([q])[0];rows=[]
        for i,j in np.argwhere(depths>.00002):
            rows.append({'link':self.labels[i],'sphere_index':int(i),'obstacle':self.meshes[j][0],
                         'overlap_m':float(depths[i,j])})
        return sorted(rows,key=lambda x:-x['overlap_m'])


def try_retreat(worker, arm, command, request, start, goal, diagnostic):
    """Return joint samples and metadata, or raise without changing the robot."""
    from .failures import state_violations
    if not request['scene'].get('allow_start_collision_retreat',False):
        raise ValueError('Recovery not enabled for this scene')
    if request['scene'].get('unexpected_contact_blocked', False):
        raise ValueError('Physical non-finger collision is blocked; planner-only retreat is not eligible')
    if diagnostic.get('violated_constraints') != ['scene_collision']:
        raise ValueError('Only isolated scene collision is eligible; self/limit violations are rejected')
    if len(request['commands']) != 1 or command['kind'] != 'cartesian_pose':
        raise ValueError('Retreat requires a single Cartesian arm command')
    if any(request['scene'].get('payloads',{}).values()):
        raise ValueError('Retreat with a carried object is not supported')
    from curobo.types import JointState
    q0=np.asarray(start.position.detach().cpu()).reshape(-1)
    current=worker.kin.forward(q0,arm)
    target=worker.kin.world_pose(command['values'],arm,command['frame'])
    delta=np.asarray(target[:3])-current[:3]
    if not (.005<=delta[2]<=.08 and np.linalg.norm(delta[:2])<=.02 and np.linalg.norm(delta)<=.08
            and pose_errors(current,target)[1]<=.05):
        raise ValueError('Only a 5–80 mm upward retreat with <=20 mm lateral shift and <=0.05 rad turn is eligible')
    geometry=SceneOverlap(worker,arm,request)
    initial=geometry.depths([q0])[0]
    diagnostic['start_overlap_pairs']=geometry.describe(q0)
    if initial.max(initial=0)>.005:
        raise ValueError('Start overlap exceeds the 5 mm retreat bound')
    ik=worker.planner.ik_solver.solve_pose(goal,return_seeds=1,current_state=start)
    if not bool(ik.success.any()):
        raise ValueError('No feasible retreat endpoint IK')
    solutions=ik.solution.detach().cpu().numpy().reshape(-1,len(q0))
    q1=min(solutions,key=lambda q:np.linalg.norm(q-q0))
    if np.max(np.abs(q1-q0))>.35:
        raise ValueError('Retreat exceeds the 0.35 rad joint displacement bound')
    # Start with a dense joint polyline and further bound sphere travel. Tests
    # are sampled (not a claim of an exact continuous collision proof).
    count=max(2,int(np.ceil(np.max(np.abs(q1-q0))/.001))+1)
    while True:
        q=np.linspace(q0,q1,count)
        spheres=np.asarray([geometry.world(row) for row in q])
        travel=float(np.linalg.norm(np.diff(spheres[:,:,:3],axis=0),axis=-1).max())
        if travel<=.0005:break
        count*=2
        if count>4096:raise ValueError('Retreat sampling bound exceeded')
    depths=geometry.depths(q);check_overlap_trace(depths.reshape(len(q),-1))
    # Independent GPU feasibility retains all non-scene constraints throughout
    # and requires the endpoint to pass the unmodified planner's full checks.
    tensor=worker.torch.tensor(q,device='cuda',dtype=worker.torch.float32)
    violations=state_violations(worker.planner,JointState.from_position(tensor,joint_names=list(worker.profile.joint_names)))
    if set(violations)-{'scene_collision'}:
        raise ValueError('Retreat violates non-scene planner constraints: '+str(violations))
    end=JointState.from_position(tensor[-1:],joint_names=list(worker.profile.joint_names))
    if state_violations(worker.planner,end):raise ValueError('Retreat endpoint fails full planner feasibility')
    return q,{'protocol':'bounded_shallow_scene_retreat_v1','initial_overlap_pairs':diagnostic['start_overlap_pairs'],
              'initial_max_overlap_m':float(initial.max()),'final_max_overlap_m':float(depths[-1].max()),
              'samples':len(q),'max_sample_sphere_travel_m':travel,'max_joint_speed_rad_s':.1,
              'collision_checks_disabled':False,'limitations':'Dense sampled approximate planner geometry; live physical collision stop remains active.'}
