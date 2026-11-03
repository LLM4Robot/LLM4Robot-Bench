"""Opt-in study worker: ordinary planner first, checked finger mesh fallback."""
import os,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import importlib.util
import hashlib
# This opt-in worker mirrors exactly two numeric precision constants in the
# frozen native worker. Refuse an unreviewed upstream implementation change.
EXPECTED_NATIVE_SHA256 = "a7ddd8cf89eaff4e767f2b6c2ae790e3945b75fdfb826af7a3f9f2340c9c84bb"
_native_path=Path(__file__).resolve().parents[2]/'src/robot_tasks/planning/worker.py'
if hashlib.sha256(_native_path.read_bytes()).hexdigest()!=EXPECTED_NATIVE_SHA256:
    raise RuntimeError('Native planner changed; review/rebase the opt-in precision worker first')
_spec=importlib.util.spec_from_file_location("robot_tasks.planning.control_revalidation_native",Path(__file__).with_name("native_precision.py"))
native=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(native)
from robot_tasks.planning.failures import PlanningFailure,state_violations
from robot_tasks.robots.transforms import pose_errors
from curobo.types import JointState
from mesh_guard import FingerMeshGuard
Base=native.CuroboWorker

class CheckedWorker(Base):
    narrow_arm=None
    original_spheres=Base.spheres
    def __init__(self,*args,**kwargs):
        directory=Path(args[1] if len(args)>1 else kwargs['directory'])
        self.audit_path=directory/'control-precision-audit.jsonl'
        self.mesh_directory=directory/'finger-mesh'
        super().__init__(*args,**kwargs)
        self.metadata.update(experimental_finger_mesh_guard='Opt-in bounded component-hull narrow phase; ordinary native plans unchanged',finger_proxy_replacement='Only candidate short paths with independently checked finger meshes; original other-link/world/self/payload checks retained')
    def spheres(self,state,arm):
        rows=super().spheres(state,arm)
        if self.narrow_arm==arm:
            ee=self.profile.ee_link;n=len(self.assets['link6_static'])+len(self.assets.get('wrist_spheres',{}).get(arm,[]))
            rows[ee]=[dict(r,radius=-1.) if i>=n else r for i,r in enumerate(rows[ee])]
        return rows
    def audit(self,request,**kw):
        path=self.audit_path;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps({'request_id':request['id'],**kw},ensure_ascii=False)+'\n')
    def plan(self,request):
        started=time.perf_counter();path=self.audit_path.with_name('planner-endpoint-audit.jsonl')
        row={'request_id':request['id'],'commands':request['commands'],'planner_tolerance_m':.0001,'endpoint_max_error_m':.00025}
        try:
            reply=self._checked_plan(request);row['diagnostics']=reply.get('diagnostics',{});return reply
        except Exception as exc:
            row['error']=str(exc);raise
        finally:
            row['elapsed_s']=time.perf_counter()-started
            with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def _checked_plan(self,request):
        try:return super().plan(request)
        except PlanningFailure as original:
            if original.diagnostic.get('code')!='start_collision':raise
            command=request['commands'][0];arm=command['arm'];state=request['states'][arm]
            if len(request['commands'])!=1 or command['kind']!='cartesian_pose':raise
            target=self.kin.world_pose(command['values'],arm,command['frame']);current=self.kin.forward(state['joint_positions_rad'],arm);distance,angle=pose_errors(current,target)
            if distance>.03 or angle>np.deg2rad(15.01):raise
            # The original collided scene proxy must become feasible when ONLY
            # this arm's finger spheres are replaced. All other checks remain.
            self.narrow_arm=arm
            try:
                rows=self.spheres(state,arm)[self.profile.ee_link]
                self.planner.kinematics.config.kinematics_config.update_link_spheres(self.profile.ee_link,self.torch.tensor([[*s['center'],s['radius']] for s in rows],device='cuda',dtype=self.torch.float32))
                self.planner.update_world(self.world(arm,request))
                start=JointState.from_position(self.torch.tensor([state['joint_positions_rad']],device='cuda',dtype=self.torch.float32),joint_names=list(self.profile.joint_names))
                violations=state_violations(self.planner,start)
                if violations:raise ValueError('Non-finger start constraints remain: '+str(violations))
                guard=FingerMeshGuard(self,arm,request,self.mesh_directory)
                if guard.distance(state['joint_positions_rad'])<=.000025:raise ValueError('Actual finger mesh start is not clear')
                reply=super().plan(request)
                proof=guard.validate(reply['paths'][arm]['positions'])
                self.audit(request,status='accepted_after_exact_finger_check',proof=proof,original_diagnostic=original.diagnostic)
                # Geometric distances/identities remain in privileged audit,
                # not in the actor's native execution observation.
                reply['diagnostics'][arm]['finger_mesh_guard']='bounded closed-mesh swept-path checked; original non-finger constraints retained'
                return reply
            except Exception as exc:
                self.audit(request,status='rejected',reason=str(exc),original_diagnostic=original.diagnostic)
                original.diagnostic['finger_mesh_check']='Short-path mesh verification did not pass; no commands applied.'
                raise original from exc
            finally:
                self.narrow_arm=None;rows=self.original_spheres(state,arm)[self.profile.ee_link]
                self.planner.kinematics.config.kinematics_config.update_link_spheres(self.profile.ee_link,self.torch.tensor([[*s['center'],s['radius']] for s in rows],device='cuda',dtype=self.torch.float32))

class PrecisionSearchWorker(CheckedWorker):
    """Resolve finite IK search failure within unchanged pose/limit criteria.

    Every candidate still passes native joint-space whole-path collision,
    self-collision, payload and timing checks before original-pose FK acceptance.
    No request is split into unvalidated motion and no joint limit is relaxed.
    """
    def _checked_plan(self,request):
        try:return super()._checked_plan(request)
        except PlanningFailure as original:
            if original.diagnostic.get('code')!='goal_ik_not_found':raise
            if len(request['commands'])!=1 or request['commands'][0]['kind']!='cartesian_pose':raise
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
            from copy import deepcopy
            from robot_tasks.robots.transforms import pose_transform
            cmd=request['commands'][0];arm=cmd['arm'];target=self.kin.world_pose(cmd['values'],arm,cmd['frame']);T=pose_transform(target)
            lower=self.kin.lower+1e-6;upper=self.kin.upper-1e-6
            def residual(q):
                F=pose_transform(self.kin.forward(q,arm))
                return np.r_[F[:3,3]-T[:3,3],.01*Rotation.from_matrix(F[:3,:3].T@T[:3,:3]).as_rotvec()]
            start=np.array(request['states'][arm]['joint_positions_rad']);rng=np.random.default_rng(0)
            seeds=[start]+[rng.uniform(lower,upper) for _ in range(3)];begin=time.monotonic();failures=[]
            for index,seed in enumerate(seeds):
                if time.monotonic()-begin>3:break
                x=least_squares(residual,np.clip(seed,lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=200,xtol=1e-11,ftol=1e-11,gtol=1e-11)
                pos,angle=pose_errors(self.kin.forward(x.x,arm),target)
                if pos>.0001 or angle>.01:continue
                candidate=deepcopy(request);candidate['commands'][0].update(kind='joint_position',values=x.x.tolist())
                try:
                    reply=super()._checked_plan(candidate)
                    end=reply['paths'][arm]['positions'][-1];pe,ae=pose_errors(self.kin.forward(end,arm),target)
                    if pe>.00025 or ae>.02:raise ValueError('Independently planned endpoint missed original pose tolerance')
                    reply['diagnostics'][arm].update(requested_endpoint_position_error_m=pe,requested_endpoint_orientation_error_rad=ae,numerical_precision_fallback='bounded IK candidate and native full-path validation; original requested pose independently checked')
                    self.audit(request,status='accepted_bounded_precision_IK',seed=index,position_error_m=pe,orientation_error_rad=ae,elapsed_s=time.monotonic()-begin)
                    return reply
                except Exception as exc:failures.append(str(exc))
            self.audit(request,status='bounded_precision_IK_not_accepted',elapsed_s=time.monotonic()-begin,native_path_failures=failures)
            raise original

native.CuroboWorker=PrecisionSearchWorker
if __name__=='__main__':raise SystemExit(native.main())
