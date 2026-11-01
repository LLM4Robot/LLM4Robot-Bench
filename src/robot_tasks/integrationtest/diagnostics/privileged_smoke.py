"""Deterministic full-state manipulation checks; no model client or API calls.

Only robot joint drive targets are commanded after reset. Props remain free
PhysX bodies/FEM: no attachments, object pose writes, force assistance or scoring
overrides. Numerical IK is intentionally privileged and is not a GPT result.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
import hashlib
from pathlib import Path
import traceback
import time

import numpy as np

from ...paths import PROJECT_ROOT
from ...robots.transforms import pose_transform, transform_pose, pose_residual


def matrix(position, approach=(0., 0., -1.), jaw=(1., 0., 0.)):
    x, y = np.asarray(approach, dtype=float), np.asarray(jaw, dtype=float)
    x /= np.linalg.norm(x)
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    out = np.eye(4)
    out[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    out[:3, 3] = position
    return out


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class SmokeFailure(RuntimeError):
    pass


class Controller:
    def __init__(self, backend, robot, task, config, output, *, video_directory=None, wall_timeout=600.):
        import cv2

        self.backend, self.robot, self.task, self.config = backend, robot, task, config
        self.output, self.cv2 = output, cv2
        self.label, self.tick_count = "initial", 0
        self.started = time.monotonic()
        self.wall_timeout, self.frames_written = wall_timeout, 0
        self.initial_image_saved = False
        self.trace, self.commands, self.checkpoints = [], [], []
        self.holds = {}
        self.assessment = None
        from ...video_delivery import video_layout
        self.video_layout = video_layout(config.task_id)
        self.writer = cv2.VideoWriter(str((video_directory or output) / f"smoke_test_{config.task_id}.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), 6., tuple(self.video_layout['resolution']))
        if not self.writer.isOpened():
            raise RuntimeError("Could not create smoke MP4")

    def read(self):
        return self.task.scene.measure()

    def before_physics_step(self, dt):
        # This controller sends joint targets itself and has no planner paths or
        # queued gripper actions. Preserve per-physics collision handling and
        # grip-rate updates, while sampling measured state at the control rate.
        # Full R5Adapter.before_physics_step otherwise synchronizes both GPU
        # articulations 1920 times/s just to fill a settling-history buffer.
        if self.robot.paths or self.robot.pending_grippers:
            raise RuntimeError('Privileged controller cannot run queued policy paths')
        self.robot._stop_on_collision()
        self.robot.tracking_time_s += dt
        for arm in self.robot.grip_elapsed:
            self.robot.grip_elapsed[arm] += dt
        self.robot._advance_drive_targets(dt)

    def object(self, name):
        return pose_transform(self.read()[name]["pose"])

    def tcp(self, arm):
        return pose_transform(self.robot.state()[arm].tcp_pose_world_wxyz)

    def capture(self):
        frames = self.backend.capture_rgb()
        width,height = self.video_layout['resolution']
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for frame in frames:
            box = self.video_layout['rois'][frame.camera]
            if box is None:
                continue
            im = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 3)
            bgr = self.cv2.cvtColor(im, self.cv2.COLOR_RGB2BGR)
            x0,y0,x1,y1 = box
            label = {'overview':'Overview', 'right_wrist':'Robot LEFT wrist / tool right',
                     'left_wrist':'Robot RIGHT wrist / tool left'}[frame.camera]
            self.cv2.putText(canvas, label, (x0+4,max(12,y0-7)), self.cv2.FONT_HERSHEY_SIMPLEX,
                             .45, (220,230,235), 1, self.cv2.LINE_AA)
            scale = min((x1-x0)/frame.width, (y1-y0)/frame.height)
            w,h = round(frame.width*scale),round(frame.height*scale)
            x,y = x0+(x1-x0-w)//2,y0+(y1-y0-h)//2
            canvas[y:y+h,x:x+w] = self.cv2.resize(bgr,(w,h))
            if self.label == 'initial' and not self.initial_image_saved:
                self.cv2.imwrite(str(self.output / ('initial_'+frame.camera+'.png')), bgr)
        status = self.assessment.status if self.assessment else "initial"
        text = f"{self.config.task_id} | privileged physics / NO GPT | {self.backend.time_s():.2f}s | {self.label} | {status}"
        self.cv2.putText(canvas, text, (8, 21), self.cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, self.cv2.LINE_AA)
        self.writer.write(canvas)
        self.frames_written += 1
        if self.label == "initial" and not self.initial_image_saved:
            self.cv2.imwrite(str(self.output / "initial.png"), canvas)
            self.initial_image_saved = True
        return canvas

    def step(self, seconds):
        from ...types import EvaluationState

        for _ in range(max(1, math.ceil(seconds / self.config.control_dt_s))):
            self.backend.step(self.config.control_dt_s)
            states = self.robot.state()
            for arm,state in states.items():
                self.robot.settling.add(self.robot.tracking_time_s,arm,
                    state.joint_positions_rad+state.finger_positions_m)
            self.assessment = self.task.evaluate(EvaluationState({
                "simulation_time_s": self.backend.time_s(), "robot": states}), self.robot.feedback())
            self.tick_count += 1
            if self.tick_count % 60 == 0:
                print('SMOKE_PROGRESS',self.config.task_id,self.label,self.backend.time_s(),flush=True)
            if self.tick_count % 2 == 0:
                rows = self.read()
                self.trace.append({"time_s": self.backend.time_s(), "phase": self.label,
                    "robot": {a: {"q": s.joint_positions_rad, "tcp": s.tcp_pose_world_wxyz,
                                   "width": s.gripper_width_m} for a, s in states.items()},
                    "objects": {n: {k: r[k] for k in ("pose", "centroid", "fingers", "principal_strain_p95") if k in r}
                                for n, r in rows.items()},
                    "assessment": {"status": self.assessment.status, "reason": self.assessment.reason,
                                   "criteria": self.assessment.metrics.get("criteria", {})}})
                self.capture()
            if self.robot.collision_stop is not None:
                raise SmokeFailure("Non-finger collision: " + str(self.robot.collision_stop))
            if self.assessment.status in {"invalid", "failed"}:
                raise SmokeFailure(self.assessment.reason)
            if self.backend.time_s() > 180:
                raise SmokeFailure("180 s simulated-time budget exceeded")
            if time.monotonic()-self.started > self.wall_timeout:
                raise SmokeFailure(f"{self.wall_timeout:g} s wall-time budget exceeded")

    def phase(self, label):
        self.label = label
        self.checkpoints.append({"phase": label, "time_s": self.backend.time_s()})
        print("SMOKE_PHASE", self.config.task_id, label, self.backend.time_s(), flush=True)

    def solve(self, arm, target):
        from scipy.optimize import least_squares

        kin = self.robot.kinematics
        current = np.array(self.robot.state()[arm].joint_positions_rad)
        actual = self.tcp(arm)
        fk = pose_transform(kin.forward(current, arm))
        # Correct the small measured FK offset without writing a physical pose.
        desired = fk @ np.linalg.inv(actual) @ target

        def residual(q):
            state = kin.bases[arm] @ kin._forward_transform(q)
            r = pose_residual(state, desired)
            return np.r_[r[:3], r[3:]*.20]

        guesses = (current, np.array([0., -.8, 1.3, 0., .8, 0.]),
                   np.array([0., .8, -1.3, 0., -.8, 0.]), np.zeros(6))
        best = None
        for guess in guesses:
            fit = least_squares(residual, np.clip(guess, kin.lower+1e-6, kin.upper-1e-6),
                                bounds=(kin.lower+1e-6, kin.upper-1e-6), max_nfev=160,
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            error = np.linalg.norm(residual(fit.x))
            if best is None or error < best[0]:
                best = error, fit.x
            if error < .001:
                break
        if best[0] > .004:
            raise SmokeFailure(f"Privileged IK cannot reach {arm} target: residual={best[0]:.4f}, target={target[:3,3].tolist()}")
        return best[1]

    def goto(self, arm, target, *, minimum_s=.5, _segment=False, _refine=0):
        from ...robots.control import width_to_joints

        target = np.array(target, dtype=float)
        if not _segment:
            from scipy.spatial.transform import Rotation, Slerp
            origin = self.tcp(arm)
            error = pose_residual(origin,target)
            count = max(1,math.ceil(np.linalg.norm(error[:3])/.025),
                        math.ceil(np.linalg.norm(error[3:])/.15))
            if count > 1:
                rotations = Slerp([0.,1.],Rotation.from_matrix([origin[:3,:3],target[:3,:3]]))
                for u in np.linspace(0.,1.,count+1)[1:]:
                    point = target.copy()
                    point[:3,3] = origin[:3,3]*(1-u)+target[:3,3]*u
                    point[:3,:3] = rotations(u).as_matrix()
                    self.goto(arm,point,minimum_s=minimum_s/count,_segment=True)
                return
        q = self.solve(arm, target)
        start = np.array(self.robot.state()[arm].joint_positions_rad)
        duration = max(minimum_s, 1.5*float(np.abs(q-start).max())/.35)
        count = max(1, math.ceil(duration/self.config.control_dt_s))
        self.commands.append({"phase": self.label, "arm": arm, "target_tcp": transform_pose(target),
                              "joint_target": q.tolist(), "duration_s": count*self.config.control_dt_s})
        for i in range(1, count+1):
            u = i/count
            u = u*u*(3-2*u)
            targets = dict(self.robot.targets)
            targets[arm] = tuple(start*(1-u)+q*u) + targets[arm][6:]
            self.robot._send_targets(targets)
            self.step(self.config.control_dt_s)
        self.step(.12)
        actual = self.tcp(arm)
        error = float(np.linalg.norm(actual[:3,3]-target[:3,3]))
        if error > .003 and _refine < 2:
            self.goto(arm,target,minimum_s=.2,_segment=True,_refine=_refine+1)
            return
        if error > .008:
            raise SmokeFailure(f"Physical TCP did not reach target: {arm}, error={error:.4f}m")

    def grip(self, arm, width):
        from ...robots.control import width_to_joints

        targets = dict(self.robot.targets)
        targets[arm] = targets[arm][:6] + width_to_joints(self.config.robot, width)
        self.robot._start_grip(arm, targets[arm][6:], self.robot.state()[arm])
        self.robot._send_targets(targets)
        self.commands.append({"phase": self.label, "arm": arm, "gripper_width": width})
        self.step(1.2)

    def pick(self, name, arm, local_point, *, jaw=(1.,0.,0.), width=.070, close=.0, lift=.06, approach=None, refresh=False):
        self.phase("grasp_" + name)
        # Raise the wrist behind the work area before rotating toward a prop.
        # A direct home-to-grasp joint interpolation can sweep link6 through
        # a tall marker even when the final TCP approach is clear.
        clear = self.tcp(arm)
        if clear[2,3] < 1.13:
            clear[2,3] = 1.14
            self.goto(arm, clear, minimum_s=1.)
        obj = self.object(name)
        position = obj[:3,:3] @ np.array(local_point) + obj[:3,3]
        if approach is None:
            base = self.robot.kinematics.bases[arm][:3,3]
            radial = position-base;radial[2]=0.;radial/=np.linalg.norm(radial)
            approach = (radial+np.array([0.,0.,-1.]))/math.sqrt(2.)
        approach = np.asarray(approach,dtype=float);approach/=np.linalg.norm(approach)
        jaw = np.asarray(jaw,dtype=float)
        # Reversing the closing axis preserves the grasp but keeps the camera
        # mounted on link6's +Z side above the tabletop during a side approach.
        if np.cross(approach,jaw)[2] < 0:
            jaw = -jaw
        goal = matrix(position, approach, jaw)
        above = goal.copy()
        above[:3,3] -= np.asarray(approach)*.045
        self.grip(arm, width)
        # Orient in the empty corridor near the base, then translate with a
        # fixed orientation. Simultaneously rotating and translating can take
        # the wrist outside the reachable set even with two reachable ends.
        aligned = goal.copy();aligned[:3,3] = clear[:3,3]
        self.goto(arm,aligned,minimum_s=1.,_segment=True)
        self.goto(arm, above, minimum_s=1.)
        if refresh:
            # Tall loose props may finish settling during the approach. Update
            # the grasp from the measured pose, not a stale pre-motion pose.
            obj = self.object(name)
            goal[:3,3] = obj[:3,:3] @ np.asarray(local_point) + obj[:3,3]
        self.goto(arm, goal)
        self.grip(arm, close)
        self.holds[name] = (arm, np.linalg.inv(self.tcp(arm)) @ self.object(name))
        if lift:
            lifted = self.object(name)
            lifted[2,3] += lift
            self.move_object(name, lifted)
            self.step(.3)
            if min(self.read()[name]["fingers"][arm]) < .005:
                raise SmokeFailure(f"No measured bilateral grip after lifting {name}")

    def move_object(self, name, goal):
        arm, relative = self.holds[name]
        self.goto(arm, np.asarray(goal) @ np.linalg.inv(relative))

    def release(self, name):
        arm, _ = self.holds[name]
        self.grip(arm, .070)
        del self.holds[name]

    def finish(self):
        self.phase("verify_final_state")
        self.step(1.5)
        return self.assessment.status == "succeeded"


def execute(c):
    from scipy.spatial.transform import Rotation

    tid = c.config.task_id
    initial = {n: c.object(n).copy() for n, row in c.read().items() if "pose" in row}
    right, left = "right", "left"  # Tool IDs, fixed to overview base positions.

    def pick_cup(name,arm,lift=.06):
        approach=c.object(name)[:3,3]-c.robot.kinematics.bases[arm][:3,3]
        approach[2]=0.;approach/=np.linalg.norm(approach)
        jaw=np.cross([0,0,1.],approach)
        c.pick(name,arm,(0,0,.055),approach=approach,jaw=jaw,close=.025,lift=lift)
    if tid == "L1-01":
        c.pick("marker", right, (0,0,.125), close=.012, lift=.04,approach=(0,0,-1))
        c.phase("place_marker_inside_mug")
        cup = pose_transform(c.task.spec.design.fixture_pose)
        p = c.task.spec.parameters
        toward_base = c.robot.kinematics.bases[right][:3,3]-cup[:3,3]
        toward_base[2]=0.;toward_base/=np.linalg.norm(toward_base)
        tilt_axis=np.cross([0,0,1.],toward_base)
        # The instruction allows either pen end and does not require upright
        # insertion. Tilt toward the base to keep the wrist in reach, and centre
        # the complete immersed segment in the actual mug cavity.
        tilted=c.object('marker')
        tilted[:3,:3]=Rotation.from_rotvec(tilt_axis*np.deg2rad(50)).as_matrix()@initial['marker'][:3,:3]
        c.move_object('marker',tilted)
        for u in np.linspace(0.,1.,10):
            angle=np.deg2rad(50.-17.*u)
            target=tilted.copy()
            target[:3,:3]=Rotation.from_rotvec(tilt_axis*angle).as_matrix()@initial['marker'][:3,:3]
            target[2,3]=(p['rim']+.012)*(1-u)+(p['floor']+.007)*u
            depth=max(0.,p['rim']-target[2,3])
            target[:2,3]=cup[:2,3]-.5*depth*np.tan(angle)*toward_base[:2]
            c.move_object('marker',target)
        c.release("marker")
    elif tid == "L1-02":
        c.pick("plug", right, (0,0,.048), close=.014, lift=.05,
               approach=(-1.,0.,0.),jaw=(0.,-1.,0.))
        c.phase("align_prongs_with_nearest_socket")
        p = c.task.spec.parameters
        fixture = pose_transform(c.task.spec.design.fixture_pose)
        centre = np.mean(p["slot_centres"],axis=0)
        goal = fixture.copy()
        # Blades separate along charger local X, target pair along strip local Y.
        goal[:3,:3] = fixture[:3,:3] @ Rotation.from_euler("z",math.pi/2).as_matrix()
        goal[:3,3] = fixture[:3,:3] @ np.r_[centre, p["rim"]-fixture[2,3]+.015] + fixture[:3,3]
        c.move_object("plug", goal)
    elif tid == "L1-04":
        centre = c.task.spec.parameters["centre_xy"]
        pick_cup('lower_cup',right)
        goal = initial["lower_cup"].copy();goal[:2,3]=centre
        c.phase("place_robot_left_cup_at_centre")
        c.move_object("lower_cup",goal);c.release("lower_cup")
        pick_cup('upper_cup',left)
        goal=c.object("lower_cup");goal[2,3]+=.060
        c.phase("nest_robot_right_cup")
        c.move_object("upper_cup",goal);c.release("upper_cup")
    elif tid == "L1-05":
        from ...video_delivery import scenes
        c.phase("clear_observer_from_transfer_corridor")
        # Rotate the observer's base outward while preserving its folded
        # posture. Cartesian IK near the home singularity can choose an
        # elbow-down branch that sweeps the table.
        observer=np.array(c.robot.state()[left].joint_positions_rad)
        parked=observer.copy();parked[0]-=.7
        c.commands.append({'phase':c.label,'arm':left,'joint_target':parked.tolist(),
                           'duration_s':5.})
        for u in np.linspace(0.,1.,61)[1:]:
            targets=dict(c.robot.targets)
            targets[left]=tuple(observer+(parked-observer)*(u*u*(3-2*u)))+targets[left][6:]
            c.robot._send_targets(targets);c.step(c.config.control_dt_s)
        point = scenes()[tid]['flower_collision_repair']['head_grasp_local_m']
        c.pick("rose",right,point,width=.060,close=0.,lift=0.,
               approach=(0.,-1.,-1.),jaw=(1.,0.,0.),refresh=True)
        c.phase("extract_flower_from_source_cup")
        from scipy.spatial.transform import Slerp
        start=c.object('rose')
        # Lean the head back toward the robot as the stem rises. This keeps
        # the long flower within the R5 reach instead of lifting the wrist
        # vertically at its furthest extension.
        rotation=Rotation.from_euler('x',-30,degrees=True).as_matrix() @ Rotation.from_euler('z',110,degrees=True).as_matrix()
        rotations=Slerp([0.,1.],Rotation.from_matrix([start[:3,:3],rotation]))
        for u in np.linspace(0.,1.,9)[1:]:
            lifted=start.copy();lifted[2,3]+=.115*u;lifted[:3,:3]=rotations(u).as_matrix()
            c.move_object('rose',lifted)
        c.step(.3)
        if min(c.read()['rose']['fingers'][right]) < .005:
            raise SmokeFailure('No measured bilateral grip after lifting rose')
        c.phase("insert_white_flower_in_other_cup")
        start=c.object('rose'); cup=c.object('target_cup')
        toward=c.robot.kinematics.bases[right][:3,3]-cup[:3,3]
        toward[2]=0.;toward/=np.linalg.norm(toward)
        axis=np.cross([0.,0.,1.],toward)
        tilt=np.deg2rad(35.)
        rotation=Rotation.from_rotvec(axis*tilt).as_matrix()@Rotation.from_euler('z',110,degrees=True).as_matrix()
        rotations=Slerp([0.,1.],Rotation.from_matrix([start[:3,:3],rotation]))
        for u in np.linspace(0.,1.,13)[1:]:
            goal=start.copy();goal[:3,:3]=rotations(u).as_matrix()
            goal[:3,3]=start[:3,3]*(1-u)+(cup[:3,3]+[0,0,.105])*u
            c.move_object('rose',goal)
        c.phase("lower_stem_and_release")
        goal[:3,3]=cup[:3,3]+[0,0,.030]
        goal[:2,3]-=.5*(.080-.030)*np.tan(tilt)*toward[:2]
        c.move_object("rose",goal);c.release("rose")
        c.phase("withdraw_gripper")
        retreat=c.tcp(right);retreat[1,3]+=.060;c.goto(right,retreat)
    elif tid == "L1-03":
        from ...tasks.bench.native_geometry import shapes,polyhedron
        hat=c.object("hat")
        points=[polyhedron('video:yellow_visor',i)[0].mean(axis=0) for i,s in enumerate(shapes('video:yellow_visor')) if s['name'].startswith('Band_')]
        grasp=max(points,key=lambda p:p[0])
        radial=hat[:3,:3] @ np.r_[grasp[:2]/np.linalg.norm(grasp[:2]),0.]
        c.pick("hat",right,grasp,jaw=radial,width=.025,close=.0,lift=.06)
        p=c.task.spec.parameters
        a,b=np.asarray(p["hook_start"]),np.asarray(p["hook_end"])
        axis=(b-a)/np.linalg.norm(b-a)
        if axis[2]<0:axis=-axis
        y=np.cross(axis,[1.,0.,0.]);y/=np.linalg.norm(y);x=np.cross(y,axis)
        goal=np.eye(4);goal[:3,:3]=np.column_stack((x,y,axis))
        goal[:3,3]=(a+b)/2-goal[:3,:3]@np.array([0,0,p["band_plane_z"]])
        c.phase("thread_hat_head_opening_over_lowest_rod")
        before=goal.copy();before[:3,3]+=axis*.10
        c.move_object("hat",before);c.move_object("hat",goal);c.release("hat")
    elif tid == "L2-01":
        axis=initial["body"][:3,2]
        jaw=np.cross(axis,[0,0,1.]);jaw/=np.linalg.norm(jaw)
        c.pick("body",left,(0,0,.050),jaw=jaw,width=.035,close=.010,lift=.060)
        c.pick("cap",right,(0,0,.133),jaw=jaw,width=.035,close=.010,lift=0.)
        c.phase("pull_cap_along_pen_axis")
        goal=c.object("cap");goal[:3,3]+=axis*.050;c.move_object("cap",goal)
    elif tid == "L2-02":
        x=initial["upper_book"][:3,0]
        c.pick("upper_book",left,(-.063,0,0),jaw=(0,0,1.),approach=x,width=.040,close=.012,lift=0.)
        c.pick("middle_book",right,(.063,0,0),jaw=(0,0,1.),approach=-x,width=.040,close=.012,lift=0.)
        c.phase("pull_middle_book_50mm_hold_top_steady")
        goal=c.object("middle_book");goal[:3,3]+=np.asarray(c.task.spec.parameters["outward_axis_world"])*.052
        c.move_object("middle_book",goal)
    elif tid == "L2-03":
        pick_cup('source_cup',right,lift=.10)
        c.phase("pour_all_balls_into_other_cup")
        receiver=c.object("receiver_cup")
        for angle in np.linspace(0,-2.10,9)[1:]:
            goal=np.eye(4);goal[:3,:3]=Rotation.from_euler("y",angle).as_matrix()
            mouth=receiver[:3,3]+[0,0,.115]
            goal[:3,3]=mouth-goal[:3,:3]@np.array([0,0,.08])
            c.move_object("source_cup",goal)
        c.step(3.)
        c.phase("return_empty_cup_and_release")
        goal=initial["source_cup"].copy();goal[2,3]+=.13
        c.move_object("source_cup",goal);c.move_object("source_cup",initial["source_cup"]);c.release("source_cup")
    elif tid == "L2-05":
        from pxr import UsdGeom
        path=c.task.scene.paths["drawer"]+"/Frames/Pull"
        position=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(c.backend.world.stage.GetPrimAtPath(path))).T[:3,3]
        outward=np.asarray(c.task.spec.parameters["outward_axis_world"])
        c.phase("grip_native_drawer_edge")
        goal=matrix(position,approach=-outward,jaw=(0,0,1.))
        approach=goal.copy();approach[:3,3]+=outward*.06
        c.grip(right,.025);c.goto(right,approach);c.goto(right,goal);c.grip(right,0.)
        c.holds["drawer"]=(right,np.linalg.inv(c.tcp(right))@c.object("drawer"))
        c.phase("pull_additional_50mm")
        goal=c.object("drawer");goal[:3,3]+=outward*.051;c.move_object("drawer",goal)
        c.phase("return_to_initial_opening");c.move_object("drawer",initial["drawer"])
    elif tid == "L3-01":
        if not c.task.spec.parameters.get("body_fixed", False):
            c.pick("bottle",right,(0,0,.075),jaw=(0,-1,0),approach=(-1,0,0),width=.070,close=.020,lift=0.)
        c.pick("cap",left,(0,0,.152),jaw=(1,0,0),width=.060,close=.015,lift=0.)
        c.phase("unscrew_cap_counterclockwise")
        base=c.object("cap")
        for angle in np.linspace(0,math.pi,13)[1:]:
            goal=base.copy();goal[:3,:3]=Rotation.from_euler("z",angle).as_matrix()@base[:3,:3]
            goal[2,3]+=angle/(2*math.pi)*.009
            c.move_object("cap",goal)
        c.phase("lift_cap_40mm_clear_of_neck")
        goal=c.object("cap");goal[2,3]+= .045;c.move_object("cap",goal)
    elif tid == "L3-02":
        axis=initial["anchor"][:3,2]
        jaw=np.cross(axis,[0,0,1.]);jaw/=np.linalg.norm(jaw)
        c.pick("anchor",left,(0,0,.025),jaw=jaw,width=.035,close=.008,lift=.060)
        c.pick("nut",right,(0,0,.00525),jaw=jaw,width=.030,close=.010,lift=0.)
        c.phase("unscrew_native_hex_nut")
        for turn in range(8):
            base=c.object("nut")
            for angle in np.linspace(0,math.pi/2,7)[1:]:
                goal=base.copy();goal[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()@base[:3,:3]
                goal[:3,3]+=axis*(angle/(2*math.pi)*.0015)
                c.move_object("nut",goal)
            if c.assessment.status=="succeeded":break
            c.grip(right,.035)
            pos=c.tcp(right)[:3,3].copy();c.goto(right,matrix(pos,jaw=jaw));c.grip(right,.010)
            c.holds["nut"]=(right,np.linalg.inv(c.tcp(right))@c.object("nut"))
        goal=c.object("nut");goal[:3,3]+=axis*.015;c.move_object("nut",goal)
    elif tid == "L2-07":
        jaw=initial["eraser"][:3,1]
        c.pick("eraser",right,(0,0,.023),jaw=jaw,width=.070,close=.035,lift=.08)
        c.phase("wipe_all_native_ink")
        board=pose_transform(c.task.spec.parameters["board_pose"])
        centres=c.task.scene.centers
        for i,y in enumerate(np.arange(centres[:,1].min()-.012,centres[:,1].max()+.030,.040)):
            xs=[centres[:,0].min()-.030,centres[:,0].max()+.030]
            if i%2:xs.reverse()
            for x in np.linspace(*xs,12):
                goal=board.copy();goal[:3,3]=board[:3,:3]@np.array([x,y,.01005])+board[:3,3]
                c.move_object("eraser",goal)
    elif tid == "L3-03":
        c.phase("push_eraser_with_gripper_edge")
        row=c.read()["eraser"];start=c.object("eraser")[:3,3]
        goal=np.asarray(c.task.spec.design.articles[0].goal[:3])
        direction=goal-start;direction[2]=0.;direction/=np.linalg.norm(direction)
        approach=(direction+np.array([0.,0.,-1.]))/math.sqrt(2)
        jaw=np.cross([0,0,1.],direction)
        offset=np.linalg.norm(np.asarray(row["corners"])[:,:2]-start[:2],axis=1).max()+.020
        contact=start-direction*offset;contact[2]=.902
        above=contact.copy();above[2]+=.08
        c.grip(right,0.);c.goto(right,matrix(above,approach,jaw));c.goto(right,matrix(contact,approach,jaw))
        distance=np.linalg.norm((goal-start)[:2])
        for d in np.linspace(0,distance+.018,16)[1:]:
            c.goto(right,matrix(contact+direction*d,approach,jaw),minimum_s=.25)
        retreat=c.tcp(right);retreat[2,3]+=.08;c.goto(right,retreat)
    elif tid == "L3-04":
        c.pick("broom",right,(0,-.055,.003),width=.035,close=.010,lift=.10)
        pan=c.object("dustpan")
        c.phase("sweep_two_blocks_through_native_pan_lip")
        for n in ("block_0","block_1"):
            block=c.object(n)[:3,3]
            destination=pan[:3,:3]@np.array([0,0,.005])+pan[:3,3]
            direction=destination-block;direction[2]=0.;direction/=np.linalg.norm(direction)
            goal=np.eye(4)
            # Native brush +Y points toward the rigid bristles.
            y=direction;x=np.cross(y,[0,0,1.]);goal[:3,:3]=np.column_stack((x,y,[0,0,1.]))
            goal[:3,3]=block-direction*.10;goal[2,3]=.881
            high=goal.copy();high[2,3]+=.08;c.move_object("broom",high);c.move_object("broom",goal)
            for u in np.linspace(0,1,12)[1:]:
                p=goal.copy();p[:2,3]+=(destination[:2]-block[:2])*u;c.move_object("broom",p)
        c.phase("set_brush_down")
        c.release("broom")
    elif tid == "L2-04":
        row=c.read()["orange"]
        centre=np.array(row["centroid"])
        c.phase("grasp_native_fem_orange")
        goal=matrix(centre+[0,0,.006],jaw=(0,1,0))
        above=goal.copy();above[2,3]+=.10
        c.grip(right,.060);c.goto(right,above);c.goto(right,goal);c.grip(right,.012)
        c.phase("lift_orange_80mm")
        raised=goal.copy();raised[2,3]+=.08;c.goto(right,raised);c.step(.8)
        c.phase("return_orange_and_release")
        c.goto(right,goal);c.grip(right,.070)
        above=goal.copy();above[2,3]+=.10;c.goto(right,above)
    else:
        raise ValueError("No privileged smoke script for " + tid)


def run(task_id, output, metadata=None, *, wall_timeout=600., gui=False, keep_open=False):
    from ...backends.isaacsim import IsaacSimBackend
    from ...config import ExecutionSettings
    from ...registry import resolve_task
    from ...robots.adapters import R5Adapter
    from ...robots.kinematics import R5Kinematics
    from ...video_delivery import scenes

    output.mkdir(parents=True, exist_ok=True)
    metadata = metadata or PROJECT_ROOT/'build/reference_alignment_20261015/smokes'/task_id
    metadata.mkdir(parents=True,exist_ok=True)
    descriptor, config = resolve_task(task_id)
    config = descriptor.factory.prepare_config(config)
    if gui:
        config = replace(config, raw={**config.raw, "simulation": {
            **config.raw.get("simulation", {}), "headless": False}})
    backend, robot, task = IsaacSimBackend(), R5Adapter(), descriptor.factory()
    backend.run_directory = metadata
    result = {"task_id": task_id, "passed": False, "model_calls": 0,
              "real_robot_commands": 0, "controller": "privileged deterministic numerical IK and physical joint drives",
              "object_pose_writes_after_reset": 0, "object_attachments": False,
              "scope": "scripted physical task execution; not GPT performance or proof of real-world fidelity",
              "scene_sha256": scenes()[task_id]["sha256"],
              "runtime_scene_sha256": scenes()[task_id]["runtime_sha256"],
              "visual_revision": scenes()[task_id].get('visual_overlay',{}).get('revision'),
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scene_binding_sha256": hashlib.sha256((PROJECT_ROOT/'src/robot_tasks/tasks/bench/delivered_scene.py').read_bytes()).hexdigest(),
              "motion_sequence_completed": False,
              "failure_category": None}
    result['video_path'] = str(output/f'smoke_test_{task_id}.mp4')
    result['wall_timeout_s'] = wall_timeout
    c = None
    try:
        backend.open(config)
        robot.config, robot.backend = config, backend
        robot.execution = ExecutionSettings.from_config(config)
        robot.kinematics = R5Kinematics(config.robot)
        robot.instances = backend.add_r5(config.robot)
        backend.scene_report['ik_solver'] = 'privileged deterministic numerical IK'
        backend.scene_report['model_calls'] = 0
        task.build_scene(backend, config)
        task.reset(backend, config.seed)
        robot.reset(config.seed)
        backend.before_physics_step = robot.before_physics_step
        c = Controller(backend, robot, task, config, metadata,video_directory=output,wall_timeout=wall_timeout)
        backend.before_physics_step = c.before_physics_step
        backend.scene_report['privileged_control_clock'] = {
            'physics_dt_s':config.physics_dt_s,'state_sampling_dt_s':config.control_dt_s,
            'collision_checks_every_physics_step':True,'gripper_updates_every_physics_step':True}
        result["pre_step_state"] = serial(task.scene.measure())
        from ...video_delivery import setup_digest, visual_setup
        result['visual_setup_sha256'] = setup_digest()
        result['visual_setup_revision'] = visual_setup()['revision']
        result['initial_recording'] = 'First capture follows physical stepping; pre-reset renderer cache is not recorded.'
        result['video_layout'] = c.video_layout
        c.step(.75)
        if task_id == 'L1-05':
            task.initialize_episode(backend, robot, lambda **_: c.step(config.control_dt_s))
        result["initial_state"] = serial(task.scene.measure())
        displacements = {name:float(np.linalg.norm(np.asarray(row['pose'][:3])-
                         np.asarray(result['pre_step_state'][name]['pose'][:3])))
                         for name,row in result['initial_state'].items()
                         if 'pose' in row and name in result['pre_step_state']}
        result['initial_displacements_m'] = displacements
        for name,row in result['initial_state'].items():
            if 'centroid' in row and name in result['pre_step_state']:
                displacements[name] = float(np.linalg.norm(np.asarray(row['centroid'])-
                    np.asarray(result['pre_step_state'][name]['centroid'])))
        if displacements and max(displacements.values()) > .05:
            name = max(displacements,key=displacements.get)
            raise SmokeFailure(f'Initial scene unstable before manipulation: {name} moved {displacements[name]:.3f} m')
        from ...tasks.bench.english_reference import initial_state_issue
        issue = initial_state_issue(task.scene.spec, task.scene.measure())
        if issue:
            raise SmokeFailure(issue)
        execute(c)
        result['motion_sequence_completed'] = True
        result["passed"] = c.finish()
        if not result["passed"]:
            result["error"] = "Script ended without independently measured task success"
            result['failure_category'] = 'goal_not_reached'
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        message = str(exc)
        result['failure_category'] = (
            'controller_unreachable_target' if 'IK cannot reach' in message else
            'controller_tracking_or_contact' if 'TCP did not reach' in message else
            'physical_collision' if 'Non-finger collision' in message else
            'grasp_not_established' if 'bilateral grip' in message else
            'execution_budget' if 'budget exceeded' in message else
            'initial_scene_physics' if c is not None and c.label == 'initial' else
            'runtime_or_controller_error')
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        if c is not None:
            if c.assessment is not None:
                result["final_assessment"] = asdict(c.assessment)
            result["simulated_duration_s"] = backend.time_s()
            result["commands"] = c.commands
            result["checkpoints"] = c.checkpoints
            try:
                c.label = "PASS" if result["passed"] else "FAIL - " + result.get("error", "")[:80]
                image = c.capture()
                c.cv2.imwrite(str(metadata / "final.png"), image)
                for _ in range(6):c.writer.write(image)
                c.frames_written += 6
            except Exception:
                pass
            c.writer.release()
            result['video_frames'] = c.frames_written
            (metadata / "trace.json").write_text(json.dumps(serial(c.trace)), encoding="utf-8")
        result["scene"] = backend.scene_report
        (metadata / f"smoke_test_{task_id}.json").write_text(json.dumps(serial(result),ensure_ascii=False,indent=2),encoding="utf-8")
        print("PRIVILEGED_SMOKE", task_id, result["passed"], result.get("error", ""),flush=True)
        if gui and keep_open and c is not None:
            print("SMOKE_GUI_READY: result saved; close the window to exit. Physics is held at the final state.", flush=True)
            try:
                while backend.app.is_running():
                    backend.pump_events()
                    time.sleep(.02)
            except KeyboardInterrupt:
                pass
        robot.close()
        backend.close()
    return result["passed"]


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",nargs='?')
    parser.add_argument('--all',action='store_true')
    parser.add_argument("--output",type=Path)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--max-wall-seconds',type=float,default=600.)
    parser.add_argument('--gui',action='store_true',help='Show the interactive Isaac Sim viewport')
    parser.add_argument('--keep-open',action='store_true',help='Keep the GUI at the final state after saving the result')
    args=parser.parse_args()
    if args.keep_open and not args.gui:
        parser.error('--keep-open requires --gui')
    if args.all and args.gui:
        parser.error('--gui is supported for a single task')
    if args.all:
        import subprocess,sys
        from ...video_delivery import scenes
        root=args.output or PROJECT_ROOT/'real_videos'
        meta=args.metadata or PROJECT_ROOT/'build/reference_alignment_20261015/smokes'
        summaries=[]
        for tid in scenes():
            folder=meta/tid;folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'smoke_test_{tid}.json'
            file.unlink(missing_ok=True)
            with (folder/'worker.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m',__package__+'.privileged_smoke',tid,
                    '--output',str(root/tid),'--metadata',str(folder),
                    '--max-wall-seconds',str(args.max_wall_seconds)],stdout=log,stderr=subprocess.STDOUT)
            row=json.loads(file.read_text()) if file.exists() else {'task_id':tid,'passed':False,'error':'No worker result'}
            summaries.append({key:row.get(key) for key in ('task_id','passed','failure_category','error','motion_sequence_completed')})
            (meta/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
            print('PRIVILEGED_BATCH',json.dumps(summaries[-1],ensure_ascii=False),flush=True)
        raise SystemExit(0 if all(row['passed'] for row in summaries) else 2)
    if not args.task:
        parser.error('Provide a task ID or --all')
    raise SystemExit(0 if run(args.task,args.output or PROJECT_ROOT/"real_videos"/args.task,
                             args.metadata,wall_timeout=args.max_wall_seconds,
                             gui=args.gui,keep_open=args.keep_open) else 2)
