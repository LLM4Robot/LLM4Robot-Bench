"""Passive scene readiness, before model observations; never moves or freezes props."""

import numpy as np

from ...robots.transforms import pose_transform


class SupportedRest:
    """Require full-body rest and support continuously, including axial rotation."""

    def __init__(self, weight_n, duration_s=1.0):
        self.weight_n = weight_n
        self.duration_s = duration_s
        self.anchor = None

    def update(self, time_s, row):
        pose = pose_transform(row['pose'])
        quiet = (np.linalg.norm(row['linear_velocity']) < .001
                 and np.linalg.norm(row['angular_velocity']) < .015
                 and .8 * self.weight_n < row['bench_force_n'] < 1.2 * self.weight_n
                 and max(row['finger_forces'], default=0.) < .01)
        if not quiet:
            self.anchor = None
            return False
        if self.anchor is not None:
            start, previous = self.anchor
            rotation = previous[:3, :3].T @ pose[:3, :3]
            angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))
            if np.linalg.norm(pose[:3, 3]-previous[:3, 3]) > .0005 or angle > .01:
                self.anchor = None
            elif time_s-start >= self.duration_s-1e-9:
                return True
        if self.anchor is None:
            self.anchor = (time_s, pose)
        return False


def settle_rose(scene, tick, *, max_wait_s=120.):
    """Only physics ticks and measurements; no pose/velocity writes or robot commands."""
    start = scene.backend.time_s()
    weight = scene.design.articles[0].mass * 9.81
    # A longer window covers the observed rocking cycle, not just a slow
    # turning point immediately before it rolls back in the other direction.
    monitor = SupportedRest(weight, duration_s=2.0)
    while scene.backend.time_s()-start < max_wait_s:
        tick(phase='task_initialization', score=False)
        row = scene.measure()['rose']
        now = scene.backend.time_s()
        if monitor.update(now, row):
            return {'protocol': 'passive_supported_rest_v1', 'ready': True,
                    'wait_s': now-start, 'stable_window_s': monitor.duration_s,
                    'linear_speed_m_s': float(np.linalg.norm(row['linear_velocity'])),
                    'angular_speed_rad_s': float(np.linalg.norm(row['angular_velocity'])),
                    'bench_force_n': row['bench_force_n'],
                    'pose_or_velocity_forced': False}
    raise RuntimeError('L1-05 rose did not reach supported rest before startup timeout; GPT was not called')


class SupportedCupFlowerRest:
    """A free flower supported by a cup/charger, including axial spin."""

    names = ('rose', 'source_cup', 'source_charger')

    def __init__(self, duration_s=2., fixed_names=()):
        self.duration_s = duration_s
        self.fixed_names = frozenset(fixed_names)
        self.anchor = None

    def update(self, time_s, rows):
        rose, cup, charger = (rows[n] for n in self.names)
        supported = (sum(rose['pair_forces'].get(n, 0.) for n in self.names[1:]) > .05
                     and ('source_cup' in self.fixed_names or cup['bench_force_n'] > .02)
                     and charger['pair_forces'].get('source_cup', 0.) > .02)
        # Contact solver velocities include tiny corrective impulses. Also
        # require a continuous measured-pose window, so low speed cannot hide
        # progressive sliding or a slowly turning flower.
        quiet = supported and all(
            np.linalg.norm(r['linear_velocity']) < .010
            and np.linalg.norm(r['angular_velocity']) < .10
            and max((f for pair in r['fingers'].values() for f in pair), default=0.) < .01
            for r in (rose, cup, charger))
        if not quiet:
            self.anchor = None
            return False
        poses = {n: pose_transform(rows[n]['pose']) for n in self.names}
        if self.anchor is not None:
            start, previous = self.anchor
            drift = any(np.linalg.norm(poses[n][:3,3]-previous[n][:3,3]) > .0005
                        or np.arccos(np.clip((np.trace(previous[n][:3,:3].T @ poses[n][:3,:3])-1)/2,-1,1)) > .01
                        for n in self.names)
            if drift:
                self.anchor = None
            elif time_s-start >= self.duration_s-1e-9:
                return True
        if self.anchor is None:
            self.anchor = (time_s, poses)
        return False


def settle_cup_flower(scene, tick, *, max_wait_s=20.):
    """Wait for genuine cup support; never pin the flower or zero velocities."""
    start = scene.backend.time_s()
    initial = pose_transform(scene.measure()['rose']['pose'])
    # A fixed cup transfers its load through the world constraint, so a
    # cup/table contact impulse is not required. Flower/charger contact and
    # continuous measured rest are still required for every moving object.
    monitor = SupportedCupFlowerRest(fixed_names=getattr(scene, 'fixed_names', ()))
    while scene.backend.time_s()-start < max_wait_s:
        tick(phase='task_initialization', score=False)
        rows = scene.measure()
        now = scene.backend.time_s()
        if monitor.update(now, rows):
            final = pose_transform(rows['rose']['pose'])
            angle = float(np.arccos(np.clip((np.trace(initial[:3,:3].T @ final[:3,:3])-1)/2,-1,1)))
            displacement = float(np.linalg.norm(final[:3,3]-initial[:3,3]))
            if angle > np.deg2rad(3.) or displacement > .005:
                raise RuntimeError('L1-05 flower drifted from its supported initial layout; GPT was not called')
            return {'protocol':'passive_cup_flower_rest_v1', 'ready':True,
                    'wait_s':now-start, 'stable_window_s':monitor.duration_s,
                    'position_drift_m':displacement, 'rotation_drift_rad':angle,
                    'fixed_cups': sorted(monitor.fixed_names & {'source_cup', 'target_cup'}),
                    'pose_or_velocity_forced':False}
    raise RuntimeError('L1-05 cup-supported flower did not settle before startup timeout; GPT was not called')
