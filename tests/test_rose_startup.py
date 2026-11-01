"""Do not begin model decisions while the native rose is still rolling."""
import unittest
from types import SimpleNamespace

import numpy as np

from robot_tasks.tasks.bench.startup_settling import SupportedRest, settle_rose
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.bench.native_geometry import shapes, transformed_poly
from robot_tasks.robots.transforms import pose_transform


def row():
    return dict(pose=[0.,0.,.8,1.,0.,0.,0.],linear_velocity=[0.,0.,0.],
                angular_velocity=[0.,0.,0.],bench_force_n=.07848,finger_forces=[0.,0.])


class RoseStartupTests(unittest.TestCase):
    def test_full_window_and_reject_axial_spin_and_missing_support(self):
        monitor=SupportedRest(.07848)
        r=row()
        self.assertFalse(monitor.update(0.,r))
        self.assertFalse(monitor.update(.9,r))
        self.assertTrue(monitor.update(1.,r))
        r['angular_velocity']=[0.,0.,.1]
        self.assertFalse(monitor.update(1.1,r))
        r['angular_velocity']=[0.,0.,0.]
        self.assertFalse(monitor.update(1.2,r))
        r['bench_force_n']=0.
        self.assertFalse(monitor.update(3.,r))
        r['bench_force_n']=.07848;r['finger_forces']=[.2,0.]
        self.assertFalse(monitor.update(4.,r))

    def test_small_reported_velocity_cannot_hide_pose_drift(self):
        monitor=SupportedRest(.07848);r=row()
        monitor.update(0.,r)
        r['pose'][0]=.0006
        self.assertFalse(monitor.update(1.,r))
        self.assertFalse(monitor.update(1.5,r))
        self.assertTrue(monitor.update(2.,r))

    def test_initial_native_colliders_are_near_support_without_penetration(self):
        spec=build_spec('L1-05');t=pose_transform(spec.design.articles[0].initial)
        lows={s['name']:transformed_poly('artificial_rose',i,t)[0][:,2].min()
              for i,s in enumerate(shapes('artificial_rose'))}
        self.assertGreaterEqual(min(lows.values()),spec.parameters['table_z'])
        self.assertLess(lows['Stem0']-spec.parameters['table_z'],.0002)
        self.assertLess(min(v for k,v in lows.items() if k.startswith('Bud'))
                        -spec.parameters['table_z'],.0002)
        self.assertGreater(t[0,2],.99)

    def test_startup_timeout_never_forces_pose_or_velocity(self):
        now=[0.];r=row();r['angular_velocity']=[1.,0.,0.]
        scene=SimpleNamespace(backend=SimpleNamespace(time_s=lambda:now[0]),
                              design=SimpleNamespace(articles=[SimpleNamespace(mass=.008)]),
                              measure=lambda:{'rose':r})
        def tick(**kwargs):
            self.assertEqual(kwargs,dict(phase='task_initialization',score=False))
            now[0]+=.1
        with self.assertRaisesRegex(RuntimeError,'GPT was not called'):
            settle_rose(scene,tick,max_wait_s=2.)
        np.testing.assert_equal(r['angular_velocity'],[1.,0.,0.])


if __name__=='__main__':unittest.main()
