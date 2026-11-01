"""Cup-supported startup must be passive, supported and continuously at rest."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from robot_tasks.tasks.bench.startup_settling import SupportedCupFlowerRest, settle_cup_flower


def rows():
    row = dict(pose=[0.,0.,.88,1.,0.,0.,0.],linear_velocity=[0.,0.,0.],
               angular_velocity=[0.,0.,0.],bench_force_n=0.,
               fingers={'left':[0.,0.],'right':[0.,0.]},pair_forces={})
    result={n:deepcopy(row) for n in SupportedCupFlowerRest.names}
    result['rose']['pair_forces']={'source_cup':.08,'source_charger':.04}
    result['source_cup']['bench_force_n']=.7
    result['source_charger']['pair_forces']={'source_cup':.54}
    return result


class CupFlowerStartupTests(unittest.TestCase):
    def test_cup_support_requires_complete_window_and_no_grip(self):
        state=rows();m=SupportedCupFlowerRest()
        self.assertFalse(m.update(0,state));self.assertFalse(m.update(1.9,state))
        self.assertTrue(m.update(2,state))
        state['rose']['fingers']['right']=[.1,.1]
        self.assertFalse(m.update(2.1,state))

    def test_free_fall_and_missing_cup_support_do_not_count(self):
        for name,key in [('rose','pair_forces'),('source_charger','pair_forces')]:
            state=rows();state[name][key]={};m=SupportedCupFlowerRest()
            self.assertFalse(m.update(0,state));self.assertFalse(m.update(3,state))
        state=rows();state['source_cup']['bench_force_n']=0;m=SupportedCupFlowerRest()
        self.assertFalse(m.update(0,state));self.assertFalse(m.update(3,state))

    def test_fixed_cup_needs_flower_and_charger_contact_but_not_table_impulse(self):
        state=rows();state['source_cup']['bench_force_n']=0.
        m=SupportedCupFlowerRest(fixed_names=('source_cup', 'target_cup'))
        self.assertFalse(m.update(0,state));self.assertTrue(m.update(2,state))
        state['source_cup']['pose'][0]+=.0006
        self.assertFalse(m.update(2.1,state))
        for name in ('rose','source_charger'):
            unsupported=deepcopy(state);unsupported[name]['pair_forces']={}
            m=SupportedCupFlowerRest(fixed_names=('source_cup', 'target_cup'))
            self.assertFalse(m.update(0,unsupported));self.assertFalse(m.update(3,unsupported))

    def test_axial_turning_and_low_velocity_pose_drift_reset_window(self):
        state=rows();m=SupportedCupFlowerRest();m.update(0,state)
        state['rose']['angular_velocity']=[0,0,.2]
        self.assertFalse(m.update(2,state))
        state['rose']['angular_velocity']=[0,0,0];m.update(3,state)
        state['rose']['pose'][0]=.0006
        self.assertFalse(m.update(5,state));self.assertFalse(m.update(6,state))
        self.assertTrue(m.update(7,state))
        state['rose']['pose'][3:]=[float(np.cos(.02)),0,0,float(np.sin(.02))]
        self.assertFalse(m.update(8,state))

    def test_timeout_never_forces_pose_velocity_or_support(self):
        now=[0.];state=rows();state['rose']['pair_forces']={};original=deepcopy(state)
        scene=SimpleNamespace(backend=SimpleNamespace(time_s=lambda:now[0]),measure=lambda:state)
        def tick(**kw):
            self.assertEqual(kw,dict(phase='task_initialization',score=False));now[0]+=.1
        with self.assertRaisesRegex(RuntimeError,'GPT was not called'):
            settle_cup_flower(scene,tick,max_wait_s=.5)
        self.assertEqual(state,original)

    def test_stable_but_wrong_final_pose_is_rejected(self):
        now=[0.];state=rows()
        scene=SimpleNamespace(backend=SimpleNamespace(time_s=lambda:now[0]),measure=lambda:state)
        def tick(**kw):
            now[0]+=.1;state['rose']['pose'][0]=.006
        with self.assertRaisesRegex(RuntimeError,'drifted'):
            settle_cup_flower(scene,tick,max_wait_s=3.)

if __name__=='__main__': unittest.main()
