import unittest
import numpy as np
from robot_tasks.tasks.bench.resistant_wiping import ResistantWipeLedger


class InkWearTests(unittest.TestCase):
    def ledger(self):
        return ResistantWipeLedger((0,0,0,1,0,0,0), [[0,0,.018]], facing_direction=1)

    def step(self, l, x, held=True, force=3., z=0, now=None):
        t = (l.last_time or 0) + .01 if now is None else now
        return l.update(t, (x,0,z,1,0,0,0), held=held, pad_contact_n=force)

    def test_three_passes(self):
        l=self.ledger()
        for x in np.linspace(-.07,.07,281): self.step(l,x)
        self.assertTrue(l.dirty[0]); self.assertGreater(l.opacity[0],.25)
        self.assertLess(l.opacity[0],.6)
        for x in np.linspace(.07,-.07,281): self.step(l,x)
        self.assertTrue(l.dirty[0]); self.assertLess(l.opacity[0],.2)
        for x in np.linspace(-.07,.07,281): self.step(l,x)
        self.assertFalse(l.dirty[0]); self.assertEqual(l.metrics()['clean_fraction'],1.)

    def test_no_contact_or_grasp_or_height(self):
        for kwargs in ({'held':False},{'force':0},{'z':.01},{'force':float('nan')}):
            l=self.ledger()
            for x in np.linspace(-.04,.04,100): self.step(l,x,**kwargs)
            self.assertEqual(l.coverage[0],0)

    def test_jitter_and_stationary(self):
        l=self.ledger(); self.step(l,0)
        for i in range(1000):self.step(l,.0001*(-1)**i)
        self.assertEqual(l.coverage[0],0)

    def test_repeat_time(self):
        l=self.ledger(); self.step(l,0,now=1)
        self.step(l,.005,now=1); self.step(l,.005,now=.5)
        self.assertEqual(l.coverage[0],0)

    def test_lift_breaks_stroke(self):
        l=self.ledger();self.step(l,-.02);self.step(l,0,force=0);self.step(l,.01)
        self.assertEqual(l.coverage[0],0)

    def test_teleport(self):
        l=self.ledger();self.step(l,-.04);self.step(l,.04)
        self.assertEqual(l.coverage[0],0)

    def test_slow_motion_counts(self):
        l=self.ledger()
        for x in np.linspace(-.04,.04,801):self.step(l,x)
        self.assertGreater(l.coverage[0],.7)

    def test_outside_footprint(self):
        l=self.ledger()
        for x in np.linspace(.2,.3,100):self.step(l,x)
        self.assertEqual(l.coverage[0],0)

if __name__=='__main__':unittest.main()
