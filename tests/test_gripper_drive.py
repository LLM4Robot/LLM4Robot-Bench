"""Drive limits and restoration authority under an obstructed closing goal."""
import unittest
import numpy as np
from robot_tasks.robots.gripper_drive import SymmetricGripDrive


class GripperDriveTests(unittest.TestCase):
    def drive(self,start=.035):
        return SymmetricGripDrive(start,.03,.15,.0015,0.,.04)

    def test_empty_close_has_symmetric_rate_and_acceleration_limits(self):
        drive=self.drive();q=(.035,.035);previous=drive.reference;v=0.;dt=1/240
        for _ in range(600):
            q=drive.advance((0.,0.),q,dt)
            self.assertEqual(q[0],q[1]);self.assertLessEqual(abs(q[0]-previous),.03*dt+1e-12)
            self.assertLessEqual(abs(drive.velocity),.03+1e-12)
            # Away from target arrival, acceleration is bounded too.
            if abs(q[0])>1e-5:self.assertLessEqual(abs(drive.velocity-v),.15*dt+1e-12)
            previous,v=q[0],drive.velocity
        self.assertAlmostEqual(q[0],0.)

    def test_obstruction_does_not_wind_reference_down_to_zero(self):
        drive=self.drive()
        for _ in range(1200):q=drive.advance((0.,0.),(.033,.033),1/240)
        self.assertAlmostEqual(q[0],.0315)
        self.assertEqual(drive.phase,'bounded_hold')
        self.assertLess(2000*(.033-q[0]),4.)
        # Equal references restore a displaced centre rather than following it.
        q=drive.advance((0.,0.),(.038,.028),1/240)
        self.assertLess(q[0],.038);self.assertGreater(q[1],.028)

    def test_opening_escapes_hold_and_stays_bounded(self):
        drive=self.drive(.0315)
        for _ in range(600):q=drive.advance((.035,.035),(.033,.033),1/240)
        self.assertEqual(drive.phase,'position');self.assertAlmostEqual(q[0],.035)

    def test_numpy_measurements_do_not_escape_as_drive_state(self):
        drive=self.drive(np.float64(.0315))
        for dt in (0., *([1/240]*600)):
            q=drive.advance(np.array([0.,0.]),np.array([.033,.033],dtype=np.float32),dt)
            for value in (*q,drive.reference,drive.velocity):
                self.assertIs(type(value),float)

if __name__=='__main__':unittest.main()
