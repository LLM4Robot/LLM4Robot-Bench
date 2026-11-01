"""Measured dimensions and physical openings of the L1-02 moulded holder."""
import unittest
import numpy as np
from scipy.spatial import ConvexHull
from robot_tasks.paths import PROJECT_ROOT
try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None

@unittest.skipIf(Usd is None, 'USD required')
class MeasuredHolder(unittest.TestCase):
    def test_dimensions_and_open_pin_channel(self):
        stage=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing/source/L1-03/runtime_aligned.usdc'))
        root='/World/Household/plug_storage_base'
        points=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(root+'/Visuals/MouldedShell')).GetPointsAttr().Get())
        np.testing.assert_allclose(np.ptp(points,axis=0),[.093,.093,.055],atol=1e-7)
        shoulder=points[np.isclose(points[:,2],.0535)]
        np.testing.assert_allclose(np.ptp(shoulder[:,:2],axis=0),[.083,.083],atol=1e-7)
        floor=points[np.isclose(points[:,2],.03)]
        np.testing.assert_allclose(np.ptp(floor[:,:2],axis=0),[.045,.045],atol=1e-7)
        self.assertAlmostEqual(float(points[:,2].max()-.03),.025,places=7)
        # Both pin cross-sections fit the through-hole. A convex hull over the
        # complete concave object would fill this region and fail this check.
        samples=np.array([[x,y,z] for x in [-.0076,-.0061,.0061,.0076]
                          for y in [-.00315,.00315] for z in [.027,.029,.031]])
        floor_support=False
        for p in stage.GetPrimAtPath(root+'/Colliders').GetChildren():
            v=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
            planes=ConvexHull(v).equations
            self.assertFalse(np.any(np.all(samples@planes[:,:3].T+planes[:,3]<=1e-8,axis=1)),str(p.GetPath()))
            floor_support |= bool(np.all(planes[:,:3]@np.array([.016,0.,.0293])+planes[:,3]<=1e-8))
        self.assertTrue(floor_support,'Housing must rest on a physical floor around the hole')
