"""The runtime repair keeps the cavity, appearance, passive fit and isolation."""
import unittest
import numpy as np
try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.video_delivery import scenes
from robot_tasks.tasks.bench.marker_contacts import refresh_delivered_groups


@unittest.skipIf(Usd is None, 'USD is required for marker geometry contracts')
class MarkerFitTests(unittest.TestCase):
    def setUp(self):
        self.stage=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/scenes()['L2-01']['runtime_path']))
        # Never author changes into source assets during validation.
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.marker='/World/L2/marker'

    def enabled(self, root):
        return [p for p in Usd.PrimRange(self.stage.GetPrimAtPath(root))
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]

    def test_preserves_source_geometry_and_repeatable_cavity(self):
        visual_paths=[str(p.GetPath()) for p in self.stage.Traverse()
                      if '/Visuals/' in str(p.GetPath()) or '/LatestVisuals/' in str(p.GetPath())]
        before={path:self.stage.GetPrimAtPath(path).GetPrimStack()[0].GetAsText() for path in visual_paths}
        neck={str(p.GetPath()):np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
              for p in self.enabled(self.marker+'/Body') if p.IsA(UsdGeom.Mesh)}
        for _ in range(2):
            report=refresh_delivered_groups(self.stage,self.marker)
            self.assertEqual(report['Support'],5)
            self.assertEqual(len(self.enabled(self.marker+'/Cap/Colliders')),17)
            self.assertEqual(len(self.enabled(self.marker+'/Body')),10)
            end=UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.marker+'/Cap/Colliders/SimpleClosedEnd'))
            m=np.asarray(UsdGeom.Xformable(end).GetLocalTransformation()).T
            self.assertGreater(m[2,3]-end.GetHeightAttr().Get()/2,.145)
            # The 106--128 mm insertion region stays hollow, above the nib.
            self.assertEqual(sum(p.GetName().startswith('Wall') for p in self.enabled(self.marker+'/Cap/Colliders')),16)
            self.assertAlmostEqual(report['support_repair']['box_floor_z_m'],.06,places=6)
            self.assertAlmostEqual(report['support_repair']['box_rim_z_m'],.07,places=6)
        for path,points in neck.items():np.testing.assert_array_equal(UsdGeom.Mesh(self.stage.GetPrimAtPath(path)).GetPointsAttr().Get(),points)
        self.assertEqual(before,{path:self.stage.GetPrimAtPath(path).GetPrimStack()[0].GetAsText() for path in visual_paths})
        self.assertFalse(self.stage.GetRootLayer().dirty)

    def test_internal_shoes_cannot_be_squeezed_by_robot(self):
        # Real robot shapes can be added after source asset construction.
        fake=UsdGeom.Cube.Define(self.stage,'/World/Robot/NewFinger').GetPrim()
        UsdPhysics.CollisionAPI.Apply(fake).CreateCollisionEnabledAttr(True)
        refresh_delivered_groups(self.stage,self.marker)
        barrel=[p.GetPath() for p in self.enabled(self.marker+'/Body')]
        for i in range(4):
            shoe=self.stage.GetPrimAtPath(self.marker+'/Shoe'+str(i))
            excluded=UsdPhysics.FilteredPairsAPI(shoe).GetFilteredPairsRel().GetTargets()
            self.assertIn(fake.GetPath(),excluded)
            self.assertFalse(set(barrel)&set(excluded))
            joint=self.stage.GetPrimAtPath(self.marker+'/RadialSpring'+str(i))
            drive=UsdPhysics.DriveAPI(joint,'linear')
            self.assertAlmostEqual(drive.GetStiffnessAttr().Get(),2000.)
            self.assertAlmostEqual(drive.GetDampingAttr().Get(),.89442719,places=6)
            self.assertAlmostEqual(UsdPhysics.PrismaticJoint(joint).GetUpperLimitAttr().Get(),.001)
        proxy=self.stage.GetPrimAtPath(self.marker+'/Cap/BoxContactEnvelope')
        excluded=UsdPhysics.FilteredPairsAPI(proxy).GetFilteredPairsRel().GetTargets()
        self.assertTrue(set(barrel)<=set(excluded))
        self.assertIn(fake.GetPath(),excluded)

    def test_robot_contact_proxy_does_not_fill_internal_fit(self):
        finger=UsdGeom.Cube.Define(self.stage,'/World/Robots/R5_Left/TestFinger').GetPrim()
        UsdPhysics.CollisionAPI.Apply(finger).CreateCollisionEnabledAttr(True)
        refresh_delivered_groups(self.stage,self.marker)
        # Re-running scene binding must preserve the separation of contact roles.
        report=refresh_delivered_groups(self.stage,self.marker)
        self.assertEqual(report['RobotEnvelope'],1)
        proxy=self.stage.GetPrimAtPath(self.marker+'/Cap/RobotContactEnvelope')
        excluded=UsdPhysics.FilteredPairsAPI(proxy).GetFilteredPairsRel().GetTargets()
        self.assertNotIn(finger.GetPath(),excluded)
        self.assertTrue({p.GetPath() for p in self.enabled(self.marker+'/Body')}<=set(excluded))
        for wall in self.enabled(self.marker+'/Cap/Colliders'):
            pairs=UsdPhysics.FilteredPairsAPI(wall).GetFilteredPairsRel().GetTargets()
            self.assertIn(finger.GetPath(),pairs)
        groups=self.marker+'/BoxContactGroups/'
        targets=UsdPhysics.CollisionGroup(self.stage.GetPrimAtPath(groups+'RobotEnvelope')).GetFilteredGroupsRel().GetTargets()
        self.assertIn(groups+'OtherColliders',list(map(str,targets)))
        self.assertNotIn(groups+'RobotColliders',list(map(str,targets)))
        # Exterior dimensions agree with the preserved cap, not a larger grasp aid.
        outer=UsdGeom.Cylinder(proxy)
        box_proxy=UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.marker+'/Cap/BoxContactEnvelope'))
        self.assertEqual(outer.GetRadiusAttr().Get(),box_proxy.GetRadiusAttr().Get())
        self.assertEqual(outer.GetHeightAttr().Get(),box_proxy.GetHeightAttr().Get())


if __name__=='__main__':unittest.main()
