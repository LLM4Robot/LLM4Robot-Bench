"""Check calibrated render geometry, collision geometry and passive joint frames."""
import unittest
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.video_delivery import scenes
from robot_tasks.initial_layout import scene_record
from robot_tasks.registry import resolve_task
from robot_tasks.tasks.bench.measured_assets import (
    JELLY_MASS_KG, MARKER_LENGTH_M, calibrate_jelly, calibrate_marker,
    marker_axial,
)
from robot_tasks.tasks.bench.marker_contacts import refresh_delivered_groups


def stage(task):
    result = Usd.Stage.Open(str(PROJECT_ROOT / 'assets/tuojing' / scenes()[task]['runtime_path']))
    result.SetEditTarget(result.GetSessionLayer())
    return result


class MeasuredAssetTests(unittest.TestCase):
    def test_marker_visual_length_and_diameter_match_physical_calibration(self):
        s = stage('L2-01'); root = '/World/L2/marker'
        refresh_delivered_groups(s, root)
        cache = UsdGeom.XformCache()
        inv = cache.GetLocalToWorldTransform(s.GetPrimAtPath(root)).GetInverse()
        def vertices():
            result = []
            for p in Usd.PrimRange(s.GetPrimAtPath(root)):
                if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility() != 'invisible':
                    transform = np.asarray(cache.GetLocalToWorldTransform(p) * inv).T
                    pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
                    result.append(pts @ transform[:3,:3].T + transform[:3,3])
            return np.concatenate(result)
        before = vertices(); source = s.GetRootLayer().ExportToString()
        report = calibrate_marker(s); cache.Clear(); after = vertices()
        self.assertAlmostEqual(np.ptp(after[:,2]), MARKER_LENGTH_M, places=7)
        np.testing.assert_allclose(after[:,:2], before[:,:2], atol=1e-8)
        self.assertAlmostEqual((after[:,2].min()+after[:,2].max())/2,
                               (before[:,2].min()+before[:,2].max())/2, places=7)
        for name in ['BoxContactEnvelope', 'RobotContactEnvelope']:
            p=s.GetPrimAtPath(root+'/Cap/'+name)
            xf=np.asarray(UsdGeom.Xformable(p).GetLocalTransformation()).T
            self.assertAlmostEqual(xf[2,2], report['axial_scale'])
        nib = UsdGeom.Capsule(s.GetPrimAtPath(root+'/Body/Colliders/Nib'))
        xf=np.asarray(cache.GetLocalToWorldTransform(nib.GetPrim()) * inv).T
        np.testing.assert_allclose(np.linalg.norm(xf[:3,:3],axis=0),1.,atol=1e-7)
        self.assertAlmostEqual(nib.GetRadiusAttr().Get(),.0024)
        self.assertAlmostEqual(nib.GetHeightAttr().Get()+2*.0024,
                               (.004+2*.0024)*report['axial_scale'])
        for i in range(4):
            j=UsdPhysics.Joint(s.GetPrimAtPath(root+f'/RadialSpring{i}'))
            pts=[]
            for rel, attr in ((j.GetBody0Rel(),j.GetLocalPos0Attr()),(j.GetBody1Rel(),j.GetLocalPos1Attr())):
                body=s.GetPrimAtPath(rel.GetTargets()[0]);m=np.asarray(cache.GetLocalToWorldTransform(body)).T
                pts.append((m@np.r_[np.asarray(attr.Get()),1.])[:3])
            np.testing.assert_allclose(pts[0],pts[1],atol=2e-7)
        session=s.GetSessionLayer().ExportToString();calibrate_marker(s)
        self.assertEqual(session,s.GetSessionLayer().ExportToString())
        self.assertEqual(source,s.GetRootLayer().ExportToString())

    def test_both_marker_placements_use_calibrated_bounds_and_cap_landmarks(self):
        from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
        for tid in ('L2-01','L2-01_variant01'):
            _,c=resolve_task(tid);row=scene_record(c.raw['task'])
            spec=resolve_bench_spec(c.raw['task'],c.robot)
            body=row['tracked']['body']['bounds'];cap=row['tracked']['cap']['bounds']
            self.assertAlmostEqual(cap[1][2]-body[0][2], MARKER_LENGTH_M, places=7)
            self.assertAlmostEqual(spec.parameters['cap_mouth_z'],marker_axial(.10625))
            self.assertEqual(spec.design.articles[0].asset_bounds, tuple(map(tuple,body)))

    def test_jelly_explicit_fem_mass_and_both_task_specs_match_net_weighing(self):
        from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
        s=stage('L2-06');p=s.GetPrimAtPath('/World/L2/jelly/Jelly')
        points=list(UsdGeom.Mesh(p).GetPointsAttr().Get());source=s.GetRootLayer().ExportToString()
        calibrate_jelly(s)
        self.assertAlmostEqual(UsdPhysics.MassAPI(p).GetMassAttr().Get(),JELLY_MASS_KG,places=8)
        self.assertEqual(points,list(UsdGeom.Mesh(p).GetPointsAttr().Get()))
        self.assertEqual(source,s.GetRootLayer().ExportToString())
        for tid in ('L2-06','L2-06_variant01'):
            _,c=resolve_task(tid);spec=resolve_bench_spec(c.raw['task'],c.robot)
            self.assertEqual(spec.design.articles[0].mass,JELLY_MASS_KG)
            self.assertEqual(scene_record(c.raw['task'])['jelly']['mass'],JELLY_MASS_KG)


if __name__=='__main__': unittest.main()
