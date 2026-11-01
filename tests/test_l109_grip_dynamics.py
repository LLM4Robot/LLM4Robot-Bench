"""A hollow visible flower must not inherit the mass of its solid grasp hull."""
import unittest
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.video_delivery import scenes

class FlowerGripDynamicsTests(unittest.TestCase):
    def test_mass_distribution_and_geometry_preserve_the_original_free_flower(self):
        row=scenes()['L1-05']
        stage=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/row['runtime_path']))
        native=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/row['path']))
        root='/World/Household/rose';body=stage.GetPrimAtPath(root);mass=UsdPhysics.MassAPI(body)
        leaves=row.get('physical_repairs',{}).get('flower_passive_leaves',{}).get('leaf_bodies',[])
        total=mass.GetMassAttr().Get()+sum(UsdPhysics.MassAPI(stage.GetPrimAtPath(p['path'])).GetMassAttr().Get()
                                        for p in leaves)
        self.assertAlmostEqual(total,.012,places=6)
        self.assertFalse(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get())
        self.assertFalse(body.GetAttribute('physxRigidBody:disableGravity').Get())
        com=np.array(mass.GetCenterOfMassAttr().Get());inertia=np.array(mass.GetDiagonalInertiaAttr().Get())
        self.assertTrue(np.isfinite(com).all())
        self.assertGreater(com[2],.15)
        self.assertLess(com[2],.25) # A hollow head must not dominate the COM.
        self.assertTrue(np.all(inertia>0))
        self.assertLess(inertia.max(),inertia.sum()-inertia.max())
        stem=UsdGeom.Capsule(stage.GetPrimAtPath(root+'/Colliders/CoatedStem'))
        self.assertAlmostEqual(stem.GetRadiusAttr().Get(),.0019,places=7)
        for prim in stage.GetPrimAtPath(root+'/Visuals').GetChildren():
            if prim.IsA(UsdGeom.Mesh):
                np.testing.assert_array_equal(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                    UsdGeom.Mesh(native.GetPrimAtPath(prim.GetPath())).GetPointsAttr().Get())

    def test_passive_leaves_preserve_visuals_mass_and_contactable_rounded_veins(self):
        row=scenes()['L1-05'];root='/World/Household/rose'
        stage=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/row['runtime_path']))
        native=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/row['path']))
        parts=row['physical_repairs']['flower_passive_leaves']['leaf_bodies']
        self.assertEqual(len(parts),2)
        for part in parts:
            body=stage.GetPrimAtPath(part['path']);name=body.GetName()
            self.assertTrue(UsdGeom.Xformable(body).GetResetXformStack())
            self.assertFalse(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get())
            self.assertGreater(UsdPhysics.MassAPI(body).GetMassAttr().Get(),0.)
            np.testing.assert_array_equal(
                UsdGeom.Mesh(stage.GetPrimAtPath(part['path']+'/Visuals/'+name)).GetPointsAttr().Get(),
                UsdGeom.Mesh(native.GetPrimAtPath(root+'/Visuals/'+name)).GetPointsAttr().Get())
            colliders=[p for p in stage.GetPrimAtPath(part['path']+'/Colliders').GetChildren()
                       if p.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertEqual(len(colliders),6)
            self.assertTrue(all(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders))
            self.assertTrue(all(p.IsA(UsdGeom.Capsule) for p in colliders))
            self.assertTrue(all(UsdGeom.Capsule(p).GetRadiusAttr().Get()<=.001 for p in colliders))
            material=UsdShade.MaterialBindingAPI(colliders[0]).ComputeBoundMaterial('physics')[0].GetPrim()
            self.assertGreater(material.GetAttribute('physxMaterial:compliantContactStiffness').Get(),0.)
            self.assertGreater(material.GetAttribute('physxMaterial:compliantContactDamping').Get(),0.)
            joint=UsdPhysics.Joint(stage.GetPrimAtPath(part['joint']))
            self.assertEqual(list(map(str,joint.GetBody0Rel().GetTargets())),[root])
            self.assertEqual(list(map(str,joint.GetBody1Rel().GetTargets())),[part['path']])
            for axis in ['transX','transY','transZ']:
                limit=UsdPhysics.LimitAPI(joint.GetPrim(),axis)
                self.assertGreater(limit.GetLowAttr().Get(),limit.GetHighAttr().Get())
            for axis in ['rotX','rotY','rotZ']:
                drive=UsdPhysics.DriveAPI(joint.GetPrim(),axis)
                self.assertEqual(drive.GetTargetPositionAttr().Get(),0.)
                self.assertEqual(drive.GetTargetVelocityAttr().Get(),0.)
                self.assertGreater(drive.GetDampingAttr().Get(),0.)
                self.assertLess(drive.GetMaxForceAttr().Get(),.01)

    def test_grip_glass_and_cup_contacts_keep_distinct_material_behavior(self):
        row=scenes()['L1-05']
        stage=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/row['runtime_path']))
        root='/World/Household/'
        def material(path):
            return UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')[0].GetPrim()
        grip=material(root+'rose/Colliders/CoatedStem')
        wall=material(root+'target_cup/Colliders/Wall_00')
        floor=material(root+'target_cup/Colliders/Floor')
        self.assertEqual(grip.GetAttribute('physxMaterial:frictionCombineMode').Get(),'min')
        self.assertLess(UsdPhysics.MaterialAPI(wall).GetDynamicFrictionAttr().Get(),
                        UsdPhysics.MaterialAPI(grip).GetDynamicFrictionAttr().Get())
        self.assertNotEqual(wall.GetPath(),floor.GetPath())
        for i in range(10):
            bead=root+'filler_'+str(i)
            mass=UsdPhysics.MassAPI(stage.GetPrimAtPath(bead)).GetMassAttr().Get()
            radius=UsdGeom.Sphere(stage.GetPrimAtPath(bead+'/Colliders/Sphere')).GetRadiusAttr().Get()
            density=mass/((4./3.)*np.pi*radius**3)
            self.assertGreater(density,2000.)
            self.assertLess(density,3000.)
        body=stage.GetPrimAtPath(root+'rose')
        self.assertFalse(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get())
        self.assertFalse(body.GetAttribute('physxRigidBody:disableGravity').Get())

if __name__=='__main__':unittest.main()
