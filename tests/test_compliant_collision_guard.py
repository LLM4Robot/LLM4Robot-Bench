"""Soft-proxy deformation is bounded; rigid and inter-arm protection stays intact."""
import unittest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
from robot_tasks.robots.collision_feedback import CollisionMonitor


class CompliantCollisionGuardTests(unittest.TestCase):
    def monitor(self, path='/World/Leaf', *, compliant=True):
        stage=Usd.Stage.CreateInMemory()
        leaf=UsdGeom.Capsule.Define(stage,path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(leaf)
        leaf.CreateAttribute('task:compliantContactDeformationLimit',Sdf.ValueTypeNames.Float).Set(.008)
        leaf.CreateAttribute('task:compliantContactForceLimit',Sdf.ValueTypeNames.Float).Set(.05)
        mat=UsdShade.Material.Define(stage,'/SoftContact')
        if compliant:
            mat.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
            mat.GetPrim().CreateAttribute('physxMaterial:compliantContactStiffness',Sdf.ValueTypeNames.Float).Set(2.)
        UsdShade.MaterialBindingAPI.Apply(leaf).Bind(mat,materialPurpose='physics')
        m=CollisionMonitor.__new__(CollisionMonitor)
        m.stage=stage;m.mounts={'right':'/World/Robots/Right','left':'/World/Robots/Left'}
        m.reset()
        return m,path

    def contact(self,m,path,force=.004,depth=.003):
        m.pending=[{'arm':'right','robot_collider':'/World/Robots/Right/link6/Camera',
                    'other_collider':path,'impulse_ns':force*.001,'separation_m':-depth}]
        m.after_step(.001)

    def test_only_marked_physical_compliance_allows_limited_deformation(self):
        m,path=self.monitor();self.contact(m,path,depth=.007)
        self.assertIsNone(m.blocked)
        m,path=self.monitor(compliant=False);self.contact(m,path)
        self.assertIsNotNone(m.blocked)

    def test_force_and_deformation_caps_cannot_use_release_grace(self):
        for force,depth in [(.051,.003),(.004,.009)]:
            m,path=self.monitor();self.contact(m,path,force,depth)
            self.assertIsNotNone(m.blocked)
            m.begin_action(release_arm='right');self.contact(m,path,force,depth)
            self.assertIsNotNone(m.blocked)

    def test_rigid_and_inter_arm_contacts_retain_two_mm_guard(self):
        m,path=self.monitor();self.contact(m,'/World/HardCup')
        self.assertIsNotNone(m.blocked)
        m,path=self.monitor('/World/Robots/Left/link6/Camera')
        self.contact(m,path)
        self.assertIsNotNone(m.blocked)

    def test_invalid_deformation_marker_fails_closed(self):
        m,path=self.monitor()
        m.stage.GetPrimAtPath(path).GetAttribute('task:compliantContactDeformationLimit').Set(.1)
        self.contact(m,path)
        self.assertIsNotNone(m.blocked)
