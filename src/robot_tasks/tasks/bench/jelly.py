"""L2-06: supplied gelatin FEM, with the reviewed Feishu English goal."""
from dataclasses import replace
from functools import lru_cache
import json

import numpy as np

from ...types import InvalidSimulationState
from ...paths import PROJECT_ROOT
from ...video_delivery import scenes
from .native_l2_specs import orange
from .orange_scene import OrangeScene
from .orange_evaluation import FINGER_ARM_ORDER
from .measured_assets import JELLY_MASS_KG, calibrate_jelly


@lru_cache(maxsize=1)
def jelly_reference():
    return json.loads((PROJECT_ROOT / 'configs/task_references/feishu_jelly_20261016.json').read_text(encoding='utf-8'))


def jelly():
    base = orange()
    reference = jelly_reference()
    goal = reference['success_conditions']
    row = scenes()['L2-06']
    prop, support, pedestal = row['jelly'], row['l2_support_measurement'], row['fixtures'][0]
    article = replace(base.design.articles[0], name='jelly', initial=tuple(prop['pose']),
                      goal=tuple(prop['pose']), asset='delivery:jelly',
                      asset_bounds=tuple(map(tuple, prop['bounds'])), mass=JELLY_MASS_KG)
    instruction = reference['simulation_prompt_en']
    return replace(base, design=replace(base.design, task_id='L2-06',
        name='果冻抬升并放回', articles=(article,), fixtures=(),
        instruction=instruction, version='l2-08-feishu-lift-return-v2'),
        mechanism='native_fem_jelly_lift', facts='',
        limitations='Supplied gelatin mesh and PhysX FEM material: E=3 kPa, Poisson ratio 0.45, friction 0.6. Total FEM mass is calibrated to the user-confirmed 26.4 g net weighing, overriding density-derived mass. Grasp, lift and placement are geometric proxies. Deformation is diagnostic; the source English goal does not require a damage measurement. No grasp attachment or forced pose changes.',
        parameters={**base.parameters, 'soft_object_name':'jelly',
            'soft_asset':'gelatin_cup_jelly', 'scoring_protocol':reference['protocol'],
            'lift_m':goal['minimum_lift_m'], 'max_lift_m':goal['maximum_lift_m'],
            'table_z':.88, 'pedestal_pose':tuple(pedestal['pose']),
            'support_z':pedestal['pose'][2]+support['inset_platform_z_m'],
            'support_rim_z':pedestal['pose'][2]+support['outer_dimensions_m'][2],
            'support_half_extents_m':tuple(support['inset_half_extents_m'])})


class JellyScene(OrangeScene):
    """Read legacy PhysX auto-cooked FEM meshes without changing the supplied asset."""
    object_name = 'jelly'

    def __init__(self, backend, config, spec):
        from pxr import Usd, UsdGeom, UsdPhysics
        from .surface_distance import link_triangles
        if not backend.settings.gpu_dynamics:
            raise ValueError('L2-06 jelly requires GPU dynamics')
        self.backend, self.config, self.spec, self.design = backend, config, spec, spec.design
        self.stage = backend.world.stage
        self.path = '/World/L2/jelly'
        self.visual = self.collision = self.stage.GetPrimAtPath(self.path+'/Jelly')
        if not self.visual:
            raise ValueError('Supplied red jelly mesh is missing')
        measurement = calibrate_jelly(self.stage)
        self.initial = {'jelly': self.design.articles[0].initial}
        self.placement = self.design.articles[0].initial
        self.previous = self.cached = None
        self.rest_inverse = None
        pedestal = self.stage.GetPrimAtPath('/World/L2/Pedestal')
        self.colliders = [str(p.GetPath()) for p in Usd.PrimRange(pedestal)
                          if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not self.colliders:
            raise ValueError('Jelly support has no collision geometry')
        self.finger_paths, self.finger_surfaces = [], []
        bounds = UsdGeom.BBoxCache(0., ['default','render'])
        for arm in FINGER_ARM_ORDER:
            mount = next(m for m in config.robot.mounts if m.name == arm)
            for link in ('link7','link8'):
                path = mount.prim_path+'/'+link
                self.finger_paths.append(path)
                self.finger_surfaces.append(link_triangles(self.stage,path))
        self.observer_bounds = {}
        for arm in config.observation_arms:
            mount = next(m for m in config.robot.mounts if m.name == arm)
            parts=[]
            for index in range(1,9):
                path=mount.prim_path+f'/link{index}'
                b=bounds.ComputeUntransformedBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
                parts.append((path,np.asarray(b.GetMin()),np.asarray(b.GetMax())))
            self.observer_bounds[arm]=parts
        backend.planning_fixture_paths=tuple(self.colliders)
        backend.planning_payloads=self.planning_payloads
        backend.scene_report['household']={
            'version':self.design.version,'mechanism':spec.mechanism,
            'limitations':spec.limitations,'parameters':spec.parameters,
            'initial_poses':self.initial,'object_attachment':False,
            'assets':[scenes()['L2-06']['derived_contents']],
            'native_physical_material_preserved':True,
            'measurement_calibration': measurement,
            'render_material_adapter':scenes()['L2-06']['jelly']['render_adapter'],
            'soft_contact_measurement':'FEM collision vertices to finger triangle surfaces; proximity, not force',
            'deformation':'95th-percentile principal element strain; diagnostic only'}

    def fem_points(self):
        def read(name, dtype=float):
            value=self.visual.GetAttribute('physxDeformable:'+name).Get()
            if value is None or len(value)==0:
                raise InvalidSimulationState('Missing cooked jelly FEM '+name)
            return np.asarray(value,dtype=dtype)
        if self.rest_inverse is None:
            self.rest=read('simulationRestPoints')
            self.tets=read('simulationIndices',int).reshape(-1,4)
            r=self.rest[self.tets]
            self.rest_inverse=np.linalg.inv(np.stack([r[:,i]-r[:,0] for i in (1,2,3)],axis=2))
        return read('simulationPoints'),read('collisionPoints')
