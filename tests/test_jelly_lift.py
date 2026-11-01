"""L2-06 uses the same held lift/return criterion with its own FEM object."""
import unittest
from copy import deepcopy
from robot_tasks.registry import resolve_task
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.bench.orange_evaluation import OrangeEvaluator
from test_native_l2_scoring import soft_row

class JellyLiftTests(unittest.TestCase):
    def test_release_on_original_box_is_required(self):
        spec=build_spec('L2-06');e=OrangeEvaluator(spec)
        def sample(**kw):return {'jelly':soft_row(spec,**kw)['orange']}
        self.assertEqual(e.update(0,sample()).status,'running')
        self.assertEqual(e.update(.1,sample(lifted=True,touching=True)).status,'running')
        self.assertTrue(e.lifted)
        wrong=sample();wrong['jelly']['centroid'][0]+=.3
        self.assertEqual(e.update(.2,wrong).status,'running')
        self.assertEqual(e.update(.3,sample()).status,'succeeded')

    def test_observer_contact_is_not_a_grasp(self):
        spec=build_spec('L2-06');e=OrangeEvaluator(spec)
        e.update(0,{'jelly':soft_row(spec)['orange']})
        row=soft_row(spec,lifted=True)['orange'];row['finger_distances_m']=[.001,.001,.1,.1]
        self.assertEqual(e.update(.1,{'jelly':row}).status,'running')
        self.assertFalse(e.lifted)

    def test_configuration_preserves_operator_and_shared_rig(self):
        from robot_tasks.video_delivery import effective_cameras,setup_for,verified_scene
        factory,c=resolve_task('L2-06');c=factory.factory.prepare_config(c)
        self.assertEqual(c.arms,('right',));self.assertIn('jelly from the green box',c.instruction)
        self.assertNotIn('orange',c.instruction)
        self.assertEqual(effective_cameras('L2-06'),effective_cameras('L2-04'))
        self.assertEqual(setup_for('L2-06')['arms'],setup_for('L2-04')['arms'])
        self.assertTrue(verified_scene('L2-06').exists())
        from robot_tasks.backends.settings import SimulationSettings
        self.assertEqual(SimulationSettings.from_config(c).renderer, 'PathTracing')
        _, orange_config = resolve_task('L2-04')
        self.assertEqual(SimulationSettings.from_config(orange_config).renderer, 'RaytracedLighting')

    def test_prompt_and_scoring_follow_the_reviewed_english_row(self):
        from robot_tasks.tasks.bench.jelly import jelly_reference
        reference = jelly_reference()
        entry, config = resolve_task('L2-06')
        prepared = entry.factory.prepare_config(config)
        spec = build_spec('L2-06')
        expected = ('Pick up the jelly from the green box with the left arm, '
                    'lift it 5–20 cm, then place it back and release it.')
        self.assertEqual(reference['source_prompt_en'], expected)
        self.assertEqual(config.instruction.strip(), expected)
        self.assertEqual(spec.design.instruction, expected)
        self.assertEqual(config.raw['task']['scoring_protocol'], spec.parameters['scoring_protocol'])
        self.assertNotIn('without damaging', prepared.instruction)
        self.assertNotIn('Declared prop facts', prepared.instruction)
        self.assertFalse(config.raw['completion']['return_home'])
        self.assertEqual(spec.arms, ('right',))
        self.assertEqual(spec.observers, ('left',))

    def test_held_lift_has_the_source_five_to_twenty_centimetre_bounds(self):
        spec = build_spec('L2-06')
        for rise, qualifies in ((.049, False), (.05, True), (.20, True), (.201, False)):
            with self.subTest(rise=rise):
                e = OrangeEvaluator(spec)
                initial = {'jelly': soft_row(spec)['orange']}
                e.update(0., initial)
                lifted = deepcopy(initial)
                row = lifted['jelly']
                for key in ('bounds_min', 'bounds_max', 'centroid',
                            'collision_bounds_min', 'collision_bounds_max'):
                    row[key][2] += rise
                row['finger_distances_m'][2:4] = [.001, .001]
                e.update(.1, lifted)
                self.assertEqual(e.lifted, qualifies)
                self.assertEqual(e.update(.2, initial).status,
                                 'succeeded' if qualifies else 'running')

    def test_released_above_the_real_box_is_not_returned(self):
        spec = build_spec('L2-06')
        e = OrangeEvaluator(spec)
        initial = {'jelly': soft_row(spec)['orange']}
        e.update(0., initial)
        e.update(.1, {'jelly': soft_row(spec, lifted=True, touching=True)['orange']})
        hovering = deepcopy(initial)
        for key in ('bounds_min', 'bounds_max', 'centroid',
                    'collision_bounds_min', 'collision_bounds_max'):
            hovering['jelly'][key][2] += .025
        self.assertEqual(e.update(.2, hovering).status, 'running')
        self.assertEqual(e.update(.3, initial).status, 'succeeded')

    def test_holding_on_the_box_is_not_release_and_damage_is_not_an_extra_gate(self):
        spec = build_spec('L2-06')
        e = OrangeEvaluator(spec)
        e.update(0., {'jelly': soft_row(spec)['orange']})
        e.update(.1, {'jelly': soft_row(spec, lifted=True, touching=True)['orange']})
        self.assertEqual(e.update(.2, {'jelly': soft_row(spec, touching=True)['orange']}).status, 'running')
        released = {'jelly': soft_row(spec, strain=.7)['orange']}
        released['jelly']['minimum_volume_ratio'] = -.01
        result = e.update(.3, released)
        self.assertEqual(result.status, 'succeeded')
        self.assertTrue(result.metrics['quality_warnings'])
