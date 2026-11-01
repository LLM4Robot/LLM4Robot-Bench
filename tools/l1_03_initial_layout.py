"""Photo-guided rigid-part placement; all edits precede episode physics."""
from pathlib import Path
import numpy as np
from pxr import Gf, Usd, UsdGeom

REVISION = 'three_visible_legs_side_on_hooks_20261016_v3'


def yaw_matrix(degrees):
    yaw = np.deg2rad(degrees)
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0.],
                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]]
    return transform


def set_world(stage, path, world):
    prim = stage.GetPrimAtPath(path)
    parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(
        Gf.Matrix4d((np.linalg.inv(parent) @ world).T.tolist()))


def apply(stage):
    # Native visor brim points toward -Y (the front of the table). Centre its
    # complete 240 mm silhouette on the 195 mm box, with symmetric overhang.
    box = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath('/World/Household/source_box'))).T
    hat = np.eye(4)
    # The box's support surface is recessed below its rim. Use the measured
    # passive resting pitch, with 1 mm settling clearance, not the rim height.
    pitch = np.deg2rad(-5.1)
    hat[:3, :3] = [[1., 0., 0.], [0., np.cos(pitch), -np.sin(pitch)],
                   [0., np.sin(pitch), np.cos(pitch)]]
    hat[:3, 3] = box[:3, 3] + [0., .03735, .0618]
    set_world(stage, '/World/Household/hat', hat)
    # Match the reference silhouette: three separated legs, with the right toe
    # toward the box. The upper hooks are side-on to the overview camera so
    # their trunk connections remain visible, rather than foreshortened.
    rack = yaw_matrix(-16.)
    rack[:3, 3] = [.065, 1.2458250975206666, .8800999998137354]
    set_world(stage, '/World/Household/Fixture', rack)
    source = Usd.Stage.Open(str(Path(__file__).resolve().parents[1] /
        'assets/tuojing/source/L1-07/scene.usdz'))
    upper_rotation = yaw_matrix(30. - (-16.))
    for group, prefix in [('Visuals', 'Hook'), ('Colliders', 'Hook'), ('Frames', 'Hang')]:
        for index in range(4):
            path = f'/World/Household/Fixture/{group}/{prefix}{index}'
            native = np.asarray(UsdGeom.Xformable(source.GetPrimAtPath(path)).GetLocalTransformation()).T
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(
                Gf.Matrix4d((upper_rotation @ native).T.tolist()))
    return {'revision': REVISION, 'reference': 'user real initial frame 2026-10-16',
            'hat_yaw_deg': 0., 'hat_roll_deg': -5.1, 'hat_pitch_deg': 0.,
            'hat_centred_on_box': True,
            'rack_translation_delta_m': [.03751232849178493, 0., 0.],
            'rack_yaw_deg': -16., 'upper_hook_assembly_yaw_deg': 30.,
            'upper_hook_rotation_relative_to_base_deg': 46.,
            'visual_collision_and_hang_frames_rotated_together': True,
            'rigid_part_shapes_mass_and_contact_materials_unchanged': True,
            'limitation': 'Photo-guided placement estimate; not calibrated real-world coordinates.'}
