"""Verify the image-target camera, shared scenery and native per-task framing."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pxr import Usd, UsdGeom
from robot_tasks.paths import PROJECT_ROOT as ROOT
from robot_tasks.cameras.geometry import project_world, resized_intrinsics
from robot_tasks.video_delivery import apply_visual_background, effective_cameras, scenes, setup_compatible, setup_digest, visual_setup
from tools.diagnostics.audit_reference_image_camera import compare_native, locked_reference


def edge_targets(camera, size):
    k = resized_intrinsics(camera['intrinsics'], camera['resolution'], size)
    targets = {}
    for side, x in [('left', -.2826577758), ('right', .7296936854)]:
        uv = project_world([[x, 1.927, .88], [x, 1.927, 2.6]], camera['pose'], k)
        targets[side] = [[float(uv[0, 0]+(v-uv[0, 1])*(uv[1, 0]-uv[0, 0])/(uv[1, 1]-uv[0, 1])), v] for v in (10, 30, 50)]
    uv = project_world([[-.7514504138, 1.6376504067, .8801500201], [.8385496308, 1.6376504067, .8801500201]], camera['pose'], k)
    targets['table'] = [[u, float(uv[0, 1]+(u-uv[0, 0])*(uv[1, 1]-uv[0, 1])/(uv[1, 0]-uv[0, 0]))] for u in (25, 60, size[0]-24)]
    return targets


def measure_edges(rgb, targets):
    pixels = rgb.astype(float); result = {}
    for side in ('left', 'right'):
        measured = []
        for u, v in targets[side]:
            # A single row can pick wall texture instead of the material seam.
            # Pool 21 rows to keep the edge estimate stable under screenshot
            # resampling and the existing task-specific illumination settings.
            u, v = round(u), round(v); line = np.median(pixels[max(0, v-10):v+11], axis=0)
            if side == 'left':
                gradient = np.linalg.norm(np.gradient(gaussian_filter1d(line, .7, axis=0), axis=0), axis=1)
            else:
                gradient = np.gradient(gaussian_filter1d(line[:, 2]-line[:, 0], .7))
            lo, hi = max(0, u-8), min(pixels.shape[1], u+9)
            measured.append([lo+int(np.argmax(gradient[lo:hi])), v])
        result[side] = measured
    result['table'] = []
    for u, v in targets['table']:
        u, v = round(u), round(v); line = np.median(pixels[:, u-3:u+4], axis=1)
        gradient = np.linalg.norm(np.gradient(gaussian_filter1d(line, .7, axis=0), axis=0), axis=1)
        lo, hi = max(0, v-6), min(pixels.shape[0], v+7)
        result['table'].append([u, lo+int(np.argmax(gradient[lo:hi]))])
    points = np.asarray(result['table']); slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
    result['table_footprint_fraction_estimate'] = float(1-((pixels.shape[1]-1)*.5*slope+intercept)/pixels.shape[0])
    return result


def main():
    lock, fit, reference_path = locked_reference()
    reference_task = visual_setup()['camera_reference_task']; reference = effective_cameras(reference_task)
    for name in ('pose', 'intrinsics'): assert reference['overview'][name] == fit[name]
    assert reference['overview']['resolution'] == fit['native_resolution']
    size = tuple(lock['target_size']); targets = edge_targets(reference['overview'], size)
    reference_edges = measure_edges(cv2.cvtColor(cv2.imread(str(reference_path)), cv2.COLOR_BGR2RGB), targets)
    canonical = None; rows = []
    for tid, scene in scenes().items():
        folder = ROOT/'build/video_visual_fit_20261015/aligned_initial'/tid
        result = json.loads((folder/'result.json').read_text())
        assert result['passed'] and setup_compatible(tid, result['visual_setup_sha256']), tid
        assert result['runtime_scene_sha256'] == scene['runtime_sha256'], tid
        cameras = effective_cameras(tid)
        for name in reference:
            for key in ('pose', 'intrinsics', 'parent', 'resolution', 'clipping_m'):
                assert cameras[name][key] == reference[name][key], (tid, name, key)
        calibration = result['camera_capture']['overview']['calibration']
        np.testing.assert_allclose(calibration['T_world_camera_pose_wxyz'], reference['overview']['pose'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(calibration['intrinsics_fx_fy_cx_cy'], reference['overview']['intrinsics'], atol=1e-3, rtol=0)
        assert calibration['resolution'] == fit['native_resolution']
        stage = Usd.Stage.Open(str(ROOT/'assets/tuojing'/scene['runtime_path'])); stage.SetEditTarget(stage.GetSessionLayer())
        apply_visual_background(stage, tid)
        assert UsdGeom.Imageable(stage.GetPrimAtPath('/World/Furniture/ThirdCameraStand')).ComputeVisibility() == 'invisible'
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']); geometry = {}
        for path in ('/World/Furniture/Worktable/Top', '/World/Room/CorkPanel', '/World/Room/PanelTrim2', '/World/Room/PanelTrim3', '/World/Room/PowerStrip'):
            box = bounds.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            geometry[path] = [list(box.GetMin()), list(box.GetMax())]
        if canonical is None: canonical = geometry
        for path in geometry: np.testing.assert_allclose(geometry[path], canonical[path], atol=1e-8, rtol=0, err_msg=tid+' '+path)
        image_path = folder/'overview.png'; native = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        assert list(native.shape[1::-1]) == fit['native_resolution']
        edges = measure_edges(cv2.resize(native, size, interpolation=cv2.INTER_AREA), targets)
        errors = {side: (np.asarray(edges[side])-reference_edges[side]).tolist() for side in ('left', 'right', 'table')}
        max_errors = {side: float(np.abs(errors[side]).max()) for side in errors}
        rows.append({'task': tid, 'recorded_setup_sha256': result['visual_setup_sha256'],
                     'image': str(image_path.relative_to(ROOT)), 'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                     'shared_camera_geometry_verified': True, 'shared_table_and_background_geometry_verified': True,
                     'old_external_camera_stand_hidden': True, 'rendered_edges': edges,
                     'edge_errors_px_against_user_image': errors, 'edge_max_errors_px': max_errors,
                     'framing_edge_check_passed': all(v <= 2 for v in max_errors.values()), 'strict_image_identity_claimed': False})
    native_match = compare_native(ROOT/'build/video_visual_fit_20261015/aligned_initial/L3-01/overview.png', ROOT/'reports/reference_image_camera_20261015.jpg')
    (ROOT/'reports/reference_image_camera_20261015.json').write_text(json.dumps(native_match, indent=2)+'\n')
    assert native_match['framing_passed'], native_match
    report = {'visual_setup_sha256': setup_digest(), 'reference_task': reference_task,
              'reference_image': lock['image'], 'reference_image_sha256': lock['image_sha256'],
              'tasks': len(rows), 'shared_geometry_verified': len(rows),
              'rendered_framing_edges_passed': sum(r['framing_edge_check_passed'] for r in rows),
              'reference_edges': reference_edges, 'reference_image_native_match': native_match,
              'scope': 'Native camera matched to the exact user-supplied L3-01 image, then shared across all tasks. Edge errors use the 594x442 reference dimensions. Table footprint excludes object occlusion. Common camera/table/background geometry is verified; task props and arm poses remain task-specific. No runtime image warp or byte-identical-pixel claim.',
              'table_geometry_modified_for_fit': False, 'rows': rows}
    (ROOT/'reports/shared_camera_framing_20261015.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'reference_edges', 'reference_image_native_match')}, indent=2))


if __name__ == '__main__':
    main()
