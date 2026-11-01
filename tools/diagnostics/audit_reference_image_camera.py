"""Compare native rendering directly with the user's locked reference image."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2
import numpy as np
from robot_tasks.paths import PROJECT_ROOT as ROOT
from robot_tasks.video_delivery import visual_setup


def locked_reference():
    lock = visual_setup()['overview_reference_image_fit']
    raw = (ROOT/lock['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lock['sha256']
    image = ROOT/lock['image']
    assert hashlib.sha256(image.read_bytes()).hexdigest() == lock['image_sha256']
    return lock, json.loads(raw), image


def compare_native(image_path, comparison_path, *, native_raster=True):
    lock, fit, reference = locked_reference()
    target = cv2.imread(str(reference))
    native = cv2.imread(str(image_path))
    if native_raster:
        assert list(native.shape[1::-1]) == fit['native_resolution']
    rendered = cv2.resize(native, tuple(lock['target_size']), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=8000)
    kt, dt = sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    kn, dn = sift.detectAndCompute(cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY), None)
    matches = [p for p, q in cv2.BFMatcher().knnMatch(dn, dt, k=2) if p.distance < .7*q.distance]
    source = np.float32([kn[m.queryIdx].pt for m in matches])
    desired = np.float32([kt[m.trainIdx].pt for m in matches])
    # RANSAC rejects descriptor mismatches; its transform is never applied.
    _, mask = cv2.findHomography(source, desired, cv2.RANSAC, 1.5)
    inliers = mask.ravel().astype(bool)
    errors = np.linalg.norm(source-desired, axis=1)
    regions = {}
    boxes = {'background_panel': [180, 0, 540, 85], 'wall_socket': [240, 105, 380, 150],
             'left_robot': [170, 85, 280, 215], 'right_robot': [375, 90, 500, 220],
             'bottle': [285, 215, 335, 330], 'foreground_table': [0, 330, 594, 442]}
    for name, (x0, y0, x1, y1) in boxes.items():
        selected = (desired[:, 0] >= x0) & (desired[:, 0] <= x1) & (desired[:, 1] >= y0) & (desired[:, 1] <= y1)
        good = selected & inliers
        regions[name] = {'descriptor_matches': int(selected.sum()), 'inliers': int(good.sum()),
                         'direct_median_px': float(np.median(errors[good])) if good.any() else None,
                         'direct_max_px': float(errors[good].max()) if good.any() else None,
                         'dense_RGB_MAE': float(np.abs(target[y0:y1, x0:x1].astype(float)-rendered[y0:y1, x0:x1]).mean())}
    median, p95 = float(np.median(errors[inliers])), float(np.quantile(errors[inliers], .95))
    mae = float(np.abs(target.astype(float)-rendered).mean())
    result = {'reference_image': str(reference.relative_to(ROOT)), 'reference_image_sha256': lock['image_sha256'],
              'native_image': str(image_path.relative_to(ROOT)),
              'native_image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
              'reference_size': lock['target_size'], 'native_resolution': fit['native_resolution'],
              'source_raster': list(native.shape[1::-1]), 'source_is_native_sensor_raster': native_raster,
              'matches': len(matches), 'inliers': int(inliers.sum()),
              'direct_no_warp_median_px': median, 'direct_no_warp_p95_px': p95,
              'direct_no_warp_max_px': float(errors[inliers].max()),
              'unfiltered_descriptor_match_median_px': float(np.median(errors)),
              'unfiltered_descriptor_match_p95_px': float(np.quantile(errors, .95)),
              'direct_RGB_MAE': mae, 'regions': regions,
              'framing_passed': int(inliers.sum()) >= 100 and median < .5 and p95 < 1.5 and mae < 4,
              'byte_identical_pixels_claimed': False, 'runtime_image_warp': False,
              'method': 'Native render resized only to attachment dimensions. Direct coordinate errors and RGB differences, no image registration transform applied. Screenshot interpolation/compression and render sampling leave small pixel differences.'}
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(comparison_path), np.hstack([target, rendered, cv2.absdiff(target, rendered)]))
    return result


if __name__ == '__main__':
    report = compare_native(ROOT/'build/video_visual_fit_20261015/aligned_initial/L3-01/overview.png',
                            ROOT/'reports/reference_image_camera_20261015.jpg')
    (ROOT/'reports/reference_image_camera_20261015.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['framing_passed']
