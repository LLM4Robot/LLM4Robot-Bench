"""Audit real/sim view correspondence and physical robot reset; never fit/warp images.

Run scene_smoke --no-preparation into OUTPUT/initial/TASK first. Real video ROIs
and reviewed first-frame annotations live in configs/task_references. This audit
does not infer hardware joint angles or camera calibration from two background
points. It preserves source videos and all scene layers.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from PIL import Image, ImageDraw

from robot_tasks.cameras.geometry import project_world
from robot_tasks.paths import PROJECT_ROOT as ROOT
from robot_tasks.robots.transforms import axis_rotation, pose_transform
from robot_tasks.video_delivery import scenes, effective_cameras, setup_for, setup_digest, setup_compatible, visual_setup


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contain(image, size):
    """Use a uniform scale and padding, never stretch or geometrically register."""
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = (image.width * scale, image.height * scale)
    offset = ((size[0] - resized[0]) / 2, (size[1] - resized[1]) / 2)
    canvas = image.transform(size, Image.Transform.AFFINE,
                             (1/scale, 0, -offset[0]/scale, 0, 1/scale, -offset[1]/scale),
                             resample=Image.Resampling.BICUBIC, fillcolor='black')
    return canvas, {"uniform_scale": scale, "resized_size": list(resized), "padding_xy": list(offset)}


def joint_closure(stage, robot_path):
    """Joint frames must meet, with motion only along the declared joint axis."""
    from pxr import UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    rows = []
    for i in range(1, 9):
        prim = stage.GetPrimAtPath(robot_path + "/joints/joint" + str(i))
        joint = UsdPhysics.Joint(prim)
        frames = []
        for body, position, orientation in (
            (joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
            (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
        ):
            q = orientation.Get()
            world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body.GetTargets()[0]))).T
            frames.append(world @ pose_transform([*position.Get(), q.GetReal(), *q.GetImaginary()]))
        relative = np.linalg.inv(frames[0]) @ frames[1]
        axis_index = {"X": 0, "Y": 1, "Z": 2}[prim.GetAttribute("physics:axis").Get()]
        axis = np.eye(3)[axis_index]
        if prim.IsA(UsdPhysics.RevoluteJoint):
            # Best rotation about this axis; all remaining rotation violates the joint.
            j, k = (axis_index + 1) % 3, (axis_index + 2) % 3
            r = relative[:3, :3]
            angle = np.arctan2(r[k, j] - r[j, k], r[j, j] + r[k, k])
            rotation_residual = axis_rotation(axis, angle).T @ r
            translation = np.linalg.norm(relative[:3, 3])
            coordinate = float(angle)
            target = prim.GetAttribute("drive:angular:physics:targetPosition").Get()
            target = None if target is None else float(np.deg2rad(target))
        else:
            coordinate = float(relative[axis_index, 3])
            translation = np.linalg.norm(relative[:3, 3] - coordinate * axis)
            rotation_residual = relative[:3, :3]
            target = prim.GetAttribute("drive:linear:physics:targetPosition").Get()
        rotation = float(np.arccos(np.clip((np.trace(rotation_residual) - 1) / 2, -1, 1)))
        rows.append({"joint": prim.GetName(), "axis_coordinate_rad_or_m": coordinate,
                     "drive_target_rad_or_m": target, "closure_translation_m": float(translation),
                     "closure_rotation_rad": rotation,
                     "kinematically_consistent": bool(translation < .0001 and rotation < .001)})
    return rows


def build(output, initial_root=None):
    from pxr import Usd
    from robot_tasks.registry import resolve_task
    from robot_tasks.robots.kinematics import R5Kinematics
    from robot_tasks.robots.transforms import transform_pose
    from robot_tasks.integrationtest.diagnostics.robot_reset_audit import pose_error

    reference = read(ROOT / "configs/task_references/view_pose_landmarks_20261015.json")
    review_path = output / 'visible_frame_review.json'
    frame_review = read(review_path) if review_path.exists() else {}
    if frame_review and frame_review.get('visual_setup_sha256') != setup_digest():
        raise ValueError('Visible frame review refers to an older visual setup')
    records, cards = [], []
    for tid, scene in scenes().items():
        annotation = reference['tasks'][tid]
        source = ROOT / annotation['source_video']
        if sha(source) != annotation['source_video_sha256']:
            raise ValueError("Annotations refer to another source video: " + tid)
        runtime = ROOT / 'assets/tuojing' / scene['runtime_path']
        if sha(runtime) != scene['runtime_sha256']:
            raise ValueError("Scene changed: " + tid)
        initial = (initial_root or output / 'initial') / tid
        check = read(initial / 'result.json')
        if check['runtime_scene_sha256'] != scene['runtime_sha256']:
            raise ValueError("Stale physical reset capture: " + tid)
        if not setup_compatible(tid, check.get('visual_setup_sha256')):
            raise ValueError('Stale video-visible setup capture: ' + tid)
        setup = setup_for(tid)
        entry, config = resolve_task(tid)
        kinematics = R5Kinematics(entry.factory.prepare_config(config).robot)
        cameras = effective_cameras(tid)
        rendered_cameras = {}
        for name, captured in check['camera_capture'].items():
            actual = captured['calibration']
            target = cameras[name]
            expected = pose_transform(target['pose'])
            if target['parent'] != 'world':
                arm = target['parent']
                joints = check['robot_reset_audit']['arms'][arm]['state']['joint_positions_rad']
                expected = pose_transform(kinematics.forward(joints, arm)) @ np.linalg.inv(kinematics.tcp) @ expected
            error = pose_error(actual['T_world_camera_pose_wxyz'], transform_pose(expected))
            lens_matches = bool(np.allclose(actual['intrinsics_fx_fy_cx_cy'], target['intrinsics'], rtol=0, atol=.001))
            passed = lens_matches and error['translation_m'] < .002 and error['rotation_rad'] < .005
            if not passed:
                raise ValueError('Rendered camera does not match the configured virtual rig: ' + tid + ' ' + name)
            rendered_cameras[name] = {'passed': passed, 'intrinsics_match': lens_matches,
                'actual_pose_vs_independent_fk': error,
                'scope': 'Configuration applied to the rendered virtual camera; not a hardware calibration accuracy measurement.'}
        folder = output / tid
        folder.mkdir(parents=True, exist_ok=True)
        # Lossless first decode cached when the original video was collected.
        raw = Image.open(ROOT / 'build/reference_alignment_20261015/smokes' / tid / 'real_initial.png').convert('RGB')
        if list(raw.size) != annotation['source_resolution']:
            raise ValueError("Unexpected real video layout: " + tid)
        images, exports = {}, {}
        for view, box in annotation['rois'].items():
            size = tuple(cameras[view]['resolution'])
            sim = Image.open(initial / (view + '.png')).convert('RGB')
            if sim.size != size:
                raise ValueError("Unexpected simulation view: " + tid + ' ' + view)
            sim.save(folder / ('sim_' + view + '.png'))
            if box is None:
                exports[view] = {"real_available": False}
                continue
            real, transform = contain(raw.crop(box), size)
            real.save(folder / ('real_' + view + '.png'))
            images[view] = (real, sim)
            exports[view] = {"real_available": True, "source_roi_xyxy": box, "canvas_size": list(size),
                             "display_transform": transform, "homography_applied": False}
        # These two correspondences were used to fit the wall reference group;
        # their residuals must not be presented as independent validation.
        camera = effective_cameras(tid)['overview']
        world = setup['background_reference']['target_anchor_world']
        predicted = project_world(world, camera['pose'], camera['intrinsics'])
        measured = np.asarray(annotation['real_overview_plug_centres_px_1280x960'])
        # The reviewed coordinates used an anisotropic 1280x960 preview. Undo
        # its subpixel aspect rounding and apply the actual uniform display.
        box = annotation['rois']['overview']
        display = exports['overview']['display_transform']
        measured = (measured + .5) / [1280, 960] * display['resized_size'] + display['padding_xy'] - .5
        errors = np.linalg.norm(measured - predicted, axis=1)
        real_delta, sim_delta = measured[1]-measured[0], predicted[1]-predicted[0]
        angle = float(np.degrees(np.arctan2(sim_delta[1], sim_delta[0]) - np.arctan2(real_delta[1], real_delta[0])))
        for kind, points in [('real', measured), ('sim', predicted)]:
            image = images['overview'][kind == 'sim'].copy()
            draw = ImageDraw.Draw(image)
            for i, (u, v) in enumerate(points):
                draw.ellipse((u-7, v-7, u+7, v+7), outline='#ff3050', width=3)
                draw.text((u+10, v), ('white', 'black')[i], fill='#ff3050')
            image.save(folder / (kind + '_anchors.png'))
        stage = Usd.Stage.Open(str(runtime))
        joints = {name: joint_closure(stage, robot['path']) for name, robot in scene['robots'].items()}
        reset = check['robot_reset_audit']
        warnings = []
        for name,item in scene['tracked'].items():
            prim = stage.GetPrimAtPath(item['path'])
            from pxr import UsdGeom
            if prim and UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                warnings.append(name + ': scoring body is invisible; visible review geometry may be separate')
        openings = '；'.join(f"控制 ID {a}：开度 {r['gripper_width_m']*1000:.1f} mm" for a,r in setup['arms'].items())
        note = '机械臂初态按真机视频估计；三路相机统一采用 L3-01 配置。' + openings + '。腕部从合法关节状态生成。'
        projected_fit = {}
        for arm, arm_setup in setup['arms'].items():
            fit = arm_setup.get('visible_fit', {})
            residuals = {}
            for name in ('housing', 'cap'):
                target, arm_prediction = fit.get(name+'_target_uv'), fit.get(name+'_predicted_uv')
                if target is not None and arm_prediction is not None:
                    residuals[name+'_error_px'] = float(np.linalg.norm(np.asarray(target)-arm_prediction))
            target, arm_prediction = fit.get('object_target_bbox'), fit.get('object_predicted_bbox')
            if target is not None and arm_prediction is not None:
                residuals['object_name'] = fit.get('object_name')
                residuals['object_bbox_max_edge_error_px'] = float(np.max(np.abs(np.asarray(target)-arm_prediction)))
            projected_fit[arm] = residuals
        if tid == 'L1-05':
            note += '此任务只有真机全局录像，腕相机沿用同套装置的估计。'
        if warnings:
            note += '源资产可见几何与评分刚体存在分离，动作结果需结合该问题查看。'
        reviewed = frame_review.get('tasks', {}).get(tid, {})
        row = {"task_id": tid, "fully_aligned": False, "real_robot_commands": 0, "model_calls": 0,
               "visible_setup_applied": True, "visual_setup_sha256": setup_digest(),
               "recorded_initial_visual_setup_sha256": check.get('visual_setup_sha256'),
               "real_video_sha256": annotation['source_video_sha256'], "runtime_scene_sha256": scene['runtime_sha256'],
               "physical_robot_reset": reset, "initial_scene_passed": check['passed'],
               "initial_scene_error": (check.get('error') or '').split("'metrics':")[0],
               "camera_parameters": effective_cameras(tid), "real_camera_calibration_available": False,
               "rendered_camera_configuration": rendered_cameras,
               "hardware_calibration_required": False, "source_asset_binding_warnings": warnings,
               "projected_fit_diagnostics": {"arms": projected_fit,
                   "scope": "Historical arm/joint fitting diagnostics in the earlier per-task image coordinates (overview 1280x960, wrists 1280x720). These are not residuals of the current shared reference-image camera or held-out validation."},
               "visual_findings": note, "prior_findings": annotation['visual_findings'], "views": exports,
               "visible_frame_review": reviewed,
               "background_check": {"landmarks": ["white plug centre", "black plug centre"],
                   "annotation_uncertainty_px": annotation['annotation_uncertainty_px'],
                   "estimated_world_points_m": world, "real_uv": measured.tolist(), "simulation_uv": predicted.tolist(),
                   "errors_px": errors.tolist(), "max_error_px": float(max(errors)),
                   "projected_separation_error_percent": float((np.linalg.norm(sim_delta)/np.linalg.norm(real_delta)-1)*100),
                   "image_line_angle_difference_deg": angle,
                   "scope": "Fitted wall reference points, not held-out evidence of whole-image fidelity; physical scene parameters are image-derived estimates."},
               "native_authored_joint_closure": joints,
               "native_pose_is_valid_articulation": all(j['kinematically_consistent'] for arm in joints.values() for j in arm)}
        records.append(row)
        prefix = '../' + folder.relative_to(ROOT).as_posix() + '/'
        states = reset['arms']
        detail = '; '.join(f"{a}: q偏差 {max(abs(v) for v in r['joint_error_rad'])*180/np.pi:.3f}°，开度 {r['state']['gripper_width_m']*1000:.2f} mm" for a,r in states.items())
        smoke = read(ROOT/'build/reference_alignment_20261015/smokes'/tid/f'smoke_test_{tid}.json')
        if not setup_compatible(tid, smoke.get('visual_setup_sha256')):
            raise ValueError('Current visual setup has not been physically recorded: ' + tid)
        row['recorded_smoke_visual_setup_sha256'] = smoke.get('visual_setup_sha256')
        recorded_layout = smoke.get('video_layout')
        cards.append({"id": tid, "prefix": prefix, "findings": row['visual_findings'], "reset": detail,
                      "reset_pass": reset['passed'], "errors": row['background_check'],
                      "projected_fit": projected_fit,
                      "remaining_visible_differences": reviewed.get('remaining_visible_differences', []),
                      "native_valid": row['native_pose_is_valid_articulation'], "views": list(images),
                      "real_video": '../' + annotation['source_video'], "real_roi": annotation['rois']['overview'],
                      "real_rois": annotation['rois'],
                      "sim_rois": recorded_layout['rois'] if recorded_layout else {'overview':[0,30,960,750]},
                      "sim_video": f'../real_videos/{tid}/smoke_test_{tid}.mp4',
                      "sim_roi": recorded_layout['rois']['overview'] if recorded_layout else [0,30,960,750],
                      "sim_start_s": 0 if recorded_layout else 1/6})
    robot_rows = [r for row in records for r in row['physical_robot_reset']['arms'].values()]
    summary = {"tasks": len(records), "robot_resets_passed": sum(r['physical_robot_reset']['passed'] for r in records),
               "rendered_camera_checks_passed": sum(c['passed'] for r in records for c in r['rendered_camera_configuration'].values()),
               "video_setups_applied": len(records), "pixel_identical_claimed": False,
               "fully_aligned_tasks": 0, "native_articulation_inconsistent_tasks": [r['task_id'] for r in records if not r['native_pose_is_valid_articulation']],
               "max_base_error_m": max(r['base_error']['translation_m'] for r in robot_rows),
               "max_tcp_fk_error_m": max(r['tcp_fk_error']['translation_m'] for r in robot_rows),
               "max_home_joint_drift_deg": max(max(abs(v) for v in r['joint_error_rad']) for r in robot_rows)*180/np.pi,
               "max_background_point_error_px": max(r['background_check']['max_error_px'] for r in records)}
    report = {"schema_version": 1, "summary": summary, "tasks": records,
              "conclusion": "All current tasks share the fixed L3-01 overview and link6-relative wrist camera rig. Initial joints and openings remain task-specific and physically checked. Real-video panes are preserved; different original recording viewpoints are no longer fitted per task. Pixel-identical geometry/materials are not claimed.",
              "user_scope": "Use the bottle-cap unscrewing task's camera rig for all current tasks; keep task-specific robot and object initial states.",
              "camera_reference_task": visual_setup()['camera_reference_task'],
              "camera_reference_image": visual_setup()['overview_reference_image_fit'],
              "visual_setup_revision": visual_setup()['revision'], "hardware_calibration_required": False,
              "evidence_reuse_policy": "Recorded whole-file SHA256 values are preserved. An older capture is reused only if the hash-verified archived configuration has an identical complete record for this task; changed tasks are physically rerun.",
              "visible_frame_review": {"method": frame_review.get('method'),
                  "common_remaining": frame_review.get('common_remaining', []),
                  "reviewed_tasks": sum(bool(r['visible_frame_review'].get('reviewed')) for r in records)},
              "existing_smoke_recording_first_frame": {
                  "issue": "Frame 0 contains the renderer cache from before the commanded physical reset; frame 1 matches the current passive-reset robot view.",
                  "evidence": "build/view_pose_alignment_20261015/smoke_first_frame_check.json",
                  "new_recordings": "Rerun after the setup changes; first frame follows physical stepping, and all panes match the source layout.",
                  "comparison_player_start_frame": 0, "real_video_modified": False,
                  "future_recorder_requirement": "Record only after PhysX has propagated reset joint transforms to the renderer; preserve pre-step numeric object state separately."},
              "display": "Uniform pane extraction and scaling, using the shared L3-01 camera rig for all simulations. No output-image warp or real-video rewriting. Original real recordings can have different viewpoints."}
    dest = ROOT / 'reports/view_pose_alignment_20261015'
    dest.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_html(dest.with_suffix('.html'), summary, cards)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_html(path, summary, cards):
    data = json.dumps(cards, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真机与仿真：初始姿态、角度和缩放校验</title>
<style>body{max-width:1500px;margin:auto;padding:24px;font:16px/1.6 system-ui;background:#eef1f5;color:#192733}section{background:white;padding:20px;margin:24px 0;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair>div{min-width:0}img,canvas{width:100%;display:block;background:#111}button,select{padding:7px 12px;margin:4px}input[type=range]{width:220px}pre{white-space:pre-wrap}.bad{color:#a52b34}.good{color:#12744b}.overlay{position:relative}.overlay img:nth-child(2){position:absolute;inset:0;width:100%}.hidden,video{display:none}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd3dc}a{color:#2055ae}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>
<h1>按真机视频调整的初态与视角</h1><p><strong>以视频可见设定为准估计调整。</strong>各任务分别设置关节初态、夹爪开度和三路视角，并在物理复位后重新渲染。视频看不到的参数保留为估计值。</p>
<p>全局画面为 4:3，腕画面为 16:9；对照显示只提取同一视窗并等比例缩放，没有输出图像配准、拉伸或镜像。真机左腕对应控制 ID right，真机右腕对应控制 ID left。静帧为仿真复位后约 2 秒，动作录像另行执行，不作时间同步。</p>
<p>墙面插头是用于调整场景的参考点，误差小不代表材质、照明和全部几何已像素一致。可用叠图检查整幅画面；源资产中可见模型与评分刚体分离的问题会单独标注。</p>
<p>新 smoke 使用与真机相同的全局/双腕排版，从物理复位生效后开始录制；原始真机录像不改动。旧录像首帧缓存的问题已在录制代码中修正。</p>
<p><a href="view_pose_alignment_20261015.json">完整数据与每个关节的检查结果</a> · <a href="reference_alignment_20261015.html">任务动作与资产报告</a></p><pre id="summary"></pre><nav id="nav"></nav><main></main>
<script id="data" type="application/json">CARDS</script><script>
const cards=JSON.parse(document.querySelector('#data').textContent);
const summary=SUMMARY;
document.querySelector('#summary').textContent=`共用 L3-01 相机配置 ${summary.video_setups_applied}/${summary.tasks}；物理复位 ${summary.robot_resets_passed}/${summary.tasks} 通过；渲染相机与配置一致 ${summary.rendered_camera_checks_passed ?? 0}/${summary.tasks*3}。\n底座最大误差 ${(summary.max_base_error_m*1000).toFixed(6)} mm；末端与独立 FK 最大误差 ${(summary.max_tcp_fk_error_m*1000).toFixed(4)} mm。\n原始 USD 关节不闭合的任务：${summary.native_articulation_inconsistent_tasks.join('、')}。运行时已从合法关节状态生成初态。`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function player(src,roi,poster,start=0){
 const box=document.createElement('div');box.innerHTML=`<canvas width="960" height="720"></canvas><video preload="none" muted playsinline></video><button>播放</button><input type="range" min="0" max="1" step=".001" value="0"><span>0.0 s</span>`;
 const canvas=box.querySelector('canvas'),ctx=canvas.getContext('2d'),v=box.querySelector('video'),b=box.querySelector('button'),seek=box.querySelector('input'),time=box.querySelector('span');const im=new Image();im.onload=()=>{canvas.height=Math.round(960*im.height/im.width);ctx.drawImage(im,0,0,960,canvas.height);if(v.readyState>=2)draw()};im.src=poster;
 v.onloadedmetadata=()=>{seek.min=start;v.currentTime=start};
 const draw=()=>{if(v.readyState<2)return;const [x,y,x1,y1]=roi,w=x1-x,h=y1-y,W=canvas.width,H=canvas.height,s=Math.min(W/w,H/h);ctx.fillStyle='black';ctx.fillRect(0,0,W,H);ctx.drawImage(v,x,y,w,h,(W-w*s)/2,(H-h*s)/2,w*s,h*s);if(Number.isFinite(v.duration)){seek.max=v.duration;seek.value=v.currentTime}time.textContent=v.currentTime.toFixed(3)+' s';if(!v.paused)requestAnimationFrame(draw)};
 b.onclick=async()=>{if(!v.src)v.src=src;if(v.paused){try{await v.play();b.textContent='暂停';draw()}catch(e){time.textContent=String(e)}}else{v.pause();b.textContent='播放'}};seek.oninput=()=>{if(!v.src)v.src=src;v.currentTime=+seek.value};v.onseeked=draw;v.onended=()=>b.textContent='播放';box.setView=(r,p)=>{roi=r;im.src=p;if(v.readyState>=2)draw()};return box;
}
for(const c of cards){
 document.querySelector('#nav').insertAdjacentHTML('beforeend',`<a href="#${c.id}">${c.id}</a>　`);
 const section=document.createElement('section');section.id=c.id;const e=c.errors;
 section.innerHTML=`<h2>${c.id}</h2><p>${esc(c.findings)}</p>${(c.remaining_visible_differences||[]).map(n=>`<p class="bad">仍有差异：${esc(n)}</p>`).join('')}<p class="good">物理复位：${c.reset_pass?'通过':'失败'}。${esc(c.reset)}</p><p>墙面参考点偏差 ${e.max_error_px.toFixed(1)} px（仿真已统一 L3-01 视角，原真机录像构图可能不同；不代表整幅画面误差）。</p><select>${c.views.map(v=>`<option value="${v}">${({overview:'全局',right_wrist:'真机左腕 / 控制 ID right',left_wrist:'真机右腕 / 控制 ID left'})[v]}</option>`).join('')}</select><button class="toggle">切换叠图</button><label>仿真透明度 <input class="blend" type="range" min="0" max="1" step=".01" value=".5"></label><div class="pair frames"><div><p>真机首帧</p><img class="real"></div><div><p>当前仿真复位后</p><img class="sim"></div></div><div class="overlay hidden"><img class="real"><img class="sim" style="opacity:.5"></div><details><summary>拟合残差与仍待复核的大小 / 位置</summary><p>以下是历史机械臂初态拟合记录，使用当时的逐任务图像坐标：全局 1280×960，腕部 1280×720。当前已统一为用户参考图视角，这些历史数值不作为当前构图误差或独立验收结果。</p><pre>${esc(JSON.stringify(c.projected_fit||{},null,2))}</pre></details><details><summary>固定背景点标注</summary><div class="pair"><img loading="lazy" src="${c.prefix}real_anchors.png"><img loading="lazy" src="${c.prefix}sim_anchors.png"></div></details><details class="movies"><summary>按当前所选视角播放真机 / smoke 视频</summary><p>分别控制播放和进度；原视频播放倍率不同，不作为动作时间对齐证据。</p><div class="pair"></div></details>`;
 const update=()=>{const view=section.querySelector('select').value;section.querySelectorAll('img.real').forEach(i=>i.src=c.prefix+'real_'+view+'.png');section.querySelectorAll('img.sim').forEach(i=>i.src=c.prefix+'sim_'+view+'.png');if(section.players){section.players[0].setView(c.real_rois[view],c.prefix+'real_'+view+'.png');section.players[1].setView(c.sim_rois[view],c.prefix+'sim_'+view+'.png')}};section.querySelector('select').onchange=update;update();section.querySelector('.toggle').onclick=()=>{section.querySelector('.frames').classList.toggle('hidden');section.querySelector('.overlay').classList.toggle('hidden')};section.querySelector('.blend').oninput=x=>section.querySelector('.overlay img.sim').style.opacity=x.target.value;
 const movies=section.querySelector('.movies');movies.ontoggle=()=>{if(!movies.open||movies.dataset.loaded)return;movies.dataset.loaded='1';const p=movies.querySelector('.pair');const view=section.querySelector('select').value;section.players=[player(c.real_video,c.real_rois[view],c.prefix+'real_'+view+'.png'),player(c.sim_video,c.sim_rois[view],c.prefix+'sim_'+view+'.png',c.sim_start_s)];p.append(...section.players)};document.querySelector('main').append(section);
}
</script></html>'''
    path.write_text(document.replace('CARDS', data).replace('SUMMARY', json.dumps(summary)), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/video_visual_fit_20261015')
    parser.add_argument('--initial-root', type=Path, default=ROOT / 'build/video_visual_fit_20261015/aligned_initial')
    args = parser.parse_args()
    build(args.output.resolve(), args.initial_root.resolve())
