"""Capture current task setups without calling a model or manipulating props.

Each task runs in a fresh Isaac process through the existing scene check,
before robot preparation. No CUDA planner is loaded. The detail camera is for
documentation only and never enters model observations.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

from ...paths import PROJECT_ROOT


def source_fingerprint():
    """Invalidate previews when task geometry, setup or camera configs change."""
    digest = hashlib.sha256()
    files = list((PROJECT_ROOT / "src/robot_tasks").rglob("*.py"))
    for folder in ("tasks", "robots", "cameras", "workflows"):
        files.extend((PROJECT_ROOT / "configs" / folder).glob("*.toml"))
    files.extend((PROJECT_ROOT / "configs/task_references").glob("*.json"))
    files.extend(
        PROJECT_ROOT / "assets/tuojing" / name
        for name in ("catalog.json", "source.lock.json", "video_20261015.json", "video_scoring_geometry.json")
    )
    for path in sorted(files):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_detail(backend, task, output):
    import numpy as np
    from PIL import Image
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    prim = backend.world.stage.GetPrimAtPath(task.scene.root)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    if not np.isfinite([low, high]).all() or np.any(high < low):
        raise ValueError("Invalid task geometry bounds for preview camera")
    target = (low + high) / 2
    distance = max(0.38, float(np.linalg.norm(high - low)) * 1.8)
    eye = target + np.array([0.65, -0.72, 0.90]) * distance
    rep = backend._rep
    camera = rep.create.camera(
        name="documentation_detail",
        position=tuple(eye),
        look_at=tuple(target),
        clipping_range=(0.005, 20),
    )
    product = rep.create.render_product(camera, (1200, 900))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    try:
        # Render the physically reset initial state; do not advance physics.
        timestamp = backend.time_s()
        for _ in range(12):
            backend.world.render()
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=2)
            if not np.isclose(backend.time_s(), timestamp, rtol=0, atol=1e-9):
                raise RuntimeError("Preview unexpectedly advanced the simulation clock")
            data = np.asarray(rgb.get_data())
        if data.ndim != 3 or data.shape[:2] != (900, 1200) or data[..., :3].std() < 1:
            raise RuntimeError("Documentation camera did not render a valid image")
        Image.fromarray(np.ascontiguousarray(data[..., :3], dtype=np.uint8)).save(
            output / "detail.png"
        )
        return {
            "image": "detail.png",
            "camera_position": eye.tolist(),
            "camera_target": target.tolist(),
            "bounds": [low.tolist(), high.tolist()],
            "simulation_time_s": backend.time_s(),
            "model_input": False,
            "source_sha256": source_fingerprint(),
        }
    finally:
        rgb.detach(product)
        product.destroy()


def write_index(output, results):
    """Local static contact sheet; all links stay beside the captured evidence."""
    from ...registry import resolve_task

    cards = []
    for row in results:
        task = row["task"]
        _, config = resolve_task(task)
        name = html.escape(config.raw["task"].get("name", task))
        status = (
            "初态校验通过"
            if row["captured"]
            else html.escape(str(row.get("error") or row["status"]))
        )
        photos = "".join(
            f'<figure><a href="{task}/{file}"><img loading="lazy" src="{task}/{file}"></a>'
            f"<figcaption>{caption}</figcaption></figure>"
            for file, caption in (
                ("overview.png", "公共第三视角 · 模型观测"),
                ("detail.png", "道具近景 · 仅用于展示"),
            )
            if (output / task / file).exists()
        )
        description = html.escape(config.raw["task"].get("description", "").strip())
        cards.append(
            f'<section id="{task}"><h2>{task} {name}</h2><p>{status}</p>'
            f'<div class="photos">{photos}</div><details><summary>布置与完成条件</summary>'
            f'<pre>{description}</pre></details><p><a href="{task}/result.json">校验记录</a></p></section>'
        )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>任务初始场景</title>
<style>body{max-width:1400px;margin:auto;padding:24px;background:#f5f6f8;color:#18242e;font:16px/1.6 system-ui}
section{background:white;border:1px solid #ddd;padding:20px;margin:24px 0;border-radius:10px}
.photos{display:flex;gap:16px}figure{flex:1;min-width:0;margin:0}img{width:100%;border-radius:6px}
pre{white-space:pre-wrap;font:inherit}a{color:#135abc}@media(max-width:700px){.photos{display:block}}</style>
<h1>任务初始场景</h1><p>机器人按任务配置的关节与夹爪初态复位，道具经过物理落稳；未调用 GPT、未执行任务。
初态通过不等于模型能完成任务。第三视角是实际输入画面；近景是独立展示相机。</p>"""
    document += (
        "<nav>"
        + " · ".join(f'<a href="#{r["task"]}">{r["task"]}</a>' for r in results)
        + "</nav>"
    )
    (output / "index.html").write_text(
        document + "".join(cards) + "</html>", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional explicit initial-layout task config")
    args = parser.parse_args()
    if args.all and args.config:
        parser.error("--config selects one task; cannot be used with --all")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.all:
        if not args.task:
            parser.error("Provide a task ID or --all")
        from .scene_smoke import check

        return check(args.task, args.output, prepare=False, capture_hook=capture_detail,
                     capture_before_observations=True, config_path=args.config)

    from ...registry import TASKS

    results = []
    for spec in TASKS.values():
        if not spec.task_id.startswith("L") or spec.base_task_id is not None:
            continue
        if not spec.implemented:
            results.append(
                {"task": spec.task_id, "status": "real_only", "captured": False}
            )
            continue
        folder = args.output / spec.task_id
        # Resume an interrupted export only when its scene capture passed.
        result_path = folder / "result.json"
        existing = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        if (
            existing.get("passed")
            and (folder / "detail.png").exists()
            and existing.get("preview", {}).get("source_sha256") == source_fingerprint()
        ):
            result = existing
            code = 0
        else:
            with (args.output / (spec.task_id + ".log")).open(
                "w", encoding="utf-8"
            ) as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        __package__ + ".initial_gallery",
                        spec.task_id,
                        "--output",
                        str(folder),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    check=False,
                )
            code = process.returncode
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else {}
            )
        entry = {
            "task": spec.task_id,
            "status": "captured" if result.get("passed") else "error",
            "captured": bool(result.get("passed")),
            "returncode": code,
            "error": result.get("error"),
        }
        results.append(entry)
        (args.output / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_index(args.output, results)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    return 0 if all(r["status"] != "error" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
