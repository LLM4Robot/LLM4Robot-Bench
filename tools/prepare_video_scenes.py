"""Make portable runtime layers from immutable USDZ scene deliveries.

The saved editor orchestration graph must not replace Kit's live graph. Geometry,
placements, lights and materials remain composed from the delivered stage.
"""

from pathlib import Path
import hashlib
import json
import argparse
import os

from pxr import Sdf, Usd, UsdGeom, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--visual-release', type=Path, help='Verified extracted PR #19 portable bundle')
parser.add_argument('--align-contents',action='store_true',help='Restore task-specific glass-ball counts using latest assets')
parser.add_argument('--task',action='append',help='Rebuild selected task IDs only; may be repeated')
args = parser.parse_args()
manifest_path = ROOT / "assets/tuojing/video_20261015.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for tid, row in manifest["tasks"].items():
    if args.task and tid not in args.task:
        continue
    original = ROOT / "assets/tuojing" / row["path"]
    source = Usd.Stage.Open(str(original))
    for name in row.get('derived_contents',{}).get('added',{}):
        row['tracked'].pop(name,None)
    row.pop('derived_contents',None)
    from video_scene_metadata import enrich
    enrich(source,row,tid)
    row.pop('effective_tracked_overrides',None)
    row.pop('visual_overlay',None)
    source.SetEditTarget(source.GetSessionLayer())
    for prim in source.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)
    layer = source.Flatten()

    def portable(path):
        basename = path.rsplit("/", 1)[-1].rstrip("]")
        if basename in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}:
            return basename
        if ".usdz[" in path:
            archive, internal = path.split('.usdz[',1)
            archive = Path(archive+'.usdz')
            prefix = os.path.relpath(archive,original.parent) if archive.is_absolute() else archive.as_posix()
            return prefix+'['+internal
        if Path(path).is_absolute() and Path(path).is_relative_to(ROOT):
            return os.path.relpath(path,original.parent)
        return path

    UsdUtils.ModifyAssetPaths(layer, portable)
    runtime = Usd.Stage.Open(layer)
    runtime.RemovePrim("/Orchestrator")
    runtime.RemovePrim("/Render")
    for name in ("Persp", "Front", "Top", "Right"):
        runtime.RemovePrim("/OmniverseKit_" + name)
    runtime.SetDefaultPrim(runtime.GetPrimAtPath("/World"))
    UsdGeom.SetStageUpAxis(runtime, "Z")
    UsdGeom.SetStageMetersPerUnit(runtime, 1.)
    if args.visual_release:
        from video_visual_r3 import apply
        row['visual_overlay'] = apply(runtime,args.visual_release.resolve(),tid,ROOT/'assets/tuojing')
        tray = row['visual_overlay'].pop('effective_tray_record',None)
        if tray:
            row['effective_tracked_overrides'] = {'plug_storage_base':tray}
            row['tracked']['plug_storage_base'] = tray
    from video_scene_repairs import apply as repair_bindings
    row['physical_repairs'] = repair_bindings(runtime, tid, row)
    if tid == 'L1-05' and row.get('flower_collision_repair'):
        from l1_05_flower_collision import apply as repair_flower
        row['flower_collision_repair'] = repair_flower(runtime)
    if tid == 'L1-05' and row.get('flower_initial_support'):
        from l1_05_supported_layout import apply as support_flower
        row['flower_initial_support'] = support_flower(runtime, row)
    if tid == 'L1-05':
        from l1_05_grip_dynamics import apply as flower_grip
        row['physical_repairs']['flower_grip_dynamics'] = flower_grip(runtime)
        from l1_05_passive_leaves import apply as flexible_leaves
        leaf_repair = flexible_leaves(runtime)
        row['physical_repairs']['flower_passive_leaves'] = leaf_repair
        row['physical_repairs']['flower_grip_dynamics'] = leaf_repair['root_grip_dynamics']
    if args.align_contents:
        from video_scene_contents import apply as apply_contents
        contents = apply_contents(runtime,tid,manifest['tasks'],ROOT/'assets/tuojing')
        if contents:
            row['derived_contents'] = contents
            row['tracked'].update(contents['added'])
        layer = runtime.Flatten()
        UsdUtils.ModifyAssetPaths(layer,portable)
    entry = original.with_name('runtime_aligned.usdc' if args.align_contents else "runtime_r3.usdc" if args.visual_release else "runtime.usdc")
    layer.Export(str(entry))
    row["runtime_path"] = entry.relative_to(ROOT / "assets/tuojing").as_posix()
    row["runtime_sha256"] = hashlib.sha256(entry.read_bytes()).hexdigest()
    row["runtime_bytes"] = entry.stat().st_size
    row["runtime_adapters"] = ["saved /Render and /Orchestrator omitted to discard stale renderer state", "editor-only root cameras omitted", "defaultPrim World and Z-up/metre metadata restored", "built-in MDL search-path reference restored", "archive texture paths made relative"]
    print(tid, entry.stat().st_size, flush=True)
if args.visual_release:
    manifest['visual_release'] = {'version':'visual_alignment_r3_20261014',
        'archive_sha256':'20608d2ef57d7f9a1b78e0f3bd8c3358cd3d1d59fccc396c908ff8dbf2727f21',
        'archive_bytes':343374105,'physical_changes':{},
        'scope':'room/background appearance only; task props and their materials remain from the October 15 complete delivery'}
    manifest['visual_textures'] = {file.relative_to(ROOT/'assets/tuojing').as_posix():
        {'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        for file in sorted((ROOT/'assets/tuojing/visual_r3/textures').glob('*')) if file.is_file()}
temporary = manifest_path.with_suffix('.tmp')
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
temporary.replace(manifest_path)
