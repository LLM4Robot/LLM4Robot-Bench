"""Export the installed measured L1-02 holder with its native collision shell."""
from pathlib import Path
import json
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'assets/tuojing/appearance/l1_03_measured_holder';out.mkdir(parents=True,exist_ok=True)
    source=Usd.Stage.Open(str(ROOT/'assets/tuojing/source/L1-03/runtime_aligned.usdc'))
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/Household')
    old='/World/Household/plug_storage_base';Sdf.CopySpec(source.GetRootLayer(),old,stage.GetRootLayer(),old)
    # Export the measured bare holder, excluding the photo-estimated scene insert.
    stage.RemovePrim(old+'/Visuals/HeightSpacer')
    for prim in list(stage.GetPrimAtPath(old+'/Colliders').GetChildren()):
        if prim.GetName().startswith('HeightSpacer_'):stage.RemovePrim(prim.GetPath())
    seat=UsdGeom.Xformable(stage.GetPrimAtPath(old+'/Frames/PlugSeat'))
    matrix=Gf.Matrix4d(1);matrix.SetTranslate(Gf.Vec3d(0,0,.030));seat.MakeMatrixXform().Set(matrix)
    editor=Usd.NamespaceEditor(stage);editor.MovePrimAtPath(old,'/Holder')
    if not editor.CanApplyEdits() or not editor.ApplyEdits():raise RuntimeError('Cannot relocate holder')
    stage.RemovePrim('/World');root=stage.GetPrimAtPath('/Holder')
    UsdGeom.Xformable(root).MakeMatrixXform().Set(Gf.Matrix4d(1));stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    file=out/'plug_holder_measured.usdc';stage.GetRootLayer().Export(str(file));package=file.with_suffix('.usdz')
    if package.exists():package.unlink()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(file)),str(package)):raise RuntimeError('USDZ export failed')
    r=json.loads((ROOT/'assets/tuojing/video_20261015.json').read_text())['tasks']['L1-02']['physical_repairs']['measured_holder']
    (out/'dimensions_and_provenance.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(package)

if __name__=='__main__':main()
