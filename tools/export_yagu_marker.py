"""Export the L1-01 video-textured marker as a self-contained USDZ asset."""
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'assets/tuojing/appearance/l1_01_yagu'


def main():
    source = Usd.Stage.Open(str(ROOT/'assets/tuojing/source/L1-01/runtime_aligned.usdc'))
    file = DIRECTORY / 'yagu_marker_video_v2.usdc'
    # Construct in memory so rerunning does not retain old asset data.
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/Household')
    old = '/World/Household/marker'
    Sdf.CopySpec(source.GetRootLayer(), old, stage.GetRootLayer(), old)
    editor = Usd.NamespaceEditor(stage)
    editor.MovePrimAtPath(old, '/Marker')
    if not editor.CanApplyEdits() or not editor.ApplyEdits():
        raise RuntimeError('Cannot relocate marker and its material bindings')
    stage.RemovePrim('/World')
    root = stage.GetPrimAtPath('/Marker')
    UsdGeom.Xformable(root).MakeMatrixXform().Set(Gf.Matrix4d(1))
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetPrimAtPath('/Marker/Body/Looks/Label/Texture').GetAttribute('inputs:file').Set(
        Sdf.AssetPath('label_video_v2.png'))
    layer = stage.GetRootLayer()
    layer.customLayerData = {'description': 'Video-reconstructed Yagu H marker; metre, Z-up, origin at native base. Microprint/barcode approximate.',
                             'isaac_translucency_world_epsilon_m': .00001}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim) and '/Visuals/' in str(prim.GetPath()):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material or not material.GetPath().HasPrefix('/Marker'):
                raise RuntimeError('Broken standalone material: ' + str(prim.GetPath()))
    layer.Export(str(file))
    package = file.with_suffix('.usdz')
    if package.exists():
        package.unlink()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(file)), str(package)):
        raise RuntimeError('USDZ export failed')
    print(package)


if __name__ == '__main__':
    main()
