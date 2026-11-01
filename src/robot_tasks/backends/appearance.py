"""Pinned visual assets. No joint, collision, or task geometry modifications."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from ..paths import PROJECT_ROOT
from ..robots.transforms import pose_transform

ASSETS = PROJECT_ROOT / "assets/appearance"
# The pinned community export shifts link5 by +3.7 mm along local X versus
# ARX_Model/R5a. Correct the visual only; every link then matches vendor STL
# vertices within 1 micrometre. Keep this revision-specific offset explicit.
VISUAL_OFFSETS = {"link5": np.array([-.0037, 0., 0.])}


def asset_path(relative):
    lock = json.loads((ASSETS / "source.lock.json").read_text(encoding="utf-8"))
    item = next(x for x in lock["files"] if x["path"] == relative)
    path = ASSETS / relative
    if not path.is_file():
        raise FileNotFoundError("Appearance assets missing: run python tools/fetch_appearance_assets.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Appearance asset checksum mismatch: {relative}")
    return path


def glb_primitives(path):
    """Read the pinned, single-node GLB exports without a Blender dependency.

    Fail on unsupported layouts instead of silently losing transforms/textures.
    These upstream R5 files contain PBR groups, not original factory UV images.
    """
    data = Path(path).read_bytes()
    if struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ValueError("Invalid GLB header")
    size, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A:
        raise ValueError("GLB JSON chunk missing")
    doc = json.loads(data[20:20 + size])
    binary_size, binary_kind = struct.unpack_from("<II", data, 20 + size)
    binary = data[28 + size:28 + size + binary_size]
    if binary_kind != 0x004E4942 or len(binary) != binary_size:
        raise ValueError("GLB binary chunk missing")
    if len(doc["nodes"]) != 1 or doc.get("images") or doc.get("extensionsRequired"):
        raise ValueError("Unsupported R5 GLB layout")
    node = doc["nodes"][0]
    if "matrix" in node or "children" in node:
        raise ValueError("Unsupported R5 node transform")
    q = node.get("rotation", [0, 0, 0, 1])
    transform = pose_transform((*node.get("translation", [0, 0, 0]), q[3], *q[:3]))
    transform[:3, :3] = transform[:3, :3] @ np.diag(node.get("scale", [1, 1, 1]))

    def accessor(index):
        a = doc["accessors"][index]
        view = doc["bufferViews"][a["bufferView"]]
        if a.get("sparse") or a.get("normalized") or view.get("buffer", 0) != 0:
            raise ValueError("Unsupported GLB accessor")
        dtype = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2"}[a["componentType"]])
        components = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        return np.ndarray((a["count"], components), dtype=dtype, buffer=binary,
                          offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
                          strides=(view.get("byteStride", components * dtype.itemsize), dtype.itemsize)).copy()

    result = []
    for primitive in doc["meshes"][node["mesh"]]["primitives"]:
        if primitive.get("mode", 4) != 4:
            raise ValueError("R5 visuals must be triangles")
        p = accessor(primitive["attributes"]["POSITION"])
        normals = accessor(primitive["attributes"]["NORMAL"]) @ np.linalg.inv(transform[:3, :3])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        indices = accessor(primitive["indices"]).ravel().astype(int)
        result.append({"points": p @ transform[:3, :3].T + transform[:3, 3],
                       "normals": normals, "indices": indices,
                       "material": doc["materials"][primitive["material"]]})
    return result


def material(stage, path, color, *, roughness=.5, metallic=0., textures=None, emission=None):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if emission:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if textures:
        st = UsdShade.Shader.Define(stage, path + "/UV")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        for suffix, input_name, value_type in (("diff", "diffuseColor", Sdf.ValueTypeNames.Color3f),
                                               ("rough", "roughness", Sdf.ValueTypeNames.Float),
                                               ("nor_gl", "normal", Sdf.ValueTypeNames.Normal3f)):
            tex = UsdShade.Shader.Define(stage, path + "/" + suffix)
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(asset_path(
                "textures/" + textures + "_" + suffix + ".jpg"))))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if suffix == "diff" else "raw")
            for axis in ("S", "T"):
                tex.CreateInput("wrap" + axis, Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            if suffix == "nor_gl":
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            shader.CreateInput(input_name, value_type).ConnectToSource(tex.ConnectableAPI(), "r" if suffix == "rough" else "rgb")
    return mat


def apply_r5_appearance(stage, root):
    from pxr import UsdGeom, UsdPhysics, UsdShade

    count = 0
    for link in ["base_link"] + [f"link{i}" for i in range(1, 9)]:
        parent = stage.GetPrimAtPath(root + "/" + link)
        old = stage.GetPrimAtPath(str(parent.GetPath()) + "/visuals")
        if not parent.HasAPI(UsdPhysics.RigidBodyAPI) or not old:
            raise ValueError(f"Unexpected R5 visual structure: {parent.GetPath()}")
        primitives = glb_primitives(asset_path("r5/" + link + ".glb"))
        for part in primitives:
            part["points"] += VISUAL_OFFSETS.get(link, np.zeros(3))
        # Only the original render subtree is hidden. Physics colliders and
        # articulation/inertia remain exactly those of the manufacturer URDF.
        UsdGeom.Imageable(old).CreateVisibilityAttr("invisible")
        for i, part in enumerate(primitives):
            mesh = UsdGeom.Mesh.Define(stage, str(parent.GetPath()) + f"/Appearance/Part_{i}")
            mesh.CreatePointsAttr(part["points"].astype(np.float32))
            mesh.CreateExtentAttr(np.asarray([part["points"].min(0), part["points"].max(0)], dtype=np.float32))
            mesh.CreateFaceVertexIndicesAttr(part["indices"])
            mesh.CreateFaceVertexCountsAttr([3] * (len(part["indices"]) // 3))
            mesh.CreateNormalsAttr(part["normals"].astype(np.float32))
            mesh.SetNormalsInterpolation("vertex")
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            pbr = part["material"]["pbrMetallicRoughness"]
            mat = material(stage, str(parent.GetPath()) + f"/Appearance/Material_{i}",
                           pbr.get("baseColorFactor", [1, 1, 1, 1])[:3],
                           roughness=pbr.get("roughnessFactor", 1.), metallic=pbr.get("metallicFactor", 1.))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            count += 1
    return {"profile": "r5_black_silver_v1", "visual_meshes": count,
            "source": "fiveages-sim/robot-descriptions-arx", "factory_uv_textures": False,
            "asset_lock_sha256": hashlib.sha256((ASSETS / "source.lock.json").read_bytes()).hexdigest(),
            "visual_offsets_m": {k: v.tolist() for k, v in VISUAL_OFFSETS.items()},
            "physics_model": "unchanged manufacturer R5a URDF"}
