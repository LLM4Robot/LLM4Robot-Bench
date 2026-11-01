"""L1-01 black Yagu H marker label, reconstructed from the user's rotation video.

Front, rear and side layouts are video-backed; microprint is approximate.
Only the label shader changes; geometry, UVs, pose and collisions are retained.
"""
import hashlib
from pathlib import Path
from pxr import Gf, Sdf

ROOT = Path(__file__).resolve().parents[1]
TEXTURE = 'appearance/l1_01_yagu/label_video_v2.png'


def apply(stage):
    prim = stage.GetPrimAtPath('/World/Household/marker/Body/Looks/Label/Texture')
    if not prim:
        raise ValueError('Missing L1-01 marker label texture shader')
    file = ROOT / 'assets/tuojing' / TEXTURE
    if not file.is_file():
        raise FileNotFoundError(file)
    # Runtime USD lives in source/L1-01; keep the reference portable.
    prim.GetAttribute('inputs:file').Set(Sdf.AssetPath('../../' + TEXTURE))
    prim.GetAttribute('inputs:scale').Set(Gf.Vec4f(1, 1, 1, 1))
    prim.GetAttribute('inputs:sourceColorSpace').Set('sRGB')
    return {'revision': 'yagu_black_h_video_wrap_v2',
            'texture': TEXTURE, 'bytes': file.stat().st_size,
            'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
            'source': 'User-supplied 22.17-second marker rotation video; imagegen reconstruction',
            'limitation': 'Complete circumference reconstructed; fine print, pictograms and barcode approximate, not exact manufacturer artwork or a scannable barcode',
            'geometry_unchanged': True}
