"""Video-derived illumination and table color, authored only on the session layer."""

import hashlib
import math
from pathlib import Path


def apply(stage, settings):
    """Adjust the renderer's lights/materials; preserve geometry and physics."""
    if not settings:
        return {}
    from pxr import Gf, UsdGeom, UsdLux

    intensity = float(settings['intensity_multiplier'])
    color = settings['light_rgb_multiplier']
    window = float(settings.get('window_multiplier', 1.))
    table = settings['table_rgb_multiplier']
    values = [intensity, window, *color, *table]
    if (len(color) != 3 or len(table) != 3
            or any(not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError('Invalid video illumination/color settings')
    stage.SetEditTarget(stage.GetSessionLayer())
    lights = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        light = UsdLux.LightAPI(prim)
        old_intensity = light.GetIntensityAttr().Get()
        old_color = light.GetColorAttr().Get()
        gain = intensity * (window if prim.GetName() == 'WindowFill' else 1.)
        new_color = [a*b for a, b in zip(old_color, color)]
        light.GetIntensityAttr().Set(old_intensity * gain)
        light.GetColorAttr().Set(Gf.Vec3f(*new_color))
        lights[str(prim.GetPath())] = {
            'before': {'intensity': old_intensity, 'color': list(old_color)},
            'after': {'intensity': old_intensity * gain, 'color': new_color},
        }
    texture = stage.GetPrimAtPath('/World/Materials/R3Table/Texture')
    if texture:
        attr = texture.GetAttribute('inputs:scale')
        before = list(attr.Get())
        after = [before[i]*table[i] for i in range(3)] + [before[3]]
        attr.Set(Gf.Vec4f(*after))
        surface = {'path': str(texture.GetPath()), 'attribute': 'inputs:scale',
                   'before': before, 'after': after}
    else:
        prim = stage.GetPrimAtPath('/World/ReferenceDetail/TableSurface')
        attr = UsdGeom.PrimvarsAPI(prim).GetPrimvar('displayColor') if prim else None
        before = attr.Get() if attr else None
        if not before:
            raise ValueError('Missing delivered table color input')
        attr.Set([Gf.Vec3f(*[v[i]*table[i] for i in range(3)]) for v in before])
        surface = {'path': str(prim.GetPath()), 'attribute': 'primvars:displayColor',
                   'rgb_multiplier': list(table), 'colors': len(before)}
    return {'method': 'Video-derived scene light intensity/color and table material tint',
            'configuration': dict(settings),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'lights': lights, 'table': surface, 'source_assets_modified': False,
            'video_postprocessing': False,
            'limitations': 'Appearance fit to recorded RGB, not a measured lighting or color-chart calibration.'}
