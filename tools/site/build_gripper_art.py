#!/usr/bin/env python3
"""Build presentation-only point/edge samples from the pinned R5 manufacturer CAD.

Uses only the Python standard library. Does not modify runtime robot assets.
Run from any directory: python tools/site/build_gripper_art.py
"""
from pathlib import Path
import bisect
import hashlib
import json
import math
import random
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/media/gripper'
OUT.mkdir(parents=True, exist_ok=True)
URDF = ROOT / 'assets/r5/R5a/urdf/R5a.urdf'
robot = ET.parse(URDF).getroot()
rng = random.Random(20261025)
parts = []
sources = []

def minus(a, b):
    return [a[i] - b[i] for i in range(3)]

def length(a):
    return math.sqrt(sum(x * x for x in a))

for link, count in [('link6', 3800), ('link7', 1800), ('link8', 1800)]:
    path = ROOT / f'assets/r5/R5a/meshes/{link}.STL'
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw, 80)[0]
    assert len(raw) == 84 + n * 50, 'Expected binary STL'
    triangles, areas, topology = [], [], {}
    total = 0
    for i in range(n):
        values = struct.unpack_from('<12fH', raw, 84 + i * 50)
        vertices = [tuple(round(x * 1000, 4) for x in values[k:k+3]) for k in (3, 6, 9)]
        a, b, c = vertices
        u, v = minus(b, a), minus(c, a)
        normal = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        size = length(normal)
        if size < 1e-8:
            continue
        normal = [x / size for x in normal]
        total += size / 2
        triangles.append((vertices, normal))
        areas.append(total)
        for j in range(3):
            edge = tuple(sorted((vertices[j], vertices[(j+1) % 3])))
            topology.setdefault(edge, []).append(normal)
    points = []
    for _ in range(count):
        vertices, normal = triangles[bisect.bisect_left(areas, rng.random() * total)]
        a, b, c = vertices
        u, v = math.sqrt(rng.random()), rng.random()
        point = [(1-u)*a[k] + u*(1-v)*b[k] + u*v*c[k] for k in range(3)]
        points.extend([round(x, 2) for x in point] + [round(x, 2) for x in normal])
    edges = []
    for edge, normals in topology.items():
        if length(minus(*edge)) < 1.2:
            continue
        sharp = len(normals) == 1 or any(sum(normals[0][k]*other[k] for k in range(3)) < .70 for other in normals[1:])
        if sharp:
            edges.extend(round(x, 2) for vertex in edge for x in vertex)
    if link == 'link6':
        origin, axis, limits = [0, 0, 0], [0, 0, 0], [0, 0]
    else:
        joint = robot.find(f"joint[@name='joint{link[-1]}']")
        assert joint.find('origin').get('rpy') == '0 0 0'
        origin = [float(x)*1000 for x in joint.find('origin').get('xyz').split()]
        axis = [float(x) for x in joint.find('axis').get('xyz').split()]
        limits = [float(joint.find('limit').get(x))*1000 for x in ('lower', 'upper')]
    parts.append(dict(name=link, origin=origin, axis=axis, limits=limits, points=points, edges=edges))
    sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), triangles=n, sampled_points=count, feature_edges=len(edges)//6))
model = dict(units='millimetres', parts=parts)
(OUT / 'model.js').write_text('// Derived from pinned ARX R5a CAD. See sources.json.\nwindow.R5_GRIPPER_MODEL=' + json.dumps(model, separators=(',', ':')) + ';\n')
(OUT / 'sources.json').write_text(json.dumps(dict(
    description='Stylized point-and-line illustration sampled from the existing ARX R5a manufacturer meshes. Presentation only; not an experimental capture or a calibrated hardware measurement.',
    repository='https://github.com/ARXroboticsX/ARX_Model', commit='74386441845f12af4da203843b6400951d6c5530',
    original_license='BSD, as declared in the retained manufacturer package.xml',
    generator='tools/site/build_gripper_art.py', urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(), sources=sources,
    articulation='Joint origins, axes and exported limits retained from URDF. UI opening is a normalized illustration control, not a measured fingertip width.'
), indent=2) + '\n')
# Static, resolution-independent artwork for no-JS fallback and reuse.
def project(raw, part):
    p = [raw[i] + part['origin'][i] + part['axis'][i]*32 for i in range(3)]
    x, y, z = p[1], p[0]-78, p[2]
    yaw, pitch, roll = -.46, -.57, -.35
    x, z = x*math.cos(yaw)+z*math.sin(yaw), -x*math.sin(yaw)+z*math.cos(yaw)
    y, z = y*math.cos(pitch)-z*math.sin(pitch), y*math.sin(pitch)+z*math.cos(pitch)
    x, y = x*math.cos(roll)-y*math.sin(roll), x*math.sin(roll)+y*math.cos(roll)
    scale = 2.0*650/(650-z)
    return 340+x*scale, 235+y*scale, z
svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 470" role="img" aria-labelledby="title"><title id="title">ARX R5 two-finger gripper, illustrated with points and contour lines from manufacturer CAD</title>']
for part, color in zip(parts, ['#a69ce9', '#81e3e4', '#c9b1ff']):
    edges = part['edges']
    lines = []
    for i in range(0, len(edges), 6):
        a, b = project(edges[i:i+3], part), project(edges[i+3:i+6], part)
        lines.append(f'M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}')
    svg.append(f'<path fill="none" stroke="{color}" stroke-width=".6" opacity=".24" d="'+''.join(lines)+'"/>')
    points = part['points']
    svg.append(f'<g fill="{color}" opacity=".72">')
    for i in range(0, len(points), 12):
        x, y, z = project(points[i:i+3], part)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r=".9"/>')
    svg.append('</g>')
svg.append('</svg>')
(OUT / 'r5-point-lines.svg').write_text(''.join(svg))
print(json.dumps({'points':sum(len(p['points'])//6 for p in parts),'edges':sum(len(p['edges'])//6 for p in parts),'model_bytes':(OUT/'model.js').stat().st_size}))
