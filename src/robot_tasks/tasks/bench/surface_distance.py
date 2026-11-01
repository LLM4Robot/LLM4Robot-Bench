"""Bounded point-to-triangle clearance without treating empty mesh bounds as contact."""

import numpy as np


def surface_clearance(points, triangles, max_distance=0.01):
    """Minimum sampled-surface separation, capped at max_distance (metres).

    The cap is above every caller's near/release threshold. AABB broad phase
    only selects candidates; it never certifies contact inside empty bounds.
    This is unsigned geometry, not a contact-force or grasp measurement.
    """
    points, triangles = np.asarray(points), np.asarray(triangles)
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    best = float(max_distance)
    gap = np.maximum(
        np.maximum(low - points.max(axis=0), points.min(axis=0) - high), 0.0
    )
    nearby = np.sum(gap * gap, axis=1) < best * best
    if not np.any(nearby):
        return best
    triangles, low, high = triangles[nearby], low[nearby], high[nearby]

    def dot(x, y):
        return np.einsum("ij,ij->i", x, y)

    for chunk in np.array_split(points, max(1, (len(points) + 63) // 64)):
        gap = np.maximum(
            np.maximum(low[None] - chunk[:, None], chunk[:, None] - high[None]), 0.0
        )
        pi, ti = np.where(np.sum(gap * gap, axis=2) < best * best)
        if not len(pi):
            continue
        p = chunk[pi]
        a, b, c = triangles[ti].transpose(1, 0, 2)
        ab, ac, ap = b - a, c - a, p - a
        aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
        pa, pc = dot(ap, ab), dot(ap, ac)
        denominator = aa * cc - bb * bb
        valid = denominator > 1e-24
        safe = np.where(valid, denominator, 1.0)
        u, v = (cc * pa - bb * pc) / safe, (aa * pc - bb * pa) / safe
        inside = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
        delta = ap - u[:, None] * ab - v[:, None] * ac
        distance2 = np.where(inside, dot(delta, delta), np.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            fraction = np.clip(
                dot(p - start, edge) / np.maximum(dot(edge, edge), 1e-30), 0.0, 1.0
            )
            delta = p - start - fraction[:, None] * edge
            distance2 = np.minimum(distance2, dot(delta, delta))
        best = min(best, float(np.sqrt(distance2.min())))
    return best


def link_triangles(stage, link_path):
    """Delivered collision source mesh transformed into the rigid link frame."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    link = stage.GetPrimAtPath(link_path)
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(link)).T)
    triangles = []
    root = stage.GetPrimAtPath(link_path + "/collisions")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        t = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices = np.asarray(mesh.GetPointsAttr().Get()) @ t[:3, :3].T + t[:3, 3]
        indices, cursor = mesh.GetFaceVertexIndicesAttr().Get(), 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[cursor : cursor + count]
            triangles.extend(
                vertices[[face[0], face[i], face[i + 1]]] for i in range(1, count - 1)
            )
            cursor += count
    if not triangles:
        raise ValueError("Missing collision surface for " + link_path)
    return np.asarray(triangles)
