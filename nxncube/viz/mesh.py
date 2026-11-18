"""Unflatten the facelet array into 3D: one square quad per sticker, merged into one mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..layout import Layout


@dataclass(frozen=True, eq=False)
class StickerMesh:
    """Rest-position geometry of every sticker, in units where the cube spans [-1, 1]³.

    Sticker ``i`` (facelet index) owns vertices ``4i .. 4i+3`` and triangles ``2i, 2i+1``.
    """

    n: int
    vertices: np.ndarray           # (4N, 3) float
    triangles: np.ndarray          # (2N, 3) int, wound so normals point outward
    sticker_of_vertex: np.ndarray  # (4N,) facelet index of each vertex
    centers: np.ndarray            # (N, 3) sticker centres (same as layout.center)
    normals: np.ndarray            # (N, 3) outward unit normals

    @classmethod
    def from_layout(cls, layout: Layout, gap: float = 0.1) -> "StickerMesh":
        if not 0 <= gap < 1:
            raise ValueError("gap must be in [0, 1)")
        n = layout.n
        centers = np.asarray(layout.center, dtype=float)
        normals = np.asarray(layout.normal, dtype=float)
        N = len(centers)
        half = (1.0 - gap) / n  # stickers are 2/n apart

        axis = np.argmax(np.abs(normals), axis=1)
        sign = normals[np.arange(N), axis]
        # Tangent axes (a+1, a+2) are cyclic, so e1 x e2 = +normal axis; swap on negative faces.
        e1_axis = np.where(sign > 0, (axis + 1) % 3, (axis + 2) % 3)
        e2_axis = np.where(sign > 0, (axis + 2) % 3, (axis + 1) % 3)
        e1 = np.eye(3)[e1_axis] * half
        e2 = np.eye(3)[e2_axis] * half

        quads = np.stack(
            [centers - e1 - e2, centers + e1 - e2, centers + e1 + e2, centers - e1 + e2], axis=1
        )  # counter-clockwise seen from outside
        base = 4 * np.arange(N)[:, None]
        triangles = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], axis=1).reshape(-1, 3)
        return cls(
            n=n,
            vertices=quads.reshape(-1, 3),
            triangles=triangles.astype(np.int32),
            sticker_of_vertex=np.repeat(np.arange(N), 4),
            centers=centers,
            normals=normals,
        )


def box(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (8, 3) and outward-wound triangles (12, 3) of an axis-aligned box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[(hi if (i >> a) & 1 else lo)[a] for a in range(3)] for i in range(8)])
    tris = []
    for a in range(3):
        for side in (0, 1):
            face = [i for i in range(8) if (i >> a) & 1 == side]  # 4 corners, bit order
            b, c = [x for x in range(3) if x != a]
            # order the 4 corners around the face: (0,0) (1,0) (1,1) (0,1) in (b, c) bits
            ring = sorted(face, key=lambda i: [0, 1, 3, 2][((i >> b) & 1) + 2 * ((i >> c) & 1)])
            t1, t2 = [ring[0], ring[1], ring[2]], [ring[0], ring[2], ring[3]]
            v = corners
            normal = np.cross(v[t1[1]] - v[t1[0]], v[t1[2]] - v[t1[0]])
            outward = 1 if side else -1
            if normal[a] * outward < 0:
                t1, t2 = t1[::-1], t2[::-1]
            tris += [t1, t2]
    return corners, np.array(tris, dtype=np.int32)
