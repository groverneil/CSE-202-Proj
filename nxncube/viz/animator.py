"""Turn a timeline Frame into vertex positions and colours (pure numpy, no renderer)."""

from __future__ import annotations

import numpy as np

from .mesh import StickerMesh, box
from .palette import DEFAULT_RGB
from .timeline import Frame

# Signed quarter turns to animate for turns = 1, 2, 3 (a prime turn goes the short way).
_ANIMATED_QUARTERS = {1: 1, 2: 2, 3: -1}
BODY_SCALE = 0.99  # black core sits just inside the sticker planes at ±1


def rotation(axis: int, quarters: float) -> np.ndarray:
    """Rotation by ``quarters`` clockwise quarter turns seen from the positive end of ``axis``.

    At quarters = 1 this equals ``geometry.QUARTER_TURN[axis]``.
    """
    theta = -quarters * np.pi / 2  # clockwise from +axis = negative right-hand angle
    c, s = np.cos(theta), np.sin(theta)
    b, d = (axis + 1) % 3, (axis + 2) % 3
    R = np.eye(3)
    R[b, b], R[b, d], R[d, b], R[d, d] = c, -s, s, c
    return R


def move_rotation(frame: Frame) -> np.ndarray:
    m = frame.move
    if m is None:
        return np.eye(3)
    return rotation(m.axis, _ANIMATED_QUARTERS[m.turns] * frame.progress)


class Animator:
    def __init__(self, cube, gap: float = 0.1, palette: np.ndarray = DEFAULT_RGB):
        self.cube = cube
        self.layout = cube.layout()
        self.mesh = StickerMesh.from_layout(self.layout, gap)
        self.palette = np.asarray(palette, dtype=float)

    def turning(self, frame: Frame) -> np.ndarray:
        """Boolean mask over stickers carried by the move in progress."""
        if frame.move is None:
            return np.zeros(len(self.mesh.centers), dtype=bool)
        return self.layout.turning(frame.move.axis, frame.move.layers)

    def sticker_vertices(self, frame: Frame) -> np.ndarray:
        v = self.mesh.vertices.copy()
        if frame.move is not None and frame.progress > 0:
            moving = self.turning(frame)[self.mesh.sticker_of_vertex]
            v[moving] = v[moving] @ move_rotation(frame).T
        return v

    def sticker_colors(self, frame: Frame) -> np.ndarray:
        values = np.frombuffer(frame.state, dtype=np.uint8)
        return self.palette[values[self.mesh.sticker_of_vertex]]

    def body(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        """Black core split into slabs along the move's axis; turning slabs rotate too."""
        n, S = self.cube.n, BODY_SCALE
        m = frame.move
        if m is None:
            return box([-S] * 3, [S] * 3)
        R = move_rotation(frame)
        verts, tris = [], []
        layer = 0
        while layer < n:  # group consecutive layers by whether they turn
            turning = layer in m.layers
            hi = layer
            while hi + 1 < n and ((hi + 1) in m.layers) == turning:
                hi += 1
            lo_v, hi_v = np.full(3, -S), np.full(3, S)
            lo_v[m.axis] = -S + 2 * S * layer / n
            hi_v[m.axis] = -S + 2 * S * (hi + 1) / n
            v, t = box(lo_v, hi_v)
            if turning:
                v = v @ R.T
            tris.append(t + 8 * len(verts))
            verts.append(v)
            layer = hi + 1
        return np.concatenate(verts), np.concatenate(tris)
