"""Sticker geometry for an n x n x n cube.

Everything about moves is derived from here: sticker points, cubie points and layer
indices. See docs/CONVENTIONS.md for the frame, face order and indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2

    def __str__(self) -> str:
        return self.name.lower()


class Face(IntEnum):
    U = 0
    R = 1
    F = 2
    D = 3
    L = 4
    B = 5

    def __str__(self) -> str:
        return self.name


# Outward normal of each face as (axis, sign).
FACE_NORMAL: dict[Face, tuple[Axis, int]] = {
    Face.U: (Axis.Y, +1),
    Face.R: (Axis.X, +1),
    Face.F: (Axis.Z, +1),
    Face.D: (Axis.Y, -1),
    Face.L: (Axis.X, -1),
    Face.B: (Axis.Z, -1),
}

# Direction of increasing column / row when the face is viewed from outside in the net.
_FACE_RIGHT = {
    Face.U: (1, 0, 0),
    Face.R: (0, 0, -1),
    Face.F: (1, 0, 0),
    Face.D: (1, 0, 0),
    Face.L: (0, 0, 1),
    Face.B: (-1, 0, 0),
}
_FACE_DOWN = {
    Face.U: (0, 0, 1),
    Face.R: (0, -1, 0),
    Face.F: (0, -1, 0),
    Face.D: (0, 0, -1),
    Face.L: (0, -1, 0),
    Face.B: (0, -1, 0),
}

# Quarter turn clockwise as seen from the positive end of each axis (-90 degrees).
QUARTER_TURN: dict[Axis, np.ndarray] = {
    Axis.X: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    Axis.Y: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    Axis.Z: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
}


def _unit(axis: Axis, sign: int) -> np.ndarray:
    v = np.zeros(3, dtype=np.int64)
    v[axis] = sign
    return v


@dataclass(frozen=True, eq=False)
class Geometry:
    """Per-sticker geometry, indexed by facelet index ``face * n² + row * n + col``."""

    n: int
    face: np.ndarray    # (N,) face of each sticker
    row: np.ndarray     # (N,)
    col: np.ndarray     # (N,)
    points: np.ndarray  # (N, 3) sticker position, face coordinate at ±n
    normals: np.ndarray  # (N, 3) outward unit normal
    cubies: np.ndarray  # (N, 3) position of the cubie the sticker belongs to
    layers: np.ndarray  # (N, 3) layer index of the sticker on each axis
    _lookup: np.ndarray  # dense position -> index table

    @classmethod
    def build(cls, n: int) -> "Geometry":
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        rows, cols = rows.ravel(), cols.ravel()
        offs_c = 2 * cols - (n - 1)
        offs_r = 2 * rows - (n - 1)

        faces, rr, cc, pts, nrm = [], [], [], [], []
        for f in Face:  # Face order fixes the index order.
            axis, sign = FACE_NORMAL[f]
            normal = _unit(axis, sign)
            right = np.array(_FACE_RIGHT[f])
            down = np.array(_FACE_DOWN[f])
            p = n * normal + offs_c[:, None] * right + offs_r[:, None] * down
            faces.append(np.full(n * n, int(f)))
            rr.append(rows)
            cc.append(cols)
            pts.append(p)
            nrm.append(np.broadcast_to(normal, p.shape))

        points = np.concatenate(pts).astype(np.int64)
        cubies = np.clip(points, -(n - 1), n - 1)
        layers = (cubies + (n - 1)) // 2

        size = 2 * n + 1
        lookup = np.full(size**3, -1, dtype=np.int64)
        lookup[cls._key(points, n)] = np.arange(len(points))
        if (lookup >= 0).sum() != 6 * n * n:
            raise AssertionError("sticker points are not distinct")

        for arr in (points, cubies, layers):
            arr.setflags(write=False)
        return cls(
            n=n,
            face=np.concatenate(faces),
            row=np.concatenate(rr),
            col=np.concatenate(cc),
            points=points,
            normals=np.concatenate(nrm).astype(np.int64),
            cubies=cubies,
            layers=layers,
            _lookup=lookup,
        )

    @staticmethod
    def _key(points: np.ndarray, n: int) -> np.ndarray:
        size = 2 * n + 1
        p = points + n
        return (p[:, 0] * size + p[:, 1]) * size + p[:, 2]

    @property
    def num_stickers(self) -> int:
        return 6 * self.n * self.n

    def index_of(self, points: np.ndarray) -> np.ndarray:
        """Facelet indices of the given sticker points (shape (k, 3))."""
        points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
        if np.abs(points).max(initial=0) > self.n:
            raise KeyError("point outside the cube")
        idx = self._lookup[self._key(points, self.n)]
        if (idx < 0).any():
            raise KeyError("point is not a sticker position")
        return idx

    def facelet(self, face: Face, row: int, col: int) -> int:
        return int(face) * self.n * self.n + row * self.n + col

    def quarter_turn_perm(self, axis: Axis, layers: Iterable[int]) -> np.ndarray:
        """Permutation for one clockwise quarter turn of ``layers`` about ``axis``.

        ``new_state = state[perm]``: ``perm[i]`` is where the sticker now at ``i`` came from.
        """
        layers = list(layers)
        src = np.flatnonzero(np.isin(self.layers[:, axis], layers))
        dst = self.index_of(self.points[src] @ QUARTER_TURN[axis].T)
        perm = np.arange(self.num_stickers)
        perm[dst] = src
        return perm
