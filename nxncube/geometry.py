"""Sticker geometry for an n x n x n cube.

Everything about moves is derived from here: sticker points, cubie points and layer
indices. See docs/CONVENTIONS.md for the frame, face order and indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
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


# Face whose outward normal is (axis, sign): _FACE_OF_NORMAL[axis, sign > 0].
_FACE_OF_NORMAL = np.zeros((3, 2), dtype=np.int64)
for _f, (_a, _s) in FACE_NORMAL.items():
    _FACE_OF_NORMAL[_a, int(_s > 0)] = int(_f)
_RIGHT_ARR = np.array([_FACE_RIGHT[f] for f in Face], dtype=np.int64)
_DOWN_ARR = np.array([_FACE_DOWN[f] for f in Face], dtype=np.int64)


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
        )

    @property
    def num_stickers(self) -> int:
        return 6 * self.n * self.n

    def index_of(self, points: np.ndarray) -> np.ndarray:
        """Facelet indices of the given sticker points (shape (k, 3)).

        Computed directly from the point (its ±n coordinate picks the face, the other two
        give row and column), so no position table is needed: that table has (2n+1)³
        entries, which is 64 GB at n = 1000.
        """
        n = self.n
        p = np.asarray(points, dtype=np.int64).reshape(-1, 3)
        if np.abs(p).max(initial=0) > n:
            raise KeyError("point outside the cube")
        on_face = np.abs(p) == n
        if (on_face.sum(axis=1) != 1).any():
            raise KeyError("point is not a sticker position")
        axis = np.argmax(on_face, axis=1)
        sign = p[np.arange(len(p)), axis] > 0
        face = _FACE_OF_NORMAL[axis, sign.astype(np.int64)]
        c = (p * _RIGHT_ARR[face]).sum(axis=1) + (n - 1)
        r = (p * _DOWN_ARR[face]).sum(axis=1) + (n - 1)
        if ((c % 2) | (r % 2)).any():
            raise KeyError("point is not a sticker position")
        return face * n * n + (r // 2) * n + c // 2

    def facelet(self, face: Face, row: int, col: int) -> int:
        return int(face) * self.n * self.n + row * self.n + col

    @cached_property
    def _layer_index(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Per axis: sticker indices sorted by layer, and where each layer starts."""
        idx_dtype = np.int32 if self.num_stickers < 2**31 else np.int64
        out = []
        for axis in Axis:
            key = self.layers[:, axis]
            order = np.argsort(key, kind="stable").astype(idx_dtype)
            bounds = np.searchsorted(key[order], np.arange(self.n + 1))
            out.append((order, bounds))
        return tuple(out)

    def stickers_in_layers(self, axis: Axis, layers: Iterable[int]) -> np.ndarray:
        """Facelet indices of every sticker in ``layers`` on ``axis`` (ascending)."""
        order, bounds = self._layer_index[axis]
        parts = [order[bounds[l]:bounds[l + 1]] for l in sorted(set(layers))]
        if not parts:
            return np.zeros(0, dtype=order.dtype)
        return np.sort(np.concatenate(parts))

    def quarter_turn_cycle(self, axis: Axis, layers: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Sparse form of ``quarter_turn_perm``: ``(dst, src)`` with ``new[dst] = old[src]``.

        Only the stickers in the turning layers appear, so its size is O(n) for an inner
        layer and O(n²) for an outer one instead of 6n².
        """
        src = self.stickers_in_layers(axis, layers)
        dst = self.index_of(self.points[src] @ QUARTER_TURN[axis].T).astype(src.dtype)
        return dst, src

    def quarter_turn_perm(self, axis: Axis, layers: Iterable[int]) -> np.ndarray:
        """Permutation for one clockwise quarter turn of ``layers`` about ``axis``.

        ``new_state = state[perm]``: ``perm[i]`` is where the sticker now at ``i`` came from.
        """
        dst, src = self.quarter_turn_cycle(axis, layers)
        perm = np.arange(self.num_stickers)
        perm[dst] = src
        return perm
