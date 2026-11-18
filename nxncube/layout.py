"""Read-only layout export for viewers (built once per n from the move geometry)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry


@dataclass(frozen=True, eq=False)
class Layout:
    """Per-sticker data, indexed by facelet index.

    ``face``/``row``/``col`` place the sticker in the standard net (already oriented as
    seen from outside). ``center`` is the 3D sticker centre scaled so the cube spans
    [-1, 1]; ``normal`` is the outward unit normal; ``layer[:, axis]`` is the layer index
    used to decide which stickers a move turns.
    """

    n: int
    face: np.ndarray
    row: np.ndarray
    col: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    layer: np.ndarray

    def turning(self, axis: int, layers) -> np.ndarray:
        """Boolean mask of the stickers a move on ``axis``/``layers`` carries."""
        return np.isin(self.layer[:, axis], list(layers))


def build_layout(geo: Geometry) -> Layout:
    return Layout(
        n=geo.n,
        face=geo.face.copy(),
        row=geo.row.copy(),
        col=geo.col.copy(),
        center=geo.points / geo.n,
        normal=geo.normals.copy(),
        layer=geo.layers.copy(),
    )
