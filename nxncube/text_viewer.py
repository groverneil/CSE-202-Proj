"""Text net viewer: a Viewer implementation and a debugging aid for the move generator."""

from __future__ import annotations

import sys
from typing import TextIO

import numpy as np

from .layout import Layout
from .move import Move

# Face values 0-5 are U R F D L B; the palette is a viewer concern.
DEFAULT_PALETTE = ("W", "R", "G", "Y", "O", "B")

# Net position (block row, block column) of each face value.
_NET = {0: (0, 1), 4: (1, 0), 2: (1, 1), 1: (1, 2), 5: (1, 3), 3: (2, 1)}


class TextNetViewer:
    def __init__(self, palette: tuple[str, ...] = DEFAULT_PALETTE, out: TextIO | None = None):
        self.palette = palette
        self.out = out
        self.layout: Layout | None = None

    def load(self, layout: Layout) -> None:
        self.layout = layout

    def render(self, s: bytes, highlight: np.ndarray | None = None) -> str:
        """Net as text. Stickers in ``highlight`` (a boolean mask) are shown in lowercase."""
        if self.layout is None:
            raise RuntimeError("call load() first")
        lay = self.layout
        n = lay.n
        values = np.frombuffer(s, dtype=np.uint8)
        grid = [[" "] * (4 * (2 * n + 1)) for _ in range(3 * n)]
        for i, v in enumerate(values):
            br, bc = _NET[int(lay.face[i])]
            ch = self.palette[v]
            if highlight is not None and highlight[i]:
                ch = ch.lower()
            grid[br * n + lay.row[i]][bc * (2 * n + 1) + 2 * lay.col[i]] = ch
        return "\n".join("".join(line).rstrip() for line in grid)

    def show(self, s: bytes) -> None:
        print(self.render(s), file=self.out or sys.stdout)

    def animate(self, s: bytes, m: Move) -> None:
        """A text net cannot animate; it shows ``s`` with the stickers ``m`` turns in lowercase."""
        assert self.layout is not None
        print(self.render(s, self.layout.turning(m.axis, m.layers)), file=self.out or sys.stdout)
