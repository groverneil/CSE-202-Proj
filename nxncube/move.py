"""The Move value type and the move-pruning rule shared by scramblers and search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .geometry import Axis


@dataclass(frozen=True)
class Move:
    """Rotate ``layers`` about ``axis`` by ``turns`` clockwise quarter turns.

    Clockwise is as seen from the positive end of the axis (R, U, F side).
    Layer bounds are checked by the cube, since a Move does not know ``n``.
    """

    axis: Axis
    layers: frozenset[int]
    turns: int

    def __init__(self, axis: Axis | int, layers: Iterable[int] | int, turns: int = 1):
        if isinstance(layers, int):
            layers = (layers,)
        layers = frozenset(int(l) for l in layers)
        if not layers:
            raise ValueError("a move needs at least one layer")
        if min(layers) < 0:
            raise ValueError(f"negative layer in {sorted(layers)}")
        if turns not in (1, 2, 3):
            raise ValueError(f"turns must be 1, 2 or 3, got {turns}")
        object.__setattr__(self, "axis", Axis(axis))
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "turns", turns)

    def inverse(self) -> "Move":
        return Move(self.axis, self.layers, 4 - self.turns)

    def __repr__(self) -> str:
        return f"Move({self.axis}, {sorted(self.layers)}, {self.turns})"


def can_follow(prev: Move | None, nxt: Move) -> bool:
    """Pruning rule: is ``nxt`` allowed directly after ``prev``?

    Same-axis moves commute, so a run of them is only allowed in strictly increasing
    layer order. This forbids turning the same layer twice in a row and forbids both
    orders of a commuting pair.
    """
    if prev is None or prev.axis != nxt.axis:
        return True
    return min(nxt.layers) > max(prev.layers)
