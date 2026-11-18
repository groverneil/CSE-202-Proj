"""Piece view (tier 2): cubie orbits derived from the same geometry as the moves."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .geometry import QUARTER_TURN, Axis, Geometry

# Display order of orbit kinds.
KINDS = ("corner", "middle_edge", "wing", "center", "fixed_center")


@dataclass(frozen=True, eq=False)
class Orbit:
    """A set of cubie positions that single-layer turns permute among themselves.

    ``facelets[i]`` lists the sticker indices of ``cubies[i]``, ordered by the axis of
    the sticker's normal (x, y, z).
    """

    kind: str
    cubies: tuple[tuple[int, int, int], ...]
    facelets: np.ndarray  # (len(cubies), stickers per cubie)

    @property
    def size(self) -> int:
        return len(self.cubies)


@dataclass(frozen=True, eq=False)
class PieceState:
    """Colours of every orbit's facelets for one state: ``colors[k]`` matches ``orbits[k]``."""

    orbits: tuple[Orbit, ...]
    colors: tuple[np.ndarray, ...]


def _kind(cubie: tuple[int, int, int], n: int) -> str:
    extreme = [abs(c) == n - 1 for c in cubie]
    free = [c for c, e in zip(cubie, extreme) if not e]
    k = sum(extreme)
    if k == 3:
        return "corner"
    if k == 2:
        return "middle_edge" if free[0] == 0 else "wing"
    if k == 1:
        return "fixed_center" if free == [0, 0] else "center"
    raise AssertionError(f"interior cubie {cubie} has stickers")


def compute_orbits(geo: Geometry) -> tuple[Orbit, ...]:
    n = geo.n
    # Stickers of each cubie, ordered by normal axis.
    normal_axis = np.argmax(np.abs(geo.normals), axis=1)
    stickers: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in np.lexsort((normal_axis,)):
        stickers[tuple(int(c) for c in geo.cubies[i])].append(int(i))

    cubies = sorted(stickers)
    pos = {c: i for i, c in enumerate(cubies)}
    parent = list(range(len(cubies)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Every cubie lies in one layer per axis, and the quarter turn of that layer moves it
    # by the axis rotation, so orbits are the components of c -- R_axis c.
    arr = np.array(cubies)
    for axis in Axis:
        rotated = arr @ QUARTER_TURN[axis].T
        for i, r in enumerate(rotated):
            a, b = find(i), find(pos[tuple(int(x) for x in r)])
            if a != b:
                parent[a] = b

    groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for c in cubies:
        groups[find(pos[c])].append(c)

    orbits = []
    for members in groups.values():
        kind = _kind(members[0], n)
        assert all(_kind(c, n) == kind for c in members)
        orbits.append(
            Orbit(
                kind=kind,
                cubies=tuple(members),
                facelets=np.array([stickers[c] for c in members], dtype=np.int64),
            )
        )
    orbits.sort(key=lambda o: (KINDS.index(o.kind), sorted(map(abs, o.cubies[0]))))
    return tuple(orbits)
