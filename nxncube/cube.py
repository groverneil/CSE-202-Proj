"""The n x n x n cube as a permutation puzzle (implements Puzzle and CubeView)."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np

from .geometry import Axis, Face, Geometry
from .layout import Layout, build_layout
from .move import Move
from .pieces import Orbit, PieceState, compute_orbits
from .scramble import RandomMoveScrambler

# A state is an immutable byte string of 6n² sticker values (0-5, see Face), so it can be
# a dictionary key during search.
State = bytes


class Cube:
    """All tables for one n: geometry, move permutations, orbits and layout.

    Holds no solving logic. States are plain ``bytes`` values; nothing here mutates them.
    """

    def __init__(self, n: int, seed: int | None = None, scrambler=None):
        self.n = n
        self.geometry = Geometry.build(n)
        self.rng = random.Random(seed)
        self.scrambler = scrambler if scrambler is not None else RandomMoveScrambler()
        self._perms: dict[Move, np.ndarray] = {}
        self._moves = tuple(
            Move(axis, layer, t) for axis in Axis for layer in range(n) for t in (1, 2, 3)
        )
        self._all_moves: tuple[Move, ...] | None = None
        self._solved = np.repeat(np.arange(6, dtype=np.uint8), n * n).tobytes()
        self._orbits: tuple[Orbit, ...] | None = None
        self._layout: Layout | None = None

    def __repr__(self) -> str:
        return f"Cube(n={self.n})"

    @property
    def num_stickers(self) -> int:
        return 6 * self.n * self.n

    # ---- tier 1: Puzzle -------------------------------------------------------------

    def solved(self) -> State:
        return self._solved

    def moves(self) -> Sequence[Move]:
        """The 9n single-layer primitives (axis, {layer}, turns)."""
        return self._moves

    def all_moves(self) -> Sequence[Move]:
        """Every contiguous block of layers on every axis, every turn amount: single
        layers, wide turns, slice blocks and whole-cube rotations (9n(n+1)/2 moves)."""
        if self._all_moves is None:
            n = self.n
            self._all_moves = tuple(
                Move(axis, range(lo, hi + 1), t)
                for axis in Axis
                for lo in range(n)
                for hi in range(lo, n)
                for t in (1, 2, 3)
            )
        return self._all_moves

    def apply(self, s: State, m: Move) -> State:
        return np.frombuffer(s, dtype=np.uint8)[self.perm(m)].tobytes()

    def is_solved(self, s: State) -> bool:
        """Every face shows one colour (any of the 24 orientations counts)."""
        faces = np.frombuffer(s, dtype=np.uint8).reshape(6, -1)
        return bool((faces == faces[:, :1]).all())

    def inverse(self, m: Move) -> Move:
        return m.inverse()

    # ---- tier 2: CubeView -----------------------------------------------------------

    def orbits(self) -> Sequence[Orbit]:
        if self._orbits is None:
            self._orbits = compute_orbits(self.geometry)
        return self._orbits

    def pieces(self, s: State) -> PieceState:
        arr = np.frombuffer(s, dtype=np.uint8)
        orbits = tuple(self.orbits())
        return PieceState(orbits, tuple(arr[o.facelets] for o in orbits))

    # ---- helpers ----------------------------------------------------------------------

    def validate(self, m: Move) -> None:
        if max(m.layers) >= self.n:
            raise ValueError(f"{m!r} has a layer outside 0..{self.n - 1}")

    def perm(self, m: Move) -> np.ndarray:
        """Read-only sticker permutation of ``m`` (``new = state[perm]``), cached."""
        p = self._perms.get(m)
        if p is None:
            self.validate(m)
            q = self.geometry.quarter_turn_perm(m.axis, m.layers)
            p = q
            for _ in range(m.turns - 1):
                p = p[q]
            p.setflags(write=False)
            self._perms[m] = p
        return p

    def sequence_perm(self, moves: Iterable[Move]) -> np.ndarray:
        """Single permutation equal to applying ``moves`` in order."""
        p = np.arange(self.num_stickers)
        for m in moves:
            p = p[self.perm(m)]
        return p

    def apply_moves(self, s: State, moves: Iterable[Move]) -> State:
        return np.frombuffer(s, dtype=np.uint8)[self.sequence_perm(moves)].tobytes()

    def rotation(self, axis: Axis | int, turns: int = 1) -> Move:
        """Whole-cube rotation x, y or z."""
        return Move(axis, range(self.n), turns)

    def is_rotation(self, m: Move) -> bool:
        return len(m.layers) == self.n

    def facelet(self, face: Face, row: int, col: int) -> int:
        return self.geometry.facelet(face, row, col)

    def layout(self) -> Layout:
        if self._layout is None:
            self._layout = build_layout(self.geometry)
        return self._layout

    def scramble(self, length: int | None = None, rng: random.Random | None = None) -> list[Move]:
        """Random move list from this cube's scrambler; uses the cube's seeded RNG by default."""
        return self.scrambler.scramble(self, rng if rng is not None else self.rng, length)
