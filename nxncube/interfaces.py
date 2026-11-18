"""Protocols that keep cubes, solvers, scramblers and viewers independent."""

from __future__ import annotations

import random
from typing import Protocol, Sequence, runtime_checkable

from .layout import Layout
from .move import Move
from .pieces import Orbit, PieceState

State = bytes


@runtime_checkable
class Puzzle(Protocol):
    """Tier 1: any permutation puzzle."""

    def solved(self) -> State: ...
    def moves(self) -> Sequence[Move]: ...
    def apply(self, s: State, m: Move) -> State: ...
    def is_solved(self, s: State) -> bool: ...
    def inverse(self, m: Move) -> Move: ...


@runtime_checkable
class CubeView(Puzzle, Protocol):
    """Tier 2: exposes cube structure for structured solvers."""

    n: int

    def orbits(self) -> Sequence[Orbit]: ...
    def pieces(self, s: State) -> PieceState: ...


class Solver(Protocol):
    def solve(self, puzzle: Puzzle, s: State) -> list[Move]: ...


class Scrambler(Protocol):
    def scramble(
        self, puzzle: Puzzle, rng: random.Random, length: int | None = None
    ) -> list[Move]: ...


class Viewer(Protocol):
    def load(self, layout: Layout) -> None: ...
    def show(self, s: State) -> None: ...
    def animate(self, s: State, m: Move) -> None: ...
