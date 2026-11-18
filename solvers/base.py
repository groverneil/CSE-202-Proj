"""Base class for every solver. Contains no solving logic.

Subclasses implement one method::

    class MySolver(Solver):
        name = "my-solver"

        def _solve(self, cube, state):
            ...                      # use cube.n, cube.moves(), cube.apply(), cube.orbits() ...
            return moves             # list[Move], or [(phase_label, list[Move]), ...]

Rules for subclasses:
- Size agnostic: a solver is constructed without ``n`` and must work out everything it
  needs from the ``cube`` it is given. One instance may solve cubes of different sizes.
- ``_solve`` sees the cube and the scrambled *state* only, never the scramble, so a
  solver cannot just undo the scramble.
- Never mutate anything: states are immutable bytes; return a new move list.

``solve()`` is the fixed public entry point: it validates the scramble, times ``_solve``,
checks the answer and records metrics.
"""

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence, Union

from nxncube.move import Move
from nxncube.notation import parse

MoveList = Sequence[Move]
Phases = Sequence[tuple[str, Sequence[Move]]]


def quarter_turns(moves: MoveList) -> int:
    """Quarter-turn metric: a half turn counts 2, every other turn counts 1."""
    return sum(2 if m.turns == 2 else 1 for m in moves)


@dataclass(frozen=True)
class SolveResult:
    solver: str
    n: int
    scramble: tuple[Move, ...]
    moves: tuple[Move, ...]                         # the solution, all phases joined
    phases: tuple[tuple[str, tuple[Move, ...]], ...]  # one ("solution", moves) if unphased
    seconds: float                                  # wall-clock time in _solve
    cpu_seconds: float                              # process CPU time in _solve
    legal: bool
    correct: bool
    info: dict[str, Any] = field(default_factory=dict)  # solver-specific extras

    @property
    def length(self) -> int:
        """Number of moves (each Move counts once, whatever its turns or width)."""
        return len(self.moves)

    @property
    def qtm(self) -> int:
        return quarter_turns(self.moves)

    def summary(self) -> str:
        status = "ok" if self.correct else ("WRONG" if self.legal else "ILLEGAL")
        return (f"{self.solver} n={self.n}: {status}, {self.length} moves "
                f"({self.qtm} qt), {self.seconds * 1000:.1f} ms")


class Solver(ABC):
    """Abstract solver. Override ``_solve``; call ``solve``."""

    name: str = ""

    def __init__(self) -> None:
        self.history: list[SolveResult] = []
        self.info: dict[str, Any] = {}  # _solve may fill this; copied into the result

    # ---- to override ------------------------------------------------------------------

    @abstractmethod
    def _solve(self, cube, state: bytes) -> Union[MoveList, Phases]:
        """Return moves that take ``state`` to solved on ``cube``, either as one list
        or as labelled phases ``[(label, moves), ...]``."""

    # ---- checks -------------------------------------------------------------------------

    @staticmethod
    def is_legal(cube, moves: MoveList) -> bool:
        """Every element is a Move whose layers exist on this cube."""
        try:
            for m in moves:
                if not isinstance(m, Move):
                    return False
                cube.validate(m)
        except (TypeError, ValueError):
            return False
        return True

    @classmethod
    def is_correct(cls, cube, scramble: MoveList | str, moves: MoveList) -> bool:
        """Legal, and scramble followed by ``moves`` leaves the cube solved
        (in any orientation, as ``cube.is_solved`` defines it)."""
        scramble = cls._scramble_moves(cube, scramble)
        if not cls.is_legal(cube, moves):
            return False
        state = cube.apply_moves(cube.apply_moves(cube.solved(), scramble), moves)
        return cube.is_solved(state)

    # ---- entry point --------------------------------------------------------------------

    def solve(self, cube, scramble: MoveList | str) -> SolveResult:
        """Solve ``cube`` after ``scramble`` (moves or a notation string); time it; record it."""
        scramble = self._scramble_moves(cube, scramble)
        state = cube.apply_moves(cube.solved(), scramble)

        self.info = {}
        wall0, cpu0 = time.perf_counter(), time.process_time()
        raw = self._solve(cube, state)
        seconds, cpu = time.perf_counter() - wall0, time.process_time() - cpu0

        phases = self._normalise(raw)
        moves = tuple(m for _, ms in phases for m in ms)
        legal = self.is_legal(cube, moves)
        correct = legal and cube.is_solved(cube.apply_moves(state, moves))
        result = SolveResult(
            solver=self.name or type(self).__name__,
            n=cube.n,
            scramble=tuple(scramble),
            moves=moves,
            phases=phases,
            seconds=seconds,
            cpu_seconds=cpu,
            legal=legal,
            correct=correct,
            info=dict(self.info),
        )
        self.history.append(result)
        return result

    # ---- metrics ------------------------------------------------------------------------

    def metrics(self, n: int | None = None) -> dict[str, Any]:
        """Summary of recorded solves (optionally only size ``n``)."""
        rs = [r for r in self.history if n is None or r.n == n]
        if not rs:
            return {"solves": 0}
        times = [r.seconds for r in rs]
        lengths = [r.length for r in rs]
        return {
            "solves": len(rs),
            "correct": sum(r.correct for r in rs),
            "success_rate": sum(r.correct for r in rs) / len(rs),
            "time_total": sum(times),
            "time_mean": statistics.fmean(times),
            "time_max": max(times),
            "length_mean": statistics.fmean(lengths),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "qtm_mean": statistics.fmean(r.qtm for r in rs),
            "sizes": sorted({r.n for r in rs}),
        }

    def reset_metrics(self) -> None:
        self.history.clear()

    # ---- helpers ------------------------------------------------------------------------

    @staticmethod
    def _scramble_moves(cube, scramble: MoveList | str) -> list[Move]:
        moves = parse(scramble, cube.n) if isinstance(scramble, str) else list(scramble)
        if not Solver.is_legal(cube, moves):
            raise ValueError(f"scramble is not legal on a {cube.n}x{cube.n}x{cube.n} cube")
        return moves

    @staticmethod
    def _normalise(raw) -> tuple[tuple[str, tuple[Move, ...]], ...]:
        raw = list(raw)
        if raw and all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) for p in raw):
            return tuple((label, tuple(ms)) for label, ms in raw)
        return (("solution", tuple(raw)),)
