"""Backend-free playback timeline: which state, which move and how far through it at time t.

A timeline is a list of labelled segments (a scramble, a solution, or a solver's phases),
each with its own speed. States come from ``cube.apply``, so no viewer ever reimplements
cube logic. Storing every state costs 6n² bytes per move (240 GB for a 40k-move n = 1000
replay), so only every k-th state is kept as a checkpoint; any other state is rebuilt by
applying moves (or their inverses) from the nearest checkpoint or the last state served.
Playback asks for consecutive states, so that is usually a single move.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..move import Move


@dataclass(frozen=True)
class Segment:
    """A labelled run of moves. ``speed`` is in quarter turns per second (None = the
    timeline default); ``move_speeds`` optionally overrides it move by move."""

    label: str
    moves: Sequence[Move]
    speed: float | None = None
    move_speeds: Sequence[float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "moves", tuple(self.moves))
        if self.move_speeds is not None:
            object.__setattr__(self, "move_speeds", tuple(self.move_speeds))
            if len(self.move_speeds) != len(self.moves):
                raise ValueError("move_speeds must have one entry per move")


@dataclass(frozen=True)
class Frame:
    state: bytes             # state before ``move`` (or the resting state when move is None)
    move: Move | None        # move in progress, or None while resting
    progress: float          # 0..1 through ``move``
    index: int               # global index of ``move`` (= moves completed so far)
    segment: int             # segment index
    index_in_segment: int


class Timeline:
    def __init__(
        self,
        cube,
        start: bytes,
        segments: Iterable[Segment | tuple[str, Sequence[Move]]],
        speed: float = 3.0,
        half_turn_cost: float = 1.5,
        pause: float = 0.6,
        checkpoint_every: int | None = None,
        checkpoint_budget: int = 512 * 2**20,
        progress: bool = False,
    ):
        """``checkpoint_every`` = k keeps states 0, k, 2k, ... (default: as many as fit in
        ``checkpoint_budget`` bytes, so every state for small cubes). ``progress`` shows a
        tqdm bar while the checkpoints are built."""
        if speed <= 0 or half_turn_cost <= 0 or pause < 0:
            raise ValueError("speed and half_turn_cost must be positive, pause non-negative")
        self.cube = cube
        self.segments = tuple(s if isinstance(s, Segment) else Segment(*s) for s in segments)
        self.default_speed = speed
        self.half_turn_cost = half_turn_cost
        self.pause = pause

        self.moves: list[Move] = []
        self._seg: list[int] = []
        self._in_seg: list[int] = []
        self.starts: list[float] = []
        self.durations: list[float] = []
        self.speeds: list[float] = []
        t = 0.0
        for si, seg in enumerate(self.segments):
            t += pause  # hold before each segment so its starting state is visible
            for j, m in enumerate(seg.moves):
                cube.validate(m)
                sp = seg.move_speeds[j] if seg.move_speeds else (seg.speed or speed)
                if sp <= 0:
                    raise ValueError("speeds must be positive")
                d = (half_turn_cost if m.turns == 2 else 1.0) / sp
                self.moves.append(m)
                self._seg.append(si)
                self._in_seg.append(j)
                self.starts.append(t)
                self.durations.append(d)
                self.speeds.append(sp)
                t += d
        self.ends = [s + d for s, d in zip(self.starts, self.durations)]
        self.duration = t + (pause if self.segments else 0.0)

        n_states = len(self.moves) + 1
        if checkpoint_every is None:
            per_state = max(len(start), 1)
            checkpoint_every = max(1, math.ceil(n_states * per_state / checkpoint_budget))
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        self.checkpoint_every = checkpoint_every
        self._checkpoints: list[bytes] = [start]
        state = start
        for i, m in enumerate(_progress(self.moves, progress, "building timeline"), 1):
            state = cube.apply(state, m)
            if i % checkpoint_every == 0:
                self._checkpoints.append(state)
        self._final = state
        self._cursor: tuple[int, bytes] = (0, start)
        self.states = _States(self)

    def state(self, k: int) -> bytes:
        """State after the first ``k`` moves."""
        if not 0 <= k < len(self.moves) + 1:
            raise IndexError(k)
        if k == len(self.moves):
            return self._final
        every = self.checkpoint_every
        c = k // every
        # Candidate starting points: (index, state); walk from the closest one.
        options = [(c * every, self._checkpoints[c]), self._cursor]
        if c + 1 < len(self._checkpoints):
            options.append(((c + 1) * every, self._checkpoints[c + 1]))
        i, s = min(options, key=lambda o: abs(o[0] - k))
        while i < k:
            s = self.cube.apply(s, self.moves[i])
            i += 1
        while i > k:
            i -= 1
            s = self.cube.apply(s, self.moves[i].inverse())
        self._cursor = (k, s)
        return s

    # ---- constructors -------------------------------------------------------------------

    @classmethod
    def from_moves(cls, cube, start: bytes, moves: Sequence[Move], label: str = "moves", **kw):
        return cls(cube, start, [Segment(label, moves)], **kw)

    @classmethod
    def replay(cls, cube, scramble: Sequence[Move], solution: Sequence[Move] | None = None, **kw):
        """Start solved, play the scramble, then (optionally) a solver's solution."""
        segs = [Segment("scramble", scramble)]
        if solution is not None:
            segs.append(Segment("solution", solution))
        return cls(cube, cube.solved(), segs, **kw)

    @classmethod
    def from_result(cls, cube, result, **kw):
        """Replay a SolveResult: the scramble, then each solver phase as its own segment."""
        segs = [Segment("scramble", result.scramble)]
        segs += [Segment(label, moves) for label, moves in result.phases]
        return cls(cube, cube.solved(), segs, **kw)

    # ---- queries ------------------------------------------------------------------------

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    @property
    def final_state(self) -> bytes:
        return self._final

    def completed_at(self, t: float) -> int:
        """Number of moves finished by time t."""
        return bisect.bisect_right(self.ends, t)

    def rest_time(self, k: int) -> float:
        """Earliest time state ``k`` (after k moves) is shown at rest."""
        k = max(0, min(k, self.num_moves))
        return 0.0 if k == 0 else self.ends[k - 1]

    def frame(self, t: float) -> Frame:
        t = max(0.0, min(t, self.duration))
        k = self.completed_at(t)
        if k < self.num_moves and t >= self.starts[k]:
            progress = (t - self.starts[k]) / self.durations[k]
            return Frame(self.state(k), self.moves[k], progress, k, self._seg[k], self._in_seg[k])
        # resting: in a pause before move k, or after the last move
        if k < self.num_moves:
            seg, j = self._seg[k], self._in_seg[k]
        else:
            seg = max(len(self.segments) - 1, 0)
            j = len(self.segments[seg].moves) if self.segments else 0
        return Frame(self.state(k), None, 0.0, k, seg, j)


class _States(Sequence[bytes]):
    """Read-only list view of every state; each lookup goes through ``Timeline.state``."""

    def __init__(self, timeline: Timeline):
        self._tl = timeline

    def __len__(self) -> int:
        return self._tl.num_moves + 1

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(len(self)))]
        if k < 0:
            k += len(self)
        return self._tl.state(k)


def _progress(items, enabled: bool, desc: str):
    """Wrap ``items`` in a tqdm bar when enabled and tqdm is installed."""
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, unit="move", leave=False)
