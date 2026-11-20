"""Pure-commutator library for the commutator solver (built once per cube size).

A *pure* macro is a move sequence whose net effect touches at most 3 pieces, all in one
orbit, and nothing else on the cube: a 3-cycle, or an in-place twist/flip. They are found
by brute force over commutators [A, b] = A b A' b' with A drawn from the shapes real
commutators use, then kept per orbit (shortest first, plus inverses).

The library also holds every setup sequence S up to ``max_setup`` moves as a permutation
table, so the solver can test all conjugates S X S' of a macro X with one numpy gather.
See README.md in this directory for the algorithm.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from nxncube.geometry import Axis
from nxncube.move import Move

# Orbits are solved in this order. Fixed centers come first: the goal orientation is chosen
# to match them, so they start solved and later steps must leave them alone.
KIND_ORDER = ("fixed_center", "corner", "middle_edge", "wing", "center")
# Orbits whose pieces can be twisted or flipped in place (a wing's position fixes its
# orientation, and center pieces have one sticker).
TWISTABLE = ("corner", "middle_edge")


def inverse_sequence(seq: Sequence[Move]) -> list[Move]:
    return [m.inverse() for m in reversed(seq)]


def simplify(seq: Sequence[Move]) -> list[Move]:
    """Merge neighbouring turns of the same axis and layers; drop the ones that cancel."""
    out: list[Move] = []
    for m in seq:
        if out and out[-1].axis == m.axis and out[-1].layers == m.layers:
            t = (out[-1].turns + m.turns) % 4
            out.pop()
            if t:
                out.append(Move(m.axis, m.layers, t))
        else:
            out.append(m)
    return out


@dataclass(frozen=True, eq=False)
class Macro:
    """A pure move sequence: applying ``moves`` sets ``new[dst] = old[src]`` and changes
    no other sticker. ``dst`` lists every sticker it changes."""

    moves: tuple[Move, ...]
    dst: np.ndarray
    src: np.ndarray


class CommutatorLibrary:
    def __init__(self, cube, per_orbit: int = 40, max_setup: int = 2):
        t0 = time.perf_counter()
        self.cube = cube
        N = cube.num_stickers
        self.orbits = tuple(cube.orbits())
        self.orbit_of = np.full(N, -1)
        self.piece_of = np.full(N, -1)
        for oi, o in enumerate(self.orbits):
            for pi, fs in enumerate(o.facelets):
                self.orbit_of[fs] = oi
                self.piece_of[fs] = pi
        self.moves: tuple[Move, ...] = tuple(cube.moves())
        self.perm = {m: np.asarray(cube.perm(m)) for m in self.moves}
        self._ident = np.arange(N)

        found = self._search_commutators()
        self.macros = self._select(found, per_orbit)
        self._add_composed_twists()

        self.setups: list[tuple[Move, ...]] = [()] + [
            tuple(s) for k in range(1, max_setup + 1) for s in self._sequences(k)
        ]
        self.setup_len = np.array([len(s) for s in self.setups])
        # rows are sorted by setup length; tier_ends[k] = number of setups of length <= k
        self.tier_ends = [int((self.setup_len <= k).sum()) for k in range(max_setup + 1)]
        # filled row by row in a small dtype: stacking int64 rows first would need ~0.6 GB at n = 20
        self.setup_perms = np.empty((len(self.setups), N), dtype=np.int16 if N < 2**15 else np.int32)
        for i, seq in enumerate(self.setups):
            self.setup_perms[i] = self.sequence_perm(seq)
        self.orientations = self._orientations()
        self.num_pure_found = len(found)
        self.build_seconds = time.perf_counter() - t0

    # ---- helpers --------------------------------------------------------------------

    def sequence_perm(self, seq: Sequence[Move]) -> np.ndarray:
        p = self._ident
        for m in seq:
            p = p[self.perm[m]]
        return p

    def _sequences(self, k: int) -> Iterator[list[Move]]:
        """Every k-move sequence with no two neighbouring turns of the same axis+layers."""
        for combo in itertools.product(self.moves, repeat=k):
            if any(a.axis == b.axis and a.layers == b.layers for a, b in zip(combo, combo[1:])):
                continue
            yield list(combo)

    def is_pure(self, p: np.ndarray) -> bool:
        moved = np.flatnonzero(p != self._ident)
        return (len(moved) > 0 and len(np.unique(self.orbit_of[moved])) == 1
                and len(np.unique(self.piece_of[moved])) <= 3)

    # ---- construction ---------------------------------------------------------------

    def _search_commutators(self) -> dict[bytes, tuple[list[Move], np.ndarray, np.ndarray]]:
        """Pure commutators [A, b] as (moves, dst, src), keyed by their effect (shortest kept).
        Stored sparsely: a full 6n²-entry key per commutator would need GBs by n = 20.

        A searching every 3-move A costs 9n * (9n)^3 commutators, too many by n = 8, so A is
        restricted to the shapes real commutators use: x, x y, the conjugate x y x', and
        (x y x' y')^2, which gives in-place corner twists and edge flips.
        """
        moves = self.moves
        As = [s for k in (1, 2) for s in self._sequences(k)]
        As += [[x, y, x.inverse()] for x in moves for y in moves if x.axis != y.axis and x.turns != 2]
        As += [[x, y, x.inverse(), y.inverse()] * 2 for x in moves for y in moves
               if x.turns == 1 and y.turns == 1 and x.axis != y.axis]
        found: dict[bytes, tuple[list[Move], np.ndarray, np.ndarray]] = {}
        for A in As:
            pa, pai = self.sequence_perm(A), self.sequence_perm(inverse_sequence(A))
            for b in moves:
                p = pa[self.perm[b]][pai][self.perm[b.inverse()]]
                if not self.is_pure(p):
                    continue
                moved = np.flatnonzero(p != self._ident).astype(np.int32)
                src = p[moved].astype(np.int32)
                key = moved.tobytes() + src.tobytes()
                seq = A + [b] + inverse_sequence(A) + [b.inverse()]
                if key not in found or len(seq) < len(found[key][0]):
                    found[key] = (seq, moved, src)
        return found

    def _select(self, found, per_orbit: int) -> dict[int, list[Macro]]:
        """Per orbit: the shortest ``per_orbit`` macros, plus some pure twists/flips (pieces
        stay put, stickers rotate), plus the inverse of each."""
        by_orbit: dict[int, list[tuple[list[Move], np.ndarray, np.ndarray]]] = {}
        for seq, dst, src in found.values():
            by_orbit.setdefault(int(self.orbit_of[dst[0]]), []).append((seq, dst, src))
        out: dict[int, list[Macro]] = {}
        for oi, entries in by_orbit.items():
            entries.sort(key=lambda e: len(e[0]))
            twists = [e for e in entries if self._is_twist(e[1], e[2])]
            chosen, seen = [], set()
            for e in entries[:per_orbit] + twists[: max(per_orbit // 4, 1)]:
                if id(e) in seen:
                    continue
                seen.add(id(e))
                seq, dst, src = e
                chosen.append(Macro(tuple(seq), dst, src))
                # new[dst] = old[src] undone: new[src] = old[dst]
                chosen.append(Macro(tuple(inverse_sequence(seq)), src, dst))
            out[oi] = chosen
        return out

    def _is_twist(self, dst: np.ndarray, src: np.ndarray) -> bool:
        """Every changed sticker stays on its own piece (pieces stay put, stickers rotate)."""
        return bool((self.piece_of[dst] == self.piece_of[src]).all())

    def _add_composed_twists(self, wanted: int = 8) -> None:
        """Corners and middle edges need in-place twists/flips. When the
        commutator search found none (middle-edge flips, for instance), look for products
        X · (S Y S') of two macros of that orbit, S at most one move, whose net effect is a
        pure twist/flip. A product of pure macros of one orbit is still confined to it."""
        setups = [[]] + [[m] for m in self.moves]
        for oi, macros in self.macros.items():
            if self.orbits[oi].kind not in TWISTABLE:
                continue
            if any(self._is_twist(X.dst, X.src) for X in macros):
                continue
            perms = []
            for X in macros:
                p = self._ident.copy()
                p[X.dst] = X.src
                perms.append((X, p))
            found: dict[bytes, list[Move]] = {}
            for S in setups:
                ps, psi = self.sequence_perm(S), self.sequence_perm(inverse_sequence(S))
                for X, px in perms:
                    for Y, py in perms:
                        p = px[ps[py][psi]]  # X, then S Y S'
                        moved = np.flatnonzero(p != self._ident)
                        if (len(moved) == 0 or len(np.unique(self.piece_of[moved])) > 3
                                or not self._is_twist(moved, p[moved])
                                or len(np.unique(self.orbit_of[moved])) != 1):
                            continue
                        key = p.astype(np.int32).tobytes()
                        seq = list(X.moves) + S + list(Y.moves) + inverse_sequence(S)
                        if key not in found or len(seq) < len(found[key]):
                            found[key] = seq
                if len(found) >= wanted:
                    break
            for key, seq in sorted(found.items(), key=lambda kv: len(kv[1]))[:wanted]:
                p = np.frombuffer(key, dtype=np.int32).astype(np.intp)
                moved = np.flatnonzero(p != self._ident)
                macros.append(Macro(tuple(seq), moved, p[moved]))
                macros.append(Macro(tuple(inverse_sequence(seq)), moved, np.argsort(p)[moved]))

    def _orientations(self) -> list[np.ndarray]:
        """The solved state in each of its 24 whole-cube orientations."""
        rots = [np.asarray(self.cube.perm(self.cube.rotation(a))) for a in Axis]
        start = np.frombuffer(self.cube.solved(), dtype=np.uint8)
        seen = {start.tobytes(): start}
        frontier = [start]
        while frontier:
            nxt = []
            for s in frontier:
                for r in rots:
                    t = s[r]
                    if t.tobytes() not in seen:
                        seen[t.tobytes()] = t
                        nxt.append(t)
            frontier = nxt
        return list(seen.values())

    def stats(self) -> dict:
        return {
            "pure_commutators_found": self.num_pure_found,
            "macros_kept": sum(len(v) for v in self.macros.values()),
            "setups": len(self.setups),
            "setup_table_mb": self.setup_perms.nbytes / 2**20,
            "build_seconds": self.build_seconds,
        }
