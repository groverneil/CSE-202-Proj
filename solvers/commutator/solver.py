"""Piece-by-piece commutator solver: works for any n and any legal scramble.

Each orbit (corners, middle edges, each wing orbit, each center orbit) is solved on its
own with pure 3-cycles: for every unsolved position p, find a piece with p's colours and the
shortest conjugate S X S' of a library macro X that brings it to p without disturbing any
solved sticker. A 3-cycle cannot fix an odd permutation, so when an orbit gets stuck on a
lone swap, one quarter turn that leaves earlier orbits alone is prepended and the orbit is
solved again. See README.md in this directory for the algorithm and pseudocode.
"""

from __future__ import annotations

import numpy as np

from nxncube.move import Move

from ..base import Solver
from .library import KIND_ORDER, CommutatorLibrary, inverse_sequence, simplify


class CommutatorSolver(Solver):
    name = "commutator"

    def __init__(self, per_orbit: int = 40, max_setup: int = 2) -> None:
        super().__init__()
        self.per_orbit = per_orbit
        self.max_setup = max_setup
        self._libraries: dict[int, CommutatorLibrary] = {}

    def library(self, cube) -> CommutatorLibrary:
        """The library for this cube size, built on first use and cached."""
        lib = self._libraries.get(cube.n)
        if lib is None:
            lib = self._libraries[cube.n] = CommutatorLibrary(cube, self.per_orbit, self.max_setup)
            self.info["library"] = lib.stats()
        return lib

    # ---- search for one placement ---------------------------------------------------

    @staticmethod
    def _scan(macros, table, cols, s, goal, need, target):
        """Best (macro length, row, X, dst, src) over conjugates S X S' for the setups in
        ``table`` (one permutation row each, sorted by length).

        If S has permutation P_S, S X S' changes exactly stickers P_S[X.dst], setting
        new[P_S[X.dst]] = old[P_S[X.src]]. So one gather per macro shows every conjugate's
        effect: it is acceptable when each changed sticker in ``need`` gets its goal colour
        and it covers every ``target`` sticker. ``cols`` (a move's permutation) composes one
        more move onto every row while gathering only the <= 9 columns a macro needs.
        """
        best = None
        for X in macros:
            d = X.dst if cols is None else cols[X.dst]
            r_ = X.src if cols is None else cols[X.src]
            dst, src = table[:, d], table[:, r_]
            ok = ((s[src] == goal[dst]) | ~need[dst]).all(axis=1)
            ok &= (dst[:, :, None] == target).any(axis=2).sum(axis=1) == len(target)
            hits = np.flatnonzero(ok)
            if len(hits):
                r = int(hits[0])
                if best is None or len(X.moves) < len(best[2].moves):
                    best = (len(X.moves), r, X, dst[r], src[r])
        return best

    def _place(self, lib: CommutatorLibrary, oi: int, s, goal, need, target, fallback=True):
        """A short conjugate S X S' that fixes ``target`` and keeps every ``need`` sticker,
        as (S, X, dst, src), or None.

        Setups are tried in tiers (0-1 moves, then up to the table's length) and the first
        tier with a hit wins; then, if ``fallback``, setups one move longer, S + [m]."""
        macros = lib.macros[oi]
        for end in lib.tier_ends[1:]:
            hit = self._scan(macros, lib.setup_perms[:end], None, s, goal, need, target)
            if hit is not None:
                _, r, X, dst, src = hit
                return lib.setups[r], X, dst, src
        if not fallback:
            return None
        longest = np.flatnonzero(lib.setup_len == lib.setup_len.max())
        base = lib.setup_perms[longest]
        best = None
        for m in lib.moves:
            hit = self._scan(macros, base, lib.perm[m], s, goal, need, target)
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], lib.setups[longest[hit[1]]] + (m,), hit[2], hit[3], hit[4])
        if best is None:
            return None
        self._fallbacks += 1
        return best[1:]

    # ---- one orbit ------------------------------------------------------------------

    def _solve_orbit(self, lib: CommutatorLibrary, oi: int, s: np.ndarray, goal: np.ndarray,
                     fast: bool = True):
        """Greedy placement. Returns (moves, new_state, solved?).

        ``fast``: with exactly two pieces left in a corner/edge/wing orbit, skip the slow
        longer-setup search. That position is almost always a lone swap (odd parity), which
        no 3-cycle can fix, so it is reported as stuck at once; the caller retries without
        ``fast`` if the parity fix does not resolve it."""
        fac = lib.orbits[oi].facelets
        kind = lib.orbits[oi].kind
        s = s.copy()
        out: list[Move] = []
        while True:
            done = (s[fac] == goal[fac]).all(axis=1)
            if done.all():
                return out, s, True
            if oi not in lib.macros:
                return out, s, False
            unsolved = np.flatnonzero(~done)
            fallback = not (fast and len(unsolved) == 2 and kind != "center")
            need = np.zeros(len(s), dtype=bool)
            need[fac[done].ravel()] = True
            for p in unsolved:
                need_p = need.copy()
                need_p[fac[p]] = True
                found = self._place(lib, oi, s, goal, need_p, fac[p], fallback)
                if found is not None:
                    S, X, dst, src = found
                    s[dst] = s[src]
                    out += list(S) + list(X.moves) + inverse_sequence(S)
                    break
            else:
                return out, s, False  # no unsolved position can be fixed: odd parity

    def _solve_orbit_with_parity(self, lib, oi, s, goal, earlier, fast):
        """Solve orbit ``oi``; if it is left with an odd permutation, prepend the first
        quarter turn that moves this orbit but leaves every ``earlier`` sticker solved."""
        moves, s_next, ok = self._solve_orbit(lib, oi, s, goal, fast)
        if ok:
            return moves, s_next, 0
        fac = lib.orbits[oi].facelets.ravel()
        for m in lib.moves:
            perm = lib.perm[m]
            if m.turns != 1 or (perm[fac] == fac).all():
                continue
            t = s[perm]
            if not np.array_equal(t[earlier], goal[earlier]):
                continue
            moves, s_next, ok = self._solve_orbit(lib, oi, t, goal, fast)
            if ok:
                return [m] + moves, s_next, 1
        return None

    # ---- entry point ----------------------------------------------------------------

    def _goal(self, lib: CommutatorLibrary, s: np.ndarray) -> np.ndarray:
        """Solved state to aim for. Odd n: the orientation the fixed centers already show.
        Even n: the orientation that already matches the most stickers."""
        candidates = lib.orientations
        fixed = [o.facelets.ravel() for o in lib.orbits if o.kind == "fixed_center"]
        if fixed:
            f = np.concatenate(fixed)
            candidates = [g for g in candidates if np.array_equal(g[f], s[f])]
        return max(candidates, key=lambda g: int((g == s).sum()))

    def _solve(self, cube, state):
        lib = self.library(cube)
        s = np.frombuffer(state, dtype=np.uint8).copy()
        goal = self._goal(lib, s)
        order = sorted(range(len(lib.orbits)), key=lambda i: KIND_ORDER.index(lib.orbits[i].kind))
        self._fallbacks = 0
        parity_fixes = 0
        phases = []
        for idx, oi in enumerate(order):
            earlier = np.concatenate([lib.orbits[j].facelets.ravel() for j in order[:idx]]
                                     or [np.zeros(0, dtype=int)])
            result = (self._solve_orbit_with_parity(lib, oi, s, goal, earlier, fast=True)
                      or self._solve_orbit_with_parity(lib, oi, s, goal, earlier, fast=False))
            if result is None:
                raise RuntimeError(f"commutator solver stuck on {lib.orbits[oi].kind} orbit {oi}")
            moves, s_next, fixes = result
            parity_fixes += fixes
            if moves:
                phases.append((lib.orbits[oi].kind, simplify(moves)))
            s = s_next
        self.info.update(parity_fixes=parity_fixes, setup_fallbacks=self._fallbacks)
        return phases or [("solution", [])]
