"""Seeded random-move scrambler that works for any puzzle through the tier-1 interface."""

from __future__ import annotations

import random

from .move import Move, can_follow


def default_scramble_length(n: int) -> int:
    if n <= 2:
        return 15
    if n == 3:
        return 25
    return 20 * (n - 2)


class RandomMoveScrambler:
    """Draws moves from ``puzzle.moves()`` with the same pruning rule the search uses.

    Moves that leave the solved state solved (whole-cube rotations) are skipped, since
    they scramble nothing.
    """

    def scramble(self, puzzle, rng: random.Random, length: int | None = None) -> list[Move]:
        if length is None:
            n = getattr(puzzle, "n", 3)
            length = default_scramble_length(n)
        solved = puzzle.solved()
        candidates = [m for m in puzzle.moves() if not puzzle.is_solved(puzzle.apply(solved, m))]
        if not candidates:
            raise ValueError("puzzle has no scrambling moves")

        out: list[Move] = []
        prev = None
        while len(out) < length:
            m = rng.choice(candidates)
            if can_follow(prev, m):
                out.append(m)
                prev = m
        return out
