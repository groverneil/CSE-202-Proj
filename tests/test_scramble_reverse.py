"""Scramble, then undo by playing the inverse moves in reverse order.

The undo sequence goes through the notation printer and parser first, so this also
checks that long mixed sequences survive a string round trip.
"""

import random

import pytest

from nxncube import can_follow, format_moves, parse

from .conftest import cube

SEEDS = range(1, 6)


def random_sequence(moves, rng, length):
    """Random moves from ``moves`` under the usual pruning rule."""
    out, prev = [], None
    while len(out) < length:
        m = rng.choice(moves)
        if can_follow(prev, m):
            out.append(m)
            prev = m
    return out


def check_scramble_and_reverse(c, scramble):
    n = c.n
    scrambled = c.apply_moves(c.solved(), scramble)
    assert not c.is_solved(scrambled)
    undo = parse(format_moves([c.inverse(m) for m in reversed(scramble)], n), n)
    assert c.apply_moves(scrambled, undo) == c.solved()  # exact start, not just any orientation


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_basic_move_scramble_reverses(n, seed):
    """The cube's own scrambler (9n single-layer moves, default length)."""
    c = cube(n)
    check_scramble_and_reverse(c, c.scramble(rng=random.Random(seed)))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9])
def test_full_move_set_scramble_reverses(n, seed):
    """Wide turns, slice blocks and rotations mixed in, 30 moves (60 for n >= 8)."""
    c = cube(n)
    rng = random.Random(seed)
    scramble = random_sequence(c.all_moves(), rng, 60 if n >= 8 else 30)
    assert sum(len(m.layers) > 1 for m in scramble) > 0
    check_scramble_and_reverse(c, scramble)


@pytest.mark.parametrize("n", [8, 9])
def test_full_move_set_uses_every_move(n):
    """Long enough sequences on n = 8, 9 that every one of the 9n(n+1)/2 moves appears."""
    c = cube(n)
    rng = random.Random(n)
    moves = c.all_moves()
    seen, total = set(), []
    while len(seen) < len(moves):
        chunk = random_sequence(moves, rng, 200)
        seen.update(chunk)
        total += chunk
    check_scramble_and_reverse(c, total)
