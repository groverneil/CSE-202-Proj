"""Step 3: notation round-trips and the seeded scrambler, for every n in 2..10."""

import random

import pytest

from nxncube import Axis, Cube, Move, can_follow, format_move, format_moves, parse, parse_move

from .conftest import cube
from .test_moves import all_moves


def test_every_contiguous_move_round_trips(any_cube):
    n = any_cube.n
    for m in all_moves(any_cube):
        text = format_move(m, n)
        assert parse_move(text, n) == m, (m, text)


@pytest.mark.parametrize(
    "token, n, expected",
    [
        ("R", 3, Move(Axis.X, 2, 1)),
        ("R'", 3, Move(Axis.X, 2, 3)),
        ("L", 3, Move(Axis.X, 0, 3)),
        ("L2", 3, Move(Axis.X, 0, 2)),
        ("M", 3, Move(Axis.X, 1, 3)),
        ("S", 3, Move(Axis.Z, 1, 1)),
        ("x'", 3, Move(Axis.X, [0, 1, 2], 3)),
        ("Rw", 4, Move(Axis.X, [2, 3], 1)),
        ("r", 4, Move(Axis.X, [2, 3], 1)),
        ("3Rw2", 5, Move(Axis.X, [2, 3, 4], 2)),
        ("2R'", 5, Move(Axis.X, 3, 3)),
        ("2L", 5, Move(Axis.X, 1, 3)),
        ("2-3Rw", 6, Move(Axis.X, [3, 4], 1)),
        ("M", 6, Move(Axis.X, [1, 2, 3, 4], 3)),
        ("2U2'", 4, Move(Axis.Y, 2, 2)),
    ],
)
def test_parse_examples(token, n, expected):
    assert parse_move(token, n) == expected


@pytest.mark.parametrize("bad, n", [("4R", 3), ("Q", 3), ("M", 2), ("2x", 3), ("R3", 3), ("0R", 3)])
def test_parse_rejects(bad, n):
    with pytest.raises(ValueError):
        parse_move(bad, n)


def test_scramble_string_round_trips(any_cube):
    """Scramble -> print -> parse gives the same moves and the same state."""
    n = any_cube.n
    moves = any_cube.scramble(rng=random.Random(42))
    text = format_moves(moves, n)
    again = parse(text, n)
    assert again == moves
    assert format_moves(again, n) == text
    assert any_cube.apply_moves(any_cube.solved(), again) == any_cube.apply_moves(any_cube.solved(), moves)


def test_same_seed_same_scramble(any_cube):
    n = any_cube.n
    a = any_cube.scramble(rng=random.Random(7))
    b = any_cube.scramble(rng=random.Random(7))
    c = any_cube.scramble(rng=random.Random(8))
    assert a == b and a != c
    # a cube seeded at construction is reproducible too
    assert Cube(n, seed=3).scramble() == Cube(n, seed=3).scramble()


def test_scramble_respects_pruning_and_length(any_cube):
    from nxncube import default_scramble_length

    moves = any_cube.scramble(rng=random.Random(1))
    assert len(moves) == default_scramble_length(any_cube.n)
    assert all(can_follow(a, b) for a, b in zip(moves, moves[1:]))
    assert not any(any_cube.is_rotation(m) for m in moves)
    assert not any_cube.is_solved(any_cube.apply_moves(any_cube.solved(), moves))


def test_can_follow_rule():
    r, l, r2 = Move(Axis.X, 2, 1), Move(Axis.X, 0, 1), Move(Axis.X, 2, 2)
    u = Move(Axis.Y, 2, 1)
    assert can_follow(None, r)
    assert can_follow(l, r) and not can_follow(r, l)
    assert not can_follow(r, r2)
    assert can_follow(r, u) and can_follow(u, r)


def test_parse_ignores_parentheses():
    assert parse("(R U R' U')", 3) == parse("R U R' U'", 3)


class _NoRotationShortcut:
    """Wraps a cube but hides ``is_rotation``, forcing the scrambler's generic path."""

    def __init__(self, c):
        self._c = c

    def __getattr__(self, name):
        if name == "is_rotation":
            raise AttributeError(name)
        return getattr(self._c, name)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_scrambler_rotation_shortcut_matches_generic_path(n):
    c = cube(n)
    fast = c.scrambler.scramble(c, random.Random(9), 40)
    slow = c.scrambler.scramble(_NoRotationShortcut(c), random.Random(9), 40)
    assert fast == slow
