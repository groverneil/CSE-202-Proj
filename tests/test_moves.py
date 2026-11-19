"""Correctness checks on the move generator for every n in 2..10 (no solver needed)."""

import itertools
import random

import numpy as np
import pytest

from nxncube import Axis, Cube, Move

from .conftest import cube


def all_moves(c: Cube):
    """The entire move set: every contiguous block (single layers, wide turns, slices, rotations)."""
    return list(c.all_moves())


def test_all_moves_contents(any_cube):
    n = any_cube.n
    ms = any_cube.all_moves()
    assert len(ms) == len(set(ms)) == 9 * n * (n + 1) // 2
    assert set(any_cube.moves()) <= set(ms)


def random_state(c: Cube, seed: int = 0, length: int = 60) -> bytes:
    rng = random.Random(seed)
    return c.apply_moves(c.solved(), [rng.choice(c.moves()) for _ in range(length)])


def test_primitive_count(any_cube):
    assert len(any_cube.moves()) == 9 * any_cube.n
    assert len(set(any_cube.moves())) == 9 * any_cube.n


def test_every_permutation_is_a_bijection(any_cube):
    N = any_cube.num_stickers
    for m in all_moves(any_cube):
        p = any_cube.perm(m)
        assert p.shape == (N,)
        assert np.array_equal(np.sort(p), np.arange(N)), m


def test_quarter_turn_four_times_is_identity(any_cube):
    ident = np.arange(any_cube.num_stickers)
    for m in all_moves(any_cube):
        if m.turns == 1:
            assert np.array_equal(any_cube.sequence_perm([m] * 4), ident), m
            # and no smaller power is the identity
            assert not np.array_equal(any_cube.perm(m), ident), m


def test_half_and_inverse_turns_compose_from_quarters(any_cube):
    for m in all_moves(any_cube):
        if m.turns == 1:
            q = Move(m.axis, m.layers, 1)
            for t in (2, 3):
                assert np.array_equal(
                    any_cube.perm(Move(m.axis, m.layers, t)), any_cube.sequence_perm([q] * t)
                )


def test_move_then_inverse_is_identity(any_cube):
    ident = np.arange(any_cube.num_stickers)
    for m in all_moves(any_cube):
        assert np.array_equal(any_cube.sequence_perm([m, any_cube.inverse(m)]), ident), m


def test_same_axis_moves_commute(any_cube):
    s = random_state(any_cube, seed=any_cube.n)
    prims = any_cube.moves()
    for a, b in itertools.product(prims, prims):
        if a.axis == b.axis:
            assert any_cube.apply_moves(s, [a, b]) == any_cube.apply_moves(s, [b, a]), (a, b)


def test_different_axis_moves_do_not_commute(any_cube):
    n = any_cube.n
    r, u = Move(Axis.X, n - 1, 1), Move(Axis.Y, n - 1, 1)
    s = any_cube.solved()
    assert any_cube.apply_moves(s, [r, u]) != any_cube.apply_moves(s, [u, r])


def test_rotation_keeps_solved(any_cube):
    for axis in Axis:
        for t in (1, 2, 3):
            s = any_cube.apply(any_cube.solved(), any_cube.rotation(axis, t))
            assert any_cube.is_solved(s)
            assert s != any_cube.solved()


def test_rotation_equals_all_layers_turned(any_cube):
    for axis in Axis:
        layers = [Move(axis, l, 1) for l in range(any_cube.n)]
        assert np.array_equal(
            any_cube.perm(any_cube.rotation(axis)), any_cube.sequence_perm(layers)
        )


def test_single_layer_moves_unsolve(any_cube):
    for m in any_cube.moves():
        assert not any_cube.is_solved(any_cube.apply(any_cube.solved(), m)), m


def test_colour_counts_preserved(any_cube):
    s = np.frombuffer(random_state(any_cube, seed=1), dtype=np.uint8)
    assert np.bincount(s, minlength=6).tolist() == [any_cube.n**2] * 6


def test_states_are_immutable_hashable_bytes(any_cube):
    s0 = any_cube.solved()
    s1 = any_cube.apply(s0, any_cube.moves()[0])
    assert isinstance(s1, bytes) and s0 == any_cube.solved()
    assert len({s0, s1, any_cube.apply(s1, any_cube.moves()[0].inverse())}) == 2


def test_corners_stay_valid_after_scramble(any_cube):
    """After scrambling, corner slots still hold exactly the eight solved colour triples."""
    s = random_state(any_cube, seed=2)
    corners = next(o for o in any_cube.orbits() if o.kind == "corner")
    solved_sets = {frozenset(row) for row in np.frombuffer(any_cube.solved(), np.uint8)[corners.facelets]}
    colors = np.frombuffer(s, np.uint8)[corners.facelets]
    assert {frozenset(row) for row in colors} == solved_sets


def test_out_of_range_layer_rejected():
    c = cube(3)
    with pytest.raises(ValueError):
        c.apply(c.solved(), Move(Axis.X, 3, 1))
    with pytest.raises(ValueError):
        Move(Axis.X, [], 1)
    with pytest.raises(ValueError):
        Move(Axis.X, 0, 4)


# ---- 3x3x3 group orders -----------------------------------------------------------------


def order(c: Cube, moves) -> int:
    p = c.sequence_perm(moves)
    q, k = p.copy(), 1
    ident = np.arange(c.num_stickers)
    while not np.array_equal(q, ident):
        q, k = q[p], k + 1
    return k


@pytest.mark.parametrize(
    "seq, expected",
    [("R U R' U'", 6), ("R U", 105), ("R", 4), ("R2 U2", 6), ("M2 E2 S2", 2)],
)
def test_orders_n3(seq, expected):
    from nxncube import parse

    assert order(cube(3), parse(seq, 3)) == expected


# ---- 2x2x2 full BFS -------------------------------------------------------------------

# Half-turn-metric depth distribution of the 2x2x2 with one corner fixed.
HTM_2x2 = [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]


def test_bfs_2x2_state_count_and_depth():
    """BFS with the DBL corner pinned (only R, U, F turns) reaches 3,674,160 states, depth 11."""
    c = cube(2)
    perms = [c.perm(Move(axis, 1, t)) for axis in Axis for t in (1, 2, 3)]
    pow6 = 6 ** np.arange(24, dtype=np.int64)

    def encode(states):
        return states.astype(np.int64) @ pow6

    def decode(codes):
        out = np.empty((len(codes), 24), dtype=np.uint8)
        for i in range(24):
            out[:, i] = (codes // pow6[i]) % 6
        return out

    frontier = np.frombuffer(c.solved(), dtype=np.uint8)[None, :]
    visited = encode(frontier)
    counts = [1]
    while True:
        codes = np.unique(np.concatenate([encode(frontier[:, p]) for p in perms]))
        new = codes[~np.isin(codes, visited, assume_unique=True)]
        if len(new) == 0:
            break
        counts.append(len(new))
        visited = np.union1d(visited, new)
        frontier = decode(new)

    assert sum(counts) == 3_674_160
    assert len(counts) - 1 == 11
    assert counts == HTM_2x2


def test_index_of_inverts_points(any_cube):
    g = any_cube.geometry
    assert np.array_equal(g.index_of(g.points), np.arange(any_cube.num_stickers))
    n = any_cube.n
    for bad in ([n + 1, 0, 0], [n, n, 0], [0, 0, 0], [n, n - 2, 0]):
        with pytest.raises(KeyError):
            g.index_of(np.array([bad]))


def test_cycle_lists_only_moved_stickers(any_cube):
    for m in all_moves(any_cube):
        dst, src = any_cube.cycle(m)
        p = any_cube.perm(m)
        moved = np.flatnonzero(p != np.arange(any_cube.num_stickers))
        assert np.array_equal(np.sort(dst), moved) and np.array_equal(p[dst], src), m
