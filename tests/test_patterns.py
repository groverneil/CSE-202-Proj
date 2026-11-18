"""Known patterns, checked against geometric descriptions rather than stored states.

An outer-turn 3x3 algorithm can be run on an n x n with every turn made k layers deep
(1 <= k <= n/2). The cube then behaves like a 3x3 whose "stickers" are blocks: k-wide
outer bands and an (n-2k)-wide middle band.
"""

import random

import numpy as np
import pytest

from nxncube import Axis, Move, format_moves, parse

from .conftest import cube

CUBE_IN_CUBE = "F L F U' R U F2 L2 U' L' B D' B' L2 U"
CUBE_IN_CUBE_IN_CUBE = "U' L' U' F' R2 B' R F U B2 U B' L U' F U R F'"

# Face values: U=0 R=1 F=2 D=3 L=4 B=5.
OPPOSITE = np.array([3, 4, 5, 0, 1, 2])
# Turning the cube 120 degrees about the URF-DLB diagonal: U shows F, R shows U, F shows R,
# D shows B, L shows D, B shows L.
DIAGONAL_TURN = np.array([2, 0, 1, 5, 3, 4])

N_AND_K = [(n, k) for n in (2, 3, 4, 5) for k in range(1, n // 2 + 1)]


def widen(alg3: str, n: int, k: int) -> list[Move]:
    """Run a 3x3 outer-turn algorithm on an n x n with each turn k layers deep."""
    out = []
    for m in parse(alg3, 3):
        (layer,) = m.layers
        assert layer in (0, 2), "only outer turns can be widened"
        out.append(Move(m.axis, range(n - k, n) if layer == 2 else range(k), m.turns))
    return out


def band(i: np.ndarray, n: int, k: int) -> np.ndarray:
    """Row/column index on n x n -> row/column on the 3x3 it imitates."""
    return np.where(i < k, 0, np.where(i >= n - k, 2, 1))


def inflate(state3: bytes, n: int, k: int) -> bytes:
    """What a 3x3 state looks like on an n x n whose turns are k layers deep."""
    s3 = np.frombuffer(state3, np.uint8).reshape(6, 3, 3)
    g = cube(n).geometry
    return s3[g.face, band(g.row, n, k), band(g.col, n, k)].astype(np.uint8).tobytes()


def cube_in_cube_target(n: int, k: int) -> bytes:
    """An (n-k)-sized block at URF and at DLB keeps its colours; the k-thick shell
    around them shows the colours of the cube turned about the URF-DLB diagonal."""
    g = cube(n).geometry
    on_urf_side = g.face < 3
    in_urf_block = (g.layers >= k).all(axis=1)
    in_dlb_block = (g.layers <= n - 1 - k).all(axis=1)
    keep = np.where(on_urf_side, in_urf_block, in_dlb_block)
    return np.where(keep, g.face, DIAGONAL_TURN[g.face]).astype(np.uint8).tobytes()


@pytest.mark.parametrize("n, k", N_AND_K, ids=lambda v: str(v))
def test_cube_in_a_cube(n, k):
    c = cube(n)
    alg = widen(CUBE_IN_CUBE, n, k)
    s = c.apply_moves(c.solved(), alg)
    assert s == cube_in_cube_target(n, k)
    assert not c.is_solved(s)
    assert c.apply_moves(s, [m.inverse() for m in reversed(alg)]) == c.solved()


@pytest.mark.parametrize("n, k", [(n, k) for n, k in N_AND_K if n - 2 * k >= 1])
def test_cube_in_a_cube_in_a_cube(n, k):
    """Three nested cubes (sizes n, n-k, k); needs a middle band, so n - 2k >= 1."""
    c3, c = cube(3), cube(n)
    s3 = c3.apply_moves(c3.solved(), parse(CUBE_IN_CUBE_IN_CUBE, 3))
    s = c.apply_moves(c.solved(), widen(CUBE_IN_CUBE_IN_CUBE, n, k))
    assert s == inflate(s3, n, k)
    # On U, F and R, the sticker at the URF corner, one in the middle band and the one at
    # the far corner show three different colours: three nested cubes.
    faces = np.frombuffer(s, np.uint8).reshape(6, n, n)
    last = n - 1
    urf_corner = {0: (last, last), 2: (0, last), 1: (0, 0)}  # U, F, R
    far_corner = {0: (0, 0), 2: (last, 0), 1: (last, last)}
    for f in (0, 1, 2):
        colours = {faces[f][urf_corner[f]], faces[f][k, k], faces[f][far_corner[f]]}
        assert len(colours) == 3, f


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_checkerboard_odd_n(n):
    """Half-turning every even-indexed layer on each axis gives a checkerboard on odd n:
    stickers with (row + col) even keep their colour, the rest show the opposite face."""
    c = cube(n)
    g = c.geometry
    alg = [Move(axis, layer, 2) for axis in Axis for layer in range(0, n, 2)]
    s = np.frombuffer(c.apply_moves(c.solved(), alg), np.uint8)
    expected = np.where((g.row + g.col) % 2 == 0, g.face, OPPOSITE[g.face])
    assert np.array_equal(s, expected)
    assert c.apply_moves(s.tobytes(), alg) == c.solved()  # half turns undo themselves


@pytest.mark.parametrize("n, k", N_AND_K, ids=lambda v: str(v))
def test_any_3x3_algorithm_inflates(n, k):
    """For random outer-turn 3x3 sequences, the widened n x n result is the inflated 3x3 result."""
    rng = random.Random(100 * n + k)
    c3, c = cube(3), cube(n)
    outer = [m for m in c3.moves() if m.layers & {0, 2}]
    for _ in range(20):
        alg3 = format_moves([rng.choice(outer) for _ in range(25)], 3)
        s3 = c3.apply_moves(c3.solved(), parse(alg3, 3))
        assert c.apply_moves(c.solved(), widen(alg3, n, k)) == inflate(s3, n, k)
