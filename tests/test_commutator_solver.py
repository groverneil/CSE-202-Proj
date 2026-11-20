"""The piece-by-piece commutator solver (solvers/commutator)."""

import functools
import random

import numpy as np
import pytest

from nxncube import Cube, parse
from solvers.commutator import CommutatorSolver, simplify
from solvers.conformance import assert_solver_conforms

from .conftest import cube


@functools.lru_cache(maxsize=None)
def solver() -> CommutatorSolver:
    """One shared instance, so each size's library is built once for the whole module."""
    return CommutatorSolver()


def test_conforms_on_small_cubes():
    # a fresh instance: conformance checks the metrics of exactly these solves
    assert_solver_conforms(CommutatorSolver(), sizes=(2, 3, 4, 5), seeds=range(2))


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_library_macros_are_pure(n):
    """Every macro changes at most 3 pieces, all in one orbit, and nothing else, and its
    recorded (dst, src) is exactly what its move sequence does."""
    c = cube(n)
    lib = solver().library(c)
    ident = np.arange(c.num_stickers)
    for oi, macros in lib.macros.items():
        assert macros, lib.orbits[oi].kind
        for X in macros[:20] + macros[-4:]:
            p = np.asarray(c.sequence_perm(X.moves))
            moved = np.flatnonzero(p != ident)
            assert np.array_equal(np.sort(X.dst), moved)
            assert np.array_equal(p[X.dst], X.src)
            assert set(lib.orbit_of[moved]) == {oi}
            assert len(set(lib.piece_of[moved])) <= 3


@pytest.mark.parametrize("n", [3, 4, 5])
def test_corners_and_middle_edges_have_a_pure_twist_or_flip(n):
    lib = solver().library(cube(n))
    for oi, macros in lib.macros.items():
        if lib.orbits[oi].kind in ("corner", "middle_edge"):
            assert any((lib.piece_of[X.dst] == lib.piece_of[X.src]).all() for X in macros), \
                lib.orbits[oi].kind


def test_solved_cube_needs_no_moves():
    r = solver().solve(cube(4), [])
    assert r.correct and r.length == 0


@pytest.mark.parametrize("n", [3, 5])
def test_odd_cube_keeps_its_fixed_centers(n):
    """On odd n the goal is the orientation the fixed centers already show, so a scramble
    that only rotates the whole cube is solved with no moves at all."""
    c = cube(n)
    r = solver().solve(c, [c.rotation(0), c.rotation(1, 3)])
    assert r.correct and r.length == 0


def test_short_scramble_is_solved_and_phased():
    c = cube(4)
    r = solver().solve(c, "Rw U 2F' L2 r")
    assert r.correct
    assert {label for label, _ in r.phases} <= {"corner", "wing", "center"}


@pytest.mark.parametrize("seed", range(3))
def test_full_scrambles_n6(seed):
    c = Cube(6)
    r = solver().solve(c, c.scramble(rng=random.Random(seed)))
    assert r.correct, r.summary()


def test_simplify_merges_and_cancels():
    assert simplify(parse("R R U", 3)) == parse("R2 U", 3)
    assert simplify(parse("R R U U' R2 L", 3)) == parse("L", 3)  # R2, then U U', then R2 R2
    assert simplify(parse("R R' U2 U2", 3)) == []
