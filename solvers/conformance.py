"""Reusable checks every real solver should pass. Use from a solver's test file:

    from solvers.conformance import assert_solver_conforms
    def test_my_solver():
        assert_solver_conforms(MySolver(), sizes=[2, 3, 4, 5])
"""

import inspect
import random

from nxncube import Cube

from .base import Solver


def conformance_scramble(cube: Cube, seed: int):
    """The scramble the conformance check uses for (n, seed); deterministic."""
    return cube.scramble(rng=random.Random(1000 * cube.n + seed))


def assert_solver_conforms(solver: Solver, sizes=(2, 3, 4, 5), seeds=range(3)):
    assert isinstance(solver, Solver)
    # size agnostic: constructible without n, and one instance handles every size
    init = inspect.signature(type(solver).__init__)
    assert "n" not in init.parameters, "solvers must not take the puzzle size"
    results = []
    for n in sizes:
        cube = Cube(n)
        for seed in seeds:
            scramble = conformance_scramble(cube, seed)
            r = solver.solve(cube, scramble)
            assert r.legal, r.summary()
            assert r.correct, r.summary()
            assert Solver.is_correct(cube, scramble, r.moves)
            results.append(r)
    assert solver.metrics()["sizes"] == sorted(set(sizes))
    return results
