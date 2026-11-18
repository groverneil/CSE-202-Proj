"""The Solver base class. The solvers here are test doubles with no solving logic."""

import random
import time

import pytest

from nxncube import Axis, Cube, Move, format_moves, parse
from solvers import Solver, SolveResult

from .conftest import cube
from solvers.conformance import assert_solver_conforms, conformance_scramble

# ---- test doubles ----------------------------------------------------------------------


class NoMoves(Solver):
    def _solve(self, cube, state):
        return []


class IllegalMove(Solver):
    def _solve(self, cube, state):
        return [Move(Axis.X, cube.n, 1)]  # layer n does not exist


class NotMoves(Solver):
    def _solve(self, cube, state):
        return ["R", "U"]


class Sleepy(Solver):
    name = "sleepy"

    def _solve(self, cube, state):
        time.sleep(0.05)
        self.info["note"] = "slept"
        return []


class Lookup(Solver):
    """Answers from a table the test fills in: state -> moves. Not a solver."""

    def __init__(self, table=None, phased=False):
        super().__init__()
        self.table = table or {}
        self.phased = phased
        self.seen = []

    def _solve(self, cube, state):
        self.seen.append(state)
        moves = self.table[state]
        if self.phased:
            half = len(moves) // 2
            return [("first", moves[:half]), ("second", moves[half:])]
        return moves


def undo(moves):
    return [m.inverse() for m in reversed(moves)]


def lookup_for(c: Cube, scramble, **kw) -> Lookup:
    state = c.apply_moves(c.solved(), scramble)
    return Lookup({state: undo(scramble)}, **kw)


# ---- the abstract class ----------------------------------------------------------------


def test_solver_is_abstract():
    with pytest.raises(TypeError):
        Solver()


def test_correct_solve_result(any_cube):
    """One result per n = 2..10: legal, correct, timed, measured."""
    scramble = any_cube.scramble(rng=random.Random(5))
    s = lookup_for(any_cube, scramble)
    r = s.solve(any_cube, scramble)
    assert isinstance(r, SolveResult)
    assert r.legal and r.correct
    assert r.n == any_cube.n and r.solver == "Lookup"
    assert r.scramble == tuple(scramble) and r.moves == tuple(undo(scramble))
    assert r.length == len(scramble)
    assert r.qtm == sum(2 if m.turns == 2 else 1 for m in scramble)
    assert r.seconds >= 0 and r.cpu_seconds >= 0
    assert r.phases == (("solution", r.moves),)


def test_solver_sees_state_not_scramble():
    c = cube(3)
    scramble = parse("R U F", 3)
    s = lookup_for(c, scramble)
    s.solve(c, scramble)
    assert s.seen == [c.apply_moves(c.solved(), scramble)]


def test_scramble_as_string():
    c = cube(4)
    text = "Rw U 2F' L2"
    s = lookup_for(c, parse(text, 4))
    assert s.solve(c, text).correct


def test_illegal_scramble_rejected():
    c = cube(3)
    with pytest.raises(ValueError):
        NoMoves().solve(c, [Move(Axis.Y, 3, 1)])
    with pytest.raises(ValueError):
        NoMoves().solve(c, "3Rw 4R")


def test_wrong_and_illegal_answers_are_flagged():
    c = cube(3)
    scramble = parse("R U", 3)
    r = NoMoves().solve(c, scramble)
    assert r.legal and not r.correct and "WRONG" in r.summary()
    r = IllegalMove().solve(c, scramble)
    assert not r.legal and not r.correct and "ILLEGAL" in r.summary()
    r = NotMoves().solve(c, scramble)
    assert not r.legal and not r.correct


def test_empty_scramble_with_no_moves_is_correct():
    assert NoMoves().solve(cube(5), []).correct


def test_rotation_finish_counts_as_solved():
    """is_solved accepts any orientation, so ending rotated is still correct."""
    c = cube(3)
    scramble = parse("R", 3)
    assert Solver.is_correct(c, scramble, parse("R' x y", 3))


def test_is_legal_and_is_correct_static_checks():
    c = cube(3)
    assert Solver.is_legal(c, parse("R U M x 2R", 3))
    assert not Solver.is_legal(c, [Move(Axis.Z, 5, 2)])
    assert not Solver.is_legal(c, "R U")  # a string is not a move list
    assert Solver.is_correct(c, "R U", parse("U' R'", 3))
    assert not Solver.is_correct(c, "R U", parse("R' U'", 3))
    assert not Solver.is_correct(c, "R U", [Move(Axis.X, 7, 1)])


def test_timing_and_info():
    r = Sleepy().solve(cube(2), parse("R", 2))
    assert r.seconds >= 0.05
    assert r.solver == "sleepy" and r.info == {"note": "slept"}


def test_phases_are_kept_and_joined():
    c = cube(4)
    scramble = c.scramble(rng=random.Random(1))
    r = lookup_for(c, scramble, phased=True).solve(c, scramble)
    assert [label for label, _ in r.phases] == ["first", "second"]
    assert r.moves == tuple(undo(scramble)) and r.correct


def test_metrics_accumulate_per_instance():
    s = Lookup()
    for n in (2, 3, 3):
        c = cube(n)
        scramble = c.scramble(rng=random.Random(len(s.history)))
        s.table[c.apply_moves(c.solved(), scramble)] = undo(scramble)
        s.solve(c, scramble)
    m = s.metrics()
    assert m["solves"] == 3 and m["correct"] == 3 and m["success_rate"] == 1.0
    assert m["sizes"] == [2, 3]
    assert m["length_min"] <= m["length_mean"] <= m["length_max"]
    assert m["time_total"] >= m["time_max"] >= m["time_mean"] >= 0
    assert s.metrics(n=3)["solves"] == 2
    assert s.metrics(n=7) == {"solves": 0}
    s.reset_metrics()
    assert s.metrics() == {"solves": 0}


def test_conformance_helper_with_lookup_double():
    """The conformance check a real solver will run, exercised with a pre-filled table."""
    sizes, seeds = [2, 3, 4, 5, 8, 9], range(2)
    table = {}
    for n in sizes:
        c = Cube(n)
        for seed in seeds:
            scr = conformance_scramble(c, seed)
            table[c.apply_moves(c.solved(), scr)] = undo(scr)
    results = assert_solver_conforms(Lookup(table), sizes=sizes, seeds=seeds)
    assert len(results) == len(sizes) * len(seeds)


def test_conformance_helper_rejects_wrong_solver():
    with pytest.raises(AssertionError):
        assert_solver_conforms(NoMoves(), sizes=[3], seeds=[0])


def test_result_plays_in_viewer_timeline():
    from nxncube.viz import Timeline

    c = cube(4)
    scramble = c.scramble(rng=random.Random(2))
    r = lookup_for(c, scramble, phased=True).solve(c, scramble)
    tl = Timeline.from_result(c, r)
    assert [s.label for s in tl.segments] == ["scramble", "first", "second"]
    assert tl.final_state == c.solved()
    assert format_moves(r.moves, 4)  # result moves print as notation
