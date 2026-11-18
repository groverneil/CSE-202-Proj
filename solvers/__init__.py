"""Solvers for nxncube cubes. One module per solution; all subclass ``Solver``.

    solvers/
        base.py          Solver, SolveResult (the interface; no solving logic)
        conformance.py   assert_solver_conforms(...) for testing any solver
        <name>.py        one solver per module, e.g. bfs.py, ida_star.py, kociemba.py, reduction.py
"""

from .base import Solver, SolveResult, quarter_turns

__all__ = ["Solver", "SolveResult", "quarter_turns"]
