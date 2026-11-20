"""Benchmark the commutator solver: n = 2..10, 100 full scrambles each.

    python solvers/commutator/experiments/run_sweep.py [--sizes 2-10] [--runs 100]

Every solve goes through ``Solver.solve`` on one ``CommutatorSolver`` instance, so timing,
legality and correctness are the solver interface's own, and ``solver.history`` /
``solver.metrics()`` are the official record. Writes to ``results/``:

    runs.csv        one row per solve (for plotting)
    summary.csv     solver.metrics(n) per n, plus library stats
    extremes.json   the shortest and longest solve per n (scramble + phased solution),
                    used by render_video.py

Scrambles are ``cube.scramble(rng=Random(1000 * n + seed))`` at the default length, the
same seeding as ``solvers.conformance``. The first solve of each n also builds that size's
library; its ``seconds`` include the build (``includes_library_build`` = True).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import resource
import time
from pathlib import Path

from nxncube import Cube, format_moves
from solvers.commutator import CommutatorSolver

RESULTS = Path(__file__).resolve().parent / "results"
KINDS = ("corner", "middle_edge", "wing", "center")

RUN_FIELDS = [
    "n", "seed", "pieces", "scramble_length", "legal", "correct", "moves", "qtm",
    "moves_per_piece", "seconds", "cpu_seconds", "includes_library_build", "parity_fixes",
    "setup_fallbacks", *(f"moves_{k}" for k in KINDS), "phases", "max_rss_mb",
]


def parse_sizes(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if r > 2**24 else r / 2**10  # bytes on macOS, KiB on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2-10")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    solver = CommutatorSolver()
    summary, extremes = [], []
    t_all = time.perf_counter()
    with open(RESULTS / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        for n in parse_sizes(args.sizes):
            cube = Cube(n)
            pieces = sum(o.size for o in cube.orbits() if o.kind != "fixed_center")
            results = []
            for seed in range(args.runs):
                scramble = cube.scramble(rng=random.Random(1000 * n + seed))
                r = solver.solve(cube, scramble)
                per = {k: 0 for k in KINDS}
                for label, ms in r.phases:
                    if label in per:
                        per[label] += len(ms)
                w.writerow({
                    "n": n, "seed": seed, "pieces": pieces, "scramble_length": len(r.scramble),
                    "legal": r.legal, "correct": r.correct, "moves": r.length, "qtm": r.qtm,
                    "moves_per_piece": round(r.length / pieces, 3),
                    "seconds": round(r.seconds, 5), "cpu_seconds": round(r.cpu_seconds, 5),
                    "includes_library_build": "library" in r.info,
                    "parity_fixes": r.info.get("parity_fixes"),
                    "setup_fallbacks": r.info.get("setup_fallbacks"),
                    **{f"moves_{k}": v for k, v in per.items()},
                    "phases": len(r.phases), "max_rss_mb": round(rss_mb()),
                })
                f.flush()
                results.append((seed, r))

            m = solver.metrics(n)
            lib = solver.library(cube)
            summary.append({"n": n, "pieces": pieces, **{k: v for k, v in m.items() if k != "sizes"},
                            **{f"library_{k}": v for k, v in lib.stats().items()},
                            "max_rss_mb": round(rss_mb())})
            # shortest / longest by move count; ties go to the lowest seed
            for label, (seed, r) in (("shortest", min(results, key=lambda e: (e[1].length, e[0]))),
                                     ("longest", max(results, key=lambda e: (e[1].length, -e[0])))):
                extremes.append({
                    "n": n, "which": label, "seed": seed, "moves": r.length, "qtm": r.qtm,
                    "seconds": r.seconds, "correct": r.correct,
                    "scramble": format_moves(r.scramble, n),
                    "phases": [[lbl, format_moves(ms, n)] for lbl, ms in r.phases],
                })
            print(f"{time.strftime('%H:%M:%S')} n={n:2d}: {m['correct']}/{m['solves']} correct, "
                  f"moves {m['length_mean']:.0f} (min {m['length_min']}, max {m['length_max']}), "
                  f"time mean {m['time_mean']:.3f}s max {m['time_max']:.3f}s, "
                  f"library {lib.build_seconds:.2f}s, rss {rss_mb():.0f} MB", flush=True)

    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (RESULTS / "extremes.json").write_text(json.dumps(extremes, indent=1))
    print(f"done: {len(solver.history)} solves in {(time.perf_counter() - t_all) / 60:.1f} min; "
          f"all correct: {all(r.correct for r in solver.history)}", flush=True)


if __name__ == "__main__":
    main()
