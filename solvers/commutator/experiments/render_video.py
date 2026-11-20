"""Render the shortest and longest solve of every n (from run_sweep.py) into one video.

    python solvers/commutator/experiments/render_video.py [--fps 30] [--size 720]

Reads ``results/extremes.json`` and records each solve with ``nxncube.viz``
(``Open3DViewer.record`` into one ``MovWriter``): the scramble played quickly, then the
solution phase by phase, with captions for n, which solve it is, and the move counters.
Writes ``results/extremes.mp4`` (H.264 via macOS ``avconvert``), or keeps the Motion-JPEG
``results/extremes.mov`` if that conversion is unavailable.

Long solutions are fast-forwarded so each one lasts at most ``--max-seconds``; several
moves can pass between frames, and the counters show exactly where the solve is.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nxncube import Cube, parse
from nxncube.viz import MovWriter, Open3DViewer, Segment, Timeline, to_mp4

RESULTS = Path(__file__).resolve().parent / "results"
PAUSE = 0.15  # seconds held before each segment (scramble and each phase)


def cost(moves) -> float:
    """Timeline duration units: a half turn counts 1.5 quarter turns."""
    return sum(1.5 if m.turns == 2 else 1.0 for m in moves)


def build(entry: dict, scramble_seconds: float, min_s: float, max_s: float, per_move: float):
    n = entry["n"]
    cube = Cube(n)
    scramble = parse(entry["scramble"], n)
    phases = [(label, parse(text, n)) for label, text in entry["phases"]]
    solution = [m for _, ms in phases for m in ms]
    target = min(max(per_move * len(solution), min_s), max_s)
    sol_speed = cost(solution) / max(target - PAUSE * len(phases), 1.0)
    segments = [Segment("scramble", scramble, speed=cost(scramble) / scramble_seconds)]
    segments += [Segment(label, ms, speed=sol_speed) for label, ms in phases if ms]
    timeline = Timeline(cube, cube.solved(), segments, pause=PAUSE)
    assert cube.is_solved(timeline.final_state), (n, entry["which"])
    return cube, timeline, len(scramble), len(solution)


def captions(entry: dict, timeline: Timeline, n_scramble: int, n_solution: int):
    head = f"n={entry['n']}  {entry['which']} of 100  (seed {entry['seed']})"
    tail = f"solution {n_solution} moves  |  solved in {entry['seconds']:.2f}s"

    def lines(frame) -> list[str]:
        seg = timeline.segments[frame.segment]
        if frame.index >= timeline.num_moves and frame.move is None:
            return [head, f"SOLVED  {n_solution}/{n_solution}", tail]
        if seg.label == "scramble":
            return [head, f"scramble  {frame.index_in_segment}/{n_scramble}", tail]
        done = frame.index - n_scramble
        return [head, f"{seg.label}  {frame.index_in_segment}/{len(seg.moves)}    "
                      f"solution {done}/{n_solution}", tail]

    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--scramble-seconds", type=float, default=3.0)
    ap.add_argument("--min-seconds", type=float, default=6.0, help="shortest a solution plays")
    ap.add_argument("--max-seconds", type=float, default=24.0, help="longest a solution plays")
    ap.add_argument("--seconds-per-move", type=float, default=0.1)
    args = ap.parse_args()

    extremes = json.loads((RESULTS / "extremes.json").read_text())
    extremes.sort(key=lambda e: (e["n"], e["which"] != "shortest"))
    mov, mp4 = RESULTS / "extremes.mov", RESULTS / "extremes.mp4"
    t0 = time.perf_counter()
    with MovWriter(mov, args.size, args.size, fps=args.fps) as writer:
        for e in extremes:
            cube, tl, n_scr, n_sol = build(e, args.scramble_seconds, args.min_seconds,
                                           args.max_seconds, args.seconds_per_move)
            frames = Open3DViewer(cube).record(tl, writer, caption=captions(e, tl, n_scr, n_sol),
                                              lead_in=1.0, hold=1.5)
            print(f"n={e['n']:2d} {e['which']:8s}: {n_sol:5d} moves -> {frames} frames "
                  f"({frames / args.fps:.1f}s)", flush=True)
        total = writer.frames
    print(f"{total} frames ({total / args.fps / 60:.1f} min of video) in "
          f"{time.perf_counter() - t0:.0f}s -> {mov.name} ({mov.stat().st_size / 2**20:.0f} MB)")
    if to_mp4(mov, mp4):
        mov.unlink()
        print(f"converted -> {mp4} ({mp4.stat().st_size / 2**20:.0f} MB)")
    else:
        print(f"avconvert unavailable; kept {mov}")


if __name__ == "__main__":
    main()
