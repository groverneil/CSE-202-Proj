"""Animate move sequences in a 3D window.

    python -m nxncube.viz 4 --scramble --seed 1 --speed 4
    python -m nxncube.viz 3 "F L F U' R U F2 L2 U' L' B D' B' L2 U" --speed 2
    python -m nxncube.viz 5 --scramble --seed 7 --then "<moves>"   # e.g. a pasted solution
    python -m nxncube.viz 3 --scramble --seed 1 --undo              # scramble, then reverse it
"""

from __future__ import annotations

import argparse

from .. import Cube, parse
from .open3d_viewer import Open3DViewer
from .timeline import Segment, Timeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m nxncube.viz")
    ap.add_argument("n", type=int)
    ap.add_argument("moves", nargs="?", default="", help="moves to play (SiGN notation)")
    ap.add_argument("--scramble", action="store_true", help="play a random scramble first")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--length", type=int, default=None, help="scramble length")
    ap.add_argument("--then", default=None, help="a further segment, e.g. a solution")
    ap.add_argument("--then-label", default="solution")
    ap.add_argument("--undo", action="store_true", help="finish by reversing everything")
    ap.add_argument("--speed", type=float, default=3.0, help="quarter turns per second")
    ap.add_argument("--paused", action="store_true", help="start paused (step with arrow keys)")
    ap.add_argument("--gap", type=float, default=0.1, help="gap between stickers, 0..1")
    args = ap.parse_args()

    cube = Cube(args.n, seed=args.seed)
    segments = []
    if args.scramble:
        segments.append(Segment("scramble", cube.scramble(args.length)))
    if args.moves:
        segments.append(Segment("moves", parse(args.moves, args.n)))
    if args.then:
        segments.append(Segment(args.then_label, parse(args.then, args.n)))
    if args.undo:
        played = [m for s in segments for m in s.moves]
        segments.append(Segment("undo", [m.inverse() for m in reversed(played)]))

    timeline = Timeline(cube, cube.solved(), segments, speed=args.speed, progress=True)
    Open3DViewer(cube, gap=args.gap).play(timeline, paused=args.paused)


if __name__ == "__main__":
    main()
