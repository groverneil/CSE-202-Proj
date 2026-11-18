"""Try moves from the command line.

    python -m nxncube 4 "R U 2Rw' x"       apply a move string and print the net
    python -m nxncube 5 --scramble --seed 7 print a seeded scramble and its net
"""

from __future__ import annotations

import argparse

from . import Cube, TextNetViewer, format_moves, parse


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m nxncube")
    ap.add_argument("n", type=int)
    ap.add_argument("moves", nargs="?", default="")
    ap.add_argument("--scramble", action="store_true", help="generate a scramble instead")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--length", type=int, default=None)
    args = ap.parse_args()

    cube = Cube(args.n, seed=args.seed)
    moves = cube.scramble(args.length) if args.scramble else parse(args.moves, args.n)
    if moves:
        print(format_moves(moves, args.n))
        print()
    viewer = TextNetViewer()
    viewer.load(cube.layout())
    state = cube.apply_moves(cube.solved(), moves)
    viewer.show(state)
    print(f"\nsolved: {cube.is_solved(state)}")


if __name__ == "__main__":
    main()
