# CSE-202-Proj

Generic n×n×n cube: moves are generated from geometry as sticker permutations, for any n ≥ 2.
Conventions (axes, face order, facelet indexing, notation, viewer) are in `docs/CONVENTIONS.md`.

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,viz]"
.venv/bin/python -m pytest -q                         # all tests (no display needed)
.venv/bin/python -m nxncube 4 "R U 2Rw' x"            # apply moves, print the text net
.venv/bin/python -m nxncube 5 --scramble --seed 7     # seeded scramble, text net
```

## 3D viewer (Open3D)

```sh
.venv/bin/python -m nxncube.viz 4 --scramble --seed 1 --speed 4
.venv/bin/python -m nxncube.viz 3 --scramble --seed 1 --undo        # scramble, then reverse it
.venv/bin/python -m nxncube.viz 5 "F L F U' R U F2 L2 U' L' B D' B' L2 U" --speed 2
.venv/bin/python -m nxncube.viz 4 --scramble --then "<moves>"       # e.g. a pasted solution
```

Keys: space pause/play · → ← step one move · + − speed · r restart · q / esc quit.

```python
from nxncube import Cube
from nxncube.viz import Open3DViewer, Segment, Timeline

cube = Cube(4, seed=1)
scramble = cube.scramble()
solution = solver.solve(cube, cube.apply_moves(cube.solved(), scramble))   # any Solver
Open3DViewer(cube).play(Timeline.replay(cube, scramble, solution, speed=4))

# solver phases, each with its own speed
Timeline(cube, start, [Segment("centers", c_moves, speed=6), Segment("3x3", m3, speed=2)])
```
