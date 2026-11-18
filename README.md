# CSE-202-Proj

Generic n×n×n cube: moves are generated from geometry as sticker permutations, for any n ≥ 2.
Conventions (axes, face order, facelet indexing, notation, viewer) are in `docs/CONVENTIONS.md`.

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,viz]"
.venv/bin/python -m pytest -q                         # all tests (no display needed)
.venv/bin/python -m nxncube 4 "R U 2Rw' x"            # apply moves, print the text net
.venv/bin/python -m nxncube 5 --scramble --seed 7     # seeded scramble, text net
```

## Reading moves

Moves use SiGN notation: **which layer**, **which face you count from**, **how far to turn**.

```
 6        F        2
 │        │        └── how far: nothing = 90° clockwise, ' = 90° counter-clockwise, 2 = 180°
 │        └─────────── face you count from (and look at to judge "clockwise"): R L U D F B
 └──────────────────── which layer, counting from that face (1 = the face itself)
```

So `6F2` turns the 6th layer from the front by 180°. "Clockwise" is always as seen looking
straight at the named face, even for a layer deep inside the cube: `6F` turns the same way
as `F`, and `L` turns the opposite way to `R` as seen from the front.

| Notation | Meaning | On a 5×5 |
|---|---|---|
| `F` | front face only | layer 1 |
| `3F` | only the 3rd layer from the front | the middle slice |
| `Fw` or `f` | front 2 layers together ("wide") | layers 1–2 |
| `3Fw` | front 3 layers together | layers 1–3 |
| `2-3Fw` | layers 2 to 3 from the front | an inner block |
| `M` `E` `S` | all inner layers; `M` turns like `L`, `E` like `D`, `S` like `F` | layers 2–4 |
| `x` `y` `z` | whole cube; `x` turns like `R`, `y` like `U`, `z` like `F` | all 5 |

A layer can be counted from either side: on a 10×10, `6F2` and `5B2` are the same move. The
parser accepts both; the printer counts from the nearer face (ties go to R, U, F). In code,
`6F2` on an n×n is `Move(axis=z, layers={n-6}, turns=2)`. See `docs/CONVENTIONS.md` for
the full reference.

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
Open3DViewer(cube).play(Timeline.replay(cube, scramble, speed=4))      # just the scramble

result = solver.solve(cube, scramble)                                   # any Solver
Open3DViewer(cube).play(Timeline.from_result(cube, result, speed=4))   # scramble + solution phases

# solver phases, each with its own speed
Timeline(cube, start, [Segment("centers", c_moves, speed=6), Segment("3x3", m3, speed=2)])
```

## Writing a solver

Subclass `Solver` and implement `_solve(cube, state)`. It sees the scrambled state, never
the scramble, and must work for any `cube.n`. Return a move list, or labelled phases.

```python
from nxncube import Cube
from solvers import Solver

class MySolver(Solver):
    name = "my-solver"
    def _solve(self, cube, state):
        ...                                   # cube.n, cube.moves(), cube.apply(), cube.orbits()
        return [("centers", m1), ("3x3", m2)]  # or just a list[Move]

solver = MySolver()
result = solver.solve(Cube(4), "Rw U 2F' L2")  # timed; checks legality and correctness
print(result.summary(), result.correct, result.length, result.seconds)
print(solver.metrics())                        # aggregate over every solve so far
```

Each solver goes in its own module under `solvers/` (e.g. `solvers/bfs.py`), with tests in
`tests/`. Check a new solver with `solvers.conformance.assert_solver_conforms`.
