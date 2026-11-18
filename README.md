# CSE-202-Proj

Generic n×n×n cube: moves are generated from geometry as sticker permutations, for any n ≥ 2.
Conventions (axes, face order, facelet indexing, notation) are in `docs/CONVENTIONS.md`.

```sh
python3 -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/python -m pytest -q                         # correctness tests, n = 2..10
.venv/bin/python -m nxncube 4 "R U 2Rw' x"            # apply moves, print the net
.venv/bin/python -m nxncube 5 --scramble --seed 7     # seeded scramble
```

```python
from nxncube import Cube, parse, format_moves
c = Cube(5, seed=1)
s = c.apply_moves(c.solved(), parse("3Rw2 U' M", 5))   # states are immutable bytes
c.is_solved(s), c.orbits(), c.pieces(s), c.layout()
```
