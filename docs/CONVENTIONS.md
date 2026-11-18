# Cube conventions

Every module follows this file. If code and this file disagree, the code has a bug.

## Axes and coordinates

Right-handed frame, viewed from the front:

| Axis | Points toward | Positive face | Negative face |
|------|---------------|---------------|---------------|
| x    | right         | R             | L             |
| y    | up            | U             | D             |
| z    | front         | F             | B             |

- A cubie's coordinates lie in `{-(n-1), -(n-3), ..., n-1}` on each axis. The step of 2
  keeps every coordinate an integer for both odd and even `n`.
- A sticker sits at its cubie's coordinates with the face coordinate pushed out to `±n`.
- **Layer index** on an axis is `(c + n - 1) / 2` where `c` is the cubie coordinate.
  Layer `0` is on the negative face (L, D, B); layer `n-1` is on the positive face (R, U, F).

## Faces and colours

Face order is `U R F D L B` (indices 0–5). The solved state gives every sticker on face
`f` the value `f`. Colours are a viewer concern; the default text palette is
`U=W R=R F=G D=Y L=O B=B`.

## Facelet indexing

`index = face * n² + row * n + col`, where row and column are read **as the face is seen
from outside in the standard net**:

```
        U
    L   F   R   B
        D
```

| Face | Column increases toward | Row increases toward |
|------|-------------------------|----------------------|
| U    | +x (R)                  | +z (F)               |
| R    | -z (B)                  | -y (D)               |
| F    | +x (R)                  | -y (D)               |
| D    | +x (R)                  | -z (B)               |
| L    | +z (F)                  | -y (D)               |
| B    | -x (L)                  | -y (D)               |

So row 0 of U touches B, row 0 of D touches F, and a face printed row by row reads exactly
as it appears in the net. No flipping is needed for display.

## Moves

A move is `Move(axis, layers, turns)`:

- `axis` ∈ {x, y, z}, `layers` a non-empty set of layer indices on that axis, `turns` ∈ {1, 2, 3}.
- **One turn is a quarter turn clockwise as seen from the positive end of the axis**
  (looking at the R, U or F face). So `turns=1` on the top x-layer is `R`, and on the bottom
  x-layer it is `L'`.
- Inverse of `(axis, layers, t)` is `(axis, layers, 4 - t)`. Moves on the same axis commute.
- Applying a move: `new_state = state[perm]`, i.e. `perm[i]` is the index the sticker now
  at `i` came from.

Quarter-turn rotation matrices (applied to column vectors):

```
x: (x, y, z) -> (x,  z, -y)      R moves F stickers to U
y: (x, y, z) -> (-z, y,  x)      U moves F stickers to L
z: (x, y, z) -> (y, -x,  z)      F moves U stickers to R
```

## Notation (SiGN)

Depth `d` counts layers from a face, starting at 1.

| Token        | Meaning                                                    |
|--------------|------------------------------------------------------------|
| `R`          | depth 1 from R                                             |
| `3R`         | depth 3 only (a single inner slice)                        |
| `Rw`, `r`    | depths 1–2                                                 |
| `3Rw`        | depths 1–3                                                 |
| `2-3Rw`, `2-3R` | depths 2–3                                              |
| `M` `E` `S`  | every inner layer (depths 2 … n-1); M follows L, E follows D, S follows F |
| `x` `y` `z`  | whole cube; follows R, U, F                                |
| suffix `'` / `2` | inverse / half turn (`2'` is read as `2`)              |

L, D, B, M and E turn clockwise as seen from *their own* side, which is `turns = 4 - t`
in axis terms.

The printer emits a canonical form: rotations as `x y z`, exactly-all-inner as `M E S`
(n ≥ 3), blocks touching a face as `R`/`Rw`/`kRw`, and inner slices from the nearer face
(ties go to R, U, F).

## Metric

The default move set (`Cube.moves()`) is the `9n` single-layer primitives
`(axis, {layer}, turns)`. Which moves count as "one move" is a property of the move set
passed to a solver, not of the cube.

## Viewer

- 3D units: the cube spans [-1, 1]³; sticker centres are `layout.center` (= sticker point / n).
  Each sticker is a square quad of half-width `(1 - gap) / n`, wound counter-clockwise seen
  from outside (`nxncube/viz/mesh.py`).
- Animation: a move with `turns = 1, 2, 3` animates `+1, +2, -1` quarter turns (prime turns
  go the short way). Angle sign follows the move convention above, so at the end of the
  animation the quads land exactly where `cube.apply` puts the stickers.
- Speed is in quarter turns per second. A half turn costs `half_turn_cost` (default 1.5)
  quarter turns of time. Per-move speed > segment speed > timeline default.
- Palette (3D and text): U white, R red, F green, D yellow, L orange, B blue.
