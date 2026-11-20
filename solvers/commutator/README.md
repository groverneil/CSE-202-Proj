# Commutator solver (piece by piece)

```python
from nxncube import Cube
from solvers.commutator import CommutatorSolver

cube = Cube(6)
solver = CommutatorSolver()
result = solver.solve(cube, cube.scramble())
print(result.summary())   # commutator n=6: ok, ... moves
print(result.phases)      # one phase per orbit: corner, wing, center, ...
print(result.info)        # parity_fixes, setup_fallbacks, library stats (first solve per n)
```

Solves **any legal scramble of any n x n x n cube** by placing one piece at a time with
short move sequences that each disturb only three pieces. It is a complete, guaranteed
baseline: solutions are long (about 6n² moves) but it never searches the whole cube, so time
and memory stay polynomial and small.

| file | contents |
|---|---|
| `library.py` | `CommutatorLibrary`: finds the pure commutators and the setup table, once per n |
| `solver.py`  | `CommutatorSolver(Solver)`: goal orientation, greedy placement, parity fixes |

## Idea

**Commutator.** For move sequences A and B, the commutator `[A, B] = A B A' B'` undoes
almost everything it does: only pieces that both A and B touch end up moved. When that
overlap is small, the net effect is a **3-cycle** (three pieces trade places) and *nothing
else on the cube changes*. Example: `[R U R', D] = R U R' D R U' R' D'` cycles three corners.

**Conjugate.** If X cycles positions (x, y, z), then `S X S'` cycles whichever positions the
setup S carries onto (x, y, z), and is still pure: S' undoes S. So a few commutators plus all
short setups give a 3-cycle on almost any three positions.

**Orbits.** Moves only ever send a piece to a position of the same kind: corners, middle
edges (odd n), each wing orbit, each center orbit, fixed centers (odd n). Each orbit is an
independent sub-puzzle, and a pure macro touches only one orbit, so orbits are solved one
after another without ever disturbing finished ones.

## Why it always works

1. **Every even permutation is a product of 3-cycles.** Greedy placement fixes at least one
   piece per step: to fill position p, cycle (q → p → r) where q holds p's piece and r is any
   other unsolved position. Solved pieces are never moved, so it terminates in at most
   |orbit| steps.
2. **Parity.** 3-cycles are even, so they cannot undo a lone swap (an odd permutation): an
   orbit can end with exactly two pieces swapped. On a real cube such a swap always has a
   hidden partner, because a quarter turn changes the parity of several orbits at once (corners
   together with centers, wing orbit i together with centers). One quarter turn that leaves
   every finished orbit intact makes the orbit even; its partner orbit absorbs the extra swap.
   Center orbits never get stuck: each colour appears 4 times per orbit, so a "swap" is a
   3-cycle through a third, already-solved center of the same colour.
3. **Orientation.** Corner twists always sum to 0 (mod 3) and middle-edge flips to 0 (mod 2),
   so leftover twists/flips pair up and are fixed with pure in-place twist/flip macros.
   A wing's orientation is fixed by its position, so wings never need flipping.
4. **Reachability.** On each orbit the moves generate every even arrangement, so for any three
   positions *some* setup moves them onto a commutator. The implementation tries setups of up
   to 3 moves; the guarantee is mathematical, the 3-move bound is a practical cap (it has
   always sufficed in testing, and the search can be lengthened).

## Algorithm

### Library (once per n)

1. **Commutator search.** For every A of the shapes real commutators use (single moves, move
   pairs `x y`, conjugates `x y x'`, and `(x y x' y')²` for in-place twists) and every single
   move b, compute the sticker permutation of `[A, b]` and keep it if it is *pure*: it moves
   at most 3 pieces, all in one orbit, and nothing else. Keep the shortest sequence per effect.
   (Searching every 3-move A would be (9n)⁴ commutators; these shapes are about 3·(9n)³.)
2. **Selection.** Per orbit keep the 40 shortest pure commutators plus a few pure
   twists/flips, and the inverse of each.
3. **Composed flips.** If a corner or middle-edge orbit still has no in-place twist/flip,
   search products `X · (S Y S')` of two of its macros (S at most one move) for one. A product
   of pure macros of one orbit is still pure. (Middle-edge flips on odd n come from here.)
4. **Setup table.** Every setup S of up to 2 moves as a permutation row, sorted by length,
   plus the solved state in all 24 whole-cube orientations.

### Placing one piece (the inner search)

A setup S with permutation P_S turns macro X (which sets `new[X.dst] = old[X.src]`) into
`S X S'`, which sets `new[P_S[X.dst]] = old[P_S[X.src]]` and changes nothing else. So for one
macro, gathering `P_S[X.dst]` and `P_S[X.src]` for **all** setups at once (one numpy
operation over the table) shows every conjugate's exact effect without applying any moves.
A conjugate is accepted when the stickers it changes that must be (or stay) correct get their
goal colours and it covers the target position. The cheapest accepted `2|S| + |X|` wins; if
none, setups one move longer are tried, built as `S + [m]` on the fly.

### Pseudocode

```text
SOLVE(cube, state):
    L    ← LIBRARY(cube.n)                       # cached per n
    goal ← orientation of solved that matches the fixed centers (odd n),
           else the orientation matching the most stickers (even n)
    moves ← []
    for orbit O in order [fixed centers, corners, middle edges, wings, centers]:
        (seq, state', ok) ← SOLVE_ORBIT(L, O, state, goal)
        if not ok:                                # odd permutation left in O
            for each quarter turn q that keeps every earlier orbit equal to goal:
                (seq, state', ok) ← SOLVE_ORBIT(L, O, apply(state, q), goal)
                if ok: seq ← [q] + seq; break
            if not ok: fail                       # never happens on a legal state
        moves ← moves + SIMPLIFY(seq)
        state ← state'
    return moves

SOLVE_ORBIT(L, O, s, goal):
    seq ← []
    loop:
        solved ← {positions p in O whose stickers equal goal}
        if solved = O: return (seq, s, true)
        for each unsolved position p in O:
            need ← stickers of solved positions ∪ stickers of p
            c ← PLACE(L, O, s, goal, need, target = stickers of p)
            if c = (S, X):
                s ← apply(s, S X S');  seq ← seq + S + X + S'
                continue loop
        return (seq, s, false)                    # stuck: lone swap, needs a parity fix

PLACE(L, O, s, goal, need, target):
    best ← none
    for each macro X of O:                        # X sets new[X.dst] = old[X.src]
        for all setups S at once:                 # rows of the setup table, shortest first
            D ← P_S[X.dst];  R ← P_S[X.src]       # S X S' sets new[D] = old[R]
            valid(S) ← target ⊆ D  and  ∀ i: (D[i] ∉ need or s[R[i]] = goal[D[i]])
        S* ← shortest valid S;  keep (S*, X) if 2|S*| + |X| is the best so far
    if best = none: repeat with setups S + [m] one move longer
    return best
```

## Complexity

With M = 9n single-layer moves and N = 6n² stickers:

| | cost |
|---|---|
| library: commutator search | ~3M³ commutators, O(N) each: O(n⁵) |
| library: setup table | ~M² rows × N: O(n⁴) memory (about 20 MB at n = 10) |
| one placement | (#macros) × (#setups) × (3-piece check): O(n²) numpy work |
| placements per solve | about the number of pieces, O(n²) |
| **solution length** | **Θ(n²)**, about 6 moves per piece |
| memory | O(n⁴) setup table + O(n²) state; no search frontier |

The optimal bound is Θ(n²/log n) moves (Demaine et al. 2011), reached by moving many
pieces with one commutator; this solver places one piece at a time, so it is a constant
plus log factor longer.

## Results

Default scramble length, M4 Pro, CPU only:

| n | scrambles solved | mean moves | moves / piece | mean solve time | library build | peak RSS |
|---|---|---|---|---|---|---|
| 2 | 5/5 | 64 | 8.0 | 0.16 s | < 0.1 s | — |
| 3 | 5/5 | 142 | 7.1 | 0.13 s | 0.4 s | — |
| 4 | 5/5 | 327 | 5.8 | 0.52 s | 0.9 s | — |
| 5 | 5/5 | 551 | 6.0 | 1.1 s | 1.4 s | — |
| 6 | 3/3 | 878 | 5.8 | 2.7 s | 1.2 s | 118 MB |
| 7 | 3/3 | 1,298 | 6.1 | 5.6 s | 1.9 s | 241 MB |
| 10 | 3/3 | 2,792 | 5.7 | 31 s | 6.2 s | 779 MB |

(Earlier sandbox runs of the same algorithm: 20/20 at n = 4, 10/10 at n = 8, 10/10 at n = 10.)

Time is dominated by parity fixes: each one re-solves its orbit from scratch, and the
failed first attempt also tries the slower longer setups before giving up.

## Limits and next steps

- **Long solutions.** About 6n² moves (reduction + Kociemba is far shorter). Cheaper piece
  ordering, cancelling across placements, and parallel commutators would cut this.
- **Parity by dry run.** Odd parity is detected by solving until stuck. Computing it directly
  from the piece permutation (corners and wings are distinguishable) would skip the failed
  attempt and roughly halve solve time.
- **Setup cap.** Setups longer than 3 moves are never tried. Every tested scramble was solved
  within the cap, but a fully general version would keep lengthening setups when needed.
