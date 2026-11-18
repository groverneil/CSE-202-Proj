"""Directions and facelet placement match docs/CONVENTIONS.md, for every n in 2..10."""

import numpy as np
import pytest

from nxncube import Face, parse

U, R, F, D, L, B = Face


def strip(n, kind, k):
    """(row, col) pairs of a row or column of a face."""
    return [(k, i) for i in range(n)] if kind == "row" else [(i, k) for i in range(n)]


# move -> list of (face, "row"/"col", which (0 or -1 for last), colour seen there after the move)
EXPECTED = {
    "R": [(U, "col", -1, F), (F, "col", -1, D), (D, "col", -1, B), (B, "col", 0, U)],
    "U": [(F, "row", 0, R), (R, "row", 0, B), (B, "row", 0, L), (L, "row", 0, F)],
    "F": [(U, "row", -1, L), (R, "col", 0, U), (D, "row", 0, R), (L, "col", -1, D)],
    "L": [(U, "col", 0, B), (F, "col", 0, U), (D, "col", 0, F), (B, "col", -1, D)],
    "D": [(F, "row", -1, L), (R, "row", -1, F), (B, "row", -1, R), (L, "row", -1, B)],
    "B": [(U, "row", 0, R), (L, "col", 0, U), (D, "row", -1, L), (R, "col", -1, D)],
}


@pytest.mark.parametrize("move", list(EXPECTED))
def test_face_turn_moves_adjacent_strips(any_cube, move):
    n = any_cube.n
    s = np.frombuffer(any_cube.apply_moves(any_cube.solved(), parse(move, n)), np.uint8)
    changed = set()
    for face, kind, k, colour in EXPECTED[move]:
        k = k % n
        for r, c in strip(n, kind, k):
            i = any_cube.facelet(face, r, c)
            assert s[i] == colour, (move, face, r, c)
            changed.add(i)
    # everything else still shows its home colour
    solved = np.frombuffer(any_cube.solved(), np.uint8)
    others = [i for i in range(any_cube.num_stickers) if i not in changed]
    assert np.array_equal(s[others], solved[others])


@pytest.mark.parametrize("face", list(Face))
def test_face_turn_is_clockwise_seen_from_outside(any_cube, face):
    """The turned face's top-left sticker moves to its top-right."""
    n = any_cube.n
    p = any_cube.perm(parse(face.name, n)[0])
    assert p[any_cube.facelet(face, 0, n - 1)] == any_cube.facelet(face, 0, 0)
    assert p[any_cube.facelet(face, n - 1, n - 1)] == any_cube.facelet(face, 0, n - 1)


def test_r_carries_f_column_straight_up(any_cube):
    n = any_cube.n
    p = any_cube.perm(parse("R", n)[0])
    for r in range(n):
        assert p[any_cube.facelet(U, r, n - 1)] == any_cube.facelet(F, r, n - 1)
        assert p[any_cube.facelet(F, r, n - 1)] == any_cube.facelet(D, r, n - 1)


@pytest.mark.parametrize("rot, face_after", [("x", {U: F, F: D}), ("y", {F: R, L: F}), ("z", {R: U, U: L})])
def test_rotations_follow_r_u_f(any_cube, rot, face_after):
    s = np.frombuffer(any_cube.apply_moves(any_cube.solved(), parse(rot, any_cube.n)), np.uint8)
    n2 = any_cube.n ** 2
    for face, colour in face_after.items():
        assert (s[face * n2 : (face + 1) * n2] == colour).all()


def test_slices_follow_their_faces():
    from .conftest import cube

    c = cube(3)
    s0 = c.solved()
    # M follows L, E follows D, S follows F, so: x = R M' L', y = U E' D', z = F S B'
    for rot, seq in [("x", "R M' L'"), ("y", "U E' D'"), ("z", "F S B'")]:
        assert c.apply_moves(s0, parse(rot, 3)) == c.apply_moves(s0, parse(seq, 3))
