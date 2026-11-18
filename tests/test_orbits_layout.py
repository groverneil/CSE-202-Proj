"""Piece view (orbits), layout export and the text net, for every n in 2..10."""

from collections import Counter

import numpy as np
import pytest

from nxncube import TextNetViewer, parse

from .conftest import cube


def expected_orbits(n):
    """(number of orbits, cubies per orbit) for each kind, from the plan's table."""
    exp = {"corner": (1, 8)}
    if n % 2 == 1 and n >= 3:
        exp["middle_edge"] = (1, 12)
        exp["fixed_center"] = (1, 6)  # inner slice moves permute the six fixed centres
    if n >= 4:
        exp["wing"] = ((n - 2) // 2, 24)
        exp["center"] = ((n - 2) ** 2 // 4, 24)
    return exp


def test_orbit_counts(any_cube):
    by_kind = {}
    for o in any_cube.orbits():
        by_kind.setdefault(o.kind, []).append(o.size)
    got = {k: (len(v), v[0]) for k, v in by_kind.items()}
    assert all(len(set(v)) == 1 for v in by_kind.values())
    assert got == expected_orbits(any_cube.n)


def test_orbits_partition_stickers(any_cube):
    all_facelets = np.concatenate([o.facelets.ravel() for o in any_cube.orbits()])
    assert sorted(all_facelets.tolist()) == list(range(any_cube.num_stickers))


def test_every_move_keeps_each_orbit_closed(any_cube):
    for m in any_cube.moves():
        p = any_cube.perm(m)
        for o in any_cube.orbits():
            assert set(p[o.facelets.ravel()].tolist()) == set(o.facelets.ravel().tolist())


def test_pieces_view(any_cube):
    s = any_cube.apply_moves(any_cube.solved(), parse("R U", any_cube.n))
    ps = any_cube.pieces(s)
    assert len(ps.colors) == len(ps.orbits)
    for o, col in zip(ps.orbits, ps.colors):
        assert col.shape == o.facelets.shape


def test_layout_fields(any_cube):
    lay = any_cube.layout()
    N = any_cube.num_stickers
    assert lay.center.shape == lay.normal.shape == lay.layer.shape == (N, 3)
    assert (np.abs(lay.center).max(axis=1) == 1).all()
    assert (np.abs(lay.normal).sum(axis=1) == 1).all()
    assert Counter(lay.face.tolist()) == {f: any_cube.n**2 for f in range(6)}


# ---- text net -------------------------------------------------------------------------

NET_R_N2 = """\
     W G
     W G
O O  G Y  R R  W B
O O  G Y  R R  W B
     Y B
     Y B"""

NET_R_N3 = """\
       W W G
       W W G
       W W G
O O O  G G Y  R R R  W B B
O O O  G G Y  R R R  W B B
O O O  G G Y  R R R  W B B
       Y Y B
       Y Y B
       Y Y B"""


def viewer(n):
    v = TextNetViewer()
    v.load(cube(n).layout())
    return v


def solved_net(n):
    pad = " " * (2 * n + 1)
    row = lambda ch: " ".join(ch * n)
    top = [pad + row("W")] * n
    mid = ["  ".join(row(ch) for ch in "OGRB")] * n
    bot = [pad + row("Y")] * n
    return "\n".join(top + mid + bot)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_solved_net(n):
    assert viewer(n).render(cube(n).solved()) == solved_net(n)


@pytest.mark.parametrize("n, expected", [(2, NET_R_N2), (3, NET_R_N3)])
def test_r_net_exact(n, expected):
    s = cube(n).apply_moves(cube(n).solved(), parse("R", n))
    assert viewer(n).render(s) == expected


@pytest.mark.parametrize("n", [4, 5])
def test_r_net_structure(n):
    s = cube(n).apply_moves(cube(n).solved(), parse("R", n))
    lines = viewer(n).render(s).split("\n")
    w = 2 * n + 1
    face = lambda line, block: line[block * w : block * w + 2 * n - 1].split(" ")
    for r in range(n):
        assert face(lines[r], 1) == ["W"] * (n - 1) + ["G"]          # U
        mid = lines[n + r]
        assert face(mid, 0) == ["O"] * n                               # L
        assert face(mid, 1) == ["G"] * (n - 1) + ["Y"]                # F
        assert face(mid, 2) == ["R"] * n                               # R
        assert face(mid, 3) == ["W"] + ["B"] * (n - 1)                # B
        assert face(lines[2 * n + r], 1) == ["Y"] * (n - 1) + ["B"]  # D
