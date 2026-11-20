"""Visualizer core: mesh, animation and timeline. Needs no Open3D and no display."""

import random
import subprocess
import sys

import numpy as np
import pytest

from nxncube import Axis, Move, parse
from nxncube.geometry import QUARTER_TURN
from nxncube.viz import Animator, Frame, Segment, StickerMesh, Timeline, rotation
from nxncube.viz.animator import BODY_SCALE

from .conftest import cube

SMALL = [2, 3, 4, 5]


def animator(n):
    return Animator(cube(n))


def scrambled(n, seed=0):
    c = cube(n)
    return c.apply_moves(c.solved(), c.scramble(rng=random.Random(seed)))


def end_frame(state, m):
    return Frame(state, m, 1.0, 0, 0, 0)


def quad_key(quads):
    """Order-independent key per quad: its 4 corners, rounded and sorted."""
    q = np.round(quads.reshape(-1, 4, 3), 6)
    return [tuple(map(tuple, sorted(map(tuple, x)))) for x in q]


# ---- mesh (unflatten) ----------------------------------------------------------------


@pytest.mark.parametrize("n", SMALL + [10])
def test_mesh_shape_and_placement(n):
    mesh = StickerMesh.from_layout(cube(n).layout(), gap=0.1)
    N = 6 * n * n
    assert mesh.vertices.shape == (4 * N, 3) and mesh.triangles.shape == (2 * N, 3)
    assert set(np.unique(mesh.triangles)) == set(range(4 * N))
    # every vertex lies on the cube surface, inside [-1, 1]
    assert np.allclose(np.abs(mesh.vertices).max(axis=1), 1.0)
    # quad centres are the sticker centres
    assert np.allclose(mesh.vertices.reshape(N, 4, 3).mean(axis=1), mesh.centers)


@pytest.mark.parametrize("n", SMALL)
def test_triangles_face_outward(n):
    mesh = StickerMesh.from_layout(cube(n).layout())
    v = mesh.vertices[mesh.triangles]
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    sticker_normal = np.repeat(mesh.normals, 2, axis=0)
    assert (np.einsum("ij,ij->i", normals, sticker_normal) > 0).all()


@pytest.mark.parametrize("n", SMALL)
def test_stickers_do_not_overlap(n):
    mesh = StickerMesh.from_layout(cube(n).layout(), gap=0.0)
    keys = quad_key(mesh.vertices)
    assert len(set(keys)) == len(keys)


def test_gap_validation():
    with pytest.raises(ValueError):
        StickerMesh.from_layout(cube(3).layout(), gap=1.0)


# ---- animation agrees with the move generator ----------------------------------------


@pytest.mark.parametrize("axis", list(Axis))
def test_rotation_matches_quarter_turn_convention(axis):
    assert np.allclose(rotation(axis, 1), QUARTER_TURN[axis])
    assert np.allclose(rotation(axis, 2), QUARTER_TURN[axis] @ QUARTER_TURN[axis])
    assert np.allclose(rotation(axis, -1), QUARTER_TURN[axis].T)
    assert np.allclose(rotation(axis, 0), np.eye(3))


@pytest.mark.parametrize("n", SMALL)
def test_animation_end_equals_applied_move(n):
    """For every move in the full set: the turned quads at progress 1, coloured with the
    before-state, are exactly the rest quads coloured with cube.apply(state, move)."""
    c, a = cube(n), animator(n)
    state = scrambled(n)
    rest_keys = {k: i for i, k in enumerate(quad_key(a.mesh.vertices))}
    before = np.frombuffer(state, np.uint8)
    for m in c.all_moves():
        after = np.frombuffer(c.apply(state, m), np.uint8)
        moved = quad_key(a.sticker_vertices(end_frame(state, m)))
        dest = np.array([rest_keys[k] for k in moved])  # where each sticker's quad landed
        assert sorted(dest) == list(range(c.num_stickers)), m
        assert np.array_equal(after[dest], before), m


@pytest.mark.parametrize("turns, quarters", [(1, 1), (2, 2), (3, -1)])
def test_animation_takes_short_way(turns, quarters):
    a = animator(3)
    m = Move(Axis.X, 2, turns)
    s = cube(3).solved()
    half = a.sticker_vertices(Frame(s, m, 0.5, 0, 0, 0))
    turning = a.turning(Frame(s, m, 0.5, 0, 0, 0))[a.mesh.sticker_of_vertex]
    expected = a.mesh.vertices[turning] @ rotation(Axis.X, quarters * 0.5).T
    assert np.allclose(half[turning], expected)
    assert np.allclose(half[~turning], a.mesh.vertices[~turning])
    assert np.allclose(a.sticker_vertices(Frame(s, m, 0.0, 0, 0, 0)), a.mesh.vertices)


def test_colours_follow_state():
    a = animator(3)
    s = cube(3).apply_moves(cube(3).solved(), parse("R U", 3))
    cols = a.sticker_colors(Frame(s, None, 0.0, 0, 0, 0))
    assert np.allclose(cols, a.palette[np.frombuffer(s, np.uint8)][a.mesh.sticker_of_vertex])


@pytest.mark.parametrize("n", SMALL)
def test_body_slabs(n):
    a = animator(n)
    s = cube(n).solved()
    v, t = a.body(Frame(s, None, 0.0, 0, 0, 0))
    assert len(v) == 8 and len(t) == 12
    assert np.allclose(np.abs(v), BODY_SCALE)
    # a wide turn on the R side splits the core into a turning slab and a still one
    m = Move(Axis.X, range(n // 2, n), 1)
    v, t = a.body(Frame(s, m, 0.0, 0, 0, 0))
    assert len(v) == 16 and t.max() == 15
    # an inner slice splits it into three
    if n >= 3:
        v, _ = a.body(Frame(s, Move(Axis.Y, 1, 1), 0.0, 0, 0, 0))
        assert len(v) == 24


# ---- timeline ------------------------------------------------------------------------


def test_timeline_durations_and_speeds():
    c = cube(3)
    r, r2, u = parse("R R2 U", 3)
    tl = Timeline(
        c, c.solved(),
        [Segment("a", [r, r2]), Segment("b", [u, r], speed=6.0), Segment("c", [r2, u], move_speeds=[1.0, 2.0])],
        speed=2.0, half_turn_cost=1.5, pause=0.5,
    )
    assert tl.durations == pytest.approx([0.5, 0.75, 1 / 6, 1 / 6, 1.5, 0.5])
    assert tl.speeds == [2.0, 2.0, 6.0, 6.0, 1.0, 2.0]
    assert tl.duration == pytest.approx(4 * 0.5 + sum(tl.durations))
    assert tl.starts[0] == pytest.approx(0.5)
    assert tl.starts[2] == pytest.approx(0.5 + 0.5 + 0.75 + 0.5)


def test_timeline_frames():
    c = cube(3)
    moves = parse("R U F", 3)
    tl = Timeline.from_moves(c, c.solved(), moves, speed=1.0, pause=1.0)
    f = tl.frame(0.5)  # opening pause
    assert f.move is None and f.state == c.solved() and f.index == 0
    f = tl.frame(1.25)  # a quarter through R
    assert f.move == moves[0] and f.progress == pytest.approx(0.25) and f.state == c.solved()
    f = tl.frame(2.5)  # half through U, after R
    assert f.move == moves[1] and f.index == 1 and f.state == c.apply(c.solved(), moves[0])
    f = tl.frame(100)  # clamped to the end
    assert f.move is None and f.index == 3 and f.state == c.apply_moves(c.solved(), moves)
    assert tl.completed_at(2.0) == 1 and tl.rest_time(1) == pytest.approx(2.0) and tl.rest_time(0) == 0


@pytest.mark.parametrize("n", SMALL)
def test_replay_scramble_then_solution_ends_solved(n):
    """What a solver will hand the viewer: a scramble segment then a solution segment."""
    c = cube(n)
    scramble = c.scramble(rng=random.Random(n))
    solution = [m.inverse() for m in reversed(scramble)]
    tl = Timeline.replay(c, scramble, solution)
    assert [s.label for s in tl.segments] == ["scramble", "solution"]
    assert tl.num_moves == 2 * len(scramble)
    assert not c.is_solved(tl.states[len(scramble)])
    assert tl.final_state == c.solved()
    assert tl.frame(tl.duration).state == c.solved()


def test_solver_phases_become_segments():
    c = cube(4)
    phases = [("centers", parse("Rw U", 4)), ("edges", parse("2R", 4)), ("3x3", parse("F2", 4))]
    tl = Timeline(c, c.solved(), phases)
    assert [s.label for s in tl.segments] == ["centers", "edges", "3x3"]
    f = tl.frame(tl.starts[2] + 1e-6)
    assert (f.segment, f.index_in_segment) == (1, 0)


def test_timeline_rejects_bad_input():
    c = cube(3)
    with pytest.raises(ValueError):
        Timeline.from_moves(c, c.solved(), parse("R", 3), speed=0)
    with pytest.raises(ValueError):
        Segment("x", parse("R U", 3), move_speeds=[1.0])
    with pytest.raises(ValueError):
        Timeline.from_moves(c, c.solved(), [Move(Axis.X, 5, 1)])


def test_empty_timeline():
    c = cube(3)
    tl = Timeline.from_moves(c, c.solved(), [])
    assert tl.num_moves == 0 and tl.frame(0).state == c.solved()


def test_viewer_module_imports_without_open3d():
    code = "import sys; sys.modules['open3d'] = None; import nxncube.viz; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr


@pytest.mark.parametrize("every", [1, 3, 7, 100])
def test_checkpointed_timeline_matches_every_state(every):
    """Sparse checkpoints rebuild exactly the states a full precompute would store."""
    c = cube(4)
    moves = c.scramble(30, rng=random.Random(every))
    expected = [c.solved()]
    for m in moves:
        expected.append(c.apply(expected[-1], m))
    tl = Timeline.from_moves(c, c.solved(), moves, checkpoint_every=every)
    assert tl.final_state == expected[-1]
    assert len(tl.states) == len(expected) and tl.states[-1] == expected[-1]
    order = list(range(len(expected))) + list(reversed(range(len(expected))))
    order += random.Random(0).sample(range(len(expected)), len(expected))  # random seeks
    for k in order:
        assert tl.state(k) == expected[k], k
    with pytest.raises(IndexError):
        tl.state(len(expected))


def test_checkpoint_budget_picks_spacing():
    c = cube(3)
    moves = c.scramble(40, rng=random.Random(1))
    assert Timeline.from_moves(c, c.solved(), moves).checkpoint_every == 1
    tl = Timeline.from_moves(c, c.solved(), moves, checkpoint_budget=54 * 10)
    assert tl.checkpoint_every == 5  # 41 states of 54 bytes in 540 bytes
    assert tl.frame(tl.duration).state == c.apply_moves(c.solved(), moves)
    with pytest.raises(ValueError):
        Timeline.from_moves(c, c.solved(), moves, checkpoint_every=0)


# ---- recording -----------------------------------------------------------------------


def _boxes(data: bytes, start: int = 0, end: int | None = None):
    """Top-level QuickTime boxes in data[start:end] as (type, payload_start, box_end)."""
    import struct

    end = len(data) if end is None else end
    i, out = start, []
    while i < end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", data[i + 8:i + 16])[0], 16
        out.append((kind.decode(), i + head, i + size))
        i += size
    return out


def test_mov_writer_container(tmp_path):
    import struct

    pytest.importorskip("open3d")
    from nxncube.viz import MovWriter

    path = tmp_path / "t.mov"
    with MovWriter(path, 32, 24, fps=10) as w:
        for k in range(3):
            w.write(np.full((24, 32, 3), k / 3.0))
    data = path.read_bytes()
    kinds = [k for k, _, _ in _boxes(data)]
    assert kinds == ["ftyp", "mdat", "moov"]
    _, moov_start, moov_end = _boxes(data)[2]
    stsz = data.index(b"stsz", moov_start)
    count = struct.unpack(">I", data[stsz + 12:stsz + 16])[0]
    assert count == 3
    co64 = data.index(b"co64", moov_start)
    offsets = struct.unpack(">3Q", data[co64 + 12:co64 + 36])
    assert all(data[o:o + 2] == b"\xff\xd8" for o in offsets)  # each sample is a JPEG


def test_record_writes_one_frame_per_tick():
    pytest.importorskip("open3d")
    from nxncube.viz import Open3DViewer

    class Frames:
        width, height, fps = 64, 48, 10

        def __init__(self):
            self.shapes = []

        def write(self, rgb):
            self.shapes.append(np.asarray(rgb).shape)

    c = cube(2)
    tl = Timeline.from_moves(c, c.solved(), parse("R U", 2), speed=2.0, pause=0.0)
    w = Frames()
    k = Open3DViewer(c).record(tl, w, lead_in=0.5, hold=0.2, caption=lambda f: ["test"])
    assert k == len(w.shapes) == 5 + (int(tl.duration * 10) + 1) + 3
    assert set(w.shapes) == {(48, 64, 3)}
