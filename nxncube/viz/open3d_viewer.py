"""Open3D window that plays a Timeline. Open3D is imported only when a window opens."""

from __future__ import annotations

import sys
import time
from typing import TextIO

import numpy as np

from ..move import Move
from ..notation import format_move
from .animator import Animator
from .palette import BACKGROUND_RGB, BODY_RGB, DEFAULT_RGB
from .timeline import Timeline

# GLFW key codes
_SPACE, _RIGHT, _LEFT, _PLUS, _KP_PLUS, _MINUS, _KP_MINUS, _R, _Q = 32, 262, 263, 61, 334, 45, 333, 82, 81
SPEED_STEP = 1.5

CONTROLS = "keys: space pause/play | → ← step | + − speed | r restart | q / esc quit"


class Open3DViewer:
    """Viewer for any Cube. Takes the cube as input; never changes states itself."""

    def __init__(
        self,
        cube,
        gap: float = 0.1,
        palette: np.ndarray = DEFAULT_RGB,
        window_size: tuple[int, int] = (900, 900),
        out: TextIO | None = None,
    ):
        self.cube = cube
        self.animator = Animator(cube, gap, palette)
        self.window_size = window_size
        self.out = out or sys.stdout

    # ---- Viewer protocol ----------------------------------------------------------------

    def load(self, layout) -> None:
        """The layout comes from the cube passed to the constructor; kept for the protocol."""

    def show(self, s: bytes) -> None:
        self.play(Timeline.from_moves(self.cube, s, [], label="state"))

    def animate(self, s: bytes, m: Move) -> None:
        self.play(Timeline.from_moves(self.cube, s, [m]))

    # ---- playback -----------------------------------------------------------------------

    def play(self, timeline: Timeline, speed_multiplier: float = 1.0, paused: bool = False,
             title: str | None = None) -> None:
        import open3d as o3d  # lazy: tests and the text viewer never need it

        anim, n = self.animator, self.cube.n
        stickers = o3d.geometry.TriangleMesh()
        stickers.triangles = o3d.utility.Vector3iVector(anim.mesh.triangles)
        body = o3d.geometry.TriangleMesh()

        clock = {"t": 0.0, "last": time.perf_counter(), "paused": paused,
                 "mult": speed_multiplier, "shown": None}

        def fill(frame) -> None:
            stickers.vertices = o3d.utility.Vector3dVector(anim.sticker_vertices(frame))
            stickers.vertex_colors = o3d.utility.Vector3dVector(anim.sticker_colors(frame))
            stickers.compute_vertex_normals()
            bv, bt = anim.body(frame)
            body.vertices = o3d.utility.Vector3dVector(bv)
            body.triangles = o3d.utility.Vector3iVector(bt)
            body.paint_uniform_color(BODY_RGB)
            body.compute_vertex_normals()

        def render(vis) -> None:
            frame = timeline.frame(clock["t"])
            fill(frame)
            vis.update_geometry(stickers)
            vis.update_geometry(body)
            self._report(timeline, frame, clock)

        def tick(vis) -> bool:
            now = time.perf_counter()
            if not clock["paused"]:
                clock["t"] = min(clock["t"] + (now - clock["last"]) * clock["mult"], timeline.duration)
            clock["last"] = now
            render(vis)
            return True

        def step(delta: int):
            def cb(vis) -> bool:
                clock["paused"] = True
                frame = timeline.frame(clock["t"])
                k = frame.index
                if delta > 0:
                    target = k + 1
                else:  # mid-move: back to its start; at rest: one move back
                    target = k if frame.move is not None and frame.progress > 0 else k - 1
                clock["t"] = timeline.rest_time(target)
                render(vis)
                return True
            return cb

        def toggle(vis) -> bool:
            clock["paused"] = not clock["paused"]
            if not clock["paused"] and clock["t"] >= timeline.duration:
                clock["t"] = 0.0
            print(f"  {'paused' if clock['paused'] else 'playing'}", file=self.out)
            return False

        def speed(factor: float):
            def cb(vis) -> bool:
                clock["mult"] *= factor
                print(f"  speed x{clock['mult']:.2f}", file=self.out)
                return False
            return cb

        def restart(vis) -> bool:
            clock["t"], clock["shown"] = 0.0, None
            return False

        def quit_(vis) -> bool:
            vis.close()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(title or f"{n}x{n}x{n}", *self.window_size)
        fill(timeline.frame(0.0))
        vis.add_geometry(body)
        vis.add_geometry(stickers)
        render(vis)
        vis.reset_view_point(True)
        opt = vis.get_render_option()
        opt.background_color = BACKGROUND_RGB
        opt.mesh_show_back_face = False
        ctrl = vis.get_view_control()
        ctrl.set_lookat([0.0, 0.0, 0.0])
        ctrl.set_front([0.55, 0.5, 0.67])  # camera up-right-front: U, F and R visible
        ctrl.set_up([0.0, 1.0, 0.0])
        ctrl.set_zoom(1.25)  # leaves room for corners swinging out mid-turn

        vis.register_animation_callback(tick)
        vis.register_key_callback(_SPACE, toggle)
        vis.register_key_callback(_RIGHT, step(+1))
        vis.register_key_callback(_LEFT, step(-1))
        for k in (_PLUS, _KP_PLUS):
            vis.register_key_callback(k, speed(SPEED_STEP))
        for k in (_MINUS, _KP_MINUS):
            vis.register_key_callback(k, speed(1 / SPEED_STEP))
        vis.register_key_callback(_R, restart)
        vis.register_key_callback(_Q, quit_)

        print(f"{n}x{n}x{n}: {timeline.num_moves} moves, {timeline.duration:.1f}s at x{speed_multiplier:g}",
              file=self.out)
        print(CONTROLS, file=self.out)
        clock["last"] = time.perf_counter()
        vis.run()
        vis.destroy_window()

    def _report(self, timeline: Timeline, frame, clock) -> None:
        key = (frame.index, frame.move is None)
        if key == clock["shown"]:
            return
        clock["shown"] = key
        if frame.move is not None:
            seg = timeline.segments[frame.segment]
            print(f"[{seg.label} {frame.index_in_segment + 1}/{len(seg.moves)}] "
                  f"{format_move(frame.move, self.cube.n):7s} "
                  f"({timeline.speeds[frame.index] * clock['mult']:.1f} qt/s)", file=self.out)
        elif frame.index == timeline.num_moves and timeline.num_moves:
            solved = self.cube.is_solved(frame.state)
            print(f"done. solved: {solved}", file=self.out)
