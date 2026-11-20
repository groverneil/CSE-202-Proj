"""Open3D window that plays a Timeline. Open3D is imported only when a window opens."""

from __future__ import annotations

import sys
import time
from typing import TextIO

import numpy as np

from ..move import Move
from ..notation import format_move
from .animator import Animator
from .palette import BACKGROUND_RGB, BODY_RGB, CAPTION_RGB, DEFAULT_RGB
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

        n = self.cube.n
        scene = _Scene(self, o3d, visible=True, title=title or f"{n}x{n}x{n}")
        vis = scene.vis

        clock = {"t": 0.0, "last": time.perf_counter(), "paused": paused,
                 "mult": speed_multiplier, "shown": None}

        def render(vis) -> None:
            frame = timeline.frame(clock["t"])
            scene.show(frame)
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

        render(vis)

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

    # ---- recording ----------------------------------------------------------------------

    def record(self, timeline: Timeline, writer, fps: int | None = None,
               speed_multiplier: float = 1.0, caption=None, lead_in: float = 0.0,
               hold: float = 0.0) -> int:
        """Render ``timeline`` into ``writer`` (e.g. ``video.MovWriter``) frame by frame in a
        hidden window: the same scene and camera as ``play``, sampled at ``fps``.

        ``caption(frame) -> list[str]`` gives text lines drawn top-left on every frame
        (default: the segment, move counter and move, like ``play``'s console report).
        ``lead_in`` / ``hold`` show the first / last frame for that many extra seconds.
        Several timelines (even of different cube sizes) can go into one writer.
        Returns the number of frames written.
        """
        import open3d as o3d

        fps = fps or writer.fps
        caption = caption or (lambda f: self._caption_lines(timeline, f))
        scene = _Scene(self, o3d, visible=False, size=(writer.width, writer.height))
        written = 0
        try:
            step = speed_multiplier / fps
            times = [0.0] * round(lead_in * fps)
            times += [k * step for k in range(int(timeline.duration / step) + 1)]
            times += [timeline.duration] * (1 + round(hold * fps))
            for t in times:
                frame = timeline.frame(t)
                scene.show(frame, caption(frame))
                writer.write(scene.capture())
                written += 1
        finally:
            scene.close()
        return written

    def _caption_lines(self, timeline: Timeline, frame) -> list[str]:
        seg = timeline.segments[frame.segment] if timeline.segments else None
        if seg is None:
            return []
        line = f"{seg.label} {frame.index_in_segment}/{len(seg.moves)}"
        if frame.move is not None:
            line += f"  {format_move(frame.move, self.cube.n)}"
        return [line]

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


class _Scene:
    """The Open3D window, cube geometry and camera shared by ``play`` and ``record``."""

    def __init__(self, viewer: Open3DViewer, o3d, visible: bool, title: str = "",
                 size: tuple[int, int] | None = None):
        self.o3d, self.anim = o3d, viewer.animator
        w, h = size or viewer.window_size
        vis_cls = o3d.visualization.VisualizerWithKeyCallback if visible else o3d.visualization.Visualizer
        self.vis = vis = vis_cls()
        vis.create_window(title, w, h, visible=visible)
        self.stickers = o3d.geometry.TriangleMesh()
        self.stickers.triangles = o3d.utility.Vector3iVector(self.anim.mesh.triangles)
        self.body = o3d.geometry.TriangleMesh()
        self._fill(None)
        vis.add_geometry(self.body)
        vis.add_geometry(self.stickers)
        vis.reset_view_point(True)
        opt = vis.get_render_option()
        opt.background_color = BACKGROUND_RGB
        opt.mesh_show_back_face = False
        ctrl = vis.get_view_control()
        ctrl.set_lookat([0.0, 0.0, 0.0])
        front = np.array([0.55, 0.5, 0.67])  # camera up-right-front: U, F and R visible
        front /= np.linalg.norm(front)
        up = np.array([0.0, 1.0, 0.0])
        up -= (up @ front) * front  # orthogonal to front, so Open3D's view matrix is a rotation
        ctrl.set_front(front)
        ctrl.set_up(up / np.linalg.norm(up))
        ctrl.set_zoom(1.25)  # leaves room for corners swinging out mid-turn
        self.caption = _Caption(o3d, vis) if not visible else None

    def _fill(self, frame) -> None:
        o3d, anim = self.o3d, self.anim
        if frame is None:
            from .timeline import Frame
            frame = Frame(anim.cube.solved(), None, 0.0, 0, 0, 0)
        self.stickers.vertices = o3d.utility.Vector3dVector(anim.sticker_vertices(frame))
        self.stickers.vertex_colors = o3d.utility.Vector3dVector(anim.sticker_colors(frame))
        self.stickers.compute_vertex_normals()
        bv, bt = anim.body(frame)
        self.body.vertices = o3d.utility.Vector3dVector(bv)
        self.body.triangles = o3d.utility.Vector3iVector(bt)
        self.body.paint_uniform_color(BODY_RGB)
        self.body.compute_vertex_normals()

    def show(self, frame, lines: list[str] | None = None) -> None:
        self._fill(frame)
        self.vis.update_geometry(self.stickers)
        self.vis.update_geometry(self.body)
        if self.caption is not None and lines is not None:
            self.caption.set(lines)

    def capture(self) -> np.ndarray:
        self.vis.poll_events()
        self.vis.update_renderer()
        return np.asarray(self.vis.capture_screen_float_buffer(do_render=True))

    def close(self) -> None:
        self.vis.destroy_window()


class _Caption:
    """Text lines pinned to the top-left of the image: flat text meshes (Open3D
    ``create_text``) placed just in front of the cube, facing the camera."""

    PIXEL_HEIGHT = 22  # line height in pixels
    MARGIN = 18

    def __init__(self, o3d, vis):
        self.o3d, self.vis = o3d, vis
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # copy now: these arrays are views into ``params``, which is freed after __init__
        K = np.array(params.intrinsic.intrinsic_matrix, dtype=float)
        E = np.array(params.extrinsic, dtype=float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # Open3D does not orthogonalise ``up`` against ``front``, so the view matrix can be
        # skewed: undo it with a true inverse, not a transpose.
        self.R_inv, self.t = np.linalg.inv(E[:3, :3]), E[:3, 3].copy()
        # depth: in front of the sphere the cube's corners sweep (radius sqrt 3), with margin
        self.eye = -self.R_inv @ self.t
        self.depth = max(0.5, float(np.linalg.norm(self.eye)) - np.sqrt(3) - 0.5)
        self.mesh = o3d.geometry.TriangleMesh()
        self.text: list[str] = []
        self._shown = False

    def _to_world(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        d = self.depth
        cam = np.stack([(px - self.cx) * d / self.fx, (py - self.cy) * d / self.fy, np.full_like(px, d)], 1)
        return (cam - self.t) @ self.R_inv.T  # inverse of x_cam = R x_world + t

    def set(self, lines: list[str]) -> None:
        if lines == self.text:
            return
        self.text = list(lines)
        o3d = self.o3d
        verts, tris, offset = [], [], 0
        for i, line in enumerate(lines):
            if not line:
                continue
            m = o3d.t.geometry.TriangleMesh.create_text(line, depth=0.0).to_legacy()
            v, f = np.asarray(m.vertices), np.asarray(m.triangles)
            if len(v) == 0:
                continue
            scale = self.PIXEL_HEIGHT * 0.8 / 14.0  # create_text glyphs are ~14 units tall
            px = self.MARGIN + v[:, 0] * scale
            py = self.MARGIN + (i + 1) * self.PIXEL_HEIGHT - v[:, 1] * scale  # image y points down
            verts.append(self._to_world(px, py))
            tris.append(f[:, ::-1] + offset)  # flipping y flips the winding back to front
            offset += len(v)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(verts) if verts else np.zeros((0, 3)))
        self.mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(tris) if tris else np.zeros((0, 3), int))
        self.mesh.paint_uniform_color(CAPTION_RGB)
        # normals straight at the camera, so the headlight lights every glyph evenly
        V = np.asarray(self.mesh.vertices)
        to_eye = self.eye - V if len(V) else np.zeros((0, 3))
        if len(V):
            to_eye /= np.linalg.norm(to_eye, axis=1, keepdims=True)
        self.mesh.vertex_normals = o3d.utility.Vector3dVector(to_eye)
        if not self._shown:
            self.vis.add_geometry(self.mesh, reset_bounding_box=False)
            self._shown = True
        else:
            self.vis.update_geometry(self.mesh)
