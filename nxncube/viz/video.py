"""Write rendered frames to a video file with no extra dependencies.

``MovWriter`` stores each frame as a JPEG (encoded by Open3D) inside a QuickTime ``.mov``
container (Motion-JPEG), which QuickTime, VLC and most players open directly. The container
is written by hand below: an ``mdat`` box holding the JPEGs back to back, then a ``moov``
box that indexes them. ``to_mp4`` optionally re-encodes it to a much smaller H.264 ``.mp4``
with macOS's built-in ``avconvert``.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)  # identity


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


class MovWriter:
    """Motion-JPEG QuickTime writer. Use as a context manager; call ``write(rgb)`` per frame
    with an (height, width, 3) uint8 or [0, 1] float array."""

    def __init__(self, path: str | os.PathLike, width: int, height: int, fps: int = 30,
                 quality: int = 85):
        self.path = Path(path)
        self.width, self.height, self.fps, self.quality = width, height, fps, quality
        self._sizes: list[int] = []
        self._offsets: list[int] = []
        self._tmp = tempfile.TemporaryDirectory()
        self._jpg = os.path.join(self._tmp.name, "frame.jpg")
        self._f = open(self.path, "wb")
        self._f.write(_box(b"ftyp", b"qt  " + struct.pack(">I", 0x200) + b"qt  "))
        self._mdat_start = self._f.tell()
        self._f.write(struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 0))  # 64-bit size, patched

    @property
    def frames(self) -> int:
        return len(self._sizes)

    def write(self, rgb: np.ndarray) -> None:
        import open3d as o3d  # lazy, like the viewer

        a = np.asarray(rgb)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        if a.shape != (self.height, self.width, 3):
            raise ValueError(f"frame is {a.shape}, expected {(self.height, self.width, 3)}")
        o3d.io.write_image(self._jpg, o3d.geometry.Image(np.ascontiguousarray(a)), quality=self.quality)
        data = Path(self._jpg).read_bytes()
        self._offsets.append(self._f.tell())
        self._sizes.append(len(data))
        self._f.write(data)

    def close(self) -> None:
        if self._f.closed:
            return
        end = self._f.tell()
        self._f.seek(self._mdat_start + 8)
        self._f.write(struct.pack(">Q", end - self._mdat_start))
        self._f.seek(end)
        self._f.write(self._moov())
        self._f.close()
        self._tmp.cleanup()

    def __enter__(self) -> "MovWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- QuickTime boxes ------------------------------------------------------------

    def _moov(self) -> bytes:
        n, fps, w, h = len(self._sizes), self.fps, self.width, self.height
        mvhd = _box(b"mvhd", struct.pack(">IIIII", 0, 0, 0, fps, n) + struct.pack(">IH", 0x10000, 0x100)
                    + bytes(10) + _MATRIX + bytes(24) + struct.pack(">I", 2))
        tkhd = _box(b"tkhd", struct.pack(">IIIIII", 3, 0, 0, 1, 0, n) + bytes(8)
                    + struct.pack(">HHHH", 0, 0, 0, 0) + _MATRIX + struct.pack(">II", w << 16, h << 16))
        mdhd = _box(b"mdhd", struct.pack(">IIIIIHH", 0, 0, 0, fps, n, 0, 0))
        hdlr = _box(b"hdlr", struct.pack(">I", 0) + b"mhlrvide" + bytes(12) + b"\x00")
        vmhd = _box(b"vmhd", struct.pack(">IH", 1, 0x40) + bytes(6))
        dhlr = _box(b"hdlr", struct.pack(">I", 0) + b"dhlralis" + bytes(12) + b"\x00")
        dinf = _box(b"dinf", _box(b"dref", struct.pack(">II", 0, 1) + _box(b"alis", struct.pack(">I", 1))))
        name = b"Photo - JPEG"
        desc = (bytes(6) + struct.pack(">H", 1)            # reserved, data reference index
                + struct.pack(">HH", 0, 0) + b"appl"       # version, revision, vendor
                + struct.pack(">II", 0, 512)              # temporal, spatial quality
                + struct.pack(">HH", w, h) + struct.pack(">II", 72 << 16, 72 << 16)
                + struct.pack(">IH", 0, 1)                 # data size, frames per sample
                + bytes([len(name)]) + name + bytes(31 - len(name))
                + struct.pack(">Hh", 24, -1))              # depth, no colour table
        stsd = _box(b"stsd", struct.pack(">II", 0, 1) + _box(b"jpeg", desc))
        stts = _box(b"stts", struct.pack(">IIII", 0, 1, n, 1))
        stsc = _box(b"stsc", struct.pack(">IIIII", 0, 1, 1, 1, 1))
        stsz = _box(b"stsz", struct.pack(">III", 0, 0, n) + struct.pack(f">{n}I", *self._sizes))
        co64 = _box(b"co64", struct.pack(">II", 0, n) + struct.pack(f">{n}Q", *self._offsets))
        stbl = _box(b"stbl", stsd + stts + stsc + stsz + co64)
        minf = _box(b"minf", vmhd + dhlr + dinf + stbl)
        mdia = _box(b"mdia", mdhd + hdlr + minf)
        return _box(b"moov", mvhd + _box(b"trak", tkhd + mdia))


def to_mp4(source: str | os.PathLike, target: str | os.PathLike,
           preset: str = "PresetHighestQuality") -> bool:
    """Re-encode ``source`` to H.264 ``target`` with macOS ``avconvert``. Returns False if
    ``avconvert`` is unavailable or fails (the source file is left untouched either way)."""
    exe = shutil.which("avconvert")
    if exe is None:
        return False
    r = subprocess.run([exe, "--preset", preset, "--source", str(source), "--output", str(target),
                        "--replace"], capture_output=True, text=True)
    return r.returncode == 0 and Path(target).exists()
