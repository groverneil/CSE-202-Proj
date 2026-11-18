"""SiGN notation parser and printer. The core never sees strings; this module maps them."""

from __future__ import annotations

import re
from typing import Iterable

from .geometry import Axis
from .move import Move

# face letter -> (axis, is the positive face)
_FACES = {
    "R": (Axis.X, True),
    "L": (Axis.X, False),
    "U": (Axis.Y, True),
    "D": (Axis.Y, False),
    "F": (Axis.Z, True),
    "B": (Axis.Z, False),
}
# slice letter -> (axis, follows the positive face)
_SLICES = {"M": (Axis.X, False), "E": (Axis.Y, False), "S": (Axis.Z, True)}
_ROTATIONS = {"x": Axis.X, "y": Axis.Y, "z": Axis.Z}
_POS_FACE = {Axis.X: "R", Axis.Y: "U", Axis.Z: "F"}
_NEG_FACE = {Axis.X: "L", Axis.Y: "D", Axis.Z: "B"}
_SLICE_OF = {Axis.X: "M", Axis.Y: "E", Axis.Z: "S"}

_TOKEN = re.compile(
    r"""^(?:(?P<lo>\d+)(?:-(?P<hi>\d+))?)?
        (?P<base>[RLUDFB]w?|[rludfb]|[MES]|[xyz])
        (?P<suffix>2'|2|'|)$""",
    re.VERBOSE,
)


def _turns(suffix: str, positive: bool) -> int:
    t = {"": 1, "'": 3, "2": 2, "2'": 2}[suffix]
    return t if positive else 4 - t


def _suffix(turns: int) -> str:
    return {1: "", 2: "2", 3: "'"}[turns]


def parse_move(token: str, n: int) -> Move:
    match = _TOKEN.match(token)
    if not match:
        raise ValueError(f"cannot parse move {token!r}")
    lo, hi, base, suffix = match["lo"], match["hi"], match["base"], match["suffix"]

    if base in _ROTATIONS:
        if lo:
            raise ValueError(f"rotation {token!r} takes no layer prefix")
        return Move(_ROTATIONS[base], range(n), _turns(suffix, True))

    if base in _SLICES:
        if lo:
            raise ValueError(f"slice {token!r} takes no layer prefix")
        if n < 3:
            raise ValueError(f"{base} needs n >= 3")
        axis, positive = _SLICES[base]
        return Move(axis, range(1, n - 1), _turns(suffix, positive))

    wide = base.endswith("w") or base.islower()
    axis, positive = _FACES[base[0].upper()]
    if hi is not None:
        depths = range(int(lo), int(hi) + 1)
    elif wide:
        depths = range(1, (int(lo) if lo else 2) + 1)
    else:
        depths = [int(lo) if lo else 1]
    depths = list(depths)
    if not depths or min(depths) < 1 or max(depths) > n:
        raise ValueError(f"{token!r} reaches depths outside 1..{n}")
    layers = [n - d if positive else d - 1 for d in depths]
    return Move(axis, layers, _turns(suffix, positive))


def parse(text: str, n: int) -> list[Move]:
    """Parse a whitespace-separated move sequence (parentheses are ignored)."""
    text = text.replace("(", " ").replace(")", " ")
    return [parse_move(tok, n) for tok in text.split()]


def format_move(m: Move, n: int) -> str:
    layers = sorted(m.layers)
    lo, hi = layers[0], layers[-1]
    if hi >= n:
        raise ValueError(f"{m!r} has a layer outside 0..{n - 1}")
    if hi - lo + 1 != len(layers):
        raise ValueError(f"{m!r} has non-contiguous layers; SiGN cannot express it")
    axis = m.axis

    if len(layers) == n:
        return f"{axis}{_suffix(m.turns)}"
    if n >= 3 and lo == 1 and hi == n - 2:
        _, positive = _SLICES[_SLICE_OF[axis]]
        return _SLICE_OF[axis] + _suffix(m.turns if positive else 4 - m.turns)

    pos_depths = (n - hi, n - lo)  # depth range measured from the positive face
    neg_depths = (lo + 1, hi + 1)
    if hi == n - 1:
        positive, (a, b) = True, pos_depths
    elif lo == 0:
        positive, (a, b) = False, neg_depths
    else:
        positive = pos_depths[0] <= neg_depths[0]
        a, b = pos_depths if positive else neg_depths

    face = _POS_FACE[axis] if positive else _NEG_FACE[axis]
    turns = m.turns if positive else 4 - m.turns
    if a == 1:
        body = {1: face, 2: f"{face}w"}.get(b, f"{b}{face}w")
    elif a == b:
        body = f"{a}{face}"
    else:
        body = f"{a}-{b}{face}w"
    return body + _suffix(turns)


def format_moves(moves: Iterable[Move], n: int) -> str:
    return " ".join(format_move(m, n) for m in moves)
