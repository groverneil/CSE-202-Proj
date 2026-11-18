import functools

import pytest

from nxncube import Cube

NS = list(range(2, 11))


@functools.lru_cache(maxsize=None)
def cube(n: int) -> Cube:
    return Cube(n)


@pytest.fixture(params=NS, ids=lambda n: f"n{n}")
def any_cube(request) -> Cube:
    return cube(request.param)
