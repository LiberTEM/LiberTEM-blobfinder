"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""

import pytest
import numpy as np


@pytest.fixture
def points():
    return np.array([
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, -1),
        (-1, 0),
        (-1, -1)
    ])


@pytest.fixture
def indices():
    return np.array([
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 0),
        (0, -1),
        (-1, -1)
    ])


@pytest.fixture
def zero():
    return np.array([0, 0])


@pytest.fixture
def a():
    return np.array([0, 1])


@pytest.fixture
def b():
    return np.array([1, 0])


@pytest.fixture(autouse=True)
def auto_ctx(doctest_namespace):
    try:
        from libertem.executor.inline import InlineJobExecutor
        from libertem import api as lt
        ctx = lt.Context(executor=InlineJobExecutor())
    except ImportError:
        ctx = None
    doctest_namespace["ctx"] = ctx


@pytest.fixture(autouse=True)
def auto_ds(doctest_namespace):
    try:
        from libertem.io.dataset.memory import MemoryDataSet
        dataset = MemoryDataSet(datashape=[16, 16, 16, 16])
    except ImportError:
        dataset = None
    doctest_namespace["dataset"] = dataset


@pytest.fixture(autouse=True)
def auto_libs(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def auto_blobfinder(doctest_namespace):
    import libertem_blobfinder
    doctest_namespace["libertem_blobfinder"] = libertem_blobfinder


@pytest.fixture(autouse=True)
def auto_libertem(doctest_namespace):
    try:
        import libertem
        import libertem.api
        import libertem_blobfinder

        doctest_namespace["libertem"] = libertem
        doctest_namespace["libertem.api"] = libertem.api
        doctest_namespace["libertem_blobfinder"] = libertem_blobfinder
    except ImportError:
        pass  # anything better to do here?
