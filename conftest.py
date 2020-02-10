"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""

import pytest
import numpy as np

from libertem.io.dataset.memory import MemoryDataSet
from libertem.executor.inline import InlineJobExecutor
from libertem import api as lt


@pytest.fixture
def inline_executor():
    return InlineJobExecutor(debug=True)


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)


@pytest.fixture(autouse=True)
def auto_ctx(doctest_namespace):
    ctx = lt.Context(executor=InlineJobExecutor())
    doctest_namespace["ctx"] = ctx


@pytest.fixture(autouse=True)
def auto_ds(doctest_namespace):
    dataset = MemoryDataSet(datashape=[16, 16, 16, 16])
    doctest_namespace["dataset"] = dataset


@pytest.fixture(autouse=True)
def auto_libs(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def auto_libertem(doctest_namespace):
    import libertem
    import libertem.utils
    import libertem.utils.generate
    import libertem.masks
    import libertem.api
    import libertem_blobfinder
    import libertem_blobfinder.base
    import libertem_blobfinder.common
    import libertem_blobfinder.udf
    import libertem_blobfinder.base.correlation
    import libertem_blobfinder.common.correlation
    import libertem_blobfinder.common.patterns
    import libertem_blobfinder.common.utils
    import libertem_blobfinder.udf.correlation
    import libertem_blobfinder.udf.refinement
    doctest_namespace["libertem"] = libertem
    doctest_namespace["libertem_blobfinder"] = libertem_blobfinder
    doctest_namespace["libertem_blobfinder.base"] = libertem_blobfinder.base
    doctest_namespace["libertem_blobfinder.common"] = libertem_blobfinder.common
    doctest_namespace["libertem_blobfinder.udf"] = libertem_blobfinder.udf
    doctest_namespace["libertem_blobfinder.base.correlation"] = libertem_blobfinder.base.correlation
    doctest_namespace["libertem_blobfinder.common.correlation"] = \
        libertem_blobfinder.common.correlation
    doctest_namespace["libertem_blobfinder.common.utils"] = libertem_blobfinder.common.utils
    doctest_namespace["libertem_blobfinder.udf.correlation"] = libertem_blobfinder.udf.correlation
    doctest_namespace["libertem_blobfinder.udf.refinement"] = libertem_blobfinder.udf.refinement
    doctest_namespace["libertem.utils"] = libertem.utils
    doctest_namespace["libertem.utils.generate"] = libertem.utils.generate
    doctest_namespace["libertem.masks"] = libertem.masks
    doctest_namespace["libertem.api"] = libertem.api
