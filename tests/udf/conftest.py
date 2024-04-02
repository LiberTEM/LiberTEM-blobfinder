import pytest

try:
    from libertem.executor.inline import InlineJobExecutor
    from libertem import api as lt
except ImportError:
    pytest.skip("LiberTEM not found, not running integration tests")


@pytest.fixture
def inline_executor():
    return InlineJobExecutor(debug=True)


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)
