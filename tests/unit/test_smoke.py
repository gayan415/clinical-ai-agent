import pytest


@pytest.mark.unit
def test_project_imports():
    """Verify all packages are importable."""
    import agent  # noqa: F401
    import mlops  # noqa: F401
    import model  # noqa: F401
    import perf  # noqa: F401
    import rag  # noqa: F401
    import sre  # noqa: F401


@pytest.mark.unit
def test_python_version():
    """Verify we're running Python 3.11+."""
    import sys

    assert sys.version_info >= (3, 11)
