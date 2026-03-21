import pytest


@pytest.mark.unit
def test_project_imports():
    """Verify all packages are importable."""
    import agent
    import model
    import rag
    import mlops
    import sre
    import perf


@pytest.mark.unit
def test_python_version():
    """Verify we're running Python 3.11+."""
    import sys

    assert sys.version_info >= (3, 11)
