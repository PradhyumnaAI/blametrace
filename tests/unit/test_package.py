"""Smoke test: the package imports and exposes a version string."""

import blametrace


def test_version_is_a_string() -> None:
    assert isinstance(blametrace.__version__, str)
    assert blametrace.__version__.count(".") == 2
