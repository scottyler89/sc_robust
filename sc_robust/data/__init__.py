"""Data subpackage for bundled resources (e.g., pathway gene sets)."""

from importlib import resources as _resources

__all__ = ["pathways", "iter_data_files"]


def iter_data_files(pattern: str = "*"):
    """
    Yield package-managed data files matching the provided pattern.

    Parameters
    ----------
    pattern:
        Glob-style pattern applied within the `sc_robust.data` package.
    """
    for entry in _resources.files(__name__).glob(pattern):
        if entry.is_file():
            yield entry
