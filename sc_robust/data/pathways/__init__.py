"""Bundled pathway gene-set resources."""

from importlib import resources as _resources
from typing import Iterator

__all__ = ["iter_pathway_files", "PATHWAY_FILE_SUFFIX"]

PATHWAY_FILE_SUFFIX = ".gmt"


def iter_pathway_files() -> Iterator[str]:
    """Yield the names of packaged pathway GMT files."""
    for entry in _resources.files(__name__).glob(f"*{PATHWAY_FILE_SUFFIX}"):
        if entry.is_file():
            yield entry.name
