"""Helpers for loading packaged pathway gene sets."""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .base import PathwayEnrichmentResult  # imported for typing/export convenience
from sc_robust.data import pathways as pathways_pkg

__all__ = [
    "list_available_pathway_libraries",
    "resolve_pathway_filename",
    "load_pathway_library",
    "load_multiple_pathway_libraries",
]


def list_available_pathway_libraries() -> List[str]:
    """
    Return the list of packaged pathway files.

    The returned list contains file names (e.g., ``c1.all.v2025.1.Hs.symbols.gmt``)
    sorted alphabetically.
    """
    return sorted(pathways_pkg.iter_pathway_files())


def resolve_pathway_filename(library: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a pathway library identifier to an on-disk Path.

    This helper searches the filesystem (optionally within ``base_dir``) and does
    not consider packaged data. Use :func:`load_pathway_library` to access bundled
    pathway resources.
    """
    candidate = Path(library)
    if candidate.exists():
        return candidate

    if base_dir is not None:
        base_dir = Path(base_dir)
        direct = base_dir / library
        if direct.exists():
            return direct
        if not library.endswith(pathways_pkg.PATHWAY_FILE_SUFFIX):
            matches = list(base_dir.glob(f"{library}*{pathways_pkg.PATHWAY_FILE_SUFFIX}"))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Could not resolve pathway library '{library}' on disk. "
        "Either provide a full path or ensure the file exists under the supplied base_dir."
    )


def _read_gmt(path: Path) -> Dict[str, List[str]]:
    """Parse a GMT pathway file into a mapping of pathway -> gene list."""
    pathways: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            genes = [gene.strip() for gene in row[2:] if gene.strip()]
            pathways[name] = genes
    return pathways


@lru_cache(maxsize=None)
def load_pathway_library(
    library: str,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Load a pathway library into memory.

    Parameters
    ----------
    library:
        Library identifier or path (see :func:`resolve_pathway_filename`).
    base_dir:
        Optional directory to search before falling back to packaged assets.

    Returns
    -------
    dict
        Mapping from pathway name to list of member genes.
    """
    try:
        resolved = resolve_pathway_filename(library, base_dir=base_dir)
    except FileNotFoundError:
        resolved = None

    if resolved is not None and resolved.exists():
        return _read_gmt(resolved)

    # Fall back to packaged data (matching by filename or prefix)
    packaged_files = list_available_pathway_libraries()
    target_name: Optional[str] = None
    if library in packaged_files:
        target_name = library
    else:
        stem = Path(library).stem
        matches = [name for name in packaged_files if Path(name).stem.startswith(stem)]
        if matches:
            target_name = matches[0]

    if target_name is None:
        raise FileNotFoundError(
            f"Could not locate pathway library '{library}'. Checked base_dir and packaged assets."
        )

    resource = resources.files(pathways_pkg.__name__) / target_name
    with resources.as_file(resource) as tmp_path:
        return _read_gmt(Path(tmp_path))


def load_multiple_pathway_libraries(
    libraries: Iterable[str],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Load multiple pathway libraries at once.

    Returns a nested mapping ``{library_id: {pathway_name: genes}}``.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for library in libraries:
        out[library] = load_pathway_library(library, base_dir=base_dir)
    return out
