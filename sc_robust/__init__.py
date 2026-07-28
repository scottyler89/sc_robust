"""Top-level public API with lazy loading for optional pipeline components."""

from importlib import import_module

from ._version import __version__
from .compatibility import *  # noqa: F401,F403
from .provenance import *  # noqa: F401,F403
from .compatibility import __all__ as _compatibility_exports
from .provenance import __all__ as _provenance_exports

_EXPORTS = {
    "robust": (".sc_robust", "robust"),
    "CountSplitValidationError": (".count_split_adapter", "CountSplitValidationError"),
    "split_counts": (".count_split_adapter", "split_counts"),
    "SpearmanArtifact": (".gene_modules", "SpearmanArtifact"),
    "read_spearman_h5": (".gene_modules", "read_spearman_h5"),
    "run_gene_modules_for_cohort": (".gene_modules", "run_gene_modules_for_cohort"),
    "run_replicated_gene_modules_for_cohort": (".gene_modules", "run_replicated_gene_modules_for_cohort"),
    "run_gene_modules_from_scratch_dir": (".gene_modules", "run_gene_modules_from_scratch_dir"),
}

__all__ = ["__version__", *_compatibility_exports, *_provenance_exports, *_EXPORTS]

def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
