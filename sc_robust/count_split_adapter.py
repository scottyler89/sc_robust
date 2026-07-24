"""Validated cells-by-genes adapter around :mod:`count_split`."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import sparse

from count_split.count_split import multi_split


class CountSplitValidationError(ValueError):
    """Raised when count-split inputs or conservation checks are invalid."""


def _validate_counts(counts: Any) -> tuple[Any, bool]:
    is_sparse = sparse.issparse(counts)
    shape = getattr(counts, "shape", None)
    if shape is None or len(shape) != 2:
        raise CountSplitValidationError("counts must be a two-dimensional cells x genes matrix.")
    values = counts.data if is_sparse else np.asarray(counts)
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise CountSplitValidationError("counts must contain only finite values.")
    if (values < 0).any():
        raise CountSplitValidationError("counts must be nonnegative.")
    if not np.equal(values, np.floor(values)).all():
        raise CountSplitValidationError("counts must contain integral values; no coercion is performed.")
    return counts, is_sparse


def _validate_proportions(proportions: Sequence[float]) -> list[float]:
    try:
        values = [float(value) for value in proportions]
    except (TypeError, ValueError) as exc:
        raise CountSplitValidationError("split proportions must be numeric.") from exc
    if not values or any(not np.isfinite(value) or value < 0 for value in values):
        raise CountSplitValidationError("split proportions must be finite and nonnegative.")
    if not np.isclose(sum(values), 1.0, rtol=0.0, atol=1e-12):
        raise CountSplitValidationError("split proportions must sum to one within 1e-12.")
    return values


def _seed_numba(seed: int, *, seed_failure: str) -> None:
    if seed_failure not in {"error", "warn"}:
        raise ValueError("seed_failure must be 'error' or 'warn'.")
    try:
        from numba import njit

        @njit
        def seed_rng(value: int) -> None:
            np.random.seed(value)

        seed_rng(int(seed))
    except Exception as exc:
        if seed_failure == "error":
            raise RuntimeError("Unable to seed count_split's numba RNG bridge deterministically.") from exc
        import warnings

        warnings.warn(
            "Unable to seed count_split's numba RNG bridge; reproducibility is not guaranteed.",
            RuntimeWarning,
            stacklevel=3,
        )


def split_counts(
    counts: Any,
    proportions: Sequence[float],
    *,
    seed: int | None = None,
    seed_failure: str = "error",
    bin_size: int = 1000,
) -> tuple[Any, ...]:
    """Split a cells x genes count matrix with exact element-wise conservation.

    The adapter delegates to ``multi_split`` using its required genes x cells
    orientation. Reproducibility means the same seed and same ordered matrix;
    no order-invariance guarantee is made.
    """
    counts, is_sparse = _validate_counts(counts)
    proportions = _validate_proportions(proportions)
    if bin_size <= 0:
        raise CountSplitValidationError("bin_size must be positive.")
    if seed is not None:
        _seed_numba(int(seed), seed_failure=seed_failure)
    parent = sparse.csc_matrix(counts) if is_sparse else np.asarray(counts)
    raw = multi_split(parent.T, percent_vect=proportions, bin_size=int(bin_size))
    outputs = tuple(sparse.csr_matrix(item.T) if is_sparse else np.asarray(item).T for item in raw)
    if any(item.shape != parent.shape for item in outputs):
        raise CountSplitValidationError("count_split returned an output with the wrong cells x genes shape.")
    total = sparse.csr_matrix(outputs[0]) if is_sparse else np.asarray(outputs[0]).copy()
    for item in outputs[1:]:
        total = total + (sparse.csr_matrix(item) if is_sparse else np.asarray(item))
    if is_sparse:
        difference = (total - sparse.csr_matrix(parent)).tocsr()
        difference.eliminate_zeros()
        if difference.nnz:
            raise CountSplitValidationError(
                f"count conservation failed with {difference.nnz} nonzero mismatches; first={difference[0].indices[:1].tolist()}"
            )
    elif not np.array_equal(total, parent):
        mismatch = np.argwhere(total != parent)
        first = mismatch[0].tolist() if len(mismatch) else None
        raise CountSplitValidationError(f"count conservation failed; first cell/gene mismatch={first}.")
    return outputs
