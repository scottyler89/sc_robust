"""Canonical design and contrast records for differential expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..provenance import canonical_json, stable_identifier


@dataclass(frozen=True)
class DesignSpec:
    """Validated, JSON-safe description of a DE design matrix."""

    formula: str
    terms: tuple[str, ...]
    columns: tuple[str, ...]
    reference: Optional[str]
    shape: tuple[int, int]
    rank: int
    condition_number: Optional[float]
    aliased_columns: tuple[str, ...]
    coefficient_map: Mapping[str, str]
    fit_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "terms": list(self.terms),
            "columns": list(self.columns),
            "reference": self.reference,
            "shape": list(self.shape),
            "rank": self.rank,
            "condition_number": self.condition_number,
            "aliased_columns": list(self.aliased_columns),
            "coefficient_map": dict(self.coefficient_map),
            "fit_id": self.fit_id,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)


def _sanitize(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").replace("/", "_").replace(":", "_")


def build_design_frame(
    metadata: pd.DataFrame,
    *,
    formula: str,
    annotation_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, tuple[str, ...], Mapping[str, str], Optional[str]]:
    """Build the supported additive formula subset without hidden metadata terms."""
    if not isinstance(formula, str) or not formula.startswith("~"):
        raise ValueError("design must be a formula string beginning with '~'.")
    expression = formula[1:].strip()
    intercept = True
    expression = expression.replace("- 1", "").replace("-1", "")
    if "0" in {part.strip() for part in expression.split("+")}:
        intercept = False
        expression = "+".join(part for part in expression.split("+") if part.strip() != "0")
    terms = tuple(part.strip() for part in expression.split("+") if part.strip())
    if not terms:
        raise ValueError("design formula must contain at least one metadata term.")
    missing = [term for term in terms if term not in metadata.columns]
    if missing:
        raise KeyError(f"Design terms not found in metadata: {missing}")
    annotation_missing = [col for col in annotation_columns if col not in metadata.columns]
    if annotation_missing:
        raise KeyError(f"Annotation columns not found in metadata: {annotation_missing}")

    parts: list[pd.DataFrame] = []
    coefficient_map: dict[str, str] = {}
    reference: Optional[str] = None
    for term in terms:
        values = metadata[term]
        if values.isna().any():
            raise ValueError(f"Design term {term!r} contains null values.")
        if pd.api.types.is_numeric_dtype(values):
            encoded = pd.DataFrame({_sanitize(term): values.astype(float)}, index=metadata.index)
            coefficient_map[term] = _sanitize(term)
        else:
            categories = list(pd.unique(values.astype(str)))
            if not categories:
                raise ValueError(f"Design term {term!r} has no levels.")
            if reference is None:
                reference = categories[0]
            kept = categories if not intercept else categories[1:]
            encoded = pd.get_dummies(values.astype(str), prefix=_sanitize(term), dtype=float)
            encoded = encoded.reindex(columns=[f"{_sanitize(term)}_{level}" for level in kept], fill_value=0.0)
            for level in categories:
                coefficient_map[f"{term}={level}"] = f"{_sanitize(term)}_{level}"
        parts.append(encoded)
    if intercept:
        parts.insert(0, pd.DataFrame({"Intercept": 1.0}, index=metadata.index))
    design = pd.concat(parts, axis=1)
    if design.columns.duplicated().any():
        raise ValueError("Sanitized design coefficient names are not unique.")
    matrix = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < design.shape[1]:
        aliased = tuple(design.columns[np.linalg.svd(matrix, full_matrices=False)[1].size:])
        raise ValueError(f"Design matrix is rank deficient; aliased columns: {list(aliased)}")
    condition_number = float(np.linalg.cond(matrix)) if matrix.size else None
    if condition_number is not None and not np.isfinite(condition_number):
        condition_number = None
    return design, terms, coefficient_map, reference


def make_design_spec(
    design: pd.DataFrame,
    *,
    formula: str,
    terms: Sequence[str],
    coefficient_map: Mapping[str, str],
    reference: Optional[str],
) -> DesignSpec:
    matrix = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    condition_number = float(np.linalg.cond(matrix)) if matrix.size else None
    if condition_number is not None and not np.isfinite(condition_number):
        condition_number = None
    payload = {
        "formula": formula,
        "terms": list(terms),
        "columns": list(design.columns),
        "reference": reference,
        "shape": list(design.shape),
        "rank": rank,
        "coefficient_map": dict(coefficient_map),
    }
    return DesignSpec(
        formula=formula,
        terms=tuple(terms),
        columns=tuple(str(col) for col in design.columns),
        reference=reference,
        shape=tuple(int(value) for value in design.shape),
        rank=rank,
        condition_number=condition_number,
        aliased_columns=(),
        coefficient_map=dict(coefficient_map),
        fit_id=stable_identifier("de-fit", payload),
    )
