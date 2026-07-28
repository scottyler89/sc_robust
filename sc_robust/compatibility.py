"""Structured compatibility validation for analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Literal, Mapping, Sequence
import warnings

import numpy as np
from scipy import sparse

from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    HashRecord,
    ProvenanceSerializationError,
    canonical_json,
    freeze_json,
    freeze_mapping,
    hash_ordered_ids,
    normalize_identifiers,
    stable_identifier,
    to_jsonable,
    utc_now,
)


CompatibilityPolicy = Literal["report", "warn", "error"]
FindingSeverity = Literal["info", "warning", "error"]
_FINDING_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


class CompatibilityWarning(UserWarning):
    """Warning emitted when compatibility policy is warn."""


class CompatibilityError(ValueError):
    """Raised when a compatibility report contains error findings."""

    def __init__(self, report: "CompatibilityReport") -> None:
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class CompatibilityFinding:
    """One deterministic and actionable compatibility finding."""

    code: str
    severity: FindingSeverity
    field: str
    message: str
    expected: Any = None
    observed: Any = None
    remediation: str = ""

    def __post_init__(self) -> None:
        if not _FINDING_CODE_RE.fullmatch(self.code):
            raise ValueError(f"Invalid compatibility finding code: {self.code!r}.")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"Invalid finding severity: {self.severity!r}.")
        if not self.field:
            raise ValueError("Compatibility finding field must be non-empty.")
        if not self.message:
            raise ValueError("Compatibility finding message must be non-empty.")
        object.__setattr__(self, "expected", freeze_json(self.expected))
        object.__setattr__(self, "observed", freeze_json(self.observed))

    @property
    def finding_id(self) -> str:
        """Return a stable ID for this finding."""

        return stable_identifier(
            "compatibility-finding",
            {
                "code": self.code,
                "severity": self.severity,
                "field": self.field,
                "message": self.message,
                "expected": self.expected,
                "observed": self.observed,
                "remediation": self.remediation,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native finding."""

        return {
            "finding_id": self.finding_id,
            "code": self.code,
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
            "expected": to_jsonable(self.expected),
            "observed": to_jsonable(self.observed),
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompatibilityFinding":
        """Reconstruct a finding and verify its stable ID when present."""

        normalized = to_jsonable(payload)
        finding_id = normalized.pop("finding_id", None)
        finding = cls(**normalized)
        if finding_id is not None and finding_id != finding.finding_id:
            raise ValueError(
                "Compatibility finding_id mismatch: "
                f"expected {finding.finding_id!r}, observed {finding_id!r}."
            )
        return finding


def _report_identity(
    *,
    schema_version: str,
    checked: Sequence[str],
    findings: Sequence[CompatibilityFinding],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "checked": list(checked),
        "findings": [finding.to_dict() for finding in findings],
        "context": context,
    }


@dataclass(frozen=True)
class CompatibilityReport:
    """Immutable findings from one complete compatibility validation."""

    report_id: str
    checked: tuple[str, ...]
    findings: tuple[CompatibilityFinding, ...]
    context: Mapping[str, Any]
    created_utc: str
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized_checked = tuple(sorted(set(str(item) for item in self.checked)))
        normalized_findings = tuple(self.findings)
        normalized_context = freeze_mapping(self.context)
        object.__setattr__(self, "checked", normalized_checked)
        object.__setattr__(self, "findings", normalized_findings)
        object.__setattr__(self, "context", normalized_context)

        expected_id = stable_identifier(
            "compatibility-report",
            _report_identity(
                schema_version=self.schema_version,
                checked=normalized_checked,
                findings=normalized_findings,
                context=normalized_context,
            ),
            schema_version=self.schema_version,
        )
        if self.report_id != expected_id:
            raise ValueError(
                f"Compatibility report_id mismatch: expected {expected_id!r}, "
                f"observed {self.report_id!r}."
            )

    @classmethod
    def create(
        cls,
        *,
        checked: Iterable[str],
        findings: Iterable[CompatibilityFinding] = (),
        context: Mapping[str, Any] | None = None,
        created_utc: str | None = None,
        schema_version: str = PROVENANCE_SCHEMA_VERSION,
    ) -> "CompatibilityReport":
        """Create a deterministic report whose timestamp is event-only metadata."""

        normalized_checked = tuple(sorted(set(str(item) for item in checked)))
        normalized_findings = tuple(findings)
        normalized_context = freeze_mapping(context)
        report_id = stable_identifier(
            "compatibility-report",
            _report_identity(
                schema_version=schema_version,
                checked=normalized_checked,
                findings=normalized_findings,
                context=normalized_context,
            ),
            schema_version=schema_version,
        )
        return cls(
            report_id=report_id,
            checked=normalized_checked,
            findings=normalized_findings,
            context=normalized_context,
            created_utc=created_utc or utc_now(),
            schema_version=schema_version,
        )

    @property
    def errors(self) -> tuple[CompatibilityFinding, ...]:
        """Return error-severity findings."""

        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[CompatibilityFinding, ...]:
        """Return warning-severity findings."""

        return tuple(item for item in self.findings if item.severity == "warning")

    @property
    def is_compatible(self) -> bool:
        """Return whether no invariant-breaking finding was detected."""

        return not self.errors

    def summary(self) -> str:
        """Return a compact deterministic summary."""

        header = (
            f"Compatibility report {self.report_id}: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)."
        )
        details = [
            f"[{finding.code}] {finding.field}: {finding.message} "
            f"Remediation: {finding.remediation or 'none provided'}"
            for finding in self.findings
            if finding.severity in {"error", "warning"}
        ]
        return " ".join([header, *details])

    def apply_policy(self, policy: CompatibilityPolicy) -> "CompatibilityReport":
        """Apply reporting policy after all checks have completed."""

        if policy not in {"report", "warn", "error"}:
            raise ValueError(
                "Compatibility policy must be 'report', 'warn', or 'error'; "
                f"received {policy!r}."
            )
        if policy == "error" and self.errors:
            raise CompatibilityError(self)
        if policy == "warn" and (self.errors or self.warnings):
            warnings.warn(self.summary(), CompatibilityWarning, stacklevel=2)
        elif policy == "error" and self.warnings:
            warnings.warn(self.summary(), CompatibilityWarning, stacklevel=2)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native report."""

        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "checked": list(self.checked),
            "findings": [finding.to_dict() for finding in self.findings],
            "context": to_jsonable(self.context),
            "created_utc": self.created_utc,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON."""

        return canonical_json(self.to_dict(), indent=indent)

    def to_dataframe(self):
        """Return findings as a pandas DataFrame."""

        import pandas as pd

        columns = [
            "finding_id",
            "code",
            "severity",
            "field",
            "message",
            "expected",
            "observed",
            "remediation",
        ]
        return pd.DataFrame(
            [finding.to_dict() for finding in self.findings],
            columns=columns,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompatibilityReport":
        """Reconstruct a report and verify its stable identity."""

        normalized = to_jsonable(payload)
        expected_fields = {
            "schema_version",
            "report_id",
            "checked",
            "findings",
            "context",
            "created_utc",
        }
        unknown = set(normalized) - expected_fields
        missing = expected_fields - set(normalized)
        if unknown or missing:
            raise ValueError(
                "Invalid compatibility report fields: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return cls(
            report_id=normalized["report_id"],
            checked=tuple(normalized["checked"]),
            findings=tuple(
                CompatibilityFinding.from_dict(item)
                for item in normalized["findings"]
            ),
            context=normalized["context"],
            created_utc=normalized["created_utc"],
            schema_version=normalized["schema_version"],
        )


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _axis_summary(values: Sequence[str], *, domain: str) -> dict[str, Any]:
    return {
        "count": len(values),
        "hash": hash_ordered_ids(values, domain=domain).to_dict(),
    }


def _first_axis_mismatch(
    expected: Sequence[str],
    observed: Sequence[str],
) -> dict[str, Any] | None:
    for index, (expected_id, observed_id) in enumerate(zip(expected, observed)):
        if expected_id != observed_id:
            return {
                "index": index,
                "expected_id": expected_id,
                "observed_id": observed_id,
            }
    if len(expected) != len(observed):
        return {
            "index": min(len(expected), len(observed)),
            "expected_id": expected[len(observed)] if len(expected) > len(observed) else None,
            "observed_id": observed[len(expected)] if len(observed) > len(expected) else None,
        }
    return None


def _coerce_hash_record(value: HashRecord | Mapping[str, Any] | str) -> HashRecord | str:
    if isinstance(value, HashRecord):
        return value
    if isinstance(value, str):
        return value
    return HashRecord.from_dict(value)


def _parameter_findings(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    prefix: str = "parameters",
) -> list[CompatibilityFinding]:
    findings: list[CompatibilityFinding] = []
    for key in sorted(expected):
        field = f"{prefix}.{key}"
        expected_value = expected[key]
        if key not in observed:
            try:
                expected_summary = to_jsonable(expected_value)
            except ProvenanceSerializationError:
                expected_summary = {"type": type(expected_value).__name__}
            findings.append(
                CompatibilityFinding(
                    code="parameters.missing",
                    severity="error",
                    field=field,
                    message="Required algorithm parameter is missing.",
                    expected=expected_summary,
                    observed=None,
                    remediation=f"Regenerate the artifact with {field} recorded.",
                )
            )
            continue

        observed_value = observed[key]
        if isinstance(expected_value, Mapping) and isinstance(observed_value, Mapping):
            findings.extend(
                _parameter_findings(
                    expected_value,
                    observed_value,
                    prefix=field,
                )
            )
            continue

        try:
            normalized_expected = to_jsonable(expected_value)
        except ProvenanceSerializationError as exc:
            findings.append(
                CompatibilityFinding(
                    code="parameters.expected_invalid",
                    severity="error",
                    field=field,
                    message=str(exc),
                    expected="strict JSON-native parameter value",
                    observed={"type": type(expected_value).__name__},
                    remediation="Replace opaque values with explicit configuration.",
                )
            )
            continue

        try:
            normalized_observed = to_jsonable(observed_value)
        except ProvenanceSerializationError as exc:
            findings.append(
                CompatibilityFinding(
                    code="parameters.observed_invalid",
                    severity="error",
                    field=field,
                    message=str(exc),
                    expected=normalized_expected,
                    observed={"type": type(observed_value).__name__},
                    remediation="Replace opaque values with explicit configuration.",
                )
            )
            continue

        if normalized_expected != normalized_observed:
            findings.append(
                CompatibilityFinding(
                    code="parameters.mismatch",
                    severity="error",
                    field=field,
                    message="Algorithm parameter does not match.",
                    expected=normalized_expected,
                    observed=normalized_observed,
                    remediation=(
                        "Use artifacts produced with identical parameters or "
                        "recompute the downstream stage."
                    ),
                )
            )
    return findings


def _conservation_observation(
    counts: Any,
    splits: Sequence[Any],
) -> dict[str, Any] | None:
    if sparse.issparse(counts) or any(sparse.issparse(item) for item in splits):
        total = sparse.csr_matrix(splits[0], dtype=None)
        for item in splits[1:]:
            total = total + sparse.csr_matrix(item, dtype=None)
        difference = total - sparse.csr_matrix(counts, dtype=None)
        difference.eliminate_zeros()
        if difference.nnz == 0:
            return None
        first = difference.tocoo()
        return {
            "mismatch_count": int(difference.nnz),
            "first_index": [int(first.row[0]), int(first.col[0])],
            "delta": to_jsonable(first.data[0]),
        }

    total = np.asarray(splits[0]).copy()
    for item in splits[1:]:
        total = total + np.asarray(item)
    expected = np.asarray(counts)
    mismatch = np.argwhere(total != expected)
    if mismatch.size == 0:
        return None
    first_index = tuple(int(item) for item in mismatch[0])
    return {
        "mismatch_count": int(mismatch.shape[0]),
        "first_index": list(first_index),
        "delta": to_jsonable(total[first_index] - expected[first_index]),
    }


def validate_compatibility(
    *,
    counts: Any | None = None,
    metadata: Any | None = None,
    graph: Any | None = None,
    splits: Sequence[Any] | None = None,
    split_proportions: Sequence[float] | None = None,
    cell_ids: Iterable[Any] | None = None,
    gene_ids: Iterable[Any] | None = None,
    expected_cell_ids: Iterable[Any] | None = None,
    expected_gene_ids: Iterable[Any] | None = None,
    expected_cell_hash: HashRecord | Mapping[str, Any] | str | None = None,
    expected_gene_hash: HashRecord | Mapping[str, Any] | str | None = None,
    expected_parameters: Mapping[str, Any] | None = None,
    observed_parameters: Mapping[str, Any] | None = None,
    additional_findings: Iterable[CompatibilityFinding] = (),
    context: Mapping[str, Any] | None = None,
    policy: CompatibilityPolicy = "error",
) -> CompatibilityReport:
    """Validate aligned analysis artifacts using cells x genes count orientation."""

    checked: set[str] = set()
    findings: list[CompatibilityFinding] = list(additional_findings)

    def add(
        *,
        code: str,
        field: str,
        message: str,
        expected: Any,
        observed: Any,
        remediation: str,
        severity: FindingSeverity = "error",
    ) -> None:
        findings.append(
            CompatibilityFinding(
                code=code,
                severity=severity,
                field=field,
                message=message,
                expected=expected,
                observed=observed,
                remediation=remediation,
            )
        )

    counts_shape = _shape(counts) if counts is not None else None
    if counts is not None:
        checked.add("counts.shape")
        if counts_shape is None or len(counts_shape) != 2:
            add(
                code="counts.shape.invalid",
                field="counts.shape",
                message="Counts must be a two-dimensional cells x genes matrix.",
                expected={"ndim": 2, "orientation": "cells_x_genes"},
                observed={"shape": counts_shape},
                remediation="Provide counts with rows as cells and columns as genes.",
            )

    normalized_cells: list[str] | None = None
    if cell_ids is not None:
        checked.add("cell_ids")
        try:
            normalized_cells = normalize_identifiers(cell_ids)
        except (TypeError, ValueError) as exc:
            add(
                code="cell_ids.invalid",
                field="cell_ids",
                message=str(exc),
                expected="unique, non-empty identifiers",
                observed="invalid",
                remediation="Use the unique AnnData observation IDs captured at first load.",
            )

    normalized_genes: list[str] | None = None
    if gene_ids is not None:
        checked.add("gene_ids")
        try:
            normalized_genes = normalize_identifiers(gene_ids)
        except (TypeError, ValueError) as exc:
            add(
                code="gene_ids.invalid",
                field="gene_ids",
                message=str(exc),
                expected="unique, non-empty identifiers",
                observed="invalid",
                remediation="Use the ordered AnnData variable IDs at the count boundary.",
            )

    if counts_shape is not None and len(counts_shape) == 2:
        if normalized_cells is not None:
            checked.add("counts.cell_ids")
            if len(normalized_cells) != counts_shape[0]:
                add(
                    code="counts.cell_count_mismatch",
                    field="counts.shape[0]",
                    message="Count rows do not match the number of cell IDs.",
                    expected=len(normalized_cells),
                    observed=counts_shape[0],
                    remediation="Subset and reorder counts and cell IDs together.",
                )
        if normalized_genes is not None:
            checked.add("counts.gene_ids")
            if len(normalized_genes) != counts_shape[1]:
                add(
                    code="counts.gene_count_mismatch",
                    field="counts.shape[1]",
                    message="Count columns do not match the number of gene IDs.",
                    expected=len(normalized_genes),
                    observed=counts_shape[1],
                    remediation="Subset and reorder counts and gene IDs together.",
                )

    if metadata is not None:
        checked.add("metadata")
        metadata_shape = _shape(metadata)
        metadata_rows = metadata_shape[0] if metadata_shape else None
        expected_rows = (
            len(normalized_cells)
            if normalized_cells is not None
            else counts_shape[0]
            if counts_shape is not None and len(counts_shape) == 2
            else None
        )
        if expected_rows is not None and metadata_rows != expected_rows:
            add(
                code="metadata.row_count_mismatch",
                field="metadata.shape[0]",
                message="Metadata rows do not match the cell axis.",
                expected=expected_rows,
                observed=metadata_rows,
                remediation="Align metadata to the canonical cell IDs before continuing.",
            )
        metadata_index = getattr(metadata, "index", None)
        if (
            normalized_cells is not None
            and metadata_index is not None
            and metadata_rows == len(normalized_cells)
        ):
            try:
                normalized_index = normalize_identifiers(metadata_index)
            except (TypeError, ValueError) as exc:
                add(
                    code="metadata.index_invalid",
                    field="metadata.index",
                    message=str(exc),
                    expected="unique canonical cell IDs",
                    observed="invalid",
                    remediation="Set metadata.index to the initial AnnData observation IDs.",
                )
            else:
                mismatch = _first_axis_mismatch(normalized_cells, normalized_index)
                if mismatch is not None:
                    add(
                        code="metadata.index_order_mismatch",
                        field="metadata.index",
                        message="Metadata index is not in canonical cell order.",
                        expected=_axis_summary(
                            normalized_cells,
                            domain="cell-axis",
                        ),
                        observed={
                            **_axis_summary(
                                normalized_index,
                                domain="cell-axis",
                            ),
                            "first_mismatch": mismatch,
                        },
                        remediation="Reindex metadata with metadata.loc[cell_ids].",
                    )

    if graph is not None:
        checked.add("graph.shape")
        graph_shape = _shape(graph)
        expected_cells = (
            len(normalized_cells)
            if normalized_cells is not None
            else counts_shape[0]
            if counts_shape is not None and len(counts_shape) == 2
            else None
        )
        expected_shape = (
            [expected_cells, expected_cells] if expected_cells is not None else "square"
        )
        if (
            graph_shape is None
            or len(graph_shape) != 2
            or graph_shape[0] != graph_shape[1]
            or (
                expected_cells is not None
                and graph_shape != (expected_cells, expected_cells)
            )
        ):
            add(
                code="graph.shape_mismatch",
                field="graph.shape",
                message="Graph must be square and aligned to the count cell axis.",
                expected=expected_shape,
                observed={"shape": graph_shape},
                remediation="Subset and reorder graph rows/columns with the same cell IDs.",
            )

    def compare_axis(
        *,
        axis: str,
        observed_ids: list[str] | None,
        expected_ids: Iterable[Any] | None,
        expected_hash: HashRecord | Mapping[str, Any] | str | None,
        domain: str,
    ) -> None:
        if expected_ids is not None:
            checked.add(f"{axis}.expected_ids")
            try:
                normalized_expected = normalize_identifiers(expected_ids)
            except (TypeError, ValueError) as exc:
                add(
                    code=f"{axis}.expected_ids_invalid",
                    field=f"expected_{axis}_ids",
                    message=str(exc),
                    expected="unique, non-empty identifiers",
                    observed="invalid",
                    remediation="Correct the expected artifact provenance.",
                )
            else:
                if observed_ids is None:
                    add(
                        code=f"{axis}.ids_missing",
                        field=f"{axis}_ids",
                        message=f"Observed {axis} IDs are required for compatibility.",
                        expected=_axis_summary(
                            normalized_expected,
                            domain=domain,
                        ),
                        observed=None,
                        remediation=f"Pass the ordered observed {axis} IDs.",
                    )
                else:
                    mismatch = _first_axis_mismatch(normalized_expected, observed_ids)
                    if mismatch is not None:
                        add(
                            code=f"{axis}.order_mismatch",
                            field=f"{axis}_ids",
                            message=f"Observed {axis} IDs differ in order or membership.",
                            expected=_axis_summary(
                                normalized_expected,
                                domain=domain,
                            ),
                            observed={
                                **_axis_summary(observed_ids, domain=domain),
                                "first_mismatch": mismatch,
                            },
                            remediation=(
                                f"Reindex the observed artifact to expected {axis} IDs."
                            ),
                        )

        if expected_hash is not None:
            checked.add(f"{axis}.expected_hash")
            try:
                expected_record = _coerce_hash_record(expected_hash)
            except (ProvenanceSerializationError, TypeError, ValueError) as exc:
                add(
                    code=f"{axis}.expected_hash_invalid",
                    field=f"expected_{axis}_hash",
                    message=str(exc),
                    expected="a valid ordered-axis HashRecord or SHA-256 digest",
                    observed={"type": type(expected_hash).__name__},
                    remediation="Correct the parent artifact hash record.",
                )
                return

            if isinstance(expected_record, str):
                if not re.fullmatch(r"[0-9a-f]{64}", expected_record):
                    add(
                        code=f"{axis}.expected_hash_invalid",
                        field=f"expected_{axis}_hash",
                        message="Expected digest is not lowercase SHA-256 hexadecimal.",
                        expected={"length": 64, "encoding": "lowercase_hex"},
                        observed={"digest": expected_record},
                        remediation="Pass a complete SHA-256 digest or HashRecord.",
                    )
                    return
                expected_digest = expected_record
                expected_summary: Any = {"digest": expected_record}
            else:
                expected_semantics = {
                    "domain": domain,
                    "ordering": "ordered",
                    "encoding": "canonical-json-v1",
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                }
                observed_semantics = {
                    "domain": expected_record.domain,
                    "ordering": expected_record.ordering,
                    "encoding": expected_record.encoding,
                    "schema_version": expected_record.schema_version,
                }
                if observed_semantics != expected_semantics:
                    add(
                        code=f"{axis}.expected_hash_semantics_invalid",
                        field=f"expected_{axis}_hash",
                        message="Expected axis hash uses incompatible semantics.",
                        expected=expected_semantics,
                        observed=observed_semantics,
                        remediation=(
                            f"Recompute the parent {axis} hash as an ordered "
                            f"{domain!r} axis hash."
                        ),
                    )
                    return
                expected_digest = expected_record.digest
                expected_summary = expected_record.to_dict()

            if observed_ids is None:
                add(
                    code=f"{axis}.ids_missing",
                    field=f"{axis}_ids",
                    message=f"Observed {axis} IDs are required to verify the hash.",
                    expected=expected_summary,
                    observed=None,
                    remediation=f"Pass the ordered observed {axis} IDs.",
                )
                return

            observed_hash = hash_ordered_ids(observed_ids, domain=domain)
            if observed_hash.digest != expected_digest:
                add(
                    code=f"{axis}.hash_mismatch",
                    field=f"{axis}_ids",
                    message=f"Observed {axis} axis hash does not match.",
                    expected=expected_summary,
                    observed=observed_hash.to_dict(),
                    remediation=f"Use the exact ordered {axis} axis from the parent artifact.",
                )

    compare_axis(
        axis="cell",
        observed_ids=normalized_cells,
        expected_ids=expected_cell_ids,
        expected_hash=expected_cell_hash,
        domain="cell-axis",
    )
    compare_axis(
        axis="gene",
        observed_ids=normalized_genes,
        expected_ids=expected_gene_ids,
        expected_hash=expected_gene_hash,
        domain="gene-axis",
    )

    normalized_splits: tuple[Any, ...] | None = None
    if splits is not None:
        checked.add("splits")
        normalized_splits = tuple(splits)
        if not normalized_splits:
            add(
                code="splits.empty",
                field="splits",
                message="At least one split matrix is required.",
                expected="one or more matrices",
                observed=0,
                remediation="Pass the split matrices produced by multi_split.",
            )
        elif counts is None or counts_shape is None or len(counts_shape) != 2:
            add(
                code="splits.counts_missing",
                field="counts",
                message="Valid parent counts are required to validate splits.",
                expected="two-dimensional cells x genes counts",
                observed={"shape": counts_shape},
                remediation="Pass the unsplit parent count matrix.",
            )
        else:
            split_shapes = [_shape(item) for item in normalized_splits]
            invalid_shapes = [
                {"index": index, "shape": shape}
                for index, shape in enumerate(split_shapes)
                if shape != counts_shape
            ]
            if invalid_shapes:
                add(
                    code="splits.shape_mismatch",
                    field="splits",
                    message="One or more split matrices do not match parent counts.",
                    expected={"shape": counts_shape},
                    observed=invalid_shapes,
                    remediation="Preserve cells x genes shape for every split matrix.",
                )
            else:
                observation = _conservation_observation(counts, normalized_splits)
                if observation is not None:
                    add(
                        code="splits.conservation_failed",
                        field="splits",
                        message="Split matrices do not exactly conserve parent counts.",
                        expected={"elementwise_sum": "counts"},
                        observed=observation,
                        remediation="Regenerate splits and verify orientation and seed bridge.",
                    )

    if split_proportions is not None:
        checked.add("split_proportions")
        parse_failed = False
        try:
            proportions = [float(item) for item in split_proportions]
        except (TypeError, ValueError):
            proportions = []
            parse_failed = True
        split_count = len(normalized_splits) if normalized_splits is not None else None
        invalid = (
            not proportions
            or any(not math.isfinite(item) or item < 0 for item in proportions)
            or not math.isclose(sum(proportions), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or (split_count is not None and len(proportions) != split_count)
        )
        if invalid:
            observed_proportions = [
                item
                if math.isfinite(item)
                else {
                    "non_finite": (
                        "nan"
                        if math.isnan(item)
                        else "positive_infinity"
                        if item > 0
                        else "negative_infinity"
                    )
                }
                for item in proportions
            ]
            add(
                code="splits.proportions_invalid",
                field="split_proportions",
                message="Split proportions must be finite, nonnegative, and sum to one.",
                expected={"sum": 1.0, "count": split_count},
                observed={
                    "values": observed_proportions,
                    "parse_failed": parse_failed,
                },
                remediation="Provide one normalized proportion per split matrix.",
            )

    if expected_parameters is not None:
        checked.add("parameters")
        invalid_expected_keys = [
            key for key in expected_parameters if not isinstance(key, str)
        ]
        if invalid_expected_keys:
            add(
                code="parameters.expected_invalid",
                field="expected_parameters",
                message="Expected parameter keys must be strings.",
                expected="string keys",
                observed={"invalid_key_types": sorted(
                    {type(key).__name__ for key in invalid_expected_keys}
                )},
                remediation="Use explicit string names for algorithm parameters.",
            )
        elif observed_parameters is None:
            add(
                code="parameters.missing",
                field="parameters",
                message="Observed algorithm parameters were not provided.",
                expected={"keys": sorted(expected_parameters)},
                observed=None,
                remediation="Record and pass the observed algorithm parameters.",
            )
        else:
            invalid_observed_keys = [
                key for key in observed_parameters if not isinstance(key, str)
            ]
            if invalid_observed_keys:
                add(
                    code="parameters.observed_invalid",
                    field="observed_parameters",
                    message="Observed parameter keys must be strings.",
                    expected="string keys",
                    observed={"invalid_key_types": sorted(
                        {type(key).__name__ for key in invalid_observed_keys}
                    )},
                    remediation="Use explicit string names for algorithm parameters.",
                )
            else:
                findings.extend(
                    _parameter_findings(expected_parameters, observed_parameters)
                )

    report = CompatibilityReport.create(
        checked=checked,
        findings=findings,
        context=context,
    )
    return report.apply_policy(policy)


__all__ = [
    "CompatibilityError",
    "CompatibilityFinding",
    "CompatibilityPolicy",
    "CompatibilityReport",
    "CompatibilityWarning",
    "FindingSeverity",
    "validate_compatibility",
]
