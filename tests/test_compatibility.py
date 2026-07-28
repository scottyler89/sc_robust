import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sc_robust.compatibility import (
    CompatibilityError,
    CompatibilityFinding,
    CompatibilityReport,
    CompatibilityWarning,
    validate_compatibility,
)
from sc_robust.provenance import hash_membership_ids, hash_ordered_ids


def _valid_inputs():
    counts = np.array(
        [
            [5, 2],
            [0, 3],
            [1, 1],
        ],
        dtype=np.int64,
    )
    splits = (
        np.array([[2, 1], [0, 1], [1, 0]], dtype=np.int64),
        np.array([[3, 1], [0, 2], [0, 1]], dtype=np.int64),
    )
    cell_ids = ["cell-a", "cell-b", "cell-c"]
    gene_ids = ["gene-a", "gene-b"]
    metadata = pd.DataFrame(
        {"sample": ["s1", "s1", "s2"]},
        index=cell_ids,
    )
    graph = sparse.eye(3, format="csr")
    return counts, splits, cell_ids, gene_ids, metadata, graph


def test_valid_artifacts_produce_compatible_report():
    counts, splits, cell_ids, gene_ids, metadata, graph = _valid_inputs()

    report = validate_compatibility(
        counts=counts,
        metadata=metadata,
        graph=graph,
        splits=splits,
        split_proportions=[0.5, 0.5],
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        expected_cell_ids=cell_ids,
        expected_gene_ids=gene_ids,
        expected_parameters={"mode": "topology"},
        observed_parameters={"mode": "topology", "seed": 7},
        context={"stage": "pseudobulk"},
    )

    assert report.is_compatible
    assert report.findings == ()
    assert "splits" in report.checked
    assert report.context == {"stage": "pseudobulk"}


def test_shape_equal_reordered_axes_fail_with_first_mismatch():
    counts, _, cell_ids, gene_ids, metadata, graph = _valid_inputs()

    report = validate_compatibility(
        counts=counts,
        metadata=metadata,
        graph=graph,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        expected_cell_ids=["cell-b", "cell-a", "cell-c"],
        expected_gene_ids=["gene-b", "gene-a"],
        policy="report",
    )

    codes = {finding.code for finding in report.errors}
    assert codes == {"cell.order_mismatch", "gene.order_mismatch"}
    cell_finding = next(
        finding for finding in report.errors if finding.code == "cell.order_mismatch"
    )
    assert cell_finding.observed["first_mismatch"] == {
        "index": 0,
        "expected_id": "cell-b",
        "observed_id": "cell-a",
    }
    assert cell_finding.remediation


def test_error_policy_raises_with_complete_report():
    counts, _, cell_ids, gene_ids, metadata, _ = _valid_inputs()
    bad_graph = sparse.eye(2, format="csr")

    with pytest.raises(CompatibilityError) as exc_info:
        validate_compatibility(
            counts=counts,
            metadata=metadata.iloc[:2],
            graph=bad_graph,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
        )

    codes = {finding.code for finding in exc_info.value.report.errors}
    assert codes == {"metadata.row_count_mismatch", "graph.shape_mismatch"}
    assert "Remediation:" in str(exc_info.value)


def test_warning_policy_emits_structured_summary_without_changing_checks():
    finding = CompatibilityFinding(
        code="pseudobulk.size_adjusted",
        severity="warning",
        field="cells_per_pb",
        message="METIS produced a larger pseudobulk.",
        expected={"target": 20},
        observed={"maximum": 27},
        remediation="Review the recorded size range; no cells were dropped.",
    )

    with pytest.warns(CompatibilityWarning, match="pseudobulk.size_adjusted"):
        warned = validate_compatibility(
            additional_findings=[finding],
            policy="warn",
        )
    reported = validate_compatibility(
        additional_findings=[finding],
        policy="report",
    )

    assert warned.report_id == reported.report_id
    assert warned.checked == reported.checked


def test_metadata_index_order_is_checked_against_canonical_cell_ids():
    counts, _, cell_ids, gene_ids, metadata, _ = _valid_inputs()
    reordered = metadata.iloc[[1, 0, 2]]

    report = validate_compatibility(
        counts=counts,
        metadata=reordered,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        policy="report",
    )

    finding = report.errors[0]
    assert finding.code == "metadata.index_order_mismatch"
    assert finding.observed["first_mismatch"]["index"] == 0


def test_dense_split_conservation_failure_reports_first_cell_and_gene():
    counts, splits, cell_ids, gene_ids, _, _ = _valid_inputs()
    broken = [item.copy() for item in splits]
    broken[1][2, 1] += 1

    report = validate_compatibility(
        counts=counts,
        splits=broken,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        policy="report",
    )

    finding = next(
        item for item in report.errors if item.code == "splits.conservation_failed"
    )
    assert finding.observed["first_index"] == (2, 1)
    assert finding.observed["delta"] == 1


def test_sparse_split_conservation_is_checked_without_dense_conversion():
    counts, splits, cell_ids, gene_ids, _, _ = _valid_inputs()
    sparse_counts = sparse.csr_matrix(counts)
    sparse_splits = [sparse.csr_matrix(item) for item in splits]
    sparse_splits[0][0, 0] += 1

    report = validate_compatibility(
        counts=sparse_counts,
        splits=sparse_splits,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        policy="report",
    )

    finding = next(
        item for item in report.errors if item.code == "splits.conservation_failed"
    )
    assert finding.observed["first_index"] == (0, 0)
    assert finding.observed["mismatch_count"] == 1


@pytest.mark.parametrize(
    "proportions",
    [
        [],
        [0.6, 0.6],
        [0.5, -0.5],
        [0.5, float("nan")],
        [1.0],
    ],
)
def test_invalid_split_proportions_are_actionable(proportions):
    counts, splits, cell_ids, gene_ids, _, _ = _valid_inputs()

    report = validate_compatibility(
        counts=counts,
        splits=splits,
        split_proportions=proportions,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        policy="report",
    )

    assert any(
        finding.code == "splits.proportions_invalid"
        for finding in report.errors
    )


def test_expected_axis_hash_is_verified():
    counts, _, cell_ids, gene_ids, _, _ = _valid_inputs()
    expected = hash_ordered_ids(cell_ids, domain="cell-axis")

    valid = validate_compatibility(
        counts=counts,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        expected_cell_hash=expected,
    )
    invalid = validate_compatibility(
        counts=counts,
        cell_ids=list(reversed(cell_ids)),
        gene_ids=gene_ids,
        expected_cell_hash=expected,
        policy="report",
    )

    assert valid.is_compatible
    assert invalid.errors[0].code == "cell.hash_mismatch"


def test_parameter_mismatches_report_nested_field():
    report = validate_compatibility(
        expected_parameters={"partition": {"mode": "topology", "seed": 7}},
        observed_parameters={"partition": {"mode": "within_cluster"}},
        policy="report",
    )

    fields = {finding.field for finding in report.errors}
    assert fields == {"parameters.partition.mode", "parameters.partition.seed"}


def test_report_identity_excludes_timestamp_and_round_trips():
    finding = CompatibilityFinding(
        code="test.failure",
        severity="error",
        field="test",
        message="Test mismatch.",
        expected=1,
        observed=2,
        remediation="Make the values equal.",
    )
    first = CompatibilityReport.create(
        checked=["b", "a"],
        findings=[finding],
        context={"stage": "test"},
        created_utc="2025-01-01T00:00:00Z",
    )
    second = CompatibilityReport.create(
        checked=["a", "b"],
        findings=[finding],
        context={"stage": "test"},
        created_utc="2025-01-02T00:00:00Z",
    )
    reconstructed = CompatibilityReport.from_dict(
        json.loads(first.to_json())
    )

    assert first.report_id == second.report_id
    assert reconstructed == first
    assert reconstructed.to_dataframe().loc[0, "code"] == "test.failure"


def test_tampered_report_identity_is_rejected():
    report = CompatibilityReport.create(checked=["counts.shape"])
    payload = report.to_dict()
    payload["checked"].append("graph.shape")

    with pytest.raises(ValueError, match="report_id mismatch"):
        CompatibilityReport.from_dict(payload)


def test_duplicate_identifiers_are_reported_not_silently_hashed():
    counts, _, _, gene_ids, _, _ = _valid_inputs()

    report = validate_compatibility(
        counts=counts,
        cell_ids=["cell-a", "cell-a", "cell-c"],
        gene_ids=gene_ids,
        policy="report",
    )

    assert report.errors[0].code == "cell_ids.invalid"

def test_expected_axis_hash_rejects_membership_semantics():
    counts, _, cell_ids, gene_ids, _, _ = _valid_inputs()
    membership_hash = hash_membership_ids(cell_ids, domain="cell-axis")

    report = validate_compatibility(
        counts=counts,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        expected_cell_hash=membership_hash,
        policy="report",
    )

    assert report.errors[0].code == "cell.expected_hash_semantics_invalid"


def test_expected_axis_hash_rejects_malformed_digest():
    counts, _, cell_ids, gene_ids, _, _ = _valid_inputs()

    report = validate_compatibility(
        counts=counts,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        expected_cell_hash="not-a-sha256-digest",
        policy="report",
    )

    assert report.errors[0].code == "cell.expected_hash_invalid"


def test_invalid_parameter_values_become_structured_findings():
    report = validate_compatibility(
        expected_parameters={"opaque": object()},
        observed_parameters={"opaque": "value"},
        policy="report",
    )

    assert report.errors[0].code == "parameters.expected_invalid"
    assert report.errors[0].field == "parameters.opaque"
