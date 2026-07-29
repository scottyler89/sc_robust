import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from sc_robust.provenance import (
    HashRecord,
    ProvenanceEnvelope,
    ProvenanceSerializationError,
    canonical_json,
    capture_dependency_versions,
    hash_file,
    hash_membership_ids,
    hash_ordered_ids,
    hash_payload,
    stable_identifier,
    to_jsonable,
)


def test_canonical_json_is_deterministic_and_json_native():
    first = {
        "z": np.int64(4),
        "a": (True, Path("relative/artifact.txt")),
        "nested": {"value": np.float64(1.25)},
    }
    second = {
        "nested": {"value": 1.25},
        "a": [True, "relative/artifact.txt"],
        "z": 4,
    }

    assert canonical_json(first) == canonical_json(second)
    assert json.loads(canonical_json(first)) == second


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 2]),
        {"value": float("nan")},
        {1: "non-string key"},
        datetime(2025, 1, 1),
        object(),
        {1, 2},
    ],
)
def test_canonical_json_rejects_lossy_or_ambiguous_values(value):
    with pytest.raises(ProvenanceSerializationError):
        canonical_json(value)


def test_canonical_json_rejects_cycles():
    value = []
    value.append(value)

    with pytest.raises(ProvenanceSerializationError, match="Cyclic"):
        to_jsonable(value)


def test_identifier_hashes_distinguish_order_from_membership():
    ordered = hash_ordered_ids(["cell-b", "cell-a"], domain="cell-axis")
    reordered = hash_ordered_ids(["cell-a", "cell-b"], domain="cell-axis")
    members = hash_membership_ids(["cell-b", "cell-a"], domain="pb-membership")
    reordered_members = hash_membership_ids(
        ["cell-a", "cell-b"],
        domain="pb-membership",
    )

    assert ordered.ordering == "ordered"
    assert ordered.digest != reordered.digest
    assert members.ordering == "sorted"
    assert members.digest == reordered_members.digest
    assert members.count == 2


def test_identifier_hashes_are_domain_separated_and_require_unique_ids():
    cell_hash = hash_ordered_ids(["id-1"], domain="cell-axis")
    gene_hash = hash_ordered_ids(["id-1"], domain="gene-axis")

    assert cell_hash.digest != gene_hash.digest

    with pytest.raises(ValueError, match="duplicate"):
        hash_membership_ids(["id-1", "id-1"], domain="pb-membership")


def test_hash_record_round_trip():
    record = hash_payload({"answer": 42}, domain="test-payload")

    assert HashRecord.from_dict(record.to_dict()) == record


def test_hash_record_rejects_unknown_semantics():
    with pytest.raises(ValueError, match="ordering"):
        HashRecord(
            algorithm="sha256",
            digest="0" * 64,
            domain="test",
            encoding="canonical-json-v1",
            ordering="ambiguous",  # type: ignore[arg-type]
        )


def test_file_hash_is_content_based(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    first_hash = hash_file(first)
    second_hash = hash_file(second)

    assert first_hash.digest == second_hash.digest
    assert first_hash.ordering == "content"
    assert first_hash.encoding == "raw-bytes"


def _envelope(**overrides):
    values = {
        "stage": "pseudobulk",
        "parent_ids": ("graph:abc",),
        "algorithm": {"mode": "topology", "partition_by": ["sample"]},
        "inputs": {"cell_axis": {"digest": "abc", "shape": [10, 5]}},
        "environment": {"sc_robust": "0.2.0"},
        "execution": {"seed": 7, "n_cpus": 2},
        "diagnostics": {"warnings": []},
        "created_utc": "2025-01-01T00:00:00Z",
    }
    values.update(overrides)
    return ProvenanceEnvelope.create(**values)


def test_provenance_identity_excludes_event_metadata():
    first = _envelope()
    second = _envelope(
        diagnostics={"warnings": ["target size adjusted"]},
        created_utc="2025-01-02T00:00:00Z",
    )

    assert first.stable_id == second.stable_id
    assert first.to_dict()["diagnostics"] == {"warnings": []}
    assert second.to_dict()["diagnostics"] == {
        "warnings": ["target size adjusted"]
    }


def test_provenance_identity_includes_execution_and_canonical_mappings():
    first = _envelope(algorithm={"mode": "topology", "partition_by": ["sample"]})
    reordered = _envelope(
        algorithm={"partition_by": ["sample"], "mode": "topology"}
    )
    changed_seed = _envelope(execution={"seed": 8, "n_cpus": 2})

    assert first.stable_id == reordered.stable_id
    assert first.stable_id != changed_seed.stable_id


def test_provenance_is_immutable_and_round_trips():
    envelope = _envelope()
    payload = envelope.to_dict()
    reconstructed = ProvenanceEnvelope.from_dict(json.loads(envelope.to_json()))

    assert reconstructed == envelope
    assert reconstructed.to_dict() == payload
    with pytest.raises(TypeError):
        envelope.algorithm["mode"] = "within_cluster"


def test_provenance_detects_tampered_identity():
    payload = _envelope().to_dict()
    payload["algorithm"]["mode"] = "within_cluster"

    with pytest.raises(ValueError, match="stable_id mismatch"):
        ProvenanceEnvelope.from_dict(payload)


def test_with_diagnostics_preserves_stable_identity():
    envelope = _envelope()
    updated = envelope.with_diagnostics(
        {"warnings": [{"code": "pb-size", "count": 1}]},
        compatibility_report_id="compatibility-report:abc",
    )

    assert updated.stable_id == envelope.stable_id
    assert updated.compatibility_report_id == "compatibility-report:abc"
    assert updated.to_dict()["diagnostics"]["warnings"][0]["code"] == "pb-size"


def test_dependency_capture_is_stable_and_immutable():
    versions = capture_dependency_versions(
        ["definitely-not-a-real-package", "definitely-not-a-real-package"]
    )

    assert versions == {"definitely-not-a-real-package": None}
    with pytest.raises(TypeError):
        versions["definitely-not-a-real-package"] = "1.0"


@pytest.mark.parametrize("kind", ["", "Upper", "has space", "1starts-with-number"])
def test_stable_identifier_validates_kind(kind):
    with pytest.raises(ValueError):
        stable_identifier(kind, {"value": 1})
