"""Canonical provenance, serialization, and hashing primitives."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np


PROVENANCE_SCHEMA_VERSION = "1"
HashOrdering = Literal["canonical", "ordered", "sorted", "content"]
_HASH_ENCODINGS = {"canonical-json-v1", "raw-bytes"}
_ID_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


class ProvenanceSerializationError(TypeError):
    """Raised when a provenance value cannot be represented canonically."""


def _normalize_json(value: Any, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, Enum):
        return _normalize_json(value.value, seen)

    if isinstance(value, np.ndarray):
        raise ProvenanceSerializationError(
            "NumPy arrays are not valid provenance values; record a shape, "
            "identifier hash, or content hash instead."
        )
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), seen)

    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ProvenanceSerializationError(
                f"Non-finite float {value!r} is not valid provenance."
            )
        return normalized

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ProvenanceSerializationError(
                "Naive datetimes are not valid provenance; use an explicit timezone."
            )
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if is_dataclass(value) and not isinstance(value, type):
        object_id = id(value)
        if object_id in seen:
            raise ProvenanceSerializationError("Cyclic dataclass provenance is invalid.")
        seen.add(object_id)
        try:
            return {
                field.name: _normalize_json(getattr(value, field.name), seen)
                for field in fields(value)
            }
        finally:
            seen.remove(object_id)

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            raise ProvenanceSerializationError("Cyclic mapping provenance is invalid.")
        seen.add(object_id)
        try:
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ProvenanceSerializationError(
                        "Provenance mapping keys must be strings; "
                        f"received {type(key).__name__}."
                    )
                normalized[key] = _normalize_json(item, seen)
            return normalized
        finally:
            seen.remove(object_id)

    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in seen:
            raise ProvenanceSerializationError("Cyclic sequence provenance is invalid.")
        seen.add(object_id)
        try:
            return [_normalize_json(item, seen) for item in value]
        finally:
            seen.remove(object_id)

    raise ProvenanceSerializationError(
        f"Unsupported provenance value {type(value).__module__}."
        f"{type(value).__qualname__}."
    )


def to_jsonable(value: Any) -> Any:
    """Return a strict JSON-native copy of a supported provenance value."""

    return _normalize_json(value, set())


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize provenance deterministically, rejecting lossy conversions."""

    return json.dumps(
        to_jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes."""

    return canonical_json(value).encode("utf-8")


def _freeze_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_normalized(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_normalized(item) for item in value)
    return value


def freeze_json(value: Any) -> Any:
    """Normalize and recursively freeze a provenance value."""

    return _freeze_normalized(to_jsonable(value))


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable, canonical provenance mapping."""

    if value is None:
        return MappingProxyType({})
    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise ProvenanceSerializationError("Expected a provenance mapping.")
    return frozen


def utc_now() -> str:
    """Return a canonical UTC event timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class HashRecord:
    """Versioned description of a provenance hash."""

    algorithm: str
    digest: str
    domain: str
    encoding: str
    ordering: HashOrdering
    schema_version: str = PROVENANCE_SCHEMA_VERSION
    count: int | None = None

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ValueError(f"Unsupported hash algorithm: {self.algorithm!r}.")
        if not self.domain:
            raise ValueError("Hash domain must be non-empty.")
        if self.encoding not in _HASH_ENCODINGS:
            raise ValueError(f"Unsupported hash encoding: {self.encoding!r}.")
        if self.ordering not in {"canonical", "ordered", "sorted", "content"}:
            raise ValueError(f"Unsupported hash ordering: {self.ordering!r}.")
        if not self.schema_version:
            raise ValueError("Hash schema_version must be non-empty.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ValueError("SHA-256 digest must be 64 lowercase hexadecimal characters.")
        if self.count is not None and self.count < 0:
            raise ValueError("Hash count must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native record."""

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HashRecord":
        """Reconstruct and validate a hash record."""

        return cls(**to_jsonable(payload))


def _domain_prefix(domain: str, schema_version: str) -> bytes:
    if not domain:
        raise ValueError("Hash domain must be non-empty.")
    return f"sc_robust:{schema_version}:{domain}\0".encode("utf-8")


def hash_payload(
    payload: Any,
    *,
    domain: str,
    schema_version: str = PROVENANCE_SCHEMA_VERSION,
) -> HashRecord:
    """Hash canonical JSON with schema and domain separation."""

    digest = hashlib.sha256()
    digest.update(_domain_prefix(domain, schema_version))
    digest.update(canonical_json_bytes(payload))
    return HashRecord(
        algorithm="sha256",
        digest=digest.hexdigest(),
        domain=domain,
        encoding="canonical-json-v1",
        ordering="canonical",
        schema_version=schema_version,
    )


def hash_array_content(
    value: Any,
    *,
    domain: str = "array-content",
    schema_version: str = PROVENANCE_SCHEMA_VERSION,
) -> HashRecord:
    """Hash dense or sparse numeric array content with shape and dtype."""
    from scipy import sparse

    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    digest = hashlib.sha256()
    digest.update(_domain_prefix(domain, schema_version))
    if sparse.issparse(value):
        matrix = value.tocsr()
        digest.update(b"sparse-csr\0")
        digest.update(str(matrix.shape).encode("ascii"))
        digest.update(str(matrix.dtype).encode("ascii"))
        for part in (matrix.indptr, matrix.indices, matrix.data):
            array = np.ascontiguousarray(part)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        count = int(matrix.shape[0] * matrix.shape[1])
    else:
        array = np.asarray(value)
        if array.ndim == 0 or array.dtype.kind == "O":
            raise ProvenanceSerializationError("Array content hashes require numeric arrays.")
        array = np.ascontiguousarray(array)
        digest.update(b"dense\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        count = int(array.size)
    return HashRecord(
        algorithm="sha256",
        digest=digest.hexdigest(),
        domain=domain,
        encoding="raw-bytes",
        ordering="content",
        schema_version=schema_version,
        count=count,
    )

def normalize_identifiers(values: Iterable[Any]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Identifier collections must not be a string or bytes value.")
    normalized = [str(value) for value in values]
    empty_positions = [index for index, value in enumerate(normalized) if not value]
    if empty_positions:
        raise ValueError(
            "Identifiers must be non-empty; "
            f"first empty identifier is at position {empty_positions[0]}."
        )
    if len(set(normalized)) != len(normalized):
        seen: set[str] = set()
        for value in normalized:
            if value in seen:
                raise ValueError(
                    f"Identifiers must be unique; duplicate {value!r}."
                )
            seen.add(value)
        raise AssertionError("Duplicate identifier detection failed.")
    return normalized


def hash_ordered_ids(
    values: Iterable[Any],
    *,
    domain: str,
    schema_version: str = PROVENANCE_SCHEMA_VERSION,
) -> HashRecord:
    """Hash unique identifiers while preserving axis order."""

    normalized = normalize_identifiers(values)
    record = hash_payload(normalized, domain=domain, schema_version=schema_version)
    return replace(record, ordering="ordered", count=len(normalized))


def hash_membership_ids(
    values: Iterable[Any],
    *,
    domain: str,
    schema_version: str = PROVENANCE_SCHEMA_VERSION,
) -> HashRecord:
    """Hash unique identifiers as an order-insensitive membership set."""

    normalized = sorted(normalize_identifiers(values))
    record = hash_payload(normalized, domain=domain, schema_version=schema_version)
    return replace(record, ordering="sorted", count=len(normalized))


def hash_file(
    path: str | Path,
    *,
    domain: str = "artifact-content",
    chunk_size: int = 1024 * 1024,
    schema_version: str = PROVENANCE_SCHEMA_VERSION,
) -> HashRecord:
    """Hash raw file content independently of its filesystem location."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return HashRecord(
        algorithm="sha256",
        digest=digest.hexdigest(),
        domain=domain,
        encoding="raw-bytes",
        ordering="content",
        schema_version=schema_version,
    )


def stable_identifier(
    kind: str,
    payload: Any,
    *,
    schema_version: str = PROVENANCE_SCHEMA_VERSION,
) -> str:
    """Create a domain-separated stable identifier."""

    if not _ID_KIND_RE.fullmatch(kind):
        raise ValueError(
            "Identifier kind must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_', or '-'."
        )
    digest = hash_payload(
        payload,
        domain=f"stable-id.{kind}",
        schema_version=schema_version,
    ).digest
    return f"{kind}:{digest}"


def capture_dependency_versions(names: Iterable[str]) -> Mapping[str, str | None]:
    """Capture installed distribution versions once in stable key order."""

    versions: dict[str, str | None] = {}
    for name in sorted(set(names)):
        if not name:
            raise ValueError("Dependency names must be non-empty.")
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return freeze_mapping(versions)


def _identity_payload(
    *,
    schema_version: str,
    stage: str,
    parent_ids: Sequence[str],
    algorithm: Mapping[str, Any],
    inputs: Mapping[str, Any],
    environment: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "stage": stage,
        "parent_ids": list(parent_ids),
        "algorithm": algorithm,
        "inputs": inputs,
        "environment": environment,
        "execution": execution,
    }


@dataclass(frozen=True)
class ProvenanceEnvelope(Mapping[str, Any]):
    """Immutable, versioned provenance for one analysis stage."""

    stage: str
    stable_id: str
    parent_ids: tuple[str, ...]
    algorithm: Mapping[str, Any]
    inputs: Mapping[str, Any]
    environment: Mapping[str, Any]
    execution: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    compatibility_report_id: str | None
    created_utc: str
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _ID_KIND_RE.fullmatch(self.stage):
            raise ValueError(
                "Provenance stage must use lowercase identifier characters."
            )
        normalized_parents = tuple(str(item) for item in self.parent_ids)
        if len(set(normalized_parents)) != len(normalized_parents):
            raise ValueError("Provenance parent IDs must be unique.")

        object.__setattr__(self, "parent_ids", normalized_parents)
        for name in ("algorithm", "inputs", "environment", "execution", "diagnostics"):
            object.__setattr__(self, name, freeze_mapping(getattr(self, name)))

        expected_id = stable_identifier(
            self.stage,
            _identity_payload(
                schema_version=self.schema_version,
                stage=self.stage,
                parent_ids=self.parent_ids,
                algorithm=self.algorithm,
                inputs=self.inputs,
                environment=self.environment,
                execution=self.execution,
            ),
            schema_version=self.schema_version,
        )
        if self.stable_id != expected_id:
            raise ValueError(
                f"Provenance stable_id mismatch: expected {expected_id!r}, "
                f"observed {self.stable_id!r}."
            )

    def __getitem__(self, key: str) -> Any:
        record = self.to_dict()
        if key in record:
            return record[key]
        if key == "deps":
            return record["environment"]
        if key == "adata":
            return record["inputs"].get("adata", {})
        if key in self.diagnostics:
            return self.diagnostics[key]
        raise KeyError(key)

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        parent_ids: Sequence[str] = (),
        algorithm: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        execution: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        compatibility_report_id: str | None = None,
        created_utc: str | None = None,
        schema_version: str = PROVENANCE_SCHEMA_VERSION,
    ) -> "ProvenanceEnvelope":
        """Create an envelope whose identity excludes event-only diagnostics."""

        normalized_parents = tuple(str(item) for item in parent_ids)
        normalized_algorithm = freeze_mapping(algorithm)
        normalized_inputs = freeze_mapping(inputs)
        normalized_environment = freeze_mapping(environment)
        normalized_execution = freeze_mapping(execution)
        stable_id = stable_identifier(
            stage,
            _identity_payload(
                schema_version=schema_version,
                stage=stage,
                parent_ids=normalized_parents,
                algorithm=normalized_algorithm,
                inputs=normalized_inputs,
                environment=normalized_environment,
                execution=normalized_execution,
            ),
            schema_version=schema_version,
        )
        return cls(
            stage=stage,
            stable_id=stable_id,
            parent_ids=normalized_parents,
            algorithm=normalized_algorithm,
            inputs=normalized_inputs,
            environment=normalized_environment,
            execution=normalized_execution,
            diagnostics=freeze_mapping(diagnostics),
            compatibility_report_id=compatibility_report_id,
            created_utc=created_utc or utc_now(),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-native provenance record."""

        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "stable_id": self.stable_id,
            "parent_ids": list(self.parent_ids),
            "algorithm": to_jsonable(self.algorithm),
            "inputs": to_jsonable(self.inputs),
            "environment": to_jsonable(self.environment),
            "execution": to_jsonable(self.execution),
            "diagnostics": to_jsonable(self.diagnostics),
            "compatibility_report_id": self.compatibility_report_id,
            "created_utc": self.created_utc,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON without lossy fallback conversion."""

        return canonical_json(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceEnvelope":
        """Reconstruct an envelope and verify its stable identity."""

        normalized = to_jsonable(payload)
        expected_fields = {
            "schema_version",
            "stage",
            "stable_id",
            "parent_ids",
            "algorithm",
            "inputs",
            "environment",
            "execution",
            "diagnostics",
            "compatibility_report_id",
            "created_utc",
        }
        unknown = set(normalized) - expected_fields
        missing = expected_fields - set(normalized)
        if unknown or missing:
            raise ValueError(
                "Invalid provenance fields: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return cls(
            stage=normalized["stage"],
            stable_id=normalized["stable_id"],
            parent_ids=tuple(normalized["parent_ids"]),
            algorithm=normalized["algorithm"],
            inputs=normalized["inputs"],
            environment=normalized["environment"],
            execution=normalized["execution"],
            diagnostics=normalized["diagnostics"],
            compatibility_report_id=normalized["compatibility_report_id"],
            created_utc=normalized["created_utc"],
            schema_version=normalized["schema_version"],
        )

    def with_diagnostics(
        self,
        diagnostics: Mapping[str, Any],
        *,
        compatibility_report_id: str | None = None,
    ) -> "ProvenanceEnvelope":
        """Return a new event snapshot without changing the stable run identity."""

        return replace(
            self,
            diagnostics=freeze_mapping(diagnostics),
            compatibility_report_id=(
                self.compatibility_report_id
                if compatibility_report_id is None
                else compatibility_report_id
            ),
        )


__all__ = [
    "HashRecord",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceEnvelope",
    "ProvenanceSerializationError",
    "canonical_json",
    "canonical_json_bytes",
    "capture_dependency_versions",
    "freeze_json",
    "freeze_mapping",
    "hash_array_content",
    "hash_file",
    "hash_membership_ids",
    "hash_ordered_ids",
    "hash_payload",
    "normalize_identifiers",
    "stable_identifier",
    "to_jsonable",
    "utc_now",
]
