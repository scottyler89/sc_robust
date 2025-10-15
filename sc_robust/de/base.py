"""Shared data structures for differential expression workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import pandas as pd


@dataclass
class PseudobulkResult:
    """Container for pseudobulk expression matrices and metadata."""

    counts: pd.DataFrame
    metadata: pd.DataFrame
    parameters: Mapping[str, Any] = field(default_factory=dict)
    graph_summary: Optional[Mapping[str, Any]] = None

    def to_dataframe(
        self,
        prefix: str = "",
        columns: Optional[Sequence[str]] = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Return a tidy DataFrame representation suitable for downstream joins.

        Parameters
        ----------
        prefix:
            Optional prefix applied to column names.
        columns:
            Subset of count columns to retain (defaults to all).
        include_metadata:
            When True, concatenate metadata columns alongside counts.
        """
        counts = self.counts if columns is None else self.counts.loc[:, list(columns)]
        if prefix:
            counts = counts.add_prefix(prefix)
        if include_metadata:
            meta = self.metadata.copy()
            if prefix:
                meta = meta.add_prefix(f"{prefix}meta_" if not prefix.endswith("_") else f"{prefix}meta")
            merged = pd.concat([meta, counts], axis=1)
            return merged
        return counts


@dataclass
class DEAnalysisResult:
    """Structured output for differential expression runs."""

    dds: Any
    contrast_results: Mapping[str, pd.DataFrame]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    design_columns: Optional[Sequence[str]] = None
    artifacts: Optional[MutableMapping[str, Any]] = None

    def get_summary(self, key: str) -> pd.DataFrame:
        """Return the results DataFrame for a specific contrast."""
        try:
            return self.contrast_results[key]
        except KeyError as exc:
            raise KeyError(f"Contrast '{key}' not found.") from exc


@dataclass
class PathwayEnrichmentResult:
    """Container for pathway enrichment outputs."""

    per_contrast: Mapping[str, pd.DataFrame]
    libraries: Sequence[str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    concatenated: Optional[pd.DataFrame] = None

    def tidy(self) -> pd.DataFrame:
        """Return a concatenated long-form DataFrame (compute if needed)."""
        if self.concatenated is not None:
            return self.concatenated
        frames = []
        for contrast, df in self.per_contrast.items():
            temp = df.copy()
            temp.insert(0, "contrast", contrast)
            frames.append(temp)
        if frames:
            self.concatenated = pd.concat(frames, ignore_index=True)
        else:
            self.concatenated = pd.DataFrame()
        return self.concatenated
