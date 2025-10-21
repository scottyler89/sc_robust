"""Shared data structures for differential expression workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from importlib import resources
import pandas as pd

GENE_ANNOTATIONS_FILENAME = "ensg_annotations_abbreviated.txt"


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


def load_default_gene_annotations() -> pd.DataFrame:
    """
    Load the packaged ENSG annotations and return a standardized DataFrame.

    The returned frame is indexed by Ensembl gene IDs (`gene_id`) and always
    exposes at least `gene_id`, `gene_name`, `chromosome`, and
    `gene_description` columns so downstream DE helpers can rely on a stable
    schema.
    """
    try:
        data_path = resources.files("sc_robust.data").joinpath(GENE_ANNOTATIONS_FILENAME)
    except AttributeError as exc:
        raise FileNotFoundError(
            f"Could not locate packaged gene annotations '{GENE_ANNOTATIONS_FILENAME}'. "
            "Verify that the resource is included in the installed distribution."
        ) from exc

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Packaged gene annotation file '{GENE_ANNOTATIONS_FILENAME}' is missing. "
            "Reinstall the package or ensure setup.py includes the data file."
        )

    with data_path.open("rb") as handle:
        df = pd.read_csv(handle, sep="\t")

    rename_map = {
        "Gene stable ID": "gene_id",
        "Chromosome/scaffold name": "chromosome",
        "Gene name": "gene_name",
        "Gene description": "gene_description",
    }
    missing = [orig for orig in rename_map if orig not in df.columns]
    if missing:
        raise KeyError(
            f"Packaged gene annotation file is missing expected columns: {missing}. "
            "Ensure the resource matches the documented schema."
        )

    df = df.rename(columns=rename_map)
    df = df.drop_duplicates(subset="gene_id", keep="first")
    df = df.set_index("gene_id", drop=False)
    df.index = df.index.astype(str)
    df["gene_name"] = df["gene_name"].astype(str)
    return df
