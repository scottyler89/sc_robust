# DE Diagnostics Schema

DE results expose diagnostics through `DEAnalysisResult.diagnostics`, each
contrast through `DEAnalysisResult.contrast_diagnostics`, and failed fits through
`DEFitError.diagnostics`. These mappings are JSON-safe and are included in the
canonical provenance envelope.

## Fit Record

The fit-level record contains:

- `status`: `prepared`, `fit`, or `failed`.
- `fit_id` and `formula`.
- `input_gene_count`, `filtered_gene_count`, and `modeled_gene_count`.
- `observations.count` and `observations.per_condition`.
- `library_size` and `zero_library_count` summaries.
- `filtering` with `min_counts`, `min_variance`, and gene-list usage.
- `design` with formula, terms, columns, rank, condition number, reference, and coefficient map.
- `design_matrix` with shape, rank, condition number, and non-finite count after fitting.
- `execution` with requested and resolved CPU counts.
- `fallbacks`, `fallback_count`, and `convergence`. `fallback_count` counts
  applied ridge/retry events, not diagnostic records.
- `size_factors`, `dispersions`, and `cooks` numeric summaries; these are `null` when the backend does not expose the field.
- `outliers.cooks_count` and `outliers.refit_count`.
- `error`, `error_type`, and terminal status on failed fits.

A numeric summary has `count`, `finite_count`, `nonfinite_count`, `min`, `median`,
and `max`; empty summaries use null extrema.

## Contrast Record

Each contrast record contains `contrast_id`, `key`, `numerator`, `denominator`,
`reference`, `coefficient_labels`, `vector`, `direction`, `status`, and `nonfinite`
counts for `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, and `padj`.

```json
{
  "status": "fit",
  "fit_id": "de-fit:...",
  "formula": "~ 0 + condition",
  "input_gene_count": 18000,
  "filtered_gene_count": 4200,
  "modeled_gene_count": 13800,
  "observations": {"count": 24, "per_condition": {"condition": {"DMSO": 12, "drug": 12}}},
  "zero_library_count": 0,
  "fallback_count": 2,
  "convergence": {"genes": 13800, "terminal_failures": 0},
  "outliers": {"cooks_count": 3, "refit_count": 3}
}
```

## Failure Record

A fail-fast caller catches `DEFitError` and serializes `exc.diagnostics` without
inspecting private PyDESeq2 objects. The record retains the preparation fields and
adds `status: "failed"`, `error_type`, and `error`. No usable fit is represented
as successful.

## Batch Usage

Persist `result.export_provenance()` beside each result table. Use the stable
`fit_id` and `contrast_id` as join keys for batch summaries; do not use filenames
or Python object identity as scientific identifiers.
