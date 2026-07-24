import pytest
import numpy as np
import pandas as pd

from sc_robust.de.differential_expression import _contrast_record
from sc_robust.de.design import build_design_frame, make_design_spec


def test_contrast_record_is_json_safe_and_stable():
    metadata = pd.DataFrame({"condition": ["control", "treated"]})
    design, terms, mapping, reference = build_design_frame(metadata, formula="~ 0 + condition")
    spec = make_design_spec(
        design,
        formula="~ 0 + condition",
        terms=terms,
        coefficient_map=mapping,
        reference=reference,
    )
    table = pd.DataFrame(
        {"log2FoldChange": [1.0, np.nan], "pvalue": [0.01, 0.5]},
        index=["g1", "g2"],
    )
    record = _contrast_record(
        "treated_vs_control",
        np.array([1.0, -1.0]),
        spec.columns,
        table,
        numerator=["treated"],
        denominator=["control"],
        reference=spec.reference,
    )
    assert record["contrast_id"].startswith("de-contrast:")
    assert record["direction"] == "numerator_minus_denominator"
    assert record["nonfinite"]["log2FoldChange"] == 1
    assert _contrast_record(
        "treated_vs_control",
        np.array([1.0, -1.0]),
        spec.columns,
        table,
        numerator=["treated"],
        denominator=["control"],
        reference=spec.reference,
    )["contrast_id"] == record["contrast_id"]


class _FailingDDS:
    refit_cooks = False

    def fit_size_factors(self):
        raise ValueError("synthetic fit failure")


def test_fit_failure_raises_with_diagnostics(monkeypatch):
    from sc_robust.de import differential_expression as de
    from sc_robust.de.design import DEFitError

    monkeypatch.setattr(de, "_import_pydeseq2", lambda: (None, None, None))
    dds = _FailingDDS()
    try:
        de.fit_deseq_dataset(dds)
    except DEFitError as exc:
        assert exc.diagnostics["status"] == "failed"
        assert exc.diagnostics["error_type"] == "ValueError"
    else:
        raise AssertionError("fit failure did not produce DEFitError")


class _SuccessfulDDS:
    refit_cooks = False
    inference = type("Inference", (), {"last_irls_diagnostics": [{"irls_ridge_applied": True}], "last_alpha_mle_diagnostics": [{"cox_reid_ridge_retry": True}]})()
    obsm = {}

    def fit_size_factors(self): pass
    def fit_genewise_dispersions(self): pass
    def fit_dispersion_prior(self): pass
    def fit_MAP_dispersions(self): pass
    def fit_LFC(self): pass
    def calculate_cooks(self): pass


def test_fit_success_collects_fallback_records(monkeypatch):
    from sc_robust.de import differential_expression as de

    monkeypatch.setattr(de, "_import_pydeseq2", lambda: (None, None, None))
    dds = _SuccessfulDDS()
    de.fit_deseq_dataset(dds)
    assert dds._sc_robust_diagnostics["fallback_count"] == 2
    assert dds._sc_robust_diagnostics["fallbacks"][0]["irls_ridge_applied"]


def test_filtering_failure_is_explicit(monkeypatch):
    from sc_robust.de import differential_expression as de
    from sc_robust.de.base import PseudobulkResult

    class FakeDDS:
        def __init__(self, **kwargs): pass
    class FakeInference:
        def __init__(self, **kwargs): pass
    monkeypatch.setattr(de, "_import_pydeseq2", lambda: (FakeDDS, None, FakeInference))
    counts = pd.DataFrame([[1, 1], [1, 1]], columns=["g1", "g2"], index=["pb1", "pb2"])
    metadata = pd.DataFrame({"condition": ["a", "b"]}, index=counts.index)
    result = PseudobulkResult(counts=counts, metadata=metadata)
    try:
        de.prepare_deseq_dataset(result, design="~ 0 + condition", min_counts=100, min_variance=None)
    except ValueError as exc:
        assert "No genes remain" in str(exc)
    else:
        raise AssertionError("empty filtered fit did not fail")


def test_pairs_evaluates_only_requested_pair_and_alias_conflict(monkeypatch):
    from sc_robust.de import differential_expression as de

    class Stats:
        def __init__(self):
            self.results_df = pd.DataFrame({"log2FoldChange": [1.0], "pvalue": [0.1], "padj": [0.1]}, index=["ENSG00000141510"])
        def plot_MA(self): return None
    dds = type("DDS", (), {"obsm": {"design_matrix": pd.DataFrame(np.eye(3), columns=["a", "b", "c"])}})()
    calls = []
    def fake_run(dds, contrast, **kwargs):
        calls.append(np.asarray(contrast).tolist())
        return Stats()
    monkeypatch.setattr(de, "_run_single_contrast", fake_run)
    result = de.run_pairwise_de(dds, pairs=[("a", "b")], n_jobs=1)
    assert list(result.contrast_results) == ["a_vs_b"]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="either pairs or cluster_pairs"):
        de.run_pairwise_de(dds, pairs=[("a", "b")], cluster_pairs=[("a", "b")])
