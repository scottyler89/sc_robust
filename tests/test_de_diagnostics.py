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
