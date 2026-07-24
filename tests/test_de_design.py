import numpy as np
import pandas as pd

from sc_robust.de.design import build_design_frame, make_design_spec


def test_explicit_no_intercept_categorical_design_is_canonical():
    metadata = pd.DataFrame(
        {"condition": ["control", "treated", "control", "treated"], "batch": ["a", "a", "b", "b"]},
        index=["s1", "s2", "s3", "s4"],
    )
    design, terms, mapping, reference = build_design_frame(
        metadata, formula="~ 0 + condition", annotation_columns=["batch"]
    )
    spec = make_design_spec(
        design,
        formula="~ 0 + condition",
        terms=terms,
        coefficient_map=mapping,
        reference=reference,
    )
    assert list(design.columns) == ["condition_control", "condition_treated"]
    assert spec.reference == "control"
    assert spec.rank == 2
    assert spec.shape == (4, 2)
    assert spec.fit_id == make_design_spec(
        design,
        formula="~ 0 + condition",
        terms=terms,
        coefficient_map=mapping,
        reference=reference,
    ).fit_id


def test_intercept_design_has_reference_level_and_rejects_aliasing():
    metadata = pd.DataFrame({"condition": ["a", "b", "a", "b"]})
    design, terms, mapping, reference = build_design_frame(metadata, formula="~ condition")
    assert list(design.columns) == ["Intercept", "condition_b"]
    assert reference == "a"
    assert np.linalg.matrix_rank(design.to_numpy()) == 2


def test_missing_term_and_null_term_fail_strictly():
    metadata = pd.DataFrame({"condition": ["a", None]})
    try:
        build_design_frame(metadata, formula="~ 0 + missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing design term did not fail")
    try:
        build_design_frame(metadata, formula="~ 0 + condition")
    except ValueError as exc:
        assert "null" in str(exc)
    else:
        raise AssertionError("null design term did not fail")
