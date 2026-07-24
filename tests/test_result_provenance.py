import pandas as pd
import pytest

from sc_robust.de.base import DEAnalysisResult, PathwayEnrichmentResult, PseudobulkResult


def test_all_stage_results_export_one_immutable_provenance_shape():
    pb = PseudobulkResult(
        counts=pd.DataFrame([[1, 2]], index=["cell-1"], columns=["gene-1", "gene-2"]),
        metadata=pd.DataFrame({"sample": ["sample-1"]}, index=["cell-1"]),
        parameters={"mode": "within_cluster"},
    )
    de = DEAnalysisResult(
        dds=None,
        contrast_results={"A_vs_B": pd.DataFrame()},
        parameters={"alpha": 0.05},
        design_columns=["cluster_A", "cluster_B"],
    )
    pathway = PathwayEnrichmentResult(
        per_contrast={"A_vs_B": pd.DataFrame()},
        libraries=["c2.all.v2025.1.Hs.symbols.gmt"],
        parameters={"stat_col": "custom_t"},
    )

    assert [item.provenance.stage for item in (pb, de, pathway)] == [
        "pseudobulk",
        "de",
        "pathway",
    ]
    for result in (pb, de, pathway):
        exported = result.export_provenance()
        assert exported["schema_version"] == "1"
        assert result.provenance_json() == result.provenance_json()
        assert result.parameters == exported["algorithm"]

    assert pb.provenance.inputs["cell_axis"]["ordering"] == "ordered"
    assert de.provenance.inputs["contrast_ids"]["ordering"] == "ordered"
    assert pathway.provenance.inputs["libraries"]["ordering"] == "ordered"


def test_stage_parent_lineage_is_explicit_and_serialized():
    root = PseudobulkResult(
        counts=pd.DataFrame([[1]], index=["cell-1"], columns=["gene-1"]),
        metadata=pd.DataFrame({"sample": ["sample-1"]}, index=["cell-1"]),
    )
    de = DEAnalysisResult(
        dds=None, contrast_results={"A_vs_B": pd.DataFrame()},
        parent_ids=(root.provenance.stable_id,),
    )
    pathway = PathwayEnrichmentResult(
        per_contrast={"A_vs_B": pd.DataFrame()}, libraries=["library.gmt"],
        parent_ids=(de.provenance.stable_id,),
    )

    assert de.provenance.parent_ids == (root.provenance.stable_id,)
    assert pathway.provenance.parent_ids == (de.provenance.stable_id,)
    assert pathway.export_provenance()["parent_ids"] == [de.provenance.stable_id]


def test_result_parameters_and_provenance_are_immutable():
    result = DEAnalysisResult(
        dds=None,
        contrast_results={"A": pd.DataFrame()},
        parameters={"alpha": 0.05},
    )

    with pytest.raises(TypeError):
        result.parameters["alpha"] = 0.1
    with pytest.raises(AttributeError):
        result.parameters = {"alpha": 0.1}
    with pytest.raises(AttributeError):
        result.provenance = None


def test_de_contrast_diagnostics_are_reconstructable_from_provenance():
    result = DEAnalysisResult(
        dds=None,
        contrast_results={"treated_vs_control": pd.DataFrame()},
        parameters={"alpha": 0.05},
        design={"formula": "~ 0 + condition", "columns": ["condition_control", "condition_treated"]},
        contrast_diagnostics={
            "treated_vs_control": {
                "contrast_id": "de-contrast:example",
                "vector": [-1.0, 1.0],
                "direction": "numerator_minus_denominator",
            }
        },
    )

    assert result.provenance.inputs["contrasts"]["treated_vs_control"]["vector"] == (-1.0, 1.0)
    assert result.provenance.inputs["contrasts"]["treated_vs_control"]["direction"] == "numerator_minus_denominator"


def test_result_rejects_two_different_configuration_sources():
    envelope = DEAnalysisResult(
        dds=None,
        contrast_results={"A": pd.DataFrame()},
        parameters={"alpha": 0.05},
    ).provenance

    with pytest.raises(ValueError, match="one canonical configuration"):
        DEAnalysisResult(
            dds=None,
            contrast_results={"A": pd.DataFrame()},
            parameters={"alpha": 0.1},
            provenance=envelope,
        )
