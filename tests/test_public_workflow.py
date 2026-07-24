import numpy as np
import pandas as pd

from sc_robust.de.base import DEAnalysisResult, PseudobulkResult
from sc_robust.de.workflow import perform_de_workflow


def test_synthetic_public_workflow_reconstructs_stage_provenance(monkeypatch):
    import sc_robust.de.workflow as workflow

    calls = {}
    counts = np.arange(12, dtype=float).reshape(4, 3) + 1
    metadata = pd.DataFrame(
        {"condition": ["control", "treated", "control", "treated"]},
        index=["cell-1", "cell-2", "cell-3", "cell-4"],
    )
    pseudobulk = PseudobulkResult(
        counts=pd.DataFrame(counts, index=["pb-1", "pb-2", "pb-3", "pb-4"], columns=["g1", "g2", "g3"]),
        metadata=metadata.rename(index={f"cell-{i}": f"pb-{i}" for i in range(1, 5)}),
        parameters={"mode": "synthetic"},
    )
    cluster_result = DEAnalysisResult(
        dds=None,
        contrast_results={"treated_vs_control": pd.DataFrame()},
        parameters={"mode": "synthetic"},
        design={"formula": "~ 0 + condition"},
        contrast_diagnostics={"treated_vs_control": {"direction": "numerator_minus_denominator"}},
    )

    def fake_build(*args, **kwargs):
        calls["pseudobulk"] = kwargs
        return pseudobulk

    def fake_prepare(*args, **kwargs):
        calls["prepare"] = kwargs
        return object()

    def fake_fit(dds):
        calls["fit"] = True
        return dds

    def fake_cluster(dds, **kwargs):
        calls["cluster"] = kwargs
        return cluster_result

    monkeypatch.setattr(workflow, "build_pseudobulk", fake_build)
    monkeypatch.setattr(workflow, "prepare_deseq_dataset", fake_prepare)
    monkeypatch.setattr(workflow, "fit_deseq_dataset", fake_fit)
    monkeypatch.setattr(workflow, "run_cluster_vs_all", fake_cluster)
    monkeypatch.setattr(workflow, "resolve_pathway_libraries", lambda *args, **kwargs: [])

    output = perform_de_workflow(
        graph=None,
        counts=counts,
        cluster_labels=["c1", "c1", "c2", "c2"],
        cell_metadata=metadata,
        design_columns=["condition"],
        pathway_libraries=None,
    )

    assert output["pseudobulk"] is pseudobulk
    assert output["cluster_vs_all_de"] is cluster_result
    assert output["pairwise_de"] is None
    assert output["cluster_vs_all_pathways"] is None
    assert calls["fit"] is True
    assert calls["pseudobulk"]["cluster_labels"] == ["c1", "c1", "c2", "c2"]
    assert output["pseudobulk"].provenance.stage == "pseudobulk"
    assert output["cluster_vs_all_de"].provenance.inputs["contrasts"]["treated_vs_control"]["direction"] == "numerator_minus_denominator"
