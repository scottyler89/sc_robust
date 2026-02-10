import numpy as np
import pandas as pd
import pytest


faiss = pytest.importorskip("faiss")
torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")


def test_robust_pipeline_null_reports_no_reproducible_pcs(tmp_path, monkeypatch):
    import sc_robust.sc_robust as sr

    def fake_get_anti_cor_genes(exprs, feature_ids, species="hsapiens", scratch_dir=None, **kwargs):
        # Select a stable subset of features to keep runtime low and deterministic.
        feature_ids = list(feature_ids)
        selected = [idx < 30 for idx in range(len(feature_ids))]
        return pd.DataFrame({"selected": selected}, index=feature_ids)

    monkeypatch.setattr(sr, "get_anti_cor_genes", fake_get_anti_cor_genes)

    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(120, 60)).astype(np.int64)
    X = sp.csr_matrix(X)
    adata = ad.AnnData(X=X)
    adata.var["gene_ids"] = [f"G{i}" for i in range(adata.n_vars)]

    ro = sr.robust(
        adata,
        gene_ids=adata.var["gene_ids"].tolist(),
        splits=[0.34, 0.33, 0.33],
        pc_max=30,
        scratch_dir=tmp_path,
        offline_mode=True,
        use_live_pathway_lookup=False,
        seed=123,
    )

    assert getattr(ro, "no_reproducible_pcs", False) is True
    assert ro.graph is None
    assert ro.provenance.get("status") == "no_reproducible_pcs"
    assert (tmp_path / "train" / "kept_features_manifest.json").exists()
    assert (tmp_path / "val" / "kept_features_manifest.json").exists()
    assert "adata" in ro.provenance
    assert "deps" in ro.provenance


def test_robust_pipeline_structured_builds_graph(tmp_path, monkeypatch):
    import sc_robust.sc_robust as sr

    def fake_get_anti_cor_genes(exprs, feature_ids, species="hsapiens", scratch_dir=None, **kwargs):
        # Keep all features for the structured test.
        feature_ids = list(feature_ids)
        return pd.DataFrame({"selected": [True] * len(feature_ids)}, index=feature_ids)

    monkeypatch.setattr(sr, "get_anti_cor_genes", fake_get_anti_cor_genes)

    rng = np.random.default_rng(1)
    n_cells = 120
    n_genes = 60
    labels = np.zeros(n_cells, dtype=int)
    labels[n_cells // 2 :] = 1

    means = np.ones((n_cells, n_genes), dtype=np.float32)
    means[labels == 0, :10] = 5.0
    means[labels == 1, 10:20] = 5.0

    X = rng.poisson(means).astype(np.int64)
    X = sp.csr_matrix(X)
    adata = ad.AnnData(X=X)
    adata.var["gene_ids"] = [f"G{i}" for i in range(n_genes)]

    ro = sr.robust(
        adata,
        gene_ids=adata.var["gene_ids"].tolist(),
        splits=[0.34, 0.33, 0.33],
        pc_max=20,
        scratch_dir=tmp_path,
        offline_mode=True,
        use_live_pathway_lookup=False,
        seed=123,
    )

    assert getattr(ro, "no_reproducible_pcs", False) is False
    assert ro.graph is not None
    assert ro.graph.shape[0] == n_cells
    assert ro.graph.shape[1] <= n_cells
    assert ro.graph.nnz > 0
    assert ro.provenance.get("graph", {}).get("k_used") == max(int(round(np.log(n_cells), 0)), 10)
