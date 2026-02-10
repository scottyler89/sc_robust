import numpy as np
import pandas as pd
import pytest


faiss = pytest.importorskip("faiss")
torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")


def test_robust_pipeline_smoke(tmp_path, monkeypatch):
    import sc_robust.sc_robust as sr

    def fake_get_anti_cor_genes(exprs, feature_ids, species="hsapiens", scratch_dir=None, **kwargs):
        # Select a stable subset of features to keep runtime low and deterministic.
        feature_ids = list(feature_ids)
        selected = [idx < 30 for idx in range(len(feature_ids))]
        return pd.DataFrame({"selected": selected}, index=feature_ids)

    monkeypatch.setattr(sr, "get_anti_cor_genes", fake_get_anti_cor_genes)

    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(40, 60)).astype(np.float32)
    X = sp.csr_matrix(X)
    adata = ad.AnnData(X=X)
    adata.var["gene_ids"] = [f"G{i}" for i in range(adata.n_vars)]

    ro = sr.robust(
        adata,
        gene_ids=adata.var["gene_ids"].tolist(),
        splits=[0.5, 0.5],
        pc_max=10,
        scratch_dir=tmp_path,
        offline_mode=True,
        use_live_pathway_lookup=False,
        seed=123,
    )

    assert ro.graph.shape == (adata.n_obs, adata.n_obs)
    assert ro.graph.nnz > 0
    assert (tmp_path / "train" / "kept_features_manifest.json").exists()
    assert (tmp_path / "val" / "kept_features_manifest.json").exists()
    assert "adata" in ro.provenance
    assert "deps" in ro.provenance

