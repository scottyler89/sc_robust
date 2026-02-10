import numpy as np


def test_call_get_anti_cor_genes_filters_kwargs(monkeypatch):
    import sc_robust.sc_robust as sr

    captured = {}

    def fake_get_anti_cor_genes(exprs, feature_ids, species="hsapiens"):
        captured["species"] = species
        captured["exprs_shape"] = tuple(np.asarray(exprs).shape)
        captured["feature_ids_len"] = len(feature_ids)
        return {"selected": np.array([], dtype=bool)}

    monkeypatch.setattr(sr, "get_anti_cor_genes", fake_get_anti_cor_genes)

    exprs = np.zeros((3, 4), dtype=float)
    feature_ids = ["A", "B", "C", "D"]

    result = sr._call_get_anti_cor_genes(
        exprs,
        feature_ids,
        species="mmusculus",
        offline_mode=True,  # not in fake signature; should be dropped
        use_live_pathway_lookup=True,  # not in fake signature; should be dropped
        scratch_dir="scratch",  # not in fake signature; should be dropped
    )

    assert captured["species"] == "mmusculus"
    assert captured["exprs_shape"] == (3, 4)
    assert captured["feature_ids_len"] == 4
    assert "selected" in result


def test_anticor_failure_message_mentions_offline_hint():
    import sc_robust.sc_robust as sr

    exc = ConnectionError("g:Profiler connection timed out")
    msg = sr._anticor_features_failure_message(
        exc,
        species="hsapiens",
        split_label="train",
        anticor_options={
            "offline_mode": False,
            "use_live_pathway_lookup": True,
            "pre_remove_pathways": ["GO:0000001"],
        },
    )
    assert "environment without internet access" in msg or "without internet" in msg
    assert "use_live_pathway_lookup=False" in msg
