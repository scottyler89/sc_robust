import numpy as np
import pytest


h5py = pytest.importorskip("h5py")


def _write_toy_spearman_h5(path, *, feature_ids, rho, cpos=0.5, cneg=-0.5, schema_version="1"):
    import json

    with h5py.File(path, "w") as h5:
        h5.create_dataset("infile", data=rho.astype(np.float32))
        h5["infile"].attrs["Cpos"] = float(cpos)
        h5["infile"].attrs["Cneg"] = float(cneg)
        ids = np.array([s.encode("utf-8") for s in feature_ids], dtype="S")
        h5.create_dataset("ids/feature_ids_kept", data=ids)
        prov = {"schema_version": schema_version, "pathway_removal": {"pathway_source": "shipped_bank"}}
        h5.create_dataset("meta/provenance_json", data=json.dumps(prov).encode("utf-8"))


def test_read_spearman_h5_and_extract_edges(tmp_path):
    import sc_robust.gene_modules as gm

    feature_ids = ["G0", "G1", "G2", "G3"]
    rho = np.array(
        [
            [1.0, 0.7, 0.1, -0.8],
            [0.7, 1.0, 0.6, -0.2],
            [0.1, 0.6, 1.0, -0.6],
            [-0.8, -0.2, -0.6, 1.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "spearman.hdf5"
    _write_toy_spearman_h5(path, feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)

    art = gm.read_spearman_h5(path)
    assert art.n_features == 4
    assert art.cpos == 0.5
    assert art.cneg == -0.5
    assert art.schema_version == "1"
    assert art.provenance_sha256 is not None

    edges = gm.extract_thresholded_edges(path, chunk_rows=2, upper_triangle_only=True)
    # Expected threshold-passing edges in upper triangle:
    # (0,1)=0.7 pos, (0,3)=-0.8 neg, (1,2)=0.6 pos, (2,3)=-0.6 neg
    assert set(map(tuple, edges[["i", "j"]].to_numpy().tolist())) == {(0, 1), (0, 3), (1, 2), (2, 3)}
    assert set(edges["sign"].unique().tolist()) == {"pos", "neg"}


def test_run_leiden_gene_modules_smoke(tmp_path):
    import pandas as pd
    import sc_robust.gene_modules as gm

    # Simple 4-node graph with two tight positives: (0-1) and (2-3)
    pos_edges = pd.DataFrame({"i": [0, 2], "j": [1, 3], "rho": [0.9, 0.9], "sign": ["pos", "pos"]})
    labels = gm.run_leiden_gene_modules(pos_edges, n_genes=4, resolution=1.0)
    assert labels.shape == (4,)
    # Should produce 2 communities in most runs; allow degenerate but require same-pair together.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]


def test_run_gene_modules_from_scratch_dir_writes_outputs(tmp_path):
    import pandas as pd
    import sc_robust.gene_modules as gm

    feature_ids = ["G0", "G1", "G2", "G3"]
    rho = np.array(
        [
            [1.0, 0.7, 0.1, -0.8],
            [0.7, 1.0, 0.6, -0.2],
            [0.1, 0.6, 1.0, -0.6],
            [-0.8, -0.2, -0.6, 1.0],
        ],
        dtype=np.float32,
    )

    scratch = tmp_path / "scratch"
    (scratch / "train").mkdir(parents=True, exist_ok=True)
    (scratch / "val").mkdir(parents=True, exist_ok=True)
    _write_toy_spearman_h5(scratch / "train" / "spearman.hdf5", feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)
    _write_toy_spearman_h5(scratch / "val" / "spearman.hdf5", feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)

    out = gm.run_gene_modules_from_scratch_dir(
        scratch,
        split_mode="union",
        resolution=1.0,
        min_k_gene=1,
        max_k_gene=10,
        persist_edges=True,
    )
    assert out["gene_modules"].exists()
    assert out["module_stats"].exists()
    assert out["gene_edges_pos"].exists()
    assert out["gene_edges_neg"].exists()
    assert out["gene_module_antagonism"].exists()
    assert out["report_json"].exists()

    df = pd.read_csv(out["gene_modules"], sep="\t")
    assert set(df.columns) >= {"gene_id", "module_id", "degree_pos", "strength_pos"}
    assert df.shape[0] == 4

    # Stats JSON should include extracted provenance fields.
    import json

    payload = json.loads(out["module_stats"].read_text(encoding="utf-8"))
    assert "source" in payload
    assert "train" in payload["source"]
    assert "provenance_fields" in payload["source"]["train"]


def test_run_gene_modules_for_cohort_writes_manifest(tmp_path):
    import sc_robust.gene_modules as gm

    feature_ids = ["G0", "G1", "G2", "G3"]
    rho = np.array(
        [
            [1.0, 0.7, 0.1, -0.8],
            [0.7, 1.0, 0.6, -0.2],
            [0.1, 0.6, 1.0, -0.6],
            [-0.8, -0.2, -0.6, 1.0],
        ],
        dtype=np.float32,
    )

    scratch1 = tmp_path / "S1"
    (scratch1 / "train").mkdir(parents=True, exist_ok=True)
    (scratch1 / "val").mkdir(parents=True, exist_ok=True)
    _write_toy_spearman_h5(scratch1 / "train" / "spearman.hdf5", feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)
    _write_toy_spearman_h5(scratch1 / "val" / "spearman.hdf5", feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)

    scratch2 = tmp_path / "S2"
    (scratch2 / "train").mkdir(parents=True, exist_ok=True)
    (scratch2 / "val").mkdir(parents=True, exist_ok=True)
    _write_toy_spearman_h5(scratch2 / "train" / "spearman.hdf5", feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)
    _write_toy_spearman_h5(scratch2 / "val" / "spearman.hdf5", feature_ids=feature_ids, rho=rho, cpos=0.5, cneg=-0.5)

    out_dir = tmp_path / "out"
    manifest = gm.run_gene_modules_for_cohort(
        [scratch1, scratch2],
        out_dir=out_dir,
        split_mode="union",
        min_k_gene=1,
        max_k_gene=10,
        persist_edges=False,
    )
    assert set(manifest["sample"].tolist()) == {"S1", "S2"}
    assert (out_dir / "gene_modules_manifest.tsv.gz").exists()
    assert (out_dir / "gene_modules_manifest.schema.json").exists()


def test_run_replicated_gene_modules_for_cohort_support_annotation(tmp_path):
    from sc_robust.gene_modules import run_replicated_gene_modules_for_cohort
    import pandas as pd

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Minimal cohort manifest + per-sample gene_modules outputs.
    manifest = pd.DataFrame(
        [
            {"sample": "S1", "scratch_dir": "NA", "status": "ok"},
            {"sample": "S2", "scratch_dir": "NA", "status": "ok"},
        ]
    )
    manifest.to_csv(out_dir / "gene_modules_manifest.tsv.gz", sep="\t", index=False, compression="gzip")

    s1 = out_dir / "S1"
    s2 = out_dir / "S2"
    s1.mkdir()
    s2.mkdir()

    pd.DataFrame(
        {
            "gene_id": ["A", "B", "C", "D", "E"],
            "module_id": [0, 0, 0, 1, 1],
            "degree_pos": [0, 0, 0, 0, 0],
            "strength_pos": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(s1 / "gene_modules.tsv.gz", sep="\t", index=False, compression="gzip")

    pd.DataFrame(
        {
            "gene_id": ["A", "B", "F", "D", "E", "G"],
            "module_id": [0, 0, 0, 2, 2, 2],
            "degree_pos": [0, 0, 0, 0, 0, 0],
            "strength_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(s2 / "gene_modules.tsv.gz", sep="\t", index=False, compression="gzip")

    paths = run_replicated_gene_modules_for_cohort(out_dir, jaccard_min=0.2, resolution=1.0)
    rep = pd.read_csv(paths["replicated_modules"], sep="\t")

    # Genes A/B appear in both samples in the same overlapped module cluster -> support==2.
    ab = rep.set_index("gene_id").loc[["A", "B"], "support_n_samples"].tolist()
    assert ab == [2, 2]
    # Sample-unique genes are still present, annotated with support==1.
    assert int(rep.set_index("gene_id").loc["C", "support_n_samples"]) == 1
    assert int(rep.set_index("gene_id").loc["F", "support_n_samples"]) == 1


def test_run_replicated_module_antagonism_for_cohort(tmp_path):
    import pandas as pd
    from sc_robust.gene_modules import run_replicated_module_antagonism_for_cohort

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Cohort manifest.
    pd.DataFrame(
        [
            {"sample": "S1", "scratch_dir": "NA", "status": "ok"},
            {"sample": "S2", "scratch_dir": "NA", "status": "ok"},
        ]
    ).to_csv(out_dir / "gene_modules_manifest.tsv.gz", sep="\t", index=False, compression="gzip")

    # Replicated instances: map sample module IDs to replicated IDs.
    pd.DataFrame(
        [
            {"sample": "S1", "module_id": 0, "replicated_module_id": 10, "n_genes": 3},
            {"sample": "S1", "module_id": 1, "replicated_module_id": 11, "n_genes": 2},
            {"sample": "S2", "module_id": 0, "replicated_module_id": 10, "n_genes": 3},
            {"sample": "S2", "module_id": 2, "replicated_module_id": 11, "n_genes": 3},
        ]
    ).to_csv(out_dir / "replicated_module_instances.tsv.gz", sep="\t", index=False, compression="gzip")

    # Per-sample antagonism summaries.
    (out_dir / "S1").mkdir()
    (out_dir / "S2").mkdir()
    pd.DataFrame(
        [
            {"module_a": 0, "module_b": 1, "n_edges": 5, "mean_rho": -0.4, "edge_density": 0.2, "size_a": 3, "size_b": 2},
        ]
    ).to_csv(out_dir / "S1" / "gene_module_antagonism.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(
        [
            {"module_a": 0, "module_b": 2, "n_edges": 3, "mean_rho": -0.3, "edge_density": 0.1, "size_a": 3, "size_b": 3},
        ]
    ).to_csv(out_dir / "S2" / "gene_module_antagonism.tsv.gz", sep="\t", index=False, compression="gzip")

    out_path = run_replicated_module_antagonism_for_cohort(out_dir)
    df = pd.read_csv(out_path, sep="\t")
    assert df.shape[0] == 1
    row = df.iloc[0].to_dict()
    assert int(row["replicated_module_a"]) == 10
    assert int(row["replicated_module_b"]) == 11
    assert int(row["n_samples_support"]) == 2


def test_compute_cluster_gene_module_scores_and_meta_profiles_cosine():
    import numpy as np
    import pandas as pd
    from sc_robust.gene_modules import compute_cluster_gene_module_scores, compute_meta_cluster_module_profiles_cosine

    # 4 cells x 5 genes
    X = np.array(
        [
            [10, 10, 0, 0, 0],
            [9, 11, 0, 0, 0],
            [0, 0, 10, 10, 0],
            [0, 0, 9, 11, 0],
        ],
        dtype=np.float32,
    )
    gene_ids = ["A", "B", "C", "D", "E"]
    clusters = ["c0", "c0", "c1", "c1"]
    replicated_modules = pd.DataFrame(
        {
            "replicated_module_id": [0, 0, 1, 1],
            "gene_id": ["A", "B", "C", "D"],
            "support_n_samples": [2, 2, 1, 1],
        }
    )

    scores = compute_cluster_gene_module_scores(
        X,
        gene_ids=gene_ids,
        cluster_labels=clusters,
        replicated_modules=replicated_modules,
        min_support_n_samples=1,
    )
    # c0 should score higher on module 0; c1 higher on module 1.
    c0 = scores[scores["cluster"] == "c0"].set_index("replicated_module_id")["score"].to_dict()
    c1 = scores[scores["cluster"] == "c1"].set_index("replicated_module_id")["score"].to_dict()
    assert c0[0] > c0[1]
    assert c1[1] > c1[0]

    cluster_to_meta = {"c0": "m0", "c1": "m1"}
    prof = compute_meta_cluster_module_profiles_cosine(scores, cluster_to_meta=cluster_to_meta)
    # Each meta-cluster has 1 contributing cluster.
    assert set(prof["meta_cluster"].unique().tolist()) == {"m0", "m1"}
    assert set(prof["n_clusters"].unique().tolist()) == {1}
