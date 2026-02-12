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

    df = pd.read_csv(out["gene_modules"], sep="\t")
    assert set(df.columns) >= {"gene_id", "module_id", "degree_pos", "strength_pos"}
    assert df.shape[0] == 4
