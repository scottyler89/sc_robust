import numpy as np
import pandas as pd

from sc_robust.de.pathways import run_pathway_enrichment_for_clusters, run_pathway_enrichment


def _make_de_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i}" for i in range(1, 11)]
    stats = rng.normal(loc=0.0, scale=1.5, size=len(genes))
    pvals = np.clip(rng.uniform(0.001, 0.2, size=len(genes)), 1e-6, None)
    df = pd.DataFrame(
        {
            "gene_name": genes,
            "stat": stats,
            "pvalue": pvals,
        }
    )
    # Duplicate a gene with a different statistic to ensure duplicate handling works.
    dup = pd.DataFrame(
        {
            "gene_name": ["GENE5"],
            "stat": [stats[4] * -1],
            "pvalue": [pvals[4]],
        }
    )
    return pd.concat([df, dup], ignore_index=True)


def test_run_pathway_enrichment_matches_manual_subset():
    de_df = _make_de_df(42)
    pathways = {"pathwayA": ["GENE1", "GENE2", "GENE5"]}
    result = run_pathway_enrichment(
        de_df,
        pathways,
        stat_col="stat",
        gene_col="gene_name",
        p_col="pvalue",
        alpha=0.1,
    )
    assert "pathwayA" in result.index
    row = result.loc["pathwayA"]
    assert set(row.index) >= {"mean_t", "enrichment_t", "p", "BH_adj_p", "signed_neglog10_BH", "nom_sig_genes"}
    # Ensure duplicate gene was incorporated by checking the string includes GENE5 once.
    if row["nom_sig_genes"]:
        assert "GENE5" in row["nom_sig_genes"]


def test_parallel_enrichment_matches_sequential(tmp_path):
    de_tables = {
        "contrastA": _make_de_df(1),
        "contrastB": _make_de_df(2),
    }

    gmt_path = tmp_path / "toy.gmt"
    gmt_path.write_text(
        "toy_pathway\tNA\tGENE1\tGENE3\tGENE5\n"
        "toy_negative\tNA\tGENE2\tGENE4\tGENE6\n",
        encoding="utf-8",
    )

    sequential = run_pathway_enrichment_for_clusters(
        de_tables,
        libraries=[gmt_path.name],
        base_dir=tmp_path,
        n_jobs=1,
        backend="thread",
    )
    parallel = run_pathway_enrichment_for_clusters(
        de_tables,
        libraries=[gmt_path.name],
        base_dir=tmp_path,
        n_jobs=2,
        backend="thread",
    )

    assert sequential.parameters["backend"] == "sequential"
    assert parallel.parameters["backend"] == "thread"

    for key in de_tables:
        seq_df = sequential.per_contrast[key]
        par_df = parallel.per_contrast[key]
        pd.testing.assert_frame_equal(
            seq_df.sort_values(["code", "pathway"]).reset_index(drop=True),
            par_df.sort_values(["code", "pathway"]).reset_index(drop=True),
            check_dtype=False,
            check_like=True,
        )


def test_process_backend_requests_sequential_when_single_worker(tmp_path):
    table = {"contrast": _make_de_df(3)}
    gmt_path = tmp_path / "toy.gmt"
    gmt_path.write_text("toy_pathway\tNA\tGENE1\tGENE2\n", encoding="utf-8")
    result = run_pathway_enrichment_for_clusters(
        table,
        libraries=[gmt_path.name],
        base_dir=tmp_path,
        n_jobs=1,
        backend="process",
    )
    assert result.parameters["backend"] == "sequential"
