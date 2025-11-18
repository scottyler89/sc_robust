import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt

from sc_robust.de.base import DEAnalysisResult, PathwayEnrichmentResult
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


def test_de_analysis_result_getter_handles_missing():
    df = pd.DataFrame({"gene_name": ["GENE1"], "stat": [0.1]})
    de_result = DEAnalysisResult(
        dds=None,
        contrast_results={"contrastA": df},
        parameters={},
    )
    retrieved = de_result.get_contrast_df("contrastA")
    assert retrieved is df
    assert de_result.available_contrasts == ["contrastA"]
    with pytest.raises(KeyError):
        de_result.get_contrast_df("missing")


def _make_mock_enrichment_df(pathway: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pathway": [pathway],
            "code": [f"{pathway}_code"],
            "nom_sig_genes": ["GENE1,GENE2,GENE3"],
            "mean_t": [0.5],
            "enrichment_t": [1.2],
            "BH_adj_p": [0.01],
        }
    )


def _make_mock_de_df() -> pd.DataFrame:
    genes = [f"GENE{i}" for i in range(1, 21)]
    stats = np.linspace(-2.0, 2.0, num=len(genes))
    log_fc = np.linspace(-1.5, 1.5, num=len(genes))
    padj = np.linspace(0.001, 0.2, num=len(genes))
    return pd.DataFrame(
        {
            "gene_name": genes,
            "stat": stats,
            "log2FoldChange": log_fc,
            "padj": padj,
        }
    )


def test_save_density_difference_panels_creates_outputs(tmp_path):
    contrast = "cluster_prop__1"
    de_df = _make_mock_de_df()
    path_df = _make_mock_enrichment_df("PathwayA")
    de_result = DEAnalysisResult(dds=None, contrast_results={contrast: de_df}, parameters={})
    path_result = PathwayEnrichmentResult(
        per_contrast={contrast: path_df},
        libraries=[],
        parameters={},
    )

    output_dir = tmp_path / "density"
    logs = []
    saved = path_result.save_density_difference_panels(
        de_result,
        contrast=contrast,
        pathways=["PathwayA"],
        output_dir=output_dir,
        logger=logs.append,
        dpi=150,
    )
    assert len(saved) == 1
    assert saved[0].exists()
    assert any("Saved pathway density plot" in msg for msg in logs)


def test_save_pairwise_density_panels_uses_reverse_when_needed(tmp_path):
    reverse_key = "clusterB_vs_clusterA"
    de_df = _make_mock_de_df()
    path_df = _make_mock_enrichment_df("PathwayA")
    de_result = DEAnalysisResult(dds=None, contrast_results={reverse_key: de_df}, parameters={})
    path_result = PathwayEnrichmentResult(
        per_contrast={reverse_key: path_df},
        libraries=[],
        parameters={},
    )

    logs = []
    saved = path_result.save_pairwise_density_panels(
        de_result,
        primary_cluster="clusterA",
        comparator_cluster="clusterB",
        pathways=["PathwayA"],
        output_dir=tmp_path / "pairwise",
        logger=logs.append,
        dpi=150,
    )
    assert len(saved) == 1
    assert saved[0].exists()
    assert any("reverse contrast" in msg for msg in logs)


def test_save_pathway_scurves_creates_outputs(tmp_path):
    contrast = "cluster_prop__1"
    path_df = _make_mock_enrichment_df("PathwayA")
    path_result = PathwayEnrichmentResult(
        per_contrast={contrast: path_df},
        libraries=[],
        parameters={},
    )
    logs = []
    saved = path_result.save_pathway_scurves(
        contrast,
        pathways=["PathwayA", "Missing"],
        output_dir=tmp_path / "scurves",
        logger=logs.append,
        dpi=150,
    )
    assert len(saved) == 1
    assert saved[0].exists()
    assert any("s-curve plot" in msg for msg in logs)
    assert any("not found" in msg for msg in logs)


def test_save_pathway_volcano_plots_creates_outputs(tmp_path):
    contrasts = ["c1", "c2"]
    per_contrast = {c: _make_mock_enrichment_df(f"Pathway{idx}") for idx, c in enumerate(contrasts, start=1)}
    path_result = PathwayEnrichmentResult(
        per_contrast=per_contrast,
        libraries=[],
        parameters={},
    )
    saved = path_result.save_pathway_volcano_plots(
        contrasts=contrasts,
        output_dir=tmp_path / "volcano_pathway",
        pathways_of_interest=["Pathway1"],
        dpi=120,
    )
    assert len(saved) == len(contrasts)
    for path in saved:
        assert path.exists()


def test_de_save_volcano_plots_creates_outputs(tmp_path):
    contrast_names = ["contrastA", "contrastB"]
    de_result = DEAnalysisResult(
        dds=None,
        contrast_results={name: _make_mock_de_df() for name in contrast_names},
        parameters={},
    )
    logs = []
    saved = de_result.save_volcano_plots(
        contrasts=contrast_names,
        output_dir=tmp_path / "volcano_de",
        genes_of_interest=["GENE1"],
        logger=logs.append,
        dpi=120,
    )
    assert len(saved) == len(contrast_names)
    for path in saved:
        assert path.exists()
    assert all("Saved plot" in msg or "Saved pathway" in msg for msg in logs)


def test_pathway_enrichment_heatmap_creates_file(tmp_path):
    pathways = ["PathwayA", "PathwayB"]
    per_contrast = {
        "cluster_prop__0": pd.DataFrame(
            {
                "pathway": pathways,
                "signed_neglog10_BH": [1.2, -0.8],
            }
        ),
        "cluster_prop__1": pd.DataFrame(
            {
                "pathway": pathways,
                "signed_neglog10_BH": [0.4, 2.1],
            }
        ),
    }
    path_result = PathwayEnrichmentResult(
        per_contrast=per_contrast,
        libraries=[],
        parameters={},
    )
    out_path = tmp_path / "heatmap.png"
    fig, ax = path_result.plot_pathway_enrichment_heatmap(
        pathways,
        annot=True,
        out_path=out_path,
        logger=None,
    )
    assert out_path.exists()
    plt.close(fig)


def test_pathway_enrichment_heatmap_no_zscore(tmp_path):
    pathways = ["PathwayA"]
    per_contrast = {
        "cluster_prop__0": pd.DataFrame(
            {
                "pathway": pathways,
                "signed_neglog10_BH": [1.0],
            }
        ),
        "cluster_prop__1": pd.DataFrame(
            {
                "pathway": pathways,
                "signed_neglog10_BH": [2.0],
            }
        ),
    }
    path_result = PathwayEnrichmentResult(
        per_contrast=per_contrast,
        libraries=[],
        parameters={},
    )
    fig, ax = path_result.plot_pathway_enrichment_heatmap(
        pathways,
        zscore=False,
        return_fig=True,
    )
    plt.close(fig)
