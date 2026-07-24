import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import coo_matrix

from sc_robust.de.pseudobulk import build_pseudobulk


def _graph(n):
    rows = list(range(n - 1)) + list(range(1, n))
    cols = list(range(1, n)) + list(range(n - 1))
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_partition_by_joint_key_preserves_source_ids_and_boundary():
    n = 8
    metadata = pd.DataFrame(
        {"sample": ["s1"] * 4 + ["s2"] * 4, "cluster": ["c1", "c1", "c2", "c2"] * 2},
        index=[f"cell-{i}" for i in range(n)],
    )
    result = build_pseudobulk(
        _graph(n),
        np.arange(n * 3).reshape(n, 3) + 1,
        mode="topology",
        cells_per_pb=2,
        cell_metadata=metadata,
        partition_by=["sample", "cluster"],
        random_state=7,
    )
    assert set(result.metadata["boundary_key"]) == {"s1|c1", "s1|c2", "s2|c1", "s2|c2"}
    observed_sources = [cell for _, row in result.metadata.iterrows() for cell in row["source_cell_ids"]]
    assert sorted(observed_sources) == sorted(metadata.index.tolist())
    assert len(observed_sources) == len(set(observed_sources))
    for _, row in result.metadata.iterrows():
        assert set(row["source_cell_ids"]).issubset(set(metadata.index))
        assert row["source_cell_hash"]["ordering"] == "sorted"
        assert len(set(metadata.loc[row["source_cell_ids"], "sample"])) == 1
        assert len(set(metadata.loc[row["source_cell_ids"], "cluster"])) == 1


def test_partition_by_rejects_null_factor_and_can_drop_source_lists():
    metadata = pd.DataFrame({"sample": ["s1", None, "s1", "s1"]}, index=["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="null"):
        build_pseudobulk(
            _graph(4), np.ones((4, 2)), mode="topology", cell_metadata=metadata, partition_by="sample"
        )
    metadata.loc["b", "sample"] = "s1"
    result = build_pseudobulk(
        _graph(4), np.ones((4, 2)), mode="topology", cell_metadata=metadata,
        partition_by="sample", retain_source_cells=False, random_state=1,
    )
    assert "source_cells" not in result.metadata


def test_sparse_counts_and_integer_boundary_factor():
    metadata = pd.DataFrame({"batch": [1, 1, 2, 2]}, index=["a", "b", "c", "d"])
    result = build_pseudobulk(
        _graph(4), sparse.csr_matrix(np.ones((4, 2), dtype=int)),
        mode="topology", cells_per_pb=2, cell_metadata=metadata, partition_by="batch", random_state=2,
    )
    assert set(result.metadata["batch"]) == {"1", "2"}
    assert set(result.metadata["source_cell_ids"].explode()) == set(metadata.index)
