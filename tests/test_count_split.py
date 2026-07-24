import numpy as np
import pytest
from scipy import sparse

from sc_robust.count_split_adapter import CountSplitValidationError, split_counts


def test_dense_and_sparse_split_conserve_exactly():
    counts = np.arange(24, dtype=int).reshape(6, 4)
    dense = split_counts(counts, [0.5, 0.5], seed=11)
    assert all(item.shape == counts.shape for item in dense)
    assert np.array_equal(dense[0] + dense[1], counts)
    sparse_out = split_counts(sparse.csr_matrix(counts), [0.5, 0.5], seed=11)
    assert all(sparse.issparse(item) for item in sparse_out)
    assert np.array_equal((sparse_out[0] + sparse_out[1]).toarray(), counts)


def test_invalid_counts_and_proportions_fail_before_delegation():
    with pytest.raises(CountSplitValidationError, match="integral"):
        split_counts(np.array([[1.5, 0.0]]), [1.0])
    with pytest.raises(CountSplitValidationError, match="finite"):
        split_counts(np.array([[np.nan, 0.0]]), [1.0])
    with pytest.raises(CountSplitValidationError, match="sum to one"):
        split_counts(np.ones((2, 2), dtype=int), [0.3, 0.3])


def test_same_seed_and_order_repeats():
    counts = np.arange(36, dtype=int).reshape(9, 4)
    first = split_counts(counts, [0.25, 0.75], seed=19)
    second = split_counts(counts, [0.25, 0.75], seed=19)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
