import numpy as np
import pytest
from scipy.sparse import coo_matrix

from sc_robust.utils import perform_leiden_clustering


def _graph():
    matrix = np.array(
        [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]],
        dtype=float,
    )
    return coo_matrix(matrix)


def test_random_state_and_seed_alias_are_deterministic():
    first = perform_leiden_clustering(_graph(), resolution_parameter=0.5, random_state=13)[2]
    second = perform_leiden_clustering(_graph(), resolution_parameter=0.5, seed=13)[2]
    assert np.array_equal(first, second)


def test_conflicting_seed_aliases_fail():
    with pytest.raises(ValueError, match="conflict"):
        perform_leiden_clustering(_graph(), random_state=1, seed=2)
