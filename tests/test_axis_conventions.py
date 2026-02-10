import numpy as np
import pytest


def test_build_pseudobulk_errors_on_transposed_counts():
    sp = pytest.importorskip("scipy.sparse")
    from sc_robust.de.pseudobulk import build_pseudobulk

    graph = sp.coo_matrix((5, 5), dtype=float)
    # Wrong orientation: genes x cells (3 x 5) instead of cells x genes (5 x 3)
    counts = np.zeros((3, 5), dtype=float)

    with pytest.raises(ValueError) as excinfo:
        build_pseudobulk(graph, counts, mode="topology")

    msg = str(excinfo.value)
    assert "cells" in msg
    assert "genes" in msg
    assert "transpose" in msg.lower()


def test_prep_sample_pseudobulk_errors_on_transposed_counts():
    sp = pytest.importorskip("scipy.sparse")
    from sc_robust.process_de_test_split import prep_sample_pseudobulk

    graph = sp.coo_matrix((5, 5), dtype=float)
    counts = np.zeros((3, 5), dtype=float)

    with pytest.raises(ValueError) as excinfo:
        prep_sample_pseudobulk(graph, counts)

    msg = str(excinfo.value)
    assert "cells" in msg
    assert "genes" in msg
    assert "transpose" in msg.lower()

