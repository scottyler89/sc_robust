import numpy as np
import pytest


faiss = pytest.importorskip("faiss")
torch = pytest.importorskip("torch")


def _safe_imports():
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        return True
    except Exception:
        return False


def test_build_single_graph_cosine_and_l2():
    from sc_robust.utils import build_single_graph

    rng = np.random.default_rng(123)
    E = rng.normal(size=(60, 8)).astype(np.float32)

    for metric in ["cosine", "l2"]:
        G = build_single_graph(E, k=None, metric=metric, symmetrize='none')
        assert G.shape == (E.shape[0], E.shape[0])
        assert G.nnz > 0
        # weights should be finite and non-negative; self-edges not included
        assert np.isfinite(G.data).all()
        assert (G.data >= 0).all()


@pytest.mark.skipif(not _safe_imports(), reason="igraph/leidenalg not available")
def test_single_graph_and_leiden_runs():
    from sc_robust.utils import single_graph_and_leiden

    rng = np.random.default_rng(42)
    E = rng.normal(size=(50, 6)).astype(np.float32)

    G, labels = single_graph_and_leiden(E, k=None, metric='cosine', resolution=0.5)
    assert G.shape[0] == E.shape[0]
    assert len(labels) == E.shape[0]
    assert np.min(labels) >= 0


def test_symmetrization_options():
    from sc_robust.utils import build_single_graph
    rng = np.random.default_rng(7)
    E = rng.normal(size=(40, 5)).astype(np.float32)

    for sym_method in ["none", "max", "avg"]:
        G = build_single_graph(E, metric='cosine', symmetrize=sym_method)
        assert G.shape == (E.shape[0], E.shape[0])


def test_build_single_graph_shape_is_square_on_ties():
    """All-zero embeddings yield tied KNN scores; adjacency must still be n×n."""
    from sc_robust.utils import build_single_graph

    E = np.zeros((40, 4), dtype=np.float32)
    G = build_single_graph(E, metric="cosine", symmetrize="none")
    assert G.shape == (E.shape[0], E.shape[0])
