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


def _circle_embedding(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

def test_build_single_graph_errors_on_too_few_samples():
    from sc_robust.utils import build_single_graph

    E = _circle_embedding(20)
    with pytest.raises(ValueError) as excinfo:
        build_single_graph(E, k=None, metric="cosine", symmetrize="none")
    msg = str(excinfo.value)
    assert "Too few samples" in msg
    assert "n_samples" in msg


def test_build_single_graph_cosine_and_l2():
    from sc_robust.utils import build_single_graph

    E = _circle_embedding(120)

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

    E = _circle_embedding(120)

    G, labels = single_graph_and_leiden(E, k=None, metric='cosine', resolution=0.5)
    assert G.shape[0] == E.shape[0]
    assert len(labels) == E.shape[0]
    assert np.min(labels) >= 0


def test_symmetrization_options():
    from sc_robust.utils import build_single_graph
    E = _circle_embedding(120)

    for sym_method in ["none", "max", "avg"]:
        G = build_single_graph(E, metric='cosine', symmetrize=sym_method)
        assert G.shape == (E.shape[0], E.shape[0])
