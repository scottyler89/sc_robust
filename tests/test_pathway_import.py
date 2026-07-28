import builtins

def test_pathway_import_does_not_eagerly_load_pipeline_dependencies(monkeypatch):
    blocked = {"anndata", "anticor_features", "count_split", "faiss", "leidenalg", "pymetis", "torch"}
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"pathway import eagerly loaded {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import sc_robust.de.pathways as pathways

    assert callable(pathways.run_pathway_enrichment)
