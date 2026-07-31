import ast
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the declared test extra.
    import tomli as tomllib
import re
import sys
from pathlib import Path




def _scan_top_level_imports(pkg_root: Path) -> set[str]:
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    imports: set[str] = set()
    for path in pkg_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is None or (node.level and node.level > 0):
                    continue
                imports.add(node.module.split(".")[0])
    return {m for m in imports if m and m not in stdlib and m != "sc_robust"}


def test_imported_dependencies_are_declared_in_pyproject():
    repo_root = Path(__file__).resolve().parents[1]
    imports = _scan_top_level_imports(repo_root / "sc_robust") | _scan_top_level_imports(repo_root / "pydeseq2")
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(project["project"]["dependencies"]) + list(project["project"]["optional-dependencies"]["full"])
    reqs = {re.split(r"[<>=~!]", value, maxsplit=1)[0].strip() for value in declared}

    # Some pip packages expose different import names.
    rename = {
        "sklearn": "scikit-learn",
        "faiss": "faiss-cpu",
        "formulaic_contrasts": "formulaic-contrasts",
    }
    imports_norm = {rename.get(m, m) for m in imports}

    # Known optional imports that should not be forced on all installs.
    optional = {
        "scanpy",  # used only by sc_robust/example.py
        "pydeseq2",  # distributed in-repo from the reviewed source commit
    }

    missing = sorted((imports_norm - reqs) - optional)
    assert missing == [], f"Missing dependencies in pyproject.toml: {missing}"


def test_full_extra_requires_corrected_count_split_release():
    repo_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "count_split>=1.0.1" in project["project"]["optional-dependencies"]["full"]
