import ast
import sys
from pathlib import Path


def _parse_requirements(path: Path) -> set[str]:
    reqs: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<"):
            if sep in line:
                line = line.split(sep, 1)[0].strip()
                break
        reqs.add(line)
    return reqs


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


def test_imported_dependencies_are_listed_in_requirements():
    repo_root = Path(__file__).resolve().parents[1]
    imports = _scan_top_level_imports(repo_root / "sc_robust")
    reqs = _parse_requirements(repo_root / "requirements.txt")

    # Some pip packages expose different import names.
    rename = {
        "sklearn": "scikit-learn",
        "faiss": "faiss-cpu",
    }
    imports_norm = {rename.get(m, m) for m in imports}

    # Known optional imports that should not be forced on all installs.
    optional = {
        "scanpy",  # used only by sc_robust/example.py
    }

    missing = sorted((imports_norm - reqs) - optional)
    assert missing == [], f"Missing dependencies in requirements.txt: {missing}"

