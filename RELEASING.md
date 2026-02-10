# Releasing `sc_robust`

This repo uses lightweight SemVer tags for published releases.

## Versioning

- Version string lives in `sc_robust/_version.py` as `__version__ = "X.Y.Z"`.
- Git release tags should be `vX.Y.Z` (e.g., `v0.2.0`).

## Release checklist

1. Update version:
   - Edit `sc_robust/_version.py`
2. Run tests:
   - `python -m pytest -q`
3. Commit:
   - `git add sc_robust/_version.py`
   - `git commit -m "Release vX.Y.Z"`
4. Tag (annotated):
   - `git tag -a vX.Y.Z -m "sc_robust vX.Y.Z"`
5. Push:
   - `git push origin main`
   - `git push origin vX.Y.Z`

## Notes

- Keep tags immutable (never retag a released version); bump the patch version if you need a quick follow-up.
- If you publish to PyPI/Conda, build artifacts should be generated from the tagged commit.

