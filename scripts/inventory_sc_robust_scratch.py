#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SplitInventory:
    split: str
    spearman_h5: Optional[str]
    exprs_h5: Optional[str]
    kept_features_manifest_json: Optional[str]
    feature_selection_tsv_gz: Optional[str]
    has_ids_feature_ids_kept: Optional[bool]
    has_meta_provenance_json: Optional[bool]
    has_attrs_cpos: Optional[bool]
    has_attrs_cneg: Optional[bool]


@dataclass(frozen=True)
class SampleInventory:
    sample: str
    scratch_dir: str
    status: str
    error: Optional[str]
    train: Optional[SplitInventory]
    val: Optional[SplitInventory]


def _maybe_path(root: Path, rel: str) -> Optional[Path]:
    path = root / rel
    return path if path.exists() else None


def _h5_flags(path: Optional[Path]) -> Tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool]]:
    if path is None:
        return None, None, None, None
    try:
        import h5py  # local import
        import numpy as np
    except Exception:
        # If h5py isn't available, treat as unknown rather than failing inventory.
        return None, None, None, None

    try:
        with h5py.File(path, "r") as h5:
            has_ids = "ids/feature_ids_kept" in h5
            has_prov = "meta/provenance_json" in h5
            if "infile" in h5:
                infile = h5["infile"]
                attrs = dict(infile.attrs)
                has_cpos = "Cpos" in attrs and not (isinstance(attrs.get("Cpos"), float) and np.isnan(attrs.get("Cpos")))
                has_cneg = "Cneg" in attrs and not (isinstance(attrs.get("Cneg"), float) and np.isnan(attrs.get("Cneg")))
            else:
                has_cpos = False
                has_cneg = False
            return bool(has_ids), bool(has_prov), bool(has_cpos), bool(has_cneg)
    except Exception:
        return None, None, None, None


def inventory_sample(sample_dir: Path) -> SampleInventory:
    sample = sample_dir.name
    try:
        train_dir = sample_dir / "train"
        val_dir = sample_dir / "val"

        def inv_split(split_dir: Path, split: str) -> SplitInventory:
            spearman_h5 = _maybe_path(split_dir, "spearman.hdf5")
            exprs_h5 = _maybe_path(split_dir, "exprs.hdf5")
            kept_manifest = _maybe_path(split_dir, "kept_features_manifest.json")
            feature_sel = _maybe_path(sample_dir, f"feature_selection_{split}.tsv.gz")
            has_ids, has_prov, has_cpos, has_cneg = _h5_flags(spearman_h5)
            return SplitInventory(
                split=split,
                spearman_h5=str(spearman_h5) if spearman_h5 is not None else None,
                exprs_h5=str(exprs_h5) if exprs_h5 is not None else None,
                kept_features_manifest_json=str(kept_manifest) if kept_manifest is not None else None,
                feature_selection_tsv_gz=str(feature_sel) if feature_sel is not None else None,
                has_ids_feature_ids_kept=has_ids,
                has_meta_provenance_json=has_prov,
                has_attrs_cpos=has_cpos,
                has_attrs_cneg=has_cneg,
            )

        train = inv_split(train_dir, "train") if train_dir.exists() else None
        val = inv_split(val_dir, "val") if val_dir.exists() else None

        ok = bool(train and train.spearman_h5 and val and val.spearman_h5)
        return SampleInventory(
            sample=sample,
            scratch_dir=str(sample_dir),
            status="ok" if ok else "missing",
            error=None,
            train=train,
            val=val,
        )
    except Exception as exc:
        return SampleInventory(
            sample=sample,
            scratch_dir=str(sample_dir),
            status="failed",
            error=str(exc),
            train=None,
            val=None,
        )


def to_rows(inv: List[SampleInventory]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in inv:
        base: Dict[str, Any] = {
            "sample": sample.sample,
            "scratch_dir": sample.scratch_dir,
            "status": sample.status,
            "error": sample.error,
        }
        for split_name in ("train", "val"):
            split: Optional[SplitInventory] = getattr(sample, split_name)
            prefix = f"{split_name}."
            if split is None:
                base.update(
                    {
                        prefix + "spearman_h5": None,
                        prefix + "exprs_h5": None,
                        prefix + "kept_features_manifest_json": None,
                        prefix + "feature_selection_tsv_gz": None,
                        prefix + "has_ids_feature_ids_kept": None,
                        prefix + "has_meta_provenance_json": None,
                        prefix + "has_attrs_cpos": None,
                        prefix + "has_attrs_cneg": None,
                    }
                )
            else:
                d = asdict(split)
                d.pop("split", None)
                base.update({prefix + k: v for k, v in d.items()})
        rows.append(base)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Inventory sc_robust scratch dirs (train/val spearman.hdf5 + metadata).")
    p.add_argument("root", type=Path, help="Directory containing per-sample scratch dirs.")
    p.add_argument("--out-json", type=Path, default=None, help="Write full JSON inventory to this path.")
    p.add_argument("--out-tsv", type=Path, default=None, help="Write flattened TSV inventory to this path.")
    args = p.parse_args()

    root = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    samples = sorted([p for p in root.iterdir() if p.is_dir()])
    inv = [inventory_sample(s) for s in samples]

    payload = {
        "root": str(root),
        "n_samples": int(len(inv)),
        "samples": [asdict(x) for x in inv],
    }
    rows = to_rows(inv)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.out_tsv is not None:
        import pandas as pd

        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_tsv, sep="\t", index=False)

    # Default: write TSV to stdout for quick copy/paste.
    if args.out_json is None and args.out_tsv is None:
        import pandas as pd

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 200)
        df = pd.DataFrame(rows)
        print(df.to_csv(sep="\t", index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

