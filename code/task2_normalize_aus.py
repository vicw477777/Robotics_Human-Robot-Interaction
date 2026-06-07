from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

AU_R_PATTERN = re.compile(r"^AU\d{2}_r$")


def normalize_one_csv(
    csv_path: Path,
    out_path: Path,
    drop_failed: bool = False,
    conf_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    # OpenFace columns may have leading spaces
    df.columns = [c.strip() for c in df.columns]

    # NOTE: The coursework doesn't require filtering.
    # Keep these optional, OFF by default (strict compliance).
    if drop_failed and "success" in df.columns:
        df = df[df["success"] == 1].copy()

    if conf_threshold is not None and "confidence" in df.columns:
        df = df[df["confidence"] >= conf_threshold].copy()

    # Use AU intensity columns as raw a_{t,j}
    au_cols = sorted([c for c in df.columns if AU_R_PATTERN.match(c)])
    if not au_cols:
        raise ValueError(
            f"No AU intensity columns like 'AU01_r' found in: {csv_path}\n"
            f"Tip: open the CSV and check column names contain AUxx_r."
        )

    aus = df[au_cols].to_numpy(dtype=float)  # raw a_{t,j} (no clipping)

    # s_t = sum_j a_{t,j}
    s_t = np.sum(aus, axis=1)

    # p_{t,j} = a_{t,j}/s_t if s_t > 0 else 0
    p = np.zeros_like(aus, dtype=float)
    mask = s_t > 0
    p[mask] = aus[mask] / s_t[mask, None]

    # Output
    keep_cols = [c for c in ["frame", "timestamp", "success", "confidence"] if c in df.columns]
    out_df = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)

    out_df["s_t"] = s_t
    for j, col in enumerate(au_cols):
        out_df[f"p_{col}"] = p[:, j]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return {
        "input_csv": str(csv_path),
        "output_csv": str(out_path),
        "num_rows_out": int(len(out_df)),
        "num_aus_k": int(len(au_cols)),
        "au_cols": au_cols,
    }


def find_csvs(in_root: Path) -> List[Path]:
    return sorted([p for p in in_root.rglob("*.csv") if not p.name.endswith("_task2_norm.csv")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True, help="Root folder containing OpenFace CSVs (e.g., openface_out)")
    ap.add_argument("--out_root", type=str, required=True, help="Output folder for normalized CSVs")

    # Optional (OFF by default for strict compliance)
    ap.add_argument("--drop_failed", action="store_true", help="Optional: keep only success==1 frames")
    ap.add_argument("--conf_threshold", type=float, default=None, help="Optional: confidence threshold (e.g., 0.8)")

    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = find_csvs(in_root)
    if not csvs:
        raise SystemExit(f"No CSV files found under: {in_root}")

    manifest: List[Dict[str, Any]] = []
    for csv_path in csvs:
        # preserve folder structure under out_root
        rel = csv_path.relative_to(in_root)
        out_path = (out_root / rel).with_name(csv_path.stem + "_task2_norm.csv")

        info = normalize_one_csv(
            csv_path=csv_path,
            out_path=out_path,
            drop_failed=args.drop_failed,
            conf_threshold=args.conf_threshold,
        )
        print(f"[OK] {csv_path.name} -> {out_path.name} | k={info['num_aus_k']} | rows={info['num_rows_out']}")
        manifest.append(info)

    (out_root / "task2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Manifest saved to: {out_root / 'task2_manifest.json'}")


if __name__ == "__main__":
    main()
