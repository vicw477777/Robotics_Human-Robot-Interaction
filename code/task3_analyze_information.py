#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


AU_P_COL_RE = re.compile(r"^p_AU\d+_r$")


def shannon_entropy_bits(p_row: np.ndarray) -> float:
    """Compute H = -sum p log2 p, with convention 0 log 0 = 0."""
    p = np.asarray(p_row, dtype=float)
    p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
    if p.sum() <= 0:
        return 0.0
    m = p > 0
    return float(-(p[m] * np.log2(p[m])).sum())


def mutual_information_approx(H: np.ndarray, delta: int) -> np.ndarray:
    """I_t = H_{t+delta} - H_t."""
    delta = int(delta)
    if delta < 1:
        delta = 1
    if H.size <= delta:
        return np.array([], dtype=float)
    return H[delta:] - H[:-delta]


def information_transfer_rate(
    H: np.ndarray,
    timestamps: Optional[np.ndarray],
    fps_if_no_ts: float = 30.0,
) -> np.ndarray:
    """R_t via forward difference (length n-1), Eq.(5)/(6)."""
    if H.size < 2:
        return np.array([], dtype=float)

    dH = H[1:] - H[:-1]

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float)[: H.size]
        dt = ts[1:] - ts[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(dt > 0, dH / dt, np.nan)
        return R.astype(float)

    return (float(fps_if_no_ts) * dH).astype(float)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average (simple), returns same length as x."""
    window = int(window)
    if window <= 1:
        return x.copy()
    if x.size == 0:
        return x.copy()

    x2 = x.copy()
    finite = np.isfinite(x2)
    x2[~finite] = 0.0

    w = np.ones(window, dtype=float)
    num = np.convolve(x2, w, mode="same")
    den = np.convolve(finite.astype(float), w, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(den > 0, num / den, np.nan)
    return y.astype(float)


def half_power_bandwidth_hz(x: np.ndarray, fs: float) -> Optional[float]:
    """
    Half-power width:
    - remove mean
    - FFT power spectrum (positive freqs only, DC excluded)
    - normalize to unit area
    - find bins where power >= 0.5 * peak
    - bandwidth = f_max - f_min over those bins
      if only one bin, return df (frequency resolution)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return None

    x = x - float(np.mean(x))

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    power = (np.abs(X) ** 2) / n

    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]
    if freqs.size < 1:
        return None

    total = float(np.sum(power))
    if (not np.isfinite(total)) or total <= 0.0:
        return None
    power = power / total

    peak = float(np.max(power))
    if (not np.isfinite(peak)) or peak <= 0.0:
        return None

    thr = 0.5 * peak
    idx = np.where(power >= thr)[0]
    if idx.size == 0:
        return None

    df = float(freqs[1] - freqs[0]) if freqs.size >= 2 else float(fs) / float(n)

    f_min = float(freqs[int(idx[0])])
    f_max = float(freqs[int(idx[-1])])

    if idx.size == 1:
        return df
    return max(0.0, f_max - f_min)


def infer_emotion_from_path(path: Path, in_root: Path) -> str:
    try:
        rel = path.relative_to(in_root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return "unknown"


def plot_bandwidth_bar(values: List[Tuple[str, Optional[float]]], title: str, out_path: Path) -> None:
    labels = [k for (k, _) in values]
    ys = [0.0 if v is None or (not np.isfinite(v)) else float(v) for (_, v) in values]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, ys)
    plt.ylabel("Bandwidth (Hz)")
    plt.xlabel("clip")
    plt.title(title)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root folder containing *_task2_norm.csv (recursively).")
    ap.add_argument("--out_root", required=True, help="Output root for Task-3 results.")
    ap.add_argument("--fps", type=float, default=30.0, help="Assumed fps when timestamps are not available.")
    ap.add_argument("--delta", type=int, default=1, help="Δ (in frames) for I_t = H(t+Δ) - H(t).")
    ap.add_argument("--smooth_R_window", type=int, default=0, help="Optional moving-average window for R(t) before bandwidth.")
    ap.add_argument("--plot_bandwidth", action="store_true", help="If set, save bandwidth bar plots.")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = sorted([p for p in in_root.rglob("*.csv") if p.name.endswith("_task2_norm.csv")])
    if not csvs:
        print(f"[WARN] No *_task2_norm.csv found under: {in_root}")
        return 0

    summary_rows: List[Dict[str, object]] = []
    bw_H_list: List[Tuple[str, Optional[float]]] = []
    bw_R_list: List[Tuple[str, Optional[float]]] = []

    for csv_path in csvs:
        stem = csv_path.name.replace("_task2_norm.csv", "")
        emotion = infer_emotion_from_path(csv_path, in_root)

        rel_dir = csv_path.parent.relative_to(in_root)
        out_clip_dir = out_root / rel_dir
        out_clip_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        p_cols = [c for c in df.columns if AU_P_COL_RE.match(c)]
        if not p_cols:
            print(f"[SKIP] No p_AU*_r columns in: {csv_path}")
            continue

        P = df[p_cols].to_numpy(dtype=float)
        H = np.array([shannon_entropy_bits(P[i, :]) for i in range(P.shape[0])], dtype=float)

        if "timestamp" in df.columns:
            time_s = df["timestamp"].to_numpy(dtype=float)
            timestamps = time_s
        else:
            time_s = np.arange(len(H), dtype=float) / float(args.fps)
            timestamps = None

        I = mutual_information_approx(H, args.delta)
        R = information_transfer_rate(H, timestamps, fps_if_no_ts=float(args.fps))

        if int(args.smooth_R_window) and int(args.smooth_R_window) > 1:
            R_smooth = moving_average(R, int(args.smooth_R_window))
        else:
            R_smooth = R.copy()

        out_df = pd.DataFrame({"time_s": time_s, "H_t": H})
        out_df["I_t"] = np.nan
        out_df.loc[: len(I) - 1, "I_t"] = I
        out_df["R_t"] = np.nan
        out_df.loc[: len(R) - 1, "R_t"] = R
        out_df["R_t_smooth"] = np.nan
        out_df.loc[: len(R_smooth) - 1, "R_t_smooth"] = R_smooth

        out_csv = out_clip_dir / f"{stem}_task3_results.csv"
        out_df.to_csv(out_csv, index=False)

        fs_assumed = 30.0
        bw_H = half_power_bandwidth_hz(H, fs_assumed)
        bw_R = half_power_bandwidth_hz(R_smooth, fs_assumed)

        summary_rows.append(
            {
                "emotion": emotion,
                "stem": stem,
                "n_frames": int(len(H)),
                "mean_H_bits": float(np.nanmean(H)) if len(H) else np.nan,
                "mean_R_bits_per_s": float(np.nanmean(R)) if len(R) else np.nan,
                "bandwidth_H_Hz": float(bw_H) if bw_H is not None else 0.0,
                "bandwidth_R_Hz": float(bw_R) if bw_R is not None else 0.0,
                "R_smooth_window": int(args.smooth_R_window),
                "fs_assumed": fs_assumed,
                "source_csv": str(csv_path),
            }
        )

        label = f"{emotion}:{stem}"
        bw_H_list.append((label, bw_H))
        bw_R_list.append((label, bw_R))

        print(
            f"[OK] {csv_path.name} -> {out_csv.name} | "
            f"BW_H={bw_H if bw_H is not None else 0.0} | "
            f"BW_R={bw_R if bw_R is not None else 0.0}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "task3_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nDone. Summary saved to: {summary_csv}")

    if args.plot_bandwidth and summary_rows:
        plot_bandwidth_bar(bw_R_list, "Bandwidth of R(t) (half-power width)", out_root / "bandwidth_R.png")
        plot_bandwidth_bar(bw_H_list, "Bandwidth of H(t) (half-power width)", out_root / "bandwidth_H.png")
        print(f"[OK] Plots saved to: {out_root}")

    manifest = {
        "in_root": str(in_root),
        "out_root": str(out_root),
        "fps": float(args.fps),
        "delta": int(args.delta),
        "smooth_R_window": int(args.smooth_R_window),
        "n_clips": int(len(summary_rows)),
        "summary_csv": str(summary_csv),
    }
    (out_root / "task3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
