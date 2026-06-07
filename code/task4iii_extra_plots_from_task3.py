#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task4(iii) extra plots (matches your Task3 outputs)

Reads:
  <task3_root>/task3_summary.csv

Generates:
  - fig_task4iii_boxplot_bandwidth_H.png
  - fig_task4iii_boxplot_bandwidth_R.png

Optional PSD plots (for one or multiple clips):
  - fig_task4iii_psd_H_<stem>.png
  - fig_task4iii_psd_R_<stem>.png

Expected per-clip file:
  <task3_root>/<emotion>/<stem>_task3_results.csv
Columns required inside per-clip file:
  - H_t
  - R_t_smooth (preferred) or R_t
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EMOTIONS = ["angry", "fearful", "happy", "sad"]


def _safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def half_power_psd(
    x: np.ndarray,
    fs: float = 30.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], Optional[float]]:
    """
    Normalized PSD on positive freqs (DC excluded) and half-power width.
    Returns: freqs, power_norm, bw, f_min, f_max
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return None, None, None, None, None

    x = x - float(np.mean(x))  # remove mean

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    power = (np.abs(X) ** 2) / n

    # positive freqs only, DC excluded
    m = freqs > 0
    freqs = freqs[m]
    power = power[m]
    if freqs.size < 1:
        return None, None, None, None, None

    total = float(np.sum(power))
    if (not np.isfinite(total)) or total <= 0:
        return freqs, power, None, None, None

    power = power / total  # normalize to unit area

    peak = float(np.max(power))
    if (not np.isfinite(peak)) or peak <= 0:
        return freqs, power, None, None, None

    thr = 0.5 * peak
    idx = np.where(power >= thr)[0]
    if idx.size == 0:
        return freqs, power, None, None, None

    df = float(freqs[1] - freqs[0]) if freqs.size >= 2 else float(fs) / float(n)

    f_min = float(freqs[int(idx[0])])
    f_max = float(freqs[int(idx[-1])])
    bw = df if idx.size == 1 else max(0.0, f_max - f_min)

    return freqs, power, float(bw), float(f_min), float(f_max)


def boxplot_by_emotion(
    df: pd.DataFrame,
    emotion_col: str,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    data = []
    labels = []
    for e in EMOTIONS:
        vals = df.loc[df[emotion_col] == e, value_col].astype(float)
        vals = vals[np.isfinite(vals)]
        data.append(vals.values)
        labels.append(e)

    plt.figure(figsize=(7.6, 3.8))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def find_clip_csv(task3_root: Path, emotion: str, stem: str) -> Optional[Path]:
    p = task3_root / emotion / f"{stem}_task3_results.csv"
    if p.exists():
        return p
    hits = list(task3_root.rglob(f"{stem}_task3_results.csv"))
    return hits[0] if hits else None


def plot_psd_with_half_power(
    x: np.ndarray,
    fs: float,
    title: str,
    out_path: Path,
) -> None:
    freqs, pnorm, bw, fmin, fmax = half_power_psd(x, fs=fs)
    if freqs is None or pnorm is None:
        print(f"[WARN] PSD not available (too short or invalid). Skip: {out_path.name}")
        return

    plt.figure(figsize=(7.2, 3.8))
    plt.plot(freqs, pnorm)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized power")
    plt.title(title)

    peak = np.nanmax(pnorm) if np.any(np.isfinite(pnorm)) else np.nan
    if np.isfinite(peak) and peak > 0:
        thr = 0.5 * peak
        plt.axhline(thr, linestyle="--")
        if bw is not None and fmin is not None and fmax is not None:
            plt.axvline(fmin, linestyle="--")
            plt.axvline(fmax, linestyle="--")
            plt.text(
                0.98, 0.95,
                f"Half-power BW = {bw:.3f} Hz",
                ha="right", va="top",
                transform=plt.gca().transAxes
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_psd_pairs(s: str) -> List[Tuple[str, str]]:
    """
    Format:
      "angry:01-...;happy:02-...;fearful:01-...;sad:01-..."
    Separator can be ';' or ','.
    """
    if not s:
        return []
    items = []
    for part in s.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --psd_pairs item: {part}. Expected emotion:stem.")
        emo, stem = part.split(":", 1)
        emo = emo.strip()
        stem = stem.strip()
        items.append((emo, stem))
    return items


def run_psd_for_clip(task3_root: Path, out_dir: Path, fs: float, emotion: str, stem: str, use_R_smooth: bool) -> None:
    clip_csv = find_clip_csv(task3_root, emotion, stem)
    if clip_csv is None:
        print(f"[WARN] Cannot locate clip results: emotion={emotion}, stem={stem}")
        return

    clip = pd.read_csv(clip_csv)
    clip.columns = [c.strip() for c in clip.columns]

    if "H_t" not in clip.columns:
        print(f"[WARN] Missing H_t in: {clip_csv.name}")
        return

    r_col = "R_t_smooth" if (use_R_smooth and "R_t_smooth" in clip.columns) else "R_t"
    if r_col not in clip.columns:
        print(f"[WARN] Missing {r_col} in: {clip_csv.name}")
        return

    H = clip["H_t"].to_numpy(dtype=float)
    R = clip[r_col].to_numpy(dtype=float)

    plot_psd_with_half_power(
        H, fs=fs,
        title=f"Normalized PSD of H(t) with half-power interval ({emotion}, {stem})",
        out_path=out_dir / f"fig_task4iii_psd_H_{stem}.png",
    )
    plot_psd_with_half_power(
        R, fs=fs,
        title=f"Normalized PSD of {r_col} with half-power interval ({emotion}, {stem})",
        out_path=out_dir / f"fig_task4iii_psd_R_{stem}.png",
    )
    print(f"[OK] PSD saved for {emotion}:{stem}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task3_root", required=True, help="Task3 output root, e.g. ...\\task3_results_song")
    ap.add_argument("--fs", type=float, default=30.0, help="Sampling rate (Hz), default 30")
    ap.add_argument("--use_R_smooth", action="store_true", help="Use R_t_smooth if available")

    # Single-clip PSD (backward compatible)
    ap.add_argument("--psd_emotion", type=str, default=None, help="Emotion for PSD plot, e.g. happy")
    ap.add_argument("--psd_stem", type=str, default=None, help="Stem for PSD plot, e.g. 01-02-05-01-01-01-01")

    # Multi-clip PSD
    ap.add_argument("--psd_pairs", type=str, default=None, help='Batch PSD, e.g. "angry:01-...;happy:02-...;fearful:01-...;sad:01-..."')

    args = ap.parse_args()

    task3_root = Path(args.task3_root).expanduser().resolve()
    summary_csv = task3_root / "task3_summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"Cannot find: {summary_csv}")

    df = pd.read_csv(summary_csv)
    df.columns = [c.strip() for c in df.columns]

    required = ["emotion", "bandwidth_H_Hz", "bandwidth_R_Hz"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}' in {summary_csv}. Columns={list(df.columns)}")

    df["bandwidth_H_Hz"] = df["bandwidth_H_Hz"].map(_safe_float)
    df["bandwidth_R_Hz"] = df["bandwidth_R_Hz"].map(_safe_float)

    out_dir = task3_root

    # Boxplots
    boxplot_by_emotion(
        df=df,
        emotion_col="emotion",
        value_col="bandwidth_H_Hz",
        ylabel="Bandwidth of H(t) [Hz]",
        title="Distribution of bandwidth of H(t) by emotion",
        out_path=out_dir / "fig_task4iii_boxplot_bandwidth_H.png",
    )
    boxplot_by_emotion(
        df=df,
        emotion_col="emotion",
        value_col="bandwidth_R_Hz",
        ylabel="Bandwidth of R(t) [Hz]",
        title="Distribution of bandwidth of R(t) by emotion",
        out_path=out_dir / "fig_task4iii_boxplot_bandwidth_R.png",
    )
    print(f"[OK] Saved boxplots under: {out_dir}")

    # PSD: batch
    if args.psd_pairs:
        pairs = parse_psd_pairs(args.psd_pairs)
        for emo, stem in pairs:
            run_psd_for_clip(task3_root, out_dir, float(args.fs), emo, stem, bool(args.use_R_smooth))

    # PSD: single (optional)
    if args.psd_emotion and args.psd_stem:
        run_psd_for_clip(task3_root, out_dir, float(args.fs), str(args.psd_emotion).strip(), str(args.psd_stem).strip(), bool(args.use_R_smooth))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
