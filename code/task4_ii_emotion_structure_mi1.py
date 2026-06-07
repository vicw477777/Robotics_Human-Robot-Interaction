#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Task2 columns look like: p_AU01_r, p_AU12_r, ...
def is_p_au_col(c: str) -> bool:
    c = c.strip()
    return c.startswith("p_AU") and c.endswith("_r")


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information_bits(e_vals: np.ndarray, x_vals: np.ndarray, n_e: int, n_x: int) -> float:
    """
    Discrete MI from joint counts:
      I(E;X) = sum_{e,x} p(e,x) log2( p(e,x)/(p(e)p(x)) )
    """
    assert e_vals.shape == x_vals.shape
    N = int(e_vals.size)
    if N == 0:
        return 0.0

    joint = np.zeros((n_e, n_x), dtype=np.int64)
    for e, x in zip(e_vals.tolist(), x_vals.tolist()):
        if 0 <= e < n_e and 0 <= x < n_x:
            joint[e, x] += 1

    pe = joint.sum(axis=1).astype(float) / N
    px = joint.sum(axis=0).astype(float) / N
    pex = joint.astype(float) / N

    I = 0.0
    for e in range(n_e):
        for x in range(n_x):
            if pex[e, x] > 0 and pe[e] > 0 and px[x] > 0:
                I += pex[e, x] * math.log2(pex[e, x] / (pe[e] * px[x]))
    return float(I)


def load_all_frames(task2_root: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Expect directory structure like:
      task2_root/
        angry/*.csv (Task2 normalized outputs, containing p_AUxx_r)
        happy/*.csv
        fearful/*.csv
        sad/*.csv
    Returns:
      df_all with columns: emotion, X (dominant AU index)
      emotions list
      au_cols list
    """
    emotions = [p.name for p in task2_root.iterdir() if p.is_dir()]
    emotions = sorted(emotions)
    rows = []
    au_cols_ref = None

    for emo in emotions:
        emo_dir = task2_root / emo
        csvs = sorted(emo_dir.rglob("*.csv"))
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            au_cols = [c for c in df.columns if is_p_au_col(c)]
            if not au_cols:
                continue
            au_cols = sorted(au_cols)
            if au_cols_ref is None:
                au_cols_ref = au_cols
            else:
                # keep intersection to be safe
                au_cols_ref = sorted(list(set(au_cols_ref).intersection(set(au_cols))))

    if au_cols_ref is None:
        raise ValueError(f"No Task2 normalized CSVs with p_AU*_r columns found under: {task2_root}")

    # second pass: actually load using shared columns
    for emo in emotions:
        emo_dir = task2_root / emo
        csvs = sorted(emo_dir.rglob("*.csv"))
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if not all(c in df.columns for c in au_cols_ref):
                continue
            P = df[au_cols_ref].to_numpy(dtype=float)

            # dominant AU index per frame; if all zeros -> -1 (ignored later)
            x = np.argmax(P, axis=1).astype(int)
            # ignore frames where sum==0 (no AU activation)
            s = np.sum(P, axis=1)
            x[s <= 0] = -1

            for xi in x.tolist():
                if xi >= 0:
                    rows.append({"emotion": emo, "X": int(xi)})

    df_all = pd.DataFrame(rows)
    return df_all, emotions, au_cols_ref


def plot_heatmap(matrix: np.ndarray, labels: List[str], out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_bar(pairs: List[str], values: List[float], out_png: Path, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(pairs)), values)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task2_root", required=True, help="Root of Task2 normalized outputs (contains emotion subfolders).")
    ap.add_argument("--out_dir", required=True, help="Output directory for Task4(ii) figures and CSV.")
    args = ap.parse_args()

    task2_root = Path(args.task2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, emotions, au_cols = load_all_frames(task2_root)

    # map emotions to indices
    emo_to_idx = {e: i for i, e in enumerate(emotions)}
    df_all["E"] = df_all["emotion"].map(emo_to_idx).astype(int)

    n_x = len(au_cols)

    # pairwise MI/NMI matrix
    n = len(emotions)
    mi_mat = np.zeros((n, n), dtype=float)
    nmi_mat = np.zeros((n, n), dtype=float)

    for i, j in itertools.product(range(n), range(n)):
        if i == j:
            mi_mat[i, j] = 0.0
            nmi_mat[i, j] = 0.0
            continue

        emo_i = emotions[i]
        emo_j = emotions[j]
        sub = df_all[(df_all["emotion"] == emo_i) | (df_all["emotion"] == emo_j)].copy()

        # binary label for the pair: 0 for emo_i, 1 for emo_j
        e_bin = (sub["emotion"] == emo_j).astype(int).to_numpy()
        x = sub["X"].astype(int).to_numpy()

        I = mutual_information_bits(e_bin, x, n_e=2, n_x=n_x)

        # H(E) for the pair (based on sample counts)
        counts_e = np.array([np.sum(e_bin == 0), np.sum(e_bin == 1)], dtype=np.int64)
        H_e = entropy_bits_from_counts(counts_e)

        mi_mat[i, j] = I
        nmi_mat[i, j] = (I / H_e) if H_e > 1e-12 else 0.0

    # save matrices
    mi_df = pd.DataFrame(mi_mat, index=emotions, columns=emotions)
    nmi_df = pd.DataFrame(nmi_mat, index=emotions, columns=emotions)
    mi_df.to_csv(out_dir / "task4ii_pairwise_MI_bits.csv")
    nmi_df.to_csv(out_dir / "task4ii_pairwise_NMI.csv")

    # plots
    plot_heatmap(nmi_mat, emotions, out_dir / "fig_task4ii_NMI_heatmap.png", "Pairwise NMI(E;X) using dominant AU state")
    plot_heatmap(mi_mat, emotions, out_dir / "fig_task4ii_MI_heatmap_bits.png", "Pairwise MI(E;X) [bits] using dominant AU state")

    # bar plot (all unordered pairs i<j)
    pairs = []
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{emotions[i]} vs {emotions[j]}")
            vals.append(float(nmi_mat[i, j]))

    plot_bar(pairs, vals, out_dir / "fig_task4ii_NMI_pairs_bar.png", "NMI(E;X)", "Pairwise emotion separability (higher = more distinguishable)")

    print("[OK] Saved:")
    print(f"  - {out_dir / 'task4ii_pairwise_MI_bits.csv'}")
    print(f"  - {out_dir / 'task4ii_pairwise_NMI.csv'}")
    print(f"  - {out_dir / 'fig_task4ii_NMI_heatmap.png'}")
    print(f"  - {out_dir / 'fig_task4ii_MI_heatmap_bits.png'}")
    print(f"  - {out_dir / 'fig_task4ii_NMI_pairs_bar.png'}")


if __name__ == "__main__":
    main()
