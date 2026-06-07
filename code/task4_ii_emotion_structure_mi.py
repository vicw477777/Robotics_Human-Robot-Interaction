#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Task2 columns look like: p_AU01_r, p_AU12_r, ...
def is_p_au_col(c: str) -> bool:
    c = c.strip()
    return c.startswith("p_AU") and c.endswith("_r")


def load_emotion_counts(task2_root: Path) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """
    Expect directory structure like:
      task2_root/
        angry/*.csv  (Task2 normalized outputs, containing p_AUxx_r)
        happy/*.csv
        fearful/*.csv
        sad/*.csv

    Returns:
      counts_by_emotion: dict emotion -> counts over X (dominant AU index), shape [K]
      emotions: sorted emotion names
      au_cols: sorted p_AU*_r columns used (K = len(au_cols))
    """
    emotions = [p.name for p in task2_root.iterdir() if p.is_dir()]
    emotions = sorted(emotions)

    # First pass: find common AU probability columns across all CSVs (intersection)
    au_cols_ref = None
    for emo in emotions:
        emo_dir = task2_root / emo
        for csv_path in sorted(emo_dir.rglob("*.csv")):
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            au_cols = sorted([c for c in df.columns if is_p_au_col(c)])
            if not au_cols:
                continue
            if au_cols_ref is None:
                au_cols_ref = au_cols
            else:
                au_cols_ref = sorted(list(set(au_cols_ref).intersection(set(au_cols))))

    if au_cols_ref is None or len(au_cols_ref) == 0:
        raise ValueError(f"No Task2 normalized CSVs with p_AU*_r columns found under: {task2_root}")

    K = len(au_cols_ref)
    counts_by_emotion: Dict[str, np.ndarray] = {emo: np.zeros(K, dtype=np.int64) for emo in emotions}

    # Second pass: accumulate dominant-AU counts per emotion
    for emo in emotions:
        emo_dir = task2_root / emo
        for csv_path in sorted(emo_dir.rglob("*.csv")):
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if not all(c in df.columns for c in au_cols_ref):
                continue

            P = df[au_cols_ref].to_numpy(dtype=float)
            s = np.sum(P, axis=1)
            valid = s > 0

            if not np.any(valid):
                continue

            x = np.argmax(P[valid], axis=1).astype(int)  # 0..K-1
            # update counts
            for xi in x.tolist():
                counts_by_emotion[emo][xi] += 1

    return counts_by_emotion, emotions, au_cols_ref


def to_prob(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Convert counts -> probability with Laplace smoothing:
      p = (counts + alpha) / (sum(counts) + alpha*K)
    """
    counts = counts.astype(float)
    K = counts.size
    denom = float(np.sum(counts) + alpha * K)
    if denom <= 0:
        # fallback: uniform
        return np.ones(K, dtype=float) / K
    return (counts + alpha) / denom


def kl_div_bits(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(p||q) in bits. Assume p,q are valid distributions, and q>0 wherever p>0.
    """
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_div_bits(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence in bits:
      JSD(p,q) = 0.5 KL(p||m) + 0.5 KL(q||m), m = 0.5(p+q)
    In log2, JSD is bounded in [0, 1] for two distributions.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_div_bits(p, m) + 0.5 * kl_div_bits(q, m)


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
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing for AU-state distributions (default: 1.0).")
    args = ap.parse_args()

    task2_root = Path(args.task2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_by_emotion, emotions, au_cols = load_emotion_counts(task2_root)

    # Convert each emotion's counts -> probability distribution q_e(x)
    q_by_emotion: Dict[str, np.ndarray] = {
        emo: to_prob(counts_by_emotion[emo], alpha=args.alpha) for emo in emotions
    }

    n = len(emotions)
    jsd_mat = np.zeros((n, n), dtype=float)
    sim_mat = np.zeros((n, n), dtype=float)  # similarity = 1 - JSD (bounded [0,1])

    for i, j in itertools.product(range(n), range(n)):
        if i == j:
            jsd_mat[i, j] = 0.0
            sim_mat[i, j] = 1.0
            continue
        p = q_by_emotion[emotions[i]]
        q = q_by_emotion[emotions[j]]
        d = js_div_bits(p, q)
        jsd_mat[i, j] = d
        sim_mat[i, j] = 1.0 - d

    # Save matrices
    jsd_df = pd.DataFrame(jsd_mat, index=emotions, columns=emotions)
    sim_df = pd.DataFrame(sim_mat, index=emotions, columns=emotions)
    jsd_df.to_csv(out_dir / "task4ii_pairwise_JSD_bits.csv")
    sim_df.to_csv(out_dir / "task4ii_pairwise_similarity_1minusJSD.csv")

    # Plots
    plot_heatmap(jsd_mat, emotions, out_dir / "fig_task4ii_JSD_heatmap_bits.png",
                 "Pairwise emotion relationship via JSD(q_e1, q_e2) [bits] (higher = more distinguishable)")
    plot_heatmap(sim_mat, emotions, out_dir / "fig_task4ii_similarity_heatmap.png",
                 "Pairwise emotion similarity = 1 - JSD (higher = more similar)")

    # Bar plot for unordered pairs i<j
    pairs = []
    jsd_vals = []
    sim_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{emotions[i]} vs {emotions[j]}")
            jsd_vals.append(float(jsd_mat[i, j]))
            sim_vals.append(float(sim_mat[i, j]))

    plot_bar(pairs, jsd_vals, out_dir / "fig_task4ii_JSD_pairs_bar.png",
             "JSD [bits]", "Pairwise emotion distinguishability (higher = more distinguishable)")
    plot_bar(pairs, sim_vals, out_dir / "fig_task4ii_similarity_pairs_bar.png",
             "Similarity (1 - JSD)", "Pairwise emotion similarity (higher = more similar)")

    print("[OK] Saved:")
    print(f"  - {out_dir / 'task4ii_pairwise_JSD_bits.csv'}")
    print(f"  - {out_dir / 'task4ii_pairwise_similarity_1minusJSD.csv'}")
    print(f"  - {out_dir / 'fig_task4ii_JSD_heatmap_bits.png'}")
    print(f"  - {out_dir / 'fig_task4ii_similarity_heatmap.png'}")
    print(f"  - {out_dir / 'fig_task4ii_JSD_pairs_bar.png'}")
    print(f"  - {out_dir / 'fig_task4ii_similarity_pairs_bar.png'}")


if __name__ == "__main__":
    main()
