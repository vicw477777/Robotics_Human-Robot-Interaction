import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="Path to task3_summary.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for Task4(iii) figures/tables")
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    df.columns = [c.strip() for c in df.columns]

    emo_col = pick_col(df, ["emotion", "label", "category", "class"])
    bwH_col = pick_col(df, ["bandwidth_H", "bandwidth_H_Hz", "bw_H_Hz"])
    bwR_col = pick_col(df, ["bandwidth_R", "bandwidth_R_Hz", "bw_R_Hz"])

    if emo_col is None:
        raise KeyError(f"Cannot find emotion column. Columns={list(df.columns)}")
    if bwH_col is None or bwR_col is None:
        raise KeyError(f"Cannot find bandwidth columns. Columns={list(df.columns)}")

    # overall averages (scenario-level)
    overall = pd.DataFrame([{
        "mean_bandwidth_H_Hz": float(df[bwH_col].mean()),
        "std_bandwidth_H_Hz": float(df[bwH_col].std(ddof=1)),
        "mean_bandwidth_R_Hz": float(df[bwR_col].mean()),
        "std_bandwidth_R_Hz": float(df[bwR_col].std(ddof=1)),
        "N_clips": int(len(df)),
    }])
    overall.to_csv(out_dir / "task4iii_bandwidth_overall.csv", index=False)

    # per-emotion summary
    g = df.groupby(emo_col)
    per_emo = g.agg(
        mean_bandwidth_H_Hz=(bwH_col, "mean"),
        std_bandwidth_H_Hz=(bwH_col, lambda x: np.std(x, ddof=1)),
        mean_bandwidth_R_Hz=(bwR_col, "mean"),
        std_bandwidth_R_Hz=(bwR_col, lambda x: np.std(x, ddof=1)),
        N_clips=(bwH_col, "count"),
    ).reset_index().rename(columns={emo_col: "emotion"})
    per_emo.to_csv(out_dir / "task4iii_bandwidth_by_emotion.csv", index=False)

    # ---- Plot 1: mean bandwidth_H by emotion with std error bars
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_emo["emotion"], per_emo["mean_bandwidth_H_Hz"], yerr=per_emo["std_bandwidth_H_Hz"])
    ax.set_ylabel("Mean bandwidth of H(t) [Hz]")
    ax.set_title("Average bandwidth of H(t) by emotion (mean ± std)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_task4iii_mean_bandwidth_H_by_emotion.png", dpi=200)
    plt.close(fig)

    # ---- Plot 2: mean bandwidth_R by emotion with std error bars
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_emo["emotion"], per_emo["mean_bandwidth_R_Hz"], yerr=per_emo["std_bandwidth_R_Hz"])
    ax.set_ylabel("Mean bandwidth of R(t) [Hz]")
    ax.set_title("Average bandwidth of R(t) by emotion (mean ± std)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_task4iii_mean_bandwidth_R_by_emotion.png", dpi=200)
    plt.close(fig)

    print("[OK] Saved:")
    print(f"  - {out_dir / 'task4iii_bandwidth_overall.csv'}")
    print(f"  - {out_dir / 'task4iii_bandwidth_by_emotion.csv'}")
    print(f"  - {out_dir / 'fig_task4iii_mean_bandwidth_H_by_emotion.png'}")
    print(f"  - {out_dir / 'fig_task4iii_mean_bandwidth_R_by_emotion.png'}")


if __name__ == "__main__":
    main()
