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
    ap.add_argument("--out_dir", required=True, help="Output directory for Task4(iv) figures/tables")
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    df.columns = [c.strip() for c in df.columns]

    emo_col = pick_col(df, ["emotion", "label", "category", "class"])

    # Try to find mean entropy column produced in Task3 summary
    Hmean_col = pick_col(df, [
        "mean_H_bits", "mean_H", "H_mean", "avg_H_bits", "avg_H"
    ])

    if emo_col is None:
        raise KeyError(f"Cannot find emotion column. Columns={list(df.columns)}")
    if Hmean_col is None:
        raise KeyError(
            "Cannot find mean entropy column in task3_summary.csv.\n"
            f"Columns found: {list(df.columns)}\n"
            "Please check your task3_summary.csv for a column like mean_H_bits / mean_H."
        )

    # Per-emotion summary over clips
    g = df.groupby(emo_col)
    per_emo = g.agg(
        mean_H_bits=(Hmean_col, "mean"),
        std_H_bits=(Hmean_col, lambda x: np.std(x, ddof=1)),
        N_clips=(Hmean_col, "count"),
    ).reset_index().rename(columns={emo_col: "emotion"})

    per_emo.to_csv(out_dir / "task4iv_mean_entropy_by_emotion.csv", index=False)

    # Plot: mean ± std
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_emo["emotion"], per_emo["mean_H_bits"], yerr=per_emo["std_H_bits"])
    ax.set_ylabel("Mean entropy $\\overline{H}$ [bits]")
    ax.set_title("Mean information content by emotion (mean ± std across clips)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_task4iv_mean_entropy_by_emotion.png", dpi=200)
    plt.close(fig)

    print("[OK] Saved:")
    print(f"  - {out_dir / 'task4iv_mean_entropy_by_emotion.csv'}")
    print(f"  - {out_dir / 'fig_task4iv_mean_entropy_by_emotion.png'}")


if __name__ == "__main__":
    main()
