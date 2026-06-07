# task4_iv_entropy_plots.py
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_EMOTIONS = ["angry", "fearful", "happy", "sad"]

def find_task3_result_files(in_root: str, emotion: str):
    emo_dir = os.path.join(in_root, emotion)
    if not os.path.isdir(emo_dir):
        return []
    patterns = [
        os.path.join(emo_dir, "*_task3_results.csv"),
        os.path.join(emo_dir, "*task3_results.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(list(set(files)))
    return files

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if x.size else float("nan")

def plot_mean_std_bar(summary_df: pd.DataFrame, out_path: str):
    emotions = summary_df["emotion"].tolist()
    means = summary_df["mean_H_bits"].to_numpy(dtype=float)
    stds = summary_df["std_H_bits"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(emotions))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, emotions, rotation=30)
    plt.ylabel("Mean entropy  H̄  [bits]")
    plt.title("Mean information content by emotion (mean ± std across clips)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_boxplot_with_points(clip_df: pd.DataFrame, out_path: str, emotions):
    data = []
    for e in emotions:
        vals = clip_df.loc[clip_df["emotion"] == e, "clip_mean_H_bits"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        data.append(vals)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=emotions, showfliers=True)

    # jittered points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=vals.size)
        plt.scatter(np.full(vals.size, i) + jitter, vals, s=18)

    plt.ylabel("Clip mean entropy  H̄(i)  [bits]")
    plt.title("Distribution of clip-level mean information content by emotion")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="task3_results root, e.g., ...\\task3_results_song")
    ap.add_argument("--out_root", required=True, help="output folder for figures/csv")
    ap.add_argument("--emotions", nargs="*", default=DEFAULT_EMOTIONS, help="emotion subfolders to process")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    rows = []
    for emo in args.emotions:
        files = find_task3_result_files(args.in_root, emo)
        if not files:
            print(f"[WARN] No task3 result files found for emotion: {emo}")
            continue

        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception as ex:
                print(f"[WARN] Failed to read: {fp} ({ex})")
                continue

            if "H_t" not in df.columns:
                print(f"[WARN] Missing column H_t in: {fp}")
                continue

            clip_mean = safe_mean(df["H_t"].to_numpy())
            clip_id = os.path.basename(fp).replace("_task3_results.csv", "").replace("task3_results.csv", "")
            rows.append({
                "emotion": emo,
                "clip_id": clip_id,
                "file": fp,
                "clip_mean_H_bits": clip_mean,
            })

    clip_df = pd.DataFrame(rows)
    clip_csv = os.path.join(args.out_root, "task4iv_clip_mean_entropy.csv")
    clip_df.to_csv(clip_csv, index=False)

    if clip_df.empty:
        print("[ERROR] No valid clip entropy values computed. Check input paths and CSV columns.")
        return

    summary = (clip_df
               .groupby("emotion")["clip_mean_H_bits"]
               .agg(["mean", "std", "count"])
               .reset_index()
               .rename(columns={"mean": "mean_H_bits", "std": "std_H_bits", "count": "N_clips"}))

    # keep requested order where available
    emo_order = [e for e in args.emotions if e in summary["emotion"].tolist()]
    summary["emotion"] = pd.Categorical(summary["emotion"], categories=emo_order, ordered=True)
    summary = summary.sort_values("emotion")

    summary_csv = os.path.join(args.out_root, "task4iv_mean_entropy_by_emotion.csv")
    summary.to_csv(summary_csv, index=False)

    fig1 = os.path.join(args.out_root, "fig_task4iv_mean_entropy_by_emotion.png")
    plot_mean_std_bar(summary, fig1)

    fig2 = os.path.join(args.out_root, "fig_task4iv_entropy_boxplot_by_emotion.png")
    plot_boxplot_with_points(clip_df, fig2, emo_order)

    print("[OK] Wrote:")
    print("  ", clip_csv)
    print("  ", summary_csv)
    print("  ", fig1)
    print("  ", fig2)

if __name__ == "__main__":
    main()
