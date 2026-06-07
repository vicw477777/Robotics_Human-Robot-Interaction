import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AU_R_RE = re.compile(r"^AU\d+_r$")

# User-specified colors
EMO_COLOR = {
    "angry": 'red',
    "happy": "orange",
    "sad": "blue",
    "fearful": "green",
}


def get_au_r_cols(df: pd.DataFrame):
    cols = [c.strip() for c in df.columns]
    au_cols = [c for c in cols if AU_R_RE.match(c)]
    return sorted(au_cols)


def load_clip_mean(csv_path: Path, au_cols_ref=None):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    au_cols = get_au_r_cols(df)
    if not au_cols:
        return None, None

    if au_cols_ref is None:
        au_cols_ref = au_cols
    else:
        au_cols_ref = sorted(list(set(au_cols_ref).intersection(set(au_cols))))
        if not au_cols_ref:
            raise ValueError("No common AU*_r columns across files; inconsistent OpenFace outputs.")

    clip_mean = df[au_cols_ref].astype(float).mean(axis=0)
    return au_cols_ref, clip_mean


def compute_emotion_stats(in_root: Path, emotion: str):
    emo_dir = in_root / emotion
    if not emo_dir.exists():
        raise FileNotFoundError(f"Emotion folder not found: {emo_dir}")

    files = sorted(emo_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {emo_dir}")

    au_cols_ref = None
    per_clip = []

    for fp in files:
        au_cols_ref, clip_mean = load_clip_mean(fp, au_cols_ref)
        if clip_mean is None:
            continue
        per_clip.append(clip_mean)

    if not per_clip:
        raise ValueError(f"No usable OpenFace AU*_r columns found under: {emo_dir}")

    mat = pd.DataFrame(per_clip)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1)
    return mean.index.tolist(), mean, std, len(mat)


def plot_single_emotion(au_cols, mean_vec, emotion_key, out_png: Path, y_max: float):
    x = np.arange(len(au_cols))
    color = EMO_COLOR.get(emotion_key.lower(), "gray")

    fig, ax = plt.subplots(figsize=(14, 5.0))
    ax.bar(x, mean_vec.values, color=color)

    ax.set_title(f"Average AU distribution (all AU*_r) — {emotion_key.capitalize()}")
    ax.set_ylabel("Average AU intensity (AU*_r)")
    ax.set_xticks(x)
    ax.set_xticklabels(au_cols, rotation=60, ha="right")
    ax.set_ylim(0.0, y_max)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True,
                    help="OpenFace output root containing emotion folders with *.csv")
    ap.add_argument("--emotions", nargs="+", required=True,
                    help="Emotion folder names, e.g., angry fearful happy sad")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", default="fig_avg_AU_all")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    au_sets = []
    counts = {}

    for emo in args.emotions:
        au_cols, mean, std, n = compute_emotion_stats(in_root, emo)
        stats[emo] = {"mean": mean, "std": std, "au_cols": au_cols}
        au_sets.append(set(au_cols))
        counts[emo] = n

    # Common AU set across all emotions (ensures identical x-axis ordering)
    au_common = sorted(list(set.intersection(*au_sets)))
    if not au_common:
        raise ValueError("No common AU*_r columns across the specified emotions.")

    # Save summary table
    table = pd.DataFrame({"AU": au_common})
    for emo in args.emotions:
        table[f"{emo}_mean"] = stats[emo]["mean"][au_common].values
        table[f"{emo}_std"] = stats[emo]["std"][au_common].values
    out_csv = out_dir / f"{args.out_prefix}_summary.csv"
    table.to_csv(out_csv, index=False)

    # Shared y-axis max across all emotions
    all_means = np.concatenate([stats[emo]["mean"][au_common].values for emo in args.emotions])
    y_max = float(np.max(all_means))
    y_max = 1.05 * y_max if y_max > 0 else 1.0

    # Plot per emotion with identical axes and fixed colors
    for emo in args.emotions:
        out_png = out_dir / f"{args.out_prefix}_{emo}.png"
        plot_single_emotion(au_common, stats[emo]["mean"][au_common], emo, out_png, y_max)
        print("[OK] saved:", out_png)

    print("[OK] saved table:", out_csv)
    print("[OK] shared y-axis max:", y_max)
    print("[OK] clip counts:", ", ".join([f"{e} n={counts[e]}" for e in args.emotions]))


if __name__ == "__main__":
    main()
