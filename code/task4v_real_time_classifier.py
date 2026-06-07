#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Task 4(v) — Encoding for Robotics: simple real-time emotion perception (dominant-AU Bayesian lookup)

This script is designed to work with YOUR Task2 outputs:
- Recursively loads all "*_task2_norm.csv" under --in_root
- Uses Task2 columns: p_AU??_r (probability-like AU distribution per frame)
- Infers emotion label from the first directory under --in_root
  (e.g., .../angry/xxx_task2_norm.csv)

Method:
1) Encode each frame as x_t = argmax_j p_{t,j}  (dominant AU state)
2) Learn P(x|e) offline from training clips with Laplace smoothing
3) Decode per frame: P(e|x_t) ∝ P(x_t|e) P(e)
4) Real-time temporal smoothing over a causal window W frames
5) Evaluate with clip-level 80/20 split (stratified by emotion)

Outputs (saved under --out_dir):
- task4v_codebook_P_x_given_e.csv
- fig_task4v_codebook_heatmap.png
- fig_task4v_topAUs_per_emotion.png
- fig_task4v_confusion_matrix.png
- task4v_metrics.txt

Example (Windows PowerShell):
python task4v_real_time_classifier.py `
  --in_root "C:/Users/Admin/Desktop/mlmi/robotics/robotic-cwD/code/task2_normalized_song" `
  --out_dir "C:/Users/Admin/Desktop/mlmi/robotics/robotic-cwD/code/task4v_results_song" `
  --test_ratio 0.2 --seed 0 --window 15 --tau 0.0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AU_P_COL_RE = re.compile(r"^p_AU\d+_r$")

EMOTIONS = ["angry", "fearful", "happy", "sad"]


def infer_emotion_from_path(path: Path, in_root: Path) -> str:
    """Infer emotion label from folder name directly under in_root."""
    try:
        rel = path.relative_to(in_root)
        if len(rel.parts) >= 2:
            emo = rel.parts[0].lower()
            return emo
    except Exception:
        pass
    return "unknown"


def list_task2_csvs(in_root: Path) -> List[Path]:
    return sorted([p for p in in_root.rglob("*.csv") if p.name.endswith("_task2_norm.csv")])


def collect_global_au_cols(csv_paths: List[Path]) -> List[str]:
    """Build a stable AU column list across files (sorted by name)."""
    cols_union = set()
    for p in csv_paths:
        try:
            df0 = pd.read_csv(p, nrows=1)
        except Exception:
            continue
        cols_union.update([c for c in df0.columns if AU_P_COL_RE.match(c)])
    return sorted(cols_union)


def load_clip_as_frames(
    csv_path: Path,
    in_root: Path,
    au_cols: List[str],
    tau: float,
) -> Optional[Dict[str, object]]:
    """
    Load a clip and return:
      - emotion (str)
      - clip_id (str)
      - X (np.ndarray of dominant AU indices, shape [T])
      - M (np.ndarray of max prob per frame, shape [T])
      - valid_mask (bool mask for frames passing gate)
    """
    emo = infer_emotion_from_path(csv_path, in_root)
    if emo not in EMOTIONS:
        return None

    df = pd.read_csv(csv_path)
    if not au_cols:
        return None

    # Align AU columns: missing -> 0, extra -> ignored
    for c in au_cols:
        if c not in df.columns:
            df[c] = 0.0
    P = df[au_cols].to_numpy(dtype=float)

    # Safety: replace non-finite, negative
    P = np.where(np.isfinite(P) & (P > 0.0), P, 0.0)

    row_sums = P.sum(axis=1, keepdims=True)
    # If row sum is 0, keep as all zeros; otherwise normalize
    P = np.where(row_sums > 0, P / row_sums, 0.0)

    M = P.max(axis=1)
    X = P.argmax(axis=1).astype(int)

    valid_mask = (M >= float(tau)) if float(tau) > 0 else np.ones_like(M, dtype=bool)

    clip_id = csv_path.name.replace("_task2_norm.csv", "")
    return {
        "emotion": emo,
        "clip_id": clip_id,
        "csv_path": str(csv_path),
        "X": X,
        "M": M,
        "valid_mask": valid_mask,
    }


def stratified_split_clips(
    clips: List[Dict[str, object]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Clip-level stratified split by emotion."""
    rng = np.random.default_rng(int(seed))
    train, test = [], []
    for emo in EMOTIONS:
        group = [c for c in clips if c["emotion"] == emo]
        if not group:
            continue
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n_test = int(np.round(float(test_ratio) * len(group)))
        n_test = min(max(n_test, 1), len(group) - 1) if len(group) >= 2 else 0
        test_idx = set(idx[:n_test].tolist())
        for i, c in enumerate(group):
            (test if i in test_idx else train).append(c)
    return train, test


def train_codebook_px_given_e(
    train_clips: List[Dict[str, object]],
    K: int,
    alpha: float = 1.0,
    prior: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Px_e: shape [E, K]  where Px_e[e, x] = P(x|e)
      Pe:   shape [E]     priors P(e)
    """
    E = len(EMOTIONS)
    counts = np.zeros((E, K), dtype=np.float64)

    for c in train_clips:
        e = EMOTIONS.index(c["emotion"])
        X = c["X"]
        mask = c["valid_mask"]
        Xv = X[mask]
        for x in Xv:
            if 0 <= int(x) < K:
                counts[e, int(x)] += 1.0

    # Laplace smoothing
    Px_e = (counts + float(alpha)) / (counts.sum(axis=1, keepdims=True) + float(alpha) * K)

    if prior == "empirical":
        total_frames = counts.sum()
        if total_frames > 0:
            Pe = counts.sum(axis=1) / total_frames
        else:
            Pe = np.ones(E, dtype=np.float64) / E
    else:
        Pe = np.ones(E, dtype=np.float64) / E

    return Px_e, Pe


def decode_posteriors_for_clip(
    X: np.ndarray,
    valid_mask: np.ndarray,
    Px_e: np.ndarray,
    Pe: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Real-time (causal) smoothing:
      raw_t(e) = P(e|x_t) ∝ P(x_t|e) P(e)
      smooth_t(e) = mean_{k=0..W-1} raw_{t-k}(e) over valid frames only
    Returns:
      post_smooth: shape [T, E]
    """
    T = int(X.shape[0])
    E, K = Px_e.shape

    raw = np.zeros((T, E), dtype=np.float64)
    for t in range(T):
        if not valid_mask[t]:
            continue
        x = int(X[t])
        if x < 0 or x >= K:
            continue
        p = Px_e[:, x] * Pe
        s = p.sum()
        if s > 0:
            raw[t, :] = p / s

    W = max(int(window), 1)
    post = np.zeros_like(raw)
    cumsum = np.cumsum(raw, axis=0)
    for t in range(T):
        t0 = max(0, t - W + 1)
        denom = (t - t0 + 1)
        if t0 == 0:
            post[t, :] = cumsum[t, :] / denom
        else:
            post[t, :] = (cumsum[t, :] - cumsum[t0 - 1, :]) / denom

    return post


def confusion_matrix_from_preds(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int,
) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < n_classes and 0 <= yp < n_classes:
            cm[yt, yp] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path) -> None:
    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(cm, aspect="equal")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (frame-level, test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_codebook_heatmap(Px_e: np.ndarray, au_cols: List[str], out_path: Path) -> None:
    max_over_e = Px_e.max(axis=0)
    order = np.argsort(-max_over_e)
    N = min(25, len(order))
    idx = order[:N]

    plt.figure(figsize=(12, 4.5))
    plt.imshow(Px_e[:, idx], aspect="auto")
    plt.colorbar()
    plt.yticks(np.arange(len(EMOTIONS)), EMOTIONS)
    plt.xticks(np.arange(N), [au_cols[i].replace("p_", "") for i in idx], rotation=60, ha="right")
    plt.title("Learned codebook P(x|e) (top AUs by max probability)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_aus_per_emotion(Px_e: np.ndarray, au_cols: List[str], out_path: Path, topk: int = 8) -> None:
    topk = int(topk)
    plt.figure(figsize=(12, 6))
    for ei, emo in enumerate(EMOTIONS):
        order = np.argsort(-Px_e[ei, :])[:topk]
        xs = np.arange(topk) + ei * (topk + 1)
        plt.bar(xs, Px_e[ei, order], label=emo)
        plt.xticks(xs, [au_cols[i].replace("p_", "") for i in order], rotation=60, ha="right", fontsize=8)
    plt.ylabel("P(x|e)")
    plt.title("Top dominant-AU states per emotion (from learned codebook)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def per_class_metrics(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for i, name in enumerate(EMOTIONS):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[name] = {"precision": prec, "recall": rec, "f1": f1}
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root containing emotion folders with *_task2_norm.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder for figures/tables")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Clip-level test fraction (default 0.2)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for split")
    ap.add_argument("--tau", type=float, default=0.0, help="Confidence gate on max p (0 disables)")
    ap.add_argument("--window", type=int, default=15, help="Causal smoothing window in frames")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha")
    ap.add_argument("--prior", choices=["uniform", "empirical"], default="uniform", help="Emotion prior P(e)")
    ap.add_argument("--topk", type=int, default=8, help="Top-K AUs per emotion for bar figure")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = list_task2_csvs(in_root)
    if not csvs:
        print(f"[WARN] No *_task2_norm.csv found under: {in_root}")
        return 0

    au_cols = collect_global_au_cols(csvs)
    if not au_cols:
        print("[ERROR] No p_AU*_r columns found in Task2 files.")
        return 1

    clips: List[Dict[str, object]] = []
    for p in csvs:
        c = load_clip_as_frames(p, in_root, au_cols, tau=float(args.tau))
        if c is not None:
            clips.append(c)

    if not clips:
        print("[ERROR] No valid clips found (emotion folders + AU columns).")
        return 1

    train_clips, test_clips = stratified_split_clips(clips, test_ratio=float(args.test_ratio), seed=int(args.seed))
    if not train_clips or not test_clips:
        print("[ERROR] Split produced empty train/test. Reduce test_ratio or check data.")
        return 1

    K = len(au_cols)
    Px_e, Pe = train_codebook_px_given_e(
        train_clips=train_clips,
        K=K,
        alpha=float(args.alpha),
        prior=str(args.prior),
    )

    # Save codebook table
    codebook_df = pd.DataFrame(Px_e, index=EMOTIONS, columns=au_cols)
    codebook_df.to_csv(out_dir / "task4v_codebook_P_x_given_e.csv", index=True)

    # Evaluate on test set (frame-level), valid frames only
    y_true: List[int] = []
    y_pred: List[int] = []

    for c in test_clips:
        yt = EMOTIONS.index(c["emotion"])
        X = c["X"]
        mask = c["valid_mask"]

        post = decode_posteriors_for_clip(X, mask, Px_e, Pe, window=int(args.window))
        pred = post.argmax(axis=1).astype(int)

        idx = np.where(mask)[0]
        for t in idx:
            y_true.append(yt)
            y_pred.append(int(pred[t]))

    cm = confusion_matrix_from_preds(y_true, y_pred, n_classes=len(EMOTIONS))
    acc = (np.trace(cm) / np.maximum(cm.sum(), 1)).item()

    # Plots
    plot_confusion_matrix(cm, EMOTIONS, out_dir / "fig_task4v_confusion_matrix.png")
    plot_codebook_heatmap(Px_e, au_cols, out_dir / "fig_task4v_codebook_heatmap.png")
    plot_top_aus_per_emotion(Px_e, au_cols, out_dir / "fig_task4v_topAUs_per_emotion.png", topk=int(args.topk))

    # Metrics txt
    m = per_class_metrics(cm)
    lines = []
    lines.append(f"Train clips: {len(train_clips)} | Test clips: {len(test_clips)}")
    lines.append(f"Frame-level test accuracy (valid frames only): {acc:.3f}")
    lines.append("")
    lines.append("Per-class metrics (from confusion matrix):")
    for emo in EMOTIONS:
        lines.append(
            f"{emo}: precision={m[emo]['precision']:.3f}, recall={m[emo]['recall']:.3f}, f1={m[emo]['f1']:.3f}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    lines.append(pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_string())

    (out_dir / "task4v_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Saved outputs to: {out_dir}")
    print(f"[OK] Accuracy: {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
