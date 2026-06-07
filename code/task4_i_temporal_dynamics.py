import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cv2  # optional
except Exception:
    cv2 = None


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(xpad, kernel, mode="valid")


def local_extrema_indices(x: np.ndarray):
    if len(x) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    xm1 = x[:-2]
    x0 = x[1:-1]
    xp1 = x[2:]
    max_idx = np.where((x0 > xm1) & (x0 > xp1))[0] + 1
    min_idx = np.where((x0 < xm1) & (x0 < xp1))[0] + 1
    return max_idx, min_idx


def detect_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_timeseries(csv_path: str, fps: float, delta: int):
    df = pd.read_csv(csv_path)

    # time axis
    tcol = detect_column(df, ["time_s", "timestamp", "t", "time", "sec", "seconds"])
    fcol = detect_column(df, ["frame", "frames", "frame_idx"])

    if tcol is None:
        if fcol is not None:
            t = df[fcol].astype(float).to_numpy() / float(fps)
        else:
            t = np.arange(len(df), dtype=float) / float(fps)
    else:
        t = df[tcol].astype(float).to_numpy()

    # H and R
    Hcol = detect_column(df, ["H_t", "H_bits", "H", "Ht", "entropy_bits", "entropy"])
    Rcol = detect_column(df, ["R_t_smooth", "R_t", "R_bits_per_s", "R", "Rt", "rate_bits_per_s", "dH_dt"])

    if Hcol is None:
        raise KeyError(f"Cannot find H column in {csv_path}. Columns={list(df.columns)}")

    H = df[Hcol].astype(float).to_numpy()

    if Rcol is not None:
        R = df[Rcol].astype(float).to_numpy()
    else:
        dt = float(delta) / float(fps)
        R = np.empty_like(H)
        if len(H) > delta:
            R[:-delta] = (H[delta:] - H[:-delta]) / dt
            R[-delta:] = R[-delta - 1]
        else:
            R[:] = 0.0

    return pd.DataFrame({"time_s": t, "H": H, "R": R})


def pick_peaks_and_dips(Rs: np.ndarray, topk: int, z_thresh: float):
    """
    先尝试：局部极值 + z阈值筛选
    如果筛完为空：退化为全局 topk 最大/最小（保证有输出）
    """
    n = len(Rs)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    max_idx, min_idx = local_extrema_indices(Rs)

    # z-score filtering
    mu = float(np.mean(Rs))
    sd = float(np.std(Rs) + 1e-12)

    def topk_from_candidates(cand_idx, largest: bool):
        if cand_idx.size == 0:
            return np.array([], dtype=int)
        z = (Rs[cand_idx] - mu) / sd
        if largest:
            keep = cand_idx[z >= z_thresh]
            if keep.size == 0:
                return np.array([], dtype=int)
            order = np.argsort(-Rs[keep])
        else:
            keep = cand_idx[z <= -z_thresh]
            if keep.size == 0:
                return np.array([], dtype=int)
            order = np.argsort(Rs[keep])
        return keep[order[:topk]]

    top_pos = topk_from_candidates(max_idx, largest=True)
    top_neg = topk_from_candidates(min_idx, largest=False)

    # fallback: global topk if empty
    if top_pos.size == 0:
        top_pos = np.argsort(-Rs)[: min(topk, n)]
    if top_neg.size == 0:
        top_neg = np.argsort(Rs)[: min(topk, n)]

    # 去重：避免同一个点既在pos又在neg（极端情况）
    top_pos_set = set(map(int, top_pos.tolist()))
    top_neg = np.array([i for i in top_neg.tolist() if int(i) not in top_pos_set], dtype=int)

    return top_pos, top_neg


def plot_ht_rt(clip_id: str, ts: pd.DataFrame, out_png: str,
               smooth_w: int, topk: int, z_thresh: float, events=None):
    t = ts["time_s"].to_numpy()
    H = ts["H"].to_numpy()
    R = ts["R"].to_numpy()

    Hs = moving_average(H, smooth_w)
    Rs = moving_average(R, smooth_w)

    top_pos, top_neg = pick_peaks_and_dips(Rs, topk=topk, z_thresh=z_thresh)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, Hs)
    axes[0].set_ylabel("H(t) [bits]")
    axes[0].set_title(f"H(t) and R(t) over time — {clip_id}")

    axes[1].plot(t, Rs)
    axes[1].set_ylabel("R(t) [bits/s]")
    axes[1].set_xlabel("time [s]")

    if top_pos.size > 0:
        axes[1].scatter(t[top_pos], Rs[top_pos], marker="o", label="top +peaks")
    if top_neg.size > 0:
        axes[1].scatter(t[top_neg], Rs[top_neg], marker="x", label="top -dips")
    axes[1].legend(loc="best")

    if events:
        for ev in events:
            et = float(ev["time_s"])
            label = ev.get("label", "")
            for ax in axes:
                ax.axvline(et)
            axes[0].text(et, np.max(Hs), label, rotation=90, va="top", ha="right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    peaks = [{"type": "peak", "time_s": float(t[i]), "R_value": float(Rs[i]), "H_value": float(Hs[i])}
             for i in top_pos.tolist()]
    dips = [{"type": "dip", "time_s": float(t[i]), "R_value": float(Rs[i]), "H_value": float(Hs[i])}
            for i in top_neg.tolist()]

    return peaks, dips


def extract_frames(video_path: str, times_s, out_dir: str, prefix: str):
    if cv2 is None:
        raise RuntimeError("opencv-python not available. Install it or extract frames manually.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for ts in times_s:
        frame_idx = int(round(float(ts) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(out_dir, f"{prefix}_{float(ts):.2f}.png")
        cv2.imwrite(out_path, frame)
        saved.append(out_path)

    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--smooth_w", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--z_thresh", type=float, default=1.0)
    ap.add_argument("--events_json", default=None)
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--extract_frames", action="store_true")
    args = ap.parse_args()

    ts = load_timeseries(args.csv_path, fps=args.fps, delta=args.delta)

    events = None
    if args.events_json:
        with open(args.events_json, "r", encoding="utf-8") as f:
            all_events = json.load(f)
        events = all_events.get(args.clip_id, None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = str(out_dir / f"fig_task4i_{args.clip_id}_HtRt.png")
    peaks, dips = plot_ht_rt(
        clip_id=args.clip_id,
        ts=ts,
        out_png=out_png,
        smooth_w=args.smooth_w,
        topk=args.topk,
        z_thresh=args.z_thresh,
        events=events,
    )

    out_table = out_dir / f"task4i_{args.clip_id}_extrema.csv"
    rows = peaks + dips
    if len(rows) == 0:
        # still write an empty CSV with the expected columns
        pd.DataFrame(columns=["type", "time_s", "R_value", "H_value"]).to_csv(out_table, index=False)
        print("[WARN] No extrema rows selected; wrote empty extrema CSV.")
    else:
        pd.DataFrame(rows).sort_values("time_s").to_csv(out_table, index=False)

    print(f"[OK] Saved plot: {out_png}")
    print(f"[OK] Saved extrema table: {out_table}")

    if args.extract_frames:
        if not args.video_path:
            raise ValueError("--video_path is required when --extract_frames is set")
        times = [x["time_s"] for x in rows]
        frames_dir = out_dir / "frames"
        saved = extract_frames(args.video_path, times, str(frames_dir), prefix=f"frame_{args.clip_id}")
        print(f"[OK] Extracted {len(saved)} frames into: {frames_dir}")


if __name__ == "__main__":
    main()
