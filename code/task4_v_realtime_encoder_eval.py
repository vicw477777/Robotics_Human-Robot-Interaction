import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def is_p_au_col(c: str) -> bool:
    c = c.strip()
    return c.startswith("p_AU") and c.endswith("_r")


def load_frames(task2_root: Path):
    emotions = sorted([p.name for p in task2_root.iterdir() if p.is_dir()])
    rows = []
    au_cols_ref = None

    # find shared p_AU*_r columns
    for emo in emotions:
        for csv_path in sorted((task2_root / emo).rglob("*.csv")):
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            au_cols = sorted([c for c in df.columns if is_p_au_col(c)])
            if not au_cols:
                continue
            au_cols_ref = au_cols if au_cols_ref is None else sorted(list(set(au_cols_ref).intersection(set(au_cols))))

    if au_cols_ref is None:
        raise ValueError(f"No Task2 normalized CSVs with p_AU*_r columns under {task2_root}")

    # load frames
    for emo in emotions:
        for csv_path in sorted((task2_root / emo).rglob("*.csv")):
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if not all(c in df.columns for c in au_cols_ref):
                continue
            P = df[au_cols_ref].to_numpy(dtype=float)
            mx = np.max(P, axis=1)
            x = np.argmax(P, axis=1).astype(int)
            # mark weak frames as -1 (unknown)
            x[mx < 0.05] = -1
            for xi in x.tolist():
                if xi >= 0:
                    rows.append((emo, int(xi)))

    X = np.array([r[1] for r in rows], dtype=int)
    y = np.array([r[0] for r in rows], dtype=object)
    return emotions, au_cols_ref, X, y


def train_nb(emotions, K, X, y, alpha=1.0):
    emo_to_i = {e:i for i,e in enumerate(emotions)}
    C = np.zeros((len(emotions), K), dtype=np.int64)
    for xi, ei in zip(X.tolist(), y.tolist()):
        C[emo_to_i[ei], xi] += 1

    # P(x|e) with Laplace smoothing
    Px_e = (C + alpha) / (C.sum(axis=1, keepdims=True) + alpha * K)

    # uniform prior
    Pe = np.ones(len(emotions), dtype=float) / len(emotions)
    return Px_e, Pe, emo_to_i


def predict(Px_e, Pe, emotions, X):
    # log-prob for stability
    logPe = np.log(Pe + 1e-12)
    logPx = np.log(Px_e + 1e-12)
    scores = logPe[None, :] + logPx[:, X].T   # (N, E)
    yhat_idx = np.argmax(scores, axis=1)
    yhat = np.array([emotions[i] for i in yhat_idx], dtype=object)
    return yhat


def confusion_matrix(emotions, y_true, y_pred):
    idx = {e:i for i,e in enumerate(emotions)}
    M = np.zeros((len(emotions), len(emotions)), dtype=int)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        M[idx[yt], idx[yp]] += 1
    return M


def plot_cm(emotions, M, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(len(emotions)))
    ax.set_yticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha="right")
    ax.set_yticklabels(emotions)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task2_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task2_root = Path(args.task2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emotions, au_cols, X, y = load_frames(task2_root)
    K = len(au_cols)

    # shuffle and split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_test = int(round(len(X) * args.test_frac))
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    Px_e, Pe, _ = train_nb(emotions, K, X_train, y_train, alpha=args.alpha)
    y_pred = predict(Px_e, Pe, emotions, X_test)

    M = confusion_matrix(emotions, y_test, y_pred)
    acc = float(np.trace(M) / np.sum(M))

    pd.DataFrame(M, index=emotions, columns=emotions).to_csv(out_dir / "task4v_confusion_matrix.csv")
    plot_cm(emotions, M, out_dir / "fig_task4v_confusion_matrix.png", f"Dominant-AU Naive Bayes (acc={acc:.3f})")

    print("[OK] accuracy =", acc)
    print("[OK] saved:", out_dir / "fig_task4v_confusion_matrix.png")


if __name__ == "__main__":
    main()
