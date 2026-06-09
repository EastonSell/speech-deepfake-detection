"""Train the speech deepfake detector.

A self-contained PyTorch training loop (no PyTorch Lightning) so the project
runs cleanly on the Talapas cluster with only the dependencies in
``requirements.txt``.

Run it as a script::

    # quick sanity run on the four bundled example clips
    python -m pm.train_model --example --epochs 5

    # full run on ASVspoof 2021 LA eval (on Talapas, after extracting the data)
    python -m pm.train_model \
        --data-dir pm/dataset/ASVspoof2021_LA_eval/flac \
        --metadata pm/dataset/ASVspoof2021_LA_eval/trial_metadata.txt \
        --audio-ext .flac \
        --epochs 20 --batch-size 64 --balance \
        --out pm/model/weights/model.pt

The trained weights are written to ``--out`` (default
``pm/model/weights/model.pt``) as a plain ``state_dict`` that the evaluation
notebook loads with ``model.load_state_dict(torch.load(...))``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from pm.dataset.dataloader import ASVspoofAudioDataset, LABEL_TO_ID
from pm.model.model import AudioClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "pm" / "model" / "weights" / "model.pt"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_eer(scores, labels):
    """Equal Error Rate -- the standard ASVspoof metric.

    ``scores`` is P(spoof); ``labels`` is 1 for spoof (positive) else 0. The EER
    is the operating point where the false-alarm rate (bonafide called spoof)
    equals the false-reject rate (spoof called bonafide). We sweep every unique
    score as a threshold and return the point where |FAR - FRR| is smallest.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    n_pos = max(int(np.sum(labels == 1)), 1)
    n_neg = max(int(np.sum(labels == 0)), 1)

    thresholds = np.sort(np.unique(scores))
    best_eer, best_thr, best_gap = 1.0, float(thresholds[0]), float("inf")
    for thr in thresholds:
        pred_pos = scores >= thr
        far = np.sum(pred_pos & (labels == 0)) / n_neg   # false alarm rate
        frr = np.sum(~pred_pos & (labels == 1)) / n_pos  # false reject rate
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (far + frr) / 2.0
            best_thr = float(thr)
    return float(best_eer), best_thr


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, eer) over a loader. Spoof (label id 0) is positive."""
    model.eval()
    y_true, y_pred, y_score_spoof = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        prob_spoof = torch.softmax(logits, dim=1)[:, LABEL_TO_ID["spoof"]].cpu()
        y_pred += logits.argmax(1).cpu().tolist()
        y_score_spoof += prob_spoof.tolist()
        y_true += [int(v) for v in y]
    acc = float(np.mean(np.array(y_pred) == np.array(y_true))) if y_true else float("nan")
    spoof_indicator = [1 if t == LABEL_TO_ID["spoof"] else 0 for t in y_true]
    eer, _ = compute_eer(y_score_spoof, spoof_indicator)
    return acc, eer


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_dataset(args):
    if args.example:
        return ASVspoofAudioDataset(max_frames=args.max_frames)
    if not args.data_dir or not args.metadata:
        raise SystemExit(
            "Provide --data-dir and --metadata (or use --example for the bundled clips)."
        )
    return ASVspoofAudioDataset(
        protocol_path=args.metadata,
        audio_dir=args.data_dir,
        max_frames=args.max_frames,
        audio_ext=args.audio_ext,
    )


def class_weights(dataset, device):
    """Inverse-frequency weights to offset ASVspoof's heavy spoof/bonafide imbalance."""
    counts = np.zeros(len(LABEL_TO_ID), dtype=float)
    for rec in dataset.records:
        counts[LABEL_TO_ID[rec["label"]]] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = build_dataset(args)
    if args.limit and args.limit < len(dataset):
        dataset.records = dataset.records[: args.limit]  # deterministic smoke-test subset

    n_val = int(len(dataset) * args.val_frac)
    if args.val_frac > 0 and len(dataset) > 1:
        n_val = max(1, min(n_val, len(dataset) - 1))
    else:
        n_val = 0
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_ds, val_ds = dataset, dataset

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = AudioClassifier(num_classes=len(LABEL_TO_ID)).to(device)
    weight = class_weights(dataset, device) if args.balance else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"device={device} | train={n_train} val={n_val} | "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.lr}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(n_train, 1)
        val_acc, val_eer = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_acc {val_acc:.3f} | val_eer {val_eer:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved weights -> {out_path}")
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the speech deepfake detector.")
    p.add_argument("--example", action="store_true",
                   help="train on the four bundled example clips (sanity run)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="directory of audio files (e.g. .../ASVspoof2021_LA_eval/flac)")
    p.add_argument("--metadata", type=str, default=None,
                   help="ASVspoof trial_metadata.txt path")
    p.add_argument("--audio-ext", type=str, default=".flac",
                   help="audio file extension for the real data (default .flac)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-frames", type=int, default=96)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--balance", action="store_true",
                   help="use inverse-frequency class weights (recommended on full ASVspoof)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of utterances (quick smoke-test)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if unset)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                   help="output weights path (default pm/model/weights/model.pt)")
    return p.parse_args(argv)


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
