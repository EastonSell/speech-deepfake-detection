#!/bin/bash
#SBATCH --job-name=spoof-train
#SBATCH --account=<PIRG>            # <-- your Talapas PIRG/account
#SBATCH --partition=gpu            # use a GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ---------------------------------------------------------------------------
# Train the speech deepfake detector on ASVspoof 2021 LA eval.
# Submit from the repo root on Talapas:  sbatch scripts/train_talapas.slurm
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

module load miniconda3 2>/dev/null || module load python 2>/dev/null || true
# source activate spoof    # <-- activate the env where you ran `pip install .`

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

DATA_DIR="pm/dataset/ASVspoof2021_LA_eval/flac"
METADATA="pm/dataset/ASVspoof2021_LA_eval/trial_metadata.txt"

python -m pm.train_model \
    --data-dir "$DATA_DIR" \
    --metadata "$METADATA" \
    --audio-ext .flac \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --balance \
    --num-workers 8 \
    --out pm/model/weights/model.pt

echo "Done. Weights at pm/model/weights/model.pt"
