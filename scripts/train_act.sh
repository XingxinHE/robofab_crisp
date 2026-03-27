#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pixi passes a literal "--" before forwarded args.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Train ACT on FR3 leader/follower datasets.

Defaults are tuned for the 3-camera dataset workflow:
  src dataset (for fix script): local/fr3_leader_follower_3cams_open
  train dataset:               local/fr3_leader_follower_3cams_open_fix_feat

Usage:
  pixi run train-act -- [options]

Options:
  --repo-id <id>                  Dataset repo id used for training.
  --prepare-from <src-id>         Clone+fix dataset metadata before training.
  --steps <n>                     Training steps (default 50000).
  --batch-size <n>                Batch size (default 8).
  --save-freq <n>                 Checkpoint save frequency (default 10000).
  --log-freq <n>                  Log frequency (default 100).
  --smoke                         Fast sanity run (2000 steps, save every 1000).
  -- <extra lerobot args>         Forward additional args to lerobot train.

Examples:
  pixi run train-act -- --prepare-from local/fr3_leader_follower_3cams_open --smoke
  pixi run train-act -- --repo-id local/fr3_leader_follower_3cams_open_fix_feat --steps 50000
EOF
  exit 0
fi

DATASET_REPO_ID="local/fr3_leader_follower_3cams_open_fix_feat"
PREPARE_FROM=""
STEPS="50000"
BATCH_SIZE="8"
SAVE_FREQ="10000"
LOG_FREQ="100"
SMOKE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      DATASET_REPO_ID="$2"
      shift 2
      ;;
    --prepare-from)
      PREPARE_FROM="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --save-freq)
      SAVE_FREQ="$2"
      shift 2
      ;;
    --log-freq)
      LOG_FREQ="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${PREPARE_FROM}" ]]; then
  python "${SCRIPT_DIR}/clone_dataset_fix_features.py" \
    --src-repo-id "${PREPARE_FROM}" \
    --dst-repo-id "${DATASET_REPO_ID}"
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  STEPS="2000"
  SAVE_FREQ="1000"
fi

echo "[train-act] dataset=${DATASET_REPO_ID} steps=${STEPS} batch=${BATCH_SIZE}"

exec python -m lerobot.scripts.train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --eval_freq=0 \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=pyav \
  "${EXTRA_ARGS[@]}"
