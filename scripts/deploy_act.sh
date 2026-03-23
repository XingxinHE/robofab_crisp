#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pixi passes an argument separator as a literal "--" for wrapped tasks.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Deploy a trained ACT policy on FR3 with CRISP.

Usage:
  pixi run deploy-act -- --model-path <.../pretrained_model> [extra deploy args]

Defaults:
  --repo-id local/fr3_dualcam_streamed_deploy
  --num-episodes 1
  --policy-config lerobot_policy
  --env-config <auto-selected by deployment preflight>
  --env-namespace "" (root topics)
  --recording-manager-type keyboard
  --fps 15

Examples:
  pixi run deploy-act -- --model-path outputs/train/2026-03-17/11-47-16_act/checkpoints/last/pretrained_model
  pixi run deploy-act -- --model-path outputs/train/2026-03-17/11-47-16_act/checkpoints/last --num-episodes 3
  pixi run deploy-act -- --model-path outputs/train/2026-03-17/11-47-16_act/checkpoints/last/pretrained_model --resume
EOF
  exit 0
fi

if [[ " $* " == *" --env-config "* ]]; then
  echo "[deploy-act] Do not pass --env-config. It is selected automatically by deployment preflight." >&2
  exit 1
fi

if [[ " $* " == *" --policy-config "* ]]; then
  echo "[deploy-act] Do not pass --policy-config. This wrapper pins sync lerobot_policy." >&2
  exit 1
fi

MODEL_PATH=""
REPO_ID="local/fr3_dualcam_streamed_deploy"
NUM_EPISODES="1"
WANTS_RESUME=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --num-episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --resume)
      WANTS_RESUME=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --no-resume)
      WANTS_RESUME=0
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(python - <<'PY'
from pathlib import Path

models = sorted(Path("outputs/train").glob("**/pretrained_model"), key=lambda p: p.stat().st_mtime, reverse=True)
print(models[0] if models else "")
PY
)"
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[deploy-act] Could not find a model automatically. Pass --model-path explicitly." >&2
  exit 1
fi

if [[ -d "${MODEL_PATH}/pretrained_model" ]]; then
  MODEL_PATH="${MODEL_PATH}/pretrained_model"
fi

if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/train_config.json" ]]; then
  echo "[deploy-act] Invalid model path: ${MODEL_PATH}" >&2
  echo "[deploy-act] Expected a pretrained_model directory containing config.json and train_config.json" >&2
  exit 1
fi

LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
DATASET_PATH="${LEROBOT_HOME}/${REPO_ID}"
if [[ -d "${DATASET_PATH}" && "${WANTS_RESUME}" -eq 0 ]]; then
  echo "[deploy-act] Dataset repo already exists: ${DATASET_PATH}" >&2
  echo "[deploy-act] Re-run with --resume to append episodes, or choose a new --repo-id." >&2
  exit 1
fi

if [[ "${WANTS_RESUME}" -eq 1 ]]; then
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[deploy-act] --resume was provided, but dataset repo does not exist: ${DATASET_PATH}" >&2
    echo "[deploy-act] Remove --resume for a fresh deployment dataset, or use an existing --repo-id." >&2
    exit 1
  fi

  if [[ ! -f "${DATASET_PATH}/meta/info.json" || ! -f "${DATASET_PATH}/meta/tasks.jsonl" || ! -f "${DATASET_PATH}/meta/episodes.jsonl" ]]; then
    echo "[deploy-act] Existing dataset repo appears incomplete/corrupted: ${DATASET_PATH}" >&2
    echo "[deploy-act] Missing one of: meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl" >&2
    echo "[deploy-act] Delete this repo directory and re-run without --resume, or pick a new --repo-id." >&2
    exit 1
  fi
fi

ENV_CONFIG="$(python "${SCRIPT_DIR}/deployment_preflight.py" --print-config-only)"
if [[ -z "${ENV_CONFIG}" ]]; then
  echo "[deploy-act] Failed to determine deployment env config from preflight." >&2
  exit 1
fi

echo "[deploy-act] Using model: ${MODEL_PATH}"
echo "[deploy-act] Using env config: ${ENV_CONFIG}"
echo "[deploy-act] Recording deployment episodes to repo: ${REPO_ID}"

exec python -m crisp_gym.scripts.deploy_policy \
  --repo-id "${REPO_ID}" \
  --num-episodes "${NUM_EPISODES}" \
  --fps 15 \
  --recording-manager-type keyboard \
  --path "${MODEL_PATH}" \
  --env-config "${ENV_CONFIG}" \
  --policy-config lerobot_policy \
  --env-namespace "" \
  --log-level INFO \
  "${EXTRA_ARGS[@]}"
