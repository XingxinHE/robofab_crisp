#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME=""
DEFAULT_REPO_ID=""
PREFLIGHT_MODULE=""
ENV_NAMESPACE=""
USE_NAMESPACE_ARG="0"
FORBID_ENV_NAMESPACE_ARG="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile-name)
      PROFILE_NAME="$2"
      shift 2
      ;;
    --default-repo-id)
      DEFAULT_REPO_ID="$2"
      shift 2
      ;;
    --preflight-module)
      PREFLIGHT_MODULE="$2"
      shift 2
      ;;
    --env-namespace)
      ENV_NAMESPACE="$2"
      shift 2
      ;;
    --use-namespace-arg)
      USE_NAMESPACE_ARG="$2"
      shift 2
      ;;
    --forbid-env-namespace-arg)
      FORBID_ENV_NAMESPACE_ARG="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${PROFILE_NAME}" || -z "${DEFAULT_REPO_ID}" || -z "${PREFLIGHT_MODULE}" ]]; then
  echo "[deploy-dp-profiled] Missing required profile parameters." >&2
  exit 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
Deploy classic Diffusion Policy with profile '${PROFILE_NAME}'.

Usage:
  pixi run ${PROFILE_NAME} -- --model-path <.../pretrained_model> [extra args]

Defaults:
  --repo-id ${DEFAULT_REPO_ID}
  --num-episodes 1
  --env-config <auto-selected by ${PREFLIGHT_MODULE}>
  --env-namespace ${ENV_NAMESPACE:-<root>}
  --recording-manager-type keyboard
  --fps 15
  --task "deploy diffusion policy"
  --home-config <name-or-path>        Optional robot YAML or homes/*.yaml for deployment homing
  --after-teleop <name-or-path>       Optional final home after all deployment episodes
  --num-inference-steps <n>           Optional inference-time reverse diffusion step override
EOF
  exit 0
fi

if [[ " $* " == *" --env-config "* ]]; then
  echo "[${PROFILE_NAME}] Do not pass --env-config. It is selected automatically." >&2
  exit 1
fi

if [[ " $* " == *" --policy-config "* ]]; then
  echo "[${PROFILE_NAME}] Do not pass --policy-config. DP deployment is pinned by this wrapper." >&2
  exit 1
fi

if [[ "${FORBID_ENV_NAMESPACE_ARG}" == "1" && " $* " == *" --env-namespace "* ]]; then
  echo "[${PROFILE_NAME}] Do not pass --env-namespace. This wrapper pins profile namespace." >&2
  exit 1
fi

MODEL_PATH=""
REPO_ID="${DEFAULT_REPO_ID}"
NUM_EPISODES="1"
WANTS_RESUME=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path|--path)
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
import json
from pathlib import Path


def policy_type(path: Path) -> str | None:
    try:
        config = json.loads((path / "config.json").read_text())
    except Exception:
        config = {}
    try:
        train_config = json.loads((path / "train_config.json").read_text())
    except Exception:
        train_config = {}
    return config.get("type") or (train_config.get("policy") or {}).get("type")


models = [
    path
    for path in Path("outputs/train").glob("**/pretrained_model")
    if policy_type(path) == "diffusion"
]
models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(models[0] if models else "")
PY
)"
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[${PROFILE_NAME}] Could not find a diffusion model automatically. Pass --model-path explicitly." >&2
  exit 1
fi

if [[ -d "${MODEL_PATH}/pretrained_model" ]]; then
  MODEL_PATH="${MODEL_PATH}/pretrained_model"
fi

python - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
config_path = model_path / "config.json"
train_config_path = model_path / "train_config.json"
if not config_path.is_file() or not train_config_path.is_file():
    raise SystemExit(
        f"Invalid model path: {model_path}\n"
        "Expected a pretrained_model directory containing config.json and train_config.json"
    )

config = json.loads(config_path.read_text())
train_config = json.loads(train_config_path.read_text())
policy_type = config.get("type") or (train_config.get("policy") or {}).get("type")
if policy_type != "diffusion":
    raise SystemExit(f"Expected a diffusion checkpoint, got policy type {policy_type!r}: {model_path}")
PY

LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
DATASET_PATH="${LEROBOT_HOME}/${REPO_ID}"
if [[ -d "${DATASET_PATH}" && "${WANTS_RESUME}" -eq 0 ]]; then
  echo "[${PROFILE_NAME}] Dataset repo already exists: ${DATASET_PATH}" >&2
  echo "[${PROFILE_NAME}] Re-run with --resume to append episodes, or choose a new --repo-id." >&2
  exit 1
fi

if [[ "${WANTS_RESUME}" -eq 1 ]]; then
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[${PROFILE_NAME}] --resume was provided, but dataset repo does not exist: ${DATASET_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${DATASET_PATH}/meta/info.json" || ! -f "${DATASET_PATH}/meta/tasks.jsonl" || ! -f "${DATASET_PATH}/meta/episodes.jsonl" ]]; then
    echo "[${PROFILE_NAME}] Existing dataset repo appears incomplete/corrupted: ${DATASET_PATH}" >&2
    echo "[${PROFILE_NAME}] Missing one of: meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl" >&2
    exit 1
  fi
fi

if [[ "${USE_NAMESPACE_ARG}" == "1" ]]; then
  ENV_CONFIG="$(python -m "${PREFLIGHT_MODULE}" --namespace "${ENV_NAMESPACE}" --print-config-only)"
else
  ENV_CONFIG="$(python -m "${PREFLIGHT_MODULE}" --print-config-only)"
fi

if [[ -z "${ENV_CONFIG}" ]]; then
  echo "[${PROFILE_NAME}] Failed to determine env config from preflight." >&2
  exit 1
fi

echo "[${PROFILE_NAME}] Using diffusion model: ${MODEL_PATH}"
echo "[${PROFILE_NAME}] Using env config: ${ENV_CONFIG}"
echo "[${PROFILE_NAME}] Using env namespace: ${ENV_NAMESPACE:-<root>}"
echo "[${PROFILE_NAME}] Recording deployment episodes to repo: ${REPO_ID}"

exec python -m deployment.dp.deploy_policy \
  --repo-id "${REPO_ID}" \
  --num-episodes "${NUM_EPISODES}" \
  --fps 15 \
  --recording-manager-type keyboard \
  --path "${MODEL_PATH}" \
  --env-config "${ENV_CONFIG}" \
  --env-namespace "${ENV_NAMESPACE}" \
  --log-level INFO \
  "${EXTRA_ARGS[@]}"
