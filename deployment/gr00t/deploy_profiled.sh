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
  echo "[deploy-gr00t-profiled] Missing required profile parameters." >&2
  exit 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
Deploy GR00T with profile '${PROFILE_NAME}'.

Start the GR00T server separately, then run this client.

Usage:
  ZMQ:  pixi run ${PROFILE_NAME} -- --groot-transport zmq --groot-host 127.0.0.1 --groot-port 5555 [extra args]
  HTTP: pixi run ${PROFILE_NAME} -- --groot-transport http --groot-server http://127.0.0.1:8000 [extra args]

Defaults:
  --groot-transport http
  --groot-server http://127.0.0.1:8000
  --groot-host 127.0.0.1
  --groot-port 5555
  --repo-id ${DEFAULT_REPO_ID}
  --num-episodes 1
  --env-config <auto-selected by ${PREFLIGHT_MODULE}>
  --env-namespace ${ENV_NAMESPACE:-<root>}
  --recording-manager-type keyboard
  --fps 5
  --task "Close the lid blender by securely placing the lid on top."
  --action-chunk-size 1
  --async-inference                Optional background horizon prefetching
  --prefetch-threshold <n>         Optional queue threshold for async prefetch
  --log-timing                     Optional frame timing logs
  --max-translation-step-m 0.006
  --max-rotation-step-rad 0.06
  --home-config <name-or-path>        Optional robot YAML or homes/*.yaml for deployment homing
  --after-teleop <name-or-path>       Optional final home after all deployment episodes
EOF
  exit 0
fi

if [[ " $* " == *" --env-config "* ]]; then
  echo "[${PROFILE_NAME}] Do not pass --env-config. It is selected automatically." >&2
  exit 1
fi

if [[ " $* " == *" --policy-config "* ]]; then
  echo "[${PROFILE_NAME}] Do not pass --policy-config. GR00T deployment is pinned by this wrapper." >&2
  exit 1
fi

if [[ "${FORBID_ENV_NAMESPACE_ARG}" == "1" && " $* " == *" --env-namespace "* ]]; then
  echo "[${PROFILE_NAME}] Do not pass --env-namespace. This wrapper pins profile namespace." >&2
  exit 1
fi

GROOT_TRANSPORT="http"
GROOT_SERVER="http://127.0.0.1:8000"
GROOT_HOST="127.0.0.1"
GROOT_PORT="5555"
GROOT_API_TOKEN=""
REPO_ID="${DEFAULT_REPO_ID}"
NUM_EPISODES="1"
WANTS_RESUME=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --groot-transport)
      GROOT_TRANSPORT="$2"
      shift 2
      ;;
    --groot-server)
      GROOT_SERVER="$2"
      shift 2
      ;;
    --groot-host)
      GROOT_HOST="$2"
      shift 2
      ;;
    --groot-port)
      GROOT_PORT="$2"
      shift 2
      ;;
    --groot-api-token)
      GROOT_API_TOKEN="$2"
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

SERVER_ARGS=(
  --groot-transport "${GROOT_TRANSPORT}"
  --groot-server "${GROOT_SERVER}"
  --groot-host "${GROOT_HOST}"
  --groot-port "${GROOT_PORT}"
)
if [[ -n "${GROOT_API_TOKEN}" ]]; then
  SERVER_ARGS+=(--groot-api-token "${GROOT_API_TOKEN}")
fi

if [[ "${GROOT_TRANSPORT}" == "zmq" ]]; then
  echo "[${PROFILE_NAME}] Using GR00T ZMQ server: tcp://${GROOT_HOST}:${GROOT_PORT}"
else
  echo "[${PROFILE_NAME}] Using GR00T HTTP server: ${GROOT_SERVER}"
fi
echo "[${PROFILE_NAME}] Using env config: ${ENV_CONFIG}"
echo "[${PROFILE_NAME}] Using env namespace: ${ENV_NAMESPACE:-<root>}"
echo "[${PROFILE_NAME}] Recording deployment episodes to repo: ${REPO_ID}"

exec python -m deployment.gr00t.deploy_policy \
  --repo-id "${REPO_ID}" \
  --num-episodes "${NUM_EPISODES}" \
  --fps 5 \
  --recording-manager-type keyboard \
  "${SERVER_ARGS[@]}" \
  --env-config "${ENV_CONFIG}" \
  --env-namespace "${ENV_NAMESPACE}" \
  --log-level INFO \
  "${EXTRA_ARGS[@]}"
