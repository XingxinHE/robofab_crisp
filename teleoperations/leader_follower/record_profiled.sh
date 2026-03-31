#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE_NAME=""
FOLLOWER_CONFIG=""
RECORDING_MANAGER_TYPE="keyboard"
BUTTONS_HINT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile-name)
      PROFILE_NAME="$2"
      shift 2
      ;;
    --follower-config)
      FOLLOWER_CONFIG="$2"
      shift 2
      ;;
    --recording-manager-type)
      RECORDING_MANAGER_TYPE="$2"
      shift 2
      ;;
    --buttons-hint)
      BUTTONS_HINT="$2"
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

if [[ -z "${PROFILE_NAME}" || -z "${FOLLOWER_CONFIG}" ]]; then
  echo "[record-leader-follower-profiled] Missing required profile parameters." >&2
  exit 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
Record LeRobot data with profile '${PROFILE_NAME}'.

Usage:
  pixi run ${PROFILE_NAME} -- --repo-id <repo_id> [extra args]

Always enforced:
  --leader-config fr3_left_leader
  --leader-namespace left
  --follower-config ${FOLLOWER_CONFIG}
  --follower-namespace right
  --recording-manager-type ${RECORDING_MANAGER_TYPE}
  --fps 15
  --no-push-to-hub
EOF
  if [[ "${BUTTONS_HINT}" == "1" ]]; then
    cat <<'EOF'

Expected button mapping:
  circle -> record
  check  -> save
  cross  -> delete
EOF
  fi
  exit 0
fi

exec python -m teleoperations.leader_follower.record_leader_follower \
  --leader-config fr3_left_leader \
  --leader-namespace left \
  --follower-config "${FOLLOWER_CONFIG}" \
  --follower-namespace right \
  --recording-manager-type "${RECORDING_MANAGER_TYPE}" \
  --fps 15 \
  --no-push-to-hub \
  "$@"
