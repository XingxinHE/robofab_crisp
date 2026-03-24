#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record FR3 data with direct Xbox gamepad teleoperation.

Usage:
  pixi run record-fr3-gamepad -- --repo-id <repo_id> [extra args]

Always enforced:
  --recording-manager-type ros
  --follower-config <auto-selected by preflight>
  --fps 15
  --no-push-to-hub

Gamepad recording controls:
  D-pad Up    : record start/stop
  D-pad Right : save episode
  D-pad Left  : delete episode
  D-pad Down  : exit
EOF
  exit 0
fi

if [[ " $* " == *" --follower-config "* ]]; then
  echo "[record-fr3-gamepad] Do not pass --follower-config. It is selected automatically by preflight." >&2
  exit 1
fi

FOLLOWER_CONFIG="$(python "${SCRIPT_DIR}/recording_preflight.py" --print-config-only --no-require-streamed-topics)"
if [[ -z "${FOLLOWER_CONFIG}" ]]; then
  echo "[record-fr3-gamepad] Failed to determine follower config from preflight." >&2
  exit 1
fi

echo "[record-fr3-gamepad] Using follower config: ${FOLLOWER_CONFIG}"

exec python "${SCRIPT_DIR}/record_lerobot_gamepad_direct.py" \
  --recording-manager-type ros \
  --follower-config "${FOLLOWER_CONFIG}" \
  --follower-namespace "" \
  --fps 15 \
  --no-push-to-hub \
  "$@"
