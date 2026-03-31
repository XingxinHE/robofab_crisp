#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pixi passes an argument separator as a literal "--" for wrapped tasks.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record FR3 data with streamed SpaceMouse teleop and keyboard episode control.

Usage:
  pixi run record-streamed-fr3 -- --repo-id <repo_id> [extra crisp-record args]

Always enforced:
  --use-streamed-teleop
  --recording-manager-type keyboard
  --follower-config <auto-selected by preflight>
  --fps 15
  --no-push-to-hub

Examples:
  pixi run record-streamed-fr3 -- --repo-id local/fr3_dualcam --tasks "pick cube" --num-episodes 1
  pixi run record-streamed-fr3 -- --repo-id local/fr3_dualcam --resume --num-episodes 50
EOF
  exit 0
fi

if [[ " $* " == *" --follower-config "* ]]; then
  echo "[record-streamed-fr3] Do not pass --follower-config. It is selected automatically by preflight." >&2
  exit 1
fi

FOLLOWER_CONFIG="$(python -m teleoperations.streamed.preflight --print-config-only)"
if [[ -z "${FOLLOWER_CONFIG}" ]]; then
  echo "[record-streamed-fr3] Failed to determine follower config from preflight." >&2
  exit 1
fi

echo "[record-streamed-fr3] Using follower config: ${FOLLOWER_CONFIG}"

exec python -m teleoperations.streamed.record \
  --use-streamed-teleop \
  --recording-manager-type keyboard \
  --follower-config "${FOLLOWER_CONFIG}" \
  --follower-namespace "" \
  --fps 15 \
  --no-push-to-hub \
  "$@"
