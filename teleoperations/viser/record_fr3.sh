#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record FR3 data with direct Viser teleoperation (no streamed leader/follower topics).

Usage:
  pixi run record-viser-fr3 -- --repo-id <repo_id> [extra args]

Always enforced:
  --recording-manager-type keyboard
  --follower-config <auto-selected by preflight>
  --fps 15
  --no-push-to-hub

Examples:
  pixi run record-viser-fr3 -- --repo-id local/fr3_dualcam_streamed --tasks "pick cube" --num-episodes 1
  pixi run record-viser-fr3 -- --repo-id local/fr3_dualcam_streamed --resume --num-episodes 50
EOF
  exit 0
fi

if [[ " $* " == *" --follower-config "* ]]; then
  echo "[record-viser-fr3] Do not pass --follower-config. It is selected automatically by preflight." >&2
  exit 1
fi

FOLLOWER_CONFIG="$(python -m teleoperations.streamed.preflight --print-config-only --no-require-streamed-topics)"
if [[ -z "${FOLLOWER_CONFIG}" ]]; then
  echo "[record-viser-fr3] Failed to determine follower config from preflight." >&2
  exit 1
fi

echo "[record-viser-fr3] Using follower config: ${FOLLOWER_CONFIG}"

exec python -m teleoperations.viser.record \
  --recording-manager-type keyboard \
  --follower-config "${FOLLOWER_CONFIG}" \
  --follower-namespace "" \
  --fps 15 \
  --no-push-to-hub \
  "$@"
