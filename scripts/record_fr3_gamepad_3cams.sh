#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record LeRobot data with one FR3 + Xbox gamepad + 3 cameras.

Usage:
  pixi run record-fr3-gamepad-3cams -- --repo-id <repo_id> [extra args]

Always enforced:
  --follower-config fr3_3cams_gamepad_recording
  --follower-namespace ""
  --recording-manager-type ros
  --fps 15
  --no-push-to-hub

Gamepad recording controls:
  D-pad Up    -> record start/stop
  D-pad Right -> save
  D-pad Left  -> delete
  D-pad Down  -> exit
EOF
  exit 0
fi

exec python scripts/record_lerobot_gamepad_direct.py \
  --follower-config fr3_3cams_gamepad_recording \
  --follower-namespace "" \
  --recording-manager-type ros \
  --fps 15 \
  --no-push-to-hub \
  "$@"
