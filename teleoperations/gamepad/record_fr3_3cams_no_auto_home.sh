#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record LeRobot data with one FR3 + Xbox gamepad + 3 cameras, with no automatic homing.

Usage:
  pixi run record-gamepad-fr3-3cams-no-auto-home -- --repo-id <repo_id> [extra args]

Always enforced:
  --follower-config fr3_3cams_gamepad_recording
  --follower-namespace ""
  --recording-manager-type ros
  --fps 20
  --no-push-to-hub

No automatic homing:
  - startup does not move to home
  - D-pad Up start/stop does not move to home
  - final exit does not move to home

Gamepad recording controls:
  D-pad Up    -> record start/stop
  D-pad Right -> save
  D-pad Left  -> delete
  D-pad Down  -> exit
EOF
  exit 0
fi

exec python -m teleoperations.gamepad.record_no_auto_home \
  --follower-config fr3_3cams_gamepad_recording \
  --follower-namespace "" \
  --recording-manager-type ros \
  --fps 20 \
  --no-push-to-hub \
  "$@"
