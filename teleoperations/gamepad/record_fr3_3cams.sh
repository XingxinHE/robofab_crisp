#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record LeRobot data with one FR3 + Xbox gamepad + 3 cameras.

Usage:
  pixi run record-gamepad-fr3-3cams -- --repo-id <repo_id> [extra args]
  pixi run record-gamepad-fr3-3cams -- --repo-id <repo_id> --home-config fr3_root_home_lab

Always enforced:
  --follower-config fr3_3cams_gamepad_recording
  --follower-namespace ""
  --recording-manager-type ros
  --fps 20
  --no-push-to-hub

Useful extra args:
  --home-config <name-or-path>    Robot YAML or homes/*.yaml for start/end homing
  --home-config-noise <rad>       Joint-space randomization around that home

Gamepad recording controls:
  D-pad Up    -> record start/stop
  D-pad Right -> save
  D-pad Left  -> delete
  D-pad Down  -> exit
EOF
  exit 0
fi

exec python -m teleoperations.gamepad.record \
  --follower-config fr3_3cams_gamepad_recording \
  --follower-namespace "" \
  --recording-manager-type ros \
  --fps 20 \
  --no-push-to-hub \
  "$@"
