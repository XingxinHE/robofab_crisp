#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record LeRobot data with FR3 leader/follower teleop + FR3 pilot buttons.

Usage:
  pixi run record-fr3-leader-follower-buttons -- --repo-id <repo_id> [extra args]

Always enforced:
  --leader-config fr3_left_leader
  --leader-namespace left
  --follower-config fr3_right_leader_follower_recording
  --follower-namespace right
  --recording-manager-type ros
  --fps 15
  --no-push-to-hub

Expected button mapping (from franka_buttons_ros2 bridge):
  circle -> record
  check  -> save
  cross  -> delete

Use keyboard helper task for quit fallback:
  pixi run record-transition-keyboard
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python "${ROOT_DIR}/teleoperations/01_leader_follower/record_lerobot_format_leader_follower.py" \
  --leader-config fr3_left_leader \
  --leader-namespace left \
  --follower-config fr3_right_leader_follower_recording \
  --follower-namespace right \
  --recording-manager-type ros \
  --fps 15 \
  --no-push-to-hub \
  "$@"
