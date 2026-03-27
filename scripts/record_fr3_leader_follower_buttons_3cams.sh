#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record LeRobot data with FR3 leader/follower and 3 cameras (pilot buttons + ROS manager).

Usage:
  pixi run record-fr3-leader-follower-buttons-3cams -- --repo-id <repo_id> [extra args]

Always enforced:
  --leader-config fr3_left_leader
  --leader-namespace left
  --follower-config fr3_3_cams_leader_follower_recording
  --follower-namespace right
  --recording-manager-type ros
  --fps 15
  --no-push-to-hub

Expected button mapping:
  circle -> record
  check  -> save
  cross  -> delete
EOF
  exit 0
fi

exec python "${ROOT_DIR}/teleoperations/01_leader_follower/record_lerobot_format_leader_follower.py" \
  --leader-config fr3_left_leader \
  --leader-namespace left \
  --follower-config fr3_3_cams_leader_follower_recording \
  --follower-namespace right \
  --recording-manager-type ros \
  --fps 15 \
  --no-push-to-hub \
  "$@"
