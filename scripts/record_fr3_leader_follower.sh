#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Record LeRobot data with FR3 leader/follower teleop.

Usage:
  pixi run record-fr3-leader-follower -- --repo-id <repo_id> [extra args]

Always enforced:
  --leader-config fr3_left_leader
  --leader-namespace left
  --follower-config fr3_right_leader_follower_recording
  --follower-namespace right
  --recording-manager-type keyboard
  --fps 15
  --no-push-to-hub

Examples:
  pixi run record-fr3-leader-follower -- --repo-id local/fr3_lf --tasks "stack block" --num-episodes 1
  pixi run record-fr3-leader-follower -- --repo-id local/fr3_lf --resume --num-episodes 20
EOF
  exit 0
fi

exec python "${ROOT_DIR}/teleoperations/01_leader_follower/record_lerobot_format_leader_follower.py" \
  --leader-config fr3_left_leader \
  --leader-namespace left \
  --follower-config fr3_right_leader_follower_recording \
  --follower-namespace right \
  --recording-manager-type keyboard \
  --fps 15 \
  --no-push-to-hub \
  "$@"
