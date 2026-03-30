#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi

exec python scripts/gamepad_teleop_fr3.py \
  --env-config fr3_3cams_gamepad_recording \
  "$@"
