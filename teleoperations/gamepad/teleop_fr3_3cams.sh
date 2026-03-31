#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi

exec python -m teleoperations.gamepad.teleop \
  --env-config fr3_3cams_gamepad_recording \
  "$@"
