# Gamepad Module

This module is for direct Xbox gamepad teleoperation/recording on a single FR3.

## Commands

- Teleop (2-cam/default env):
  - `pixi run teleop-gamepad-fr3`
- Teleop (3-cam env):
  - `pixi run teleop-gamepad-fr3-3cams`
- Recording (2-cam/default env):
  - `pixi run record-gamepad-fr3 -- --repo-id <repo_id> ...`
- Recording (3-cam env):
  - `pixi run record-gamepad-fr3-3cams -- --repo-id <repo_id> ...`

pixi run teleop-gamepad-fr3-3cams -- --home-config robots/fr3_root_home_year2.yaml

## D-pad recording controls

- Up: record start/stop
- Right: save episode
- Left: delete episode
- Down: exit
