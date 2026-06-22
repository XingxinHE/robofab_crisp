# Gamepad Module

This module is for direct Xbox gamepad teleoperation/recording on a single FR3.

## Commands Example

- Teleop (2-cam/default env):
  - `pixi run teleop-gamepad-fr3`
- Teleop (3-cam env):
  - `pixi run teleop-gamepad-fr3-3cams`
- Recording (2-cam/default env):
  - `pixi run record-gamepad-fr3 -- --repo-id <repo_id> ...`
- Recording (3-cam env):
  - `pixi run record-gamepad-fr3-3cams -- --repo-id <repo_id> ...`
- Recording (3-cam env, manual start pose / no automatic homing):
  - `pixi run record-gamepad-fr3-3cams-no-auto-home -- --repo-id <repo_id> ...`

pixi run teleop-gamepad-fr3-3cams -- --home-config robots/fr3_root_home_year2.yaml

pixi run record-gamepad-fr3-3cams -- --repo-id <repo_id> --home-config robots/fr3_root_home_year2.yaml --after-teleop robots/fr3_root_home_year2.yaml

pixi run record-gamepad-fr3-3cams -- \
--repo-id local/open_close_blenderlid \
--home-config robots/fr3_root_home_robocasa.yaml \
--after-teleop robots/fr3_root_home_robocasa.yaml \
--fps 20 \
--tasks "	Open the blender by taking off the lid and placing it on the counter." \
--num-episodes 60

pixi run record-gamepad-fr3-3cams-no-auto-home -- \
--repo-id local/custom_start_pose \
--tasks "custom start pose demonstration" \
--num-episodes 10

## D-pad recording controls

- Up: record start/stop
- Right: save episode
- Left: delete episode
- Down: exit
