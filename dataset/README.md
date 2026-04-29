# Dataset Conversion Module

This folder hosts conversion code to transform CRISP real-robot datasets into a RoboCasa-like schema for co-training.

## Goal

Convert local CRISP LeRobot datasets (recorded from real FR3) from:

- `observation.state`: 20D CRISP layout
- `action`: 7D CRISP layout

to RoboCasa-like:

- `observation.state`: 16D layout
- `action`: 12D layout

without changing the original recorded dataset in place.

## Canonical Input (CRISP)

`observation.state` (20D):

0-5: `x,y,z,roll,pitch,yaw`
6: `gripper`
7-13: `joint_0..joint_6`
14-19: `target_x,target_y,target_z,target_roll,target_pitch,target_yaw`

`action` (7D):

0-5: `x,y,z,roll,pitch,yaw`
6: `gripper`

## Canonical Output (RoboCasa-like)

`observation.state` (16D):

0-2: base position (zero)
3-6: base rotation quaternion `xyzw` = identity
7-9: end-effector position relative (from CRISP `x,y,z`)
10-13: end-effector orientation quaternion `xyzw` (converted from CRISP `roll,pitch,yaw`)
14-15: gripper qpos pair

`action` (12D):

0-3: base motion (zero)
4: `control_mode = -1.0`
5-7: end-effector translation delta (`x,y,z` from CRISP action)
8-10: end-effector rotation delta as rotvec (converted from CRISP `roll,pitch,yaw` action)
11: `gripper_close` in RoboCasa convention (`+1` close, `-1` open)

## Locked Gripper Assumption (confirmed)

For the current gamepad 3-camera workflow:

- CRISP gripper is normalized in `[0,1]`
- `1.0` means open, `0.0` means close
- max physical width is `0.08 m`

Conversion formulas:

- `width_m = gripper_norm * 0.08`
- `gripper_qpos = [width_m / 2, -width_m / 2]`
- `gripper_close = +1` if `crisp_action_gripper <= 0.5`, else `-1`

## Orientation Conventions

- Base rotation and EE rotation quaternions use `xyzw` order.
- State orientation conversion:
  - `quat_xyzw = Rotation.from_euler("xyz", [roll,pitch,yaw]).as_quat()`
- Action rotation conversion:
  - `rotvec = Rotation.from_euler("xyz", [droll,dpitch,dyaw]).as_rotvec()`

## Implementation Order

1. Implement pure conversion helpers (RPY->quat, delta RPY->rotvec, gripper helpers).
2. Implement vector-level converters:
   - `convert_crisp_state_to_robocasa_state`
   - `convert_crisp_action_to_robocasa_action`
3. Implement frame-level converter preserving non-converted fields.
4. Implement metadata/features conversion helpers.
5. Implement end-to-end dataset conversion entrypoint.

## Test Framework

Tests are defined in:

- `tests/dataset/test_crisp_to_robocasa.py`

They cover:

- orientation conversion correctness
- gripper conversion correctness
- shape/layout mapping checks
- fixed constants (`base_motion`, `control_mode`, base pose)
- frame field preservation checks
- metadata conversion contract checks
- local fixture checks using:
  - `/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema`

Run tests with:

```bash
pytest tests/dataset/test_crisp_to_robocasa.py -q
```

## Non-goals

- Do not change CRISP recording pipeline here.
- Do not mutate source dataset in place.
- Codec/FPS policy for collection remains separate from schema conversion.
