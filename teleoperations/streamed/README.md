# Streamed Teleop Module

This module is for streamed leader topics (`/phone_pose`, `/phone_gripper`) with a single FR3 follower.

## Commands

- Teleop publisher (web):
  - `pixi run teleop-streamed-web`
- Recording with streamed leader:
  - `pixi run record-streamed-fr3 -- --repo-id <repo_id> ...`

## Expected topics

- Follower robot:
  - `/current_pose`
  - `/joint_states`
- Cameras:
  - `/wrist/color/image_raw`
  - `/third_person/color/image_raw`
- Streamed leader:
  - `/phone_pose`
  - `/phone_gripper`
