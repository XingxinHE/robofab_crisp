# Leader/Follower Module

This module is for dual FR3 teleoperation:
- leader arm in `left` namespace
- follower arm in `right` namespace

## Commands

- Teleop loop:
  - `pixi run teleop-leader-follower-fr3`
- Record (keyboard manager):
  - `pixi run record-leader-follower-fr3 -- --repo-id <repo_id> ...`
- Record (pilot buttons via ROS manager):
  - `pixi run record-leader-follower-fr3-buttons -- --repo-id <repo_id> ...`
- 3-camera variants:
  - `pixi run record-leader-follower-fr3-3cams -- --repo-id <repo_id> ...`
  - `pixi run record-leader-follower-fr3-buttons-3cams -- --repo-id <repo_id> ...`

## Notes

- Recording wrappers are profile-driven via `teleoperations/leader_follower/record_profiled.sh`.
- The core recorder is `teleoperations/leader_follower/record_leader_follower.py`.
