# Direct Viser Module

This module is for direct browser-based teleop that commands follower targets directly.

## Commands

- Recording with direct Viser teleop:
  - `pixi run record-viser-fr3 -- --repo-id <repo_id> ...`

## Notes

- This mode does not require streamed leader topics.
- Shared direct recording logic is in `teleoperations/shared/direct_recording_common.py`.
