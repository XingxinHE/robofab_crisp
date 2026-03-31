# Scripts Layout

The `scripts/` folder now contains only repository-wide utilities.

## Utilities kept here

- `cleanup_lerobot_resume.py`
- `clone_dataset_fix_features.py`
- `record_transition_keyboard.py`
- `train_act.sh`

## Canonical workflow entrypoints moved

- teleop/record: `teleoperations/{streamed,leader_follower,gamepad,viser}`
- deploy: `deployment/act`

Use `README_WORKFLOWS.md` for the canonical Pixi task surface.
