# Workflow Index

This repository now follows a modular CRISP config style:
- reusable hardware/config modules in `config/{robots,grippers,cameras,control}`
- thin scenario files in `config/envs`
- workflow commands exposed in Pixi with a single normalized task surface
- script responsibilities grouped and documented in `scripts/README.md`
- modality runbooks grouped in `teleoperations/README.md`

## Config Composition Rules

Environment files in `config/envs/` should only keep scenario-level settings:
- `control_frequency`, `gripper_mode`, `max_episode_steps`
- observation-state include list
- control parameter references

Everything else should be referenced with `from_yaml`:
- `robot_config: { from_yaml: robots/... }`
- `gripper_config: { from_yaml: grippers/... }`
- `camera_configs: [ { from_yaml: cameras/... }, ... ]`

This keeps camera/gripper/robot changes reusable across teleop and deployment modes.

## Pixi Task Surface

Teleop:
- `teleop-streamed-web`
- `teleop-leader-follower-fr3`
- `teleop-gamepad-fr3`
- `teleop-gamepad-fr3-3cams`

Record:
- `record-streamed-fr3`
- `record-viser-fr3`
- `record-gamepad-fr3`
- `record-gamepad-fr3-3cams`
- `record-leader-follower-fr3`
- `record-leader-follower-fr3-buttons`
- `record-leader-follower-fr3-3cams`
- `record-leader-follower-fr3-buttons-3cams`

Deploy:
- `deploy-act-fr3`
- `deploy-act-fr3-3cams`
- `deploy-act-fr3-3cams-gamepad`

Preflight:
- `preflight-recording-fr3`
- `preflight-deploy-fr3`
- `preflight-deploy-fr3-3cams`
- `preflight-deploy-fr3-3cams-gamepad`

## Teleoperation Modules

- `teleoperations/streamed/README.md`
- `teleoperations/leader_follower/README.md`
- `teleoperations/gamepad/README.md`
- `teleoperations/viser/README.md`

## Deployment Module

- `deployment/act/README.md`

## Camera Role Reuse

Per-camera YAML files are role-specific (`camera_name` must match dataset keys), but you can still reuse the same physical camera by wiring topics accordingly in launch.

Example:
- In one setup, a camera can publish to `/third_person/...` and use `cameras/third_person_256.yaml`.
- In another setup, the same device can publish to `/robot0_agentview_left/...` and use `cameras/robot0_agentview_left_256.yaml`.
