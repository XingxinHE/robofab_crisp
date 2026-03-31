# FR3 + Gamepad + 3 Cameras

Target setup:
- robot: FR3 at `172.16.0.3`
- teleop/record input: Xbox gamepad
- cameras: `robot0_eye_in_hand`, `robot0_agentview_left`, `robot0_agentview_right`

Required dataset camera keys:
- `observation.images.robot0_eye_in_hand`
- `observation.images.robot0_agentview_left`
- `observation.images.robot0_agentview_right`

## 1) Robot bringup (RT PC)

In `module_run_on_RT_pc/pixi_franka_ros2`:

```bash
# terminal 0
pixi run zenoh-router
```

```bash
# terminal 1
pixi run -e humble franka \
  robot_ip:=172.16.0.3 \
  load_gripper:=true \
  controllers_yaml:=config/controllers.yaml
```

```bash
# terminal 2
pixi run -e humble ros2 control switch_controllers --activate cartesian_impedance_controller
```

## 2) Camera bringup (camera PC)

In `pixi_realsense_ros2`:

```bash
# terminal 0
pixi run camera-triple
```

Optional check:

```bash
pixi run test-image-triple
```

## 3) Teleop sanity check (GPU/ops PC)

In `robofab_crisp`:

```bash
pixi run teleop-gamepad-fr3-3cams -- --home-on-start
```

## 4) Record dataset

```bash
pixi run record-gamepad-fr3-3cams -- \
  --repo-id local/fr3_gamepad_3cams_open \
  --tasks "open the microwave" \
  --num-episodes 10
```

Gamepad recording controls:
- D-pad Up: record start/stop
- D-pad Right: save
- D-pad Left: delete
- D-pad Down: exit

Resume rule:
- `--num-episodes` is the total target.
- example: if dataset has 18 episodes, use `--resume --num-episodes 30` to add 12.

## 5) If recording was interrupted

```bash
# dry run
pixi run dataset-cleanup-resume -- \
  --repo-id local/fr3_gamepad_3cams_open

# apply
pixi run dataset-cleanup-resume -- \
  --repo-id local/fr3_gamepad_3cams_open \
  --apply
```

## 6) Validate dataset

```bash
pixi run episode-stats -- --repo-id local/fr3_gamepad_3cams_open
pixi run episode-video-info -- --repo-id local/fr3_gamepad_3cams_open --episode 0
```

## 7) Train ACT

```bash
# smoke
pixi run train-act -- \
  --prepare-from local/fr3_gamepad_3cams_open \
  --repo-id local/fr3_gamepad_3cams_open_fix_feat \
  --smoke

# full
pixi run train-act -- \
  --repo-id local/fr3_gamepad_3cams_open_fix_feat \
  --steps 50000 \
  --batch-size 8
```

## 8) Deploy ACT

```bash
pixi run preflight-deploy-fr3-3cams-gamepad

pixi run deploy-act-fr3-3cams-gamepad -- \
  --model-path outputs/train/<date>/<job>/checkpoints/last/pretrained_model \
  --repo-id local/fr3_gamepad_3cams_open_deploy \
  --num-episodes 1
```

If `--model-path` is omitted, latest `outputs/train/**/pretrained_model` is used.
