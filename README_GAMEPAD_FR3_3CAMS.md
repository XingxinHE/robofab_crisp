# FR3 + Gamepad + 3 Cameras Runbook

Target setup:
- Robot: FR3 at `172.16.0.3`
- Teleop input: Xbox gamepad (single operator)
- Cameras:
  - `robot0_agentview_left`  -> serial `342522074350`
  - `robot0_agentview_right` -> serial `347622071856`
  - `robot0_eye_in_hand`     -> serial `336222070633`

Dataset camera keys:
- `observation.images.robot0_agentview_left`
- `observation.images.robot0_agentview_right`
- `observation.images.robot0_eye_in_hand`

## 1) Robot bringup (single FR3)

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

## 2) Camera bringup (three RealSense)

In `pixi_realsense_ros2`:

```bash
# terminal 0
pixi run camera-triple
```

Optional visual check:

```bash
# terminal 1
pixi run test-image-triple
```

## 3) Gamepad teleop sanity check

In `robofab_crisp`:

```bash
pixi run gamepad-teleop-fr3-3cams -- --home-on-start
```

## 4) Record dataset (gamepad controls recording)

In `robofab_crisp`:

```bash
pixi run record-fr3-gamepad-3cams -- \
  --repo-id local/fr3_gamepad_3cams_open \
  --tasks "open the microwave" \
  --num-episodes 10
```

Gamepad recording mapping:
- D-pad Up: record start/stop
- D-pad Right: save
- D-pad Left: delete
- D-pad Down: exit

Resume note:
- `--num-episodes` is the total target, not "additional episodes".
- Example: if dataset currently has 18 episodes, use `--resume --num-episodes 30` to add 12 more.

If recording is interrupted (collision, power/network drop), run resume cleanup first:

```bash
# dry run
pixi run dataset-cleanup-resume -- \
  --repo-id local/fr3_gamepad_3cams_open

# apply cleanup
pixi run dataset-cleanup-resume -- \
  --repo-id local/fr3_gamepad_3cams_open \
  --apply
```

## 5) Validate the recorded dataset

```bash
pixi run episode-stats -- --repo-id local/fr3_gamepad_3cams_open
pixi run episode-video-info -- --repo-id local/fr3_gamepad_3cams_open --episode 0
```

## 6) Train ACT

Smoke run first:

```bash
pixi run train-act -- \
  --prepare-from local/fr3_gamepad_3cams_open \
  --repo-id local/fr3_gamepad_3cams_open_fix_feat \
  --smoke
```

Full run:

```bash
pixi run train-act -- \
  --repo-id local/fr3_gamepad_3cams_open_fix_feat \
  --steps 50000 \
  --batch-size 8
```

## 7) Deploy ACT

Before deploy, keep robot and three camera topics alive.

```bash
pixi run deploy-act-3cams-gamepad -- \
  --model-path outputs/train/<date>/<job>/checkpoints/last/pretrained_model \
  --repo-id local/fr3_gamepad_3cams_open_deploy \
  --num-episodes 1
```

If `--model-path` is omitted, latest `outputs/train/**/pretrained_model` is used.
