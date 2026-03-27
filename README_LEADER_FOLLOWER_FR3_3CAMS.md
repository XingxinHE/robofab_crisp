# FR3 Leader/Follower 3-Camera Runbook

Target setup:
- Leader FR3: namespace `left`, IP `172.16.0.33`
- Follower FR3: namespace `right`, IP `172.16.0.3`
- Cameras:
  - `robot0_agentview_left`  -> serial `342522074350`
  - `robot0_agentview_right` -> serial `347622071856`
  - `robot0_eye_in_hand`     -> serial `336222070633`

Dataset camera keys will be:
- `observation.images.robot0_agentview_left`
- `observation.images.robot0_agentview_right`
- `observation.images.robot0_eye_in_hand`

## 1) RT PC (dual FR3)

In `module_run_on_RT_pc/pixi_franka_ros2`:

```bash
# terminal 0
pixi run zenoh-router
```

```bash
# terminal 1
pixi run -e humble franka-dual \
  leader_robot_ip:=172.16.0.33 \
  follower_robot_ip:=172.16.0.3 \
  leader_namespace:=left \
  follower_namespace:=right \
  load_gripper:=true \
  controllers_yaml:=config/controllers.yaml
```

```bash
# terminal 2
pixi run -e humble ros2 control switch_controllers \
  --controller-manager /right/controller_manager \
  --activate cartesian_impedance_controller
```

## 2) Camera PC (`pixi_realsense_ros2`)

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

Topic checks:

```bash
pixi run ros2 topic echo /robot0_agentview_left/color/camera_info --once
pixi run ros2 topic echo /robot0_agentview_right/color/camera_info --once
pixi run ros2 topic echo /robot0_eye_in_hand/color/camera_info --once
```

## 3) (Optional) FR3 pilot buttons bridge

In `franka_buttons_ros2`:

```bash
docker compose up franka_buttons
```

Verify recording transitions:

```bash
ros2 topic echo /record_transition
```

## 4) Recording (keyboard manager)

In `robofab_crisp`:

```bash
pixi run record-fr3-leader-follower-3cams -- \
  --repo-id local/fr3_leader_follower_3cams \
  --tasks "pick and place" \
  --num-episodes 1
```

Keyboard controls:
- `r`: record start/stop
- `s`: save
- `d`: delete
- `q`: quit

## 5) Recording (pilot buttons + ROS manager)

In `robofab_crisp`:

```bash
pixi run record-fr3-leader-follower-buttons-3cams -- \
  --repo-id local/fr3_leader_follower_3cams \
  --tasks "turn on microwave" \
  --num-episodes 10
```

Expected button mapping:
- `circle`: record
- `check`: save
- `cross`: delete

Keyboard fallback for quit:

```bash
pixi run record-transition-keyboard
```

## 6) Post-record validation

```bash
pixi run episode-stats -- --repo-id local/fr3_leader_follower_3cams
pixi run episode-video-info -- --repo-id local/fr3_leader_follower_3cams --episode 0
```

You should see three video keys with exact names:
- `observation.images.robot0_agentview_left`
- `observation.images.robot0_agentview_right`
- `observation.images.robot0_eye_in_hand`

## 7) Train a simple ACT policy

`train-act` supports dataset metadata cleanup for ACT state-shape compatibility.

Recommended first run (smoke test):

```bash
pixi run train-act -- \
  --prepare-from local/fr3_leader_follower_3cams_open \
  --repo-id local/fr3_leader_follower_3cams_open_fix_feat \
  --smoke
```

Full run:

```bash
pixi run train-act -- \
  --repo-id local/fr3_leader_follower_3cams_open_fix_feat \
  --steps 50000 \
  --batch-size 8
```

Notes:
- `--prepare-from` clones the dataset and removes duplicated state subfeatures in metadata.
- The original dataset remains untouched.

## 8) Deploy the trained ACT (3 cameras)

Before deploy, ensure all runtime topics are alive:
- robot topics in right namespace:
  - `/right/current_pose`, `/right/joint_states`
- `/robot0_eye_in_hand/color/image_raw`
- `/robot0_agentview_left/color/image_raw`
- `/robot0_agentview_right/color/image_raw`

If you run a single FR3 bringup, include namespace:

```bash
pixi run -e humble franka \
  robot_ip:=172.16.0.3 \
  namespace:=right \
  load_gripper:=true \
  controllers_yaml:=config/controllers.yaml
```

Then deploy:

```bash
pixi run deploy-act-3cams -- \
  --model-path outputs/train/2026-03-26/22-09-39_act/checkpoints/last/pretrained_model \
  --repo-id local/fr3_leader_follower_3cams_open_deploy \
  --num-episodes 1
```

If `--model-path` is omitted, the newest `outputs/train/**/pretrained_model` is used.
