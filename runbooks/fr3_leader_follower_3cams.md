# FR3 Leader/Follower + 3 Cameras

Target setup:
- leader FR3: namespace `left`, IP `172.16.0.33`
- follower FR3: namespace `right`, IP `172.16.0.3`
- cameras: `robot0_eye_in_hand`, `robot0_agentview_left`, `robot0_agentview_right`

Required dataset camera keys:
- `observation.images.robot0_eye_in_hand`
- `observation.images.robot0_agentview_left`
- `observation.images.robot0_agentview_right`

## 1) Dual robot bringup (RT PC)

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

## 2) Camera bringup (camera PC)

In `pixi_realsense_ros2`:

```bash
pixi run camera-triple
```

Optional checks:

```bash
pixi run test-image-triple
pixi run ros2 topic echo /robot0_agentview_left/color/camera_info --once
pixi run ros2 topic echo /robot0_agentview_right/color/camera_info --once
pixi run ros2 topic echo /robot0_eye_in_hand/color/camera_info --once
```

## 3) Record dataset (keyboard manager)

In `robofab_crisp`:

```bash
pixi run record-leader-follower-fr3-3cams -- \
  --repo-id local/fr3_leader_follower_3cams_open \
  --tasks "pick and place" \
  --num-episodes 10
```

Keyboard controls:
- `r`: record
- `s`: save
- `d`: delete
- `q`: quit

## 4) Record dataset (pilot buttons + ROS manager)

If using `franka_buttons_ros2` bridge:

```bash
pixi run record-leader-follower-fr3-buttons-3cams -- \
  --repo-id local/fr3_leader_follower_3cams_open \
  --tasks "turn on microwave" \
  --num-episodes 10
```

Expected mapping:
- `circle`: record
- `check`: save
- `cross`: delete

Keyboard fallback for quit:

```bash
pixi run record-transition-keyboard
```

## 5) Validate dataset

```bash
pixi run episode-stats -- --repo-id local/fr3_leader_follower_3cams_open
pixi run episode-video-info -- --repo-id local/fr3_leader_follower_3cams_open --episode 0
```

## 6) Train ACT

```bash
# smoke
pixi run train-act -- \
  --prepare-from local/fr3_leader_follower_3cams_open \
  --repo-id local/fr3_leader_follower_3cams_open_fix_feat \
  --smoke

# full
pixi run train-act -- \
  --repo-id local/fr3_leader_follower_3cams_open_fix_feat \
  --steps 50000 \
  --batch-size 8
```

## 7) Deploy ACT

```bash
pixi run preflight-deploy-fr3-3cams

pixi run deploy-act-fr3-3cams -- \
  --model-path outputs/train/<date>/<job>/checkpoints/last/pretrained_model \
  --repo-id local/fr3_leader_follower_3cams_open_deploy \
  --num-episodes 1
```

If `--model-path` is omitted, latest `outputs/train/**/pretrained_model` is used.
