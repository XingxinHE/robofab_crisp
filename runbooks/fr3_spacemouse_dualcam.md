# FR3 + SpaceMouse Stream + Dual Cameras

This is the secondary setup (kept for completeness). Primary setups are in:
- `runbooks/fr3_gamepad_3cams.md`
- `runbooks/fr3_leader_follower_3cams.md`

Target setup:
- single FR3 at `172.16.0.3`
- cameras: `wrist`, `third_person`
- streamed leader topics: `/phone_pose`, `/phone_gripper`

## 1) Bringup order

Realtime PC (`module_run_on_RT_pc/pixi_franka_ros2`):

```bash
pixi run zenoh-router
pixi run -e humble franka robot_ip:=172.16.0.3 load_gripper:=true controllers_yaml:=config/controllers.yaml
pixi run -e humble ros2 control switch_controllers --activate cartesian_impedance_controller
```

SpaceMouse recording publisher (`pixi_franka_spacemouse`):

```bash
pixi run run-spacemouse-recording
```

Camera PC (`pixi_realsense_ros2`):

```bash
pixi run camera-dual
```

## 2) Record

In `robofab_crisp`:

```bash
pixi run preflight-recording-fr3

pixi run record-streamed-fr3 -- \
  --repo-id local/fr3_dualcam_streamed \
  --tasks "pick and place the object" \
  --num-episodes 10
```

Keyboard controls:
- `r`: record
- `s`: save
- `d`: delete
- `q`: quit

Resume:

```bash
pixi run record-streamed-fr3 -- \
  --repo-id local/fr3_dualcam_streamed \
  --resume \
  --num-episodes 50 \
  --tasks "pick and place the object"
```

## 3) Optional: web streamed teleop publisher

```bash
pixi run teleop-streamed-web
```

## 4) Optional: direct Viser recording fallback

```bash
pixi run record-viser-fr3 -- \
  --repo-id local/fr3_dualcam_streamed \
  --tasks "pick and place the object" \
  --num-episodes 10
```

## 5) Validate, train, deploy

```bash
pixi run episode-stats -- --repo-id local/fr3_dualcam_streamed
pixi run episode-video-info -- --repo-id local/fr3_dualcam_streamed --episode 0

pixi run train-act

pixi run preflight-deploy-fr3
pixi run deploy-act-fr3 -- \
  --model-path outputs/train/<date>/<job>/checkpoints/last/pretrained_model \
  --repo-id local/fr3_dualcam_streamed_deploy \
  --num-episodes 1
```
