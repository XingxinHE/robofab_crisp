# FR3 Data Collection Runbook (CRISP + SpaceMouse + Dual D435)

This runbook records local LeRobot-format data with:
- FR3 state + gripper state from ROS2 topics
- dual camera observations from `/wrist/color/image_raw` and `/third_person/color/image_raw`
- streamed teleop leader topics (`/phone_pose`, `/phone_gripper`)

## Prerequisites

- Realtime PC (`172.16.0.100`) and GPU PC (`172.16.0.11`) are connected to the same Zenoh router.
- Realtime PC can already control FR3 with SpaceMouse in testing mode.
- GPU PC can run dual RealSense streams via `pixi_realsense_ros2`.

## Mode Rules (SpaceMouse)

Use exactly one mode at a time on the realtime PC:
- Testing mode: `pixi run run-spacemouse` (publishes direct robot commands)
- Recording mode: `pixi run run-spacemouse-recording` (publishes streamed teleop topics only)

Do not run both simultaneously.

## Terminal Order

### Realtime PC (`172.16.0.100`)

In `pixi_franka_ros2`:

```bash
# terminal 1
pixi run -e humble franka robot_ip:=172.16.0.3 load_gripper:=true controllers_yaml:=config/controllers.yaml
```

```bash
# terminal 2
pixi run -e humble ros2 control switch_controllers --activate cartesian_impedance_controller
```

In `pixi_franka_spacemouse`:

```bash
# terminal 3 (recording mode)
pixi run run-spacemouse-recording
```

### GPU PC (`172.16.0.11`)

In `pixi_realsense_ros2`:

```bash
# terminal 1
pixi run camera-dual
```

In `robofab_crisp`:

```bash
# terminal 2
pixi run recording-preflight
```

```bash
# terminal 3
pixi run record-fr3 -- \
  --repo-id local/fr3_dualcam_streamed \
  --tasks "pick and place the object" \
  --num-episodes 1
```

## Smoke Test (first episode)

```bash
pixi run record-fr3 -- \
  --repo-id local/fr3_dualcam_streamed \
  --tasks "smoke test task" \
  --num-episodes 1
```

During recording:
- `r`: start/stop recording
- `s`: save episode
- `d`: delete episode
- `q`: quit

## Resume Recording

```bash
pixi run record-fr3 -- \
  --repo-id local/fr3_dualcam_streamed \
  --resume \
  --num-episodes 50 \
  --tasks "pick and place the object"
```

## Notes

- `record-fr3` enforces:
  - `--use-streamed-teleop`
  - `--recording-manager-type keyboard`
  - `--fps 15`
  - `--no-push-to-hub`
- The preflight step auto-selects the follower config:
  - primary: `fr3_recording_streamed` (`/franka_gripper/joint_states`)
  - fallback: `fr3_recording_streamed_joint_states_fallback` (`/joint_states`, index `7`)

## Post-Recording Analysis

Episode statistics:

```bash
pixi run episode-stats -- --repo-id local/fr3_dualcam_streamed
```

Video metadata extraction (auto-detect all camera/video keys):

```bash
pixi run episode-video-info -- --repo-id local/fr3_dualcam_streamed
```

Single episode example:

```bash
pixi run episode-video-info -- --repo-id local/fr3_dualcam_streamed --episode 0
```

## Robot Motion Playback (PyBullet)

Quick start:

```bash
pixi run episode-playback -- \
  --repo-id local/fr3_dualcam_streamed \
  --episode 0
```

Common commands:

```bash
# select a specific frame range and loop playback
pixi run episode-playback -- \
  --repo-id local/fr3_dualcam_streamed \
  --episode 0 \
  --start-frame 30 \
  --end-frame 220 \
  --loop

# use an explicit dataset path
pixi run episode-playback -- \
  --dataset-dir /home/hex/.cache/huggingface/lerobot/local/fr3_dualcam_streamed \
  --episode 0

# force action-based gripper source
pixi run episode-playback -- \
  --repo-id local/fr3_dualcam_streamed \
  --episode 0 \
  --gripper-source action
```

Keyboard controls in the PyBullet window:
- `space`: pause/resume
- `left/right`: step one frame when paused
- `,` / `.`: slow down / speed up playback
- `r`: restart from `start-frame`
- `l`: toggle looping
- `q`: quit (`esc` also works on PyBullet builds exposing `B3G_ESCAPE`)

Troubleshooting:
- If URDF cannot be loaded, pass `--urdf /absolute/path/to/fr3_franka_hand_d435.urdf`.
- If a dataset column is missing, verify `--joint-column` and `--gripper-source`.
- If the GUI does not open, ensure you are in a desktop/X11 session and `DISPLAY` is set.
