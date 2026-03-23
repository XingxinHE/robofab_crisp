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
# terminal 0
pixi run zenoh-router
```

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

### Optional: web teleop leader (replace SpaceMouse)

If you want to control streamed teleop from a browser instead of SpaceMouse, run this on GPU PC before `record-fr3`:

```bash
# terminal 0 (in robofab_crisp)
pixi run web-teleop
```

Then open the printed URL (default `http://0.0.0.0:8080` from that machine's IP), drag the end-effector handle, and use:
- `Open gripper` -> publishes `1.0` to `/phone_gripper`
- `Close gripper` -> publishes `0.0` to `/phone_gripper`

The web teleop now defaults to translation-only mode for stability (`--translation-only`), with a live `/end_effector_current` frame in the scene to show actual robot pose feedback.

This publishes the same streamed leader topics used by recording:
- `/phone_pose`
- `/phone_gripper`

### Optional: direct Viser recording (no leader/follower stream)

If streamed leader/follower teleop is unstable, use direct Viser teleop recording:

```bash
pixi run record-fr3-viser -- \
  --repo-id local/fr3_dualcam_streamed \
  --tasks "pick and place the object" \
  --num-episodes 1
```

This mode drives `env.robot.set_target(...)` directly from the Viser gumball and records the same LeRobot-format dataset schema.
Unlike streamed teleop recording, this mode does not require `/phone_pose` or `/phone_gripper` topics.

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
- `train-act` enforces local-only training (`--policy.push_to_hub=false`).
- `deploy-act` pins CRISP sync policy mode (`lerobot_policy`) and auto-selects deployment env config from preflight.
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

## Training (ACT)

Before training, strip duplicated sub-state features from dataset metadata. This avoids ACT state-shape mismatch in LeRobot.

```bash
pixi run python - <<'PY'
import json
from pathlib import Path

info_path = Path.home() / ".cache/huggingface/lerobot/local/fr3_dualcam_streamed/meta/info.json"
data = json.loads(info_path.read_text())
features = data["features"]

for key in [
    "observation.state.cartesian",
    "observation.state.gripper",
    "observation.state.joints",
    "observation.state.target",
]:
    features.pop(key, None)

info_path.write_text(json.dumps(data, indent=4) + "\n")
print(f"Updated {info_path}")
PY
```

Run ACT training:

```bash
pixi run train-act
```

Default training config (`scripts/train_act.sh`):
- dataset: `local/fr3_dualcam_streamed`
- policy: `act`
- batch size: `8`
- steps: `50000`
- save frequency: `10000`
- eval: disabled (`--eval_freq=0`)

Checkpoint output:

```bash
robofab_crisp/outputs/train/<date>/<time>_act/checkpoints/last/pretrained_model
```

Quick sanity check:

```bash
pixi run python - <<'PY'
from pathlib import Path
import json

ckpt = Path("outputs/train/2026-03-17/11-47-16_act/checkpoints/last")
print("exists:", ckpt.exists())
print("step:", json.loads((ckpt / "training_state/training_step.json").read_text())["step"])
PY
```

## Deployment (ACT on real FR3)

Use this section after you have a trained checkpoint at:

```bash
outputs/train/<date>/<time>_act/checkpoints/last/pretrained_model
```

### Safety rules

- Keep only one active motion source for FR3.
- Do not run `run-spacemouse` or `run-spacemouse-recording` while policy deployment is active.
- Keep the robot in a clear workspace and start with conservative behavior checks.

### Terminal order for deployment

#### Realtime PC (`172.16.0.100`)

In `pixi_franka_ros2`:

```bash
# terminal 0
pixi run zenoh-router
```

```bash
# terminal 1
pixi run -e humble franka robot_ip:=172.16.0.3 load_gripper:=true controllers_yaml:=config/controllers.yaml
```

```bash
# terminal 2
pixi run -e humble ros2 control switch_controllers --activate cartesian_impedance_controller
```

#### GPU PC (`172.16.0.11`)

In `pixi_realsense_ros2`:

```bash
# terminal 1
pixi run camera-dual
```

In `robofab_crisp`:

```bash
# terminal 2
pixi run deployment-preflight
```

```bash
# terminal 3
pixi run deploy-act -- \
  --model-path outputs/train/<date>/<time>_act/checkpoints/last/pretrained_model \
  --repo-id local/fr3_dualcam_streamed_deploy \
  --num-episodes 1

pixi run deploy-act -- \
  --model-path outputs/train/2026-03-17/11-47-16_act/checkpoints/last/pretrained_model \
  --repo-id local/fr3_dualcam_streamed_deploy \
  --num-episodes 1

pixi run deploy-act -- \
  --model-path outputs/train/2026-03-17/11-47-16_act/checkpoints/last/pretrained_model \
  --repo-id local/fr3_dualcam_streamed_deploy \
  --resume \
  --num-episodes 5
```

If `--model-path` is omitted, `deploy-act` auto-selects the most recent `outputs/train/**/pretrained_model`.
If `--repo-id` already exists and `--resume` is not provided, `deploy-act` fails early with a clear message.
If `--resume` is provided but the existing repo is incomplete (for example missing `meta/tasks.jsonl` after an interrupted run), `deploy-act` now fails early and asks you to delete that repo directory or use a new `--repo-id`.

### Deployment controls

`deploy-act` uses CRISP keyboard recording manager. During deployment episodes:
- `r`: start/stop an episode rollout
- `s`: save the rollout
- `d`: delete the rollout
- `q`: quit

### What `deploy-act` enforces

- policy wrapper: `lerobot_policy` (sync)
- env config: auto-selected by preflight
  - primary: `fr3_deploy_streamed` (`/franka_gripper/joint_states`)
  - fallback: `fr3_deploy_streamed_joint_states_fallback` (`/joint_states`, index `7`)
- env namespace: `""` (root topics)
- fps: `15`

### Quick dry-run checklist

- `/wrist/color/image_raw` and `/third_person/color/image_raw` are publishing on GPU PC.
- `/current_pose` and `/joint_states` are visible on GPU PC over Zenoh.
- model path contains `config.json`, `train_config.json`, and `model.safetensors`.
- active controller on RT PC is `cartesian_impedance_controller`.

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
