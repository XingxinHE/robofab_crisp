# FR3 + GR00T N1.7 (DROID) + 3 Cameras + Gamepad

This runbook covers zero-shot deployment of the `nvidia/GR00T-N1.7-DROID`
checkpoint (or the base `nvidia/GR00T-N1.7-3B` model) on the FR3 with the
standard 3-camera gamepad recording setup.

> The DROID checkpoint was trained on a Franka Panda with two cameras and a
> 7-DOF arm. This deployment maps the closest available CRISP cameras/state to
> the DROID embodiment schema.

---

## Target setup

- robot: FR3 at `172.16.0.3`
- teleop/record input: Xbox gamepad
- cameras: `robot0_eye_in_hand`, `robot0_agentview_left`, `robot0_agentview_right`

GR00T N1.7 DROID only uses two of these cameras:

| CRISP camera | DROID camera key (`observation["video"]`) | Role |
|---|---|---|
| `robot0_agentview_left` | `exterior_image_1_left` | exterior / third-person |
| `robot0_eye_in_hand` | `wrist_image_left` | wrist / eye-in-hand |
| `robot0_agentview_right` | — | ignored for zero-shot DROID |

---

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

---

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

---

## 3) Teleop sanity check (GPU/ops PC)

In `robofab_crisp`:

```bash
pixi run teleop-gamepad-fr3-3cams -- --home-on-start
```

---

## 4) Start the GR00T N1.7 inference server

The server can run anywhere Isaac-GR00T N1.7 is installed with the correct
Python/CUDA dependencies.

**Option A: standalone Isaac-GR00T repo**

```bash
cd /home/hex/Documents/github/icra-2027-ws/post-train-vla/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-DROID \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda
```

**Option B: the existing `robofab_robocasa` UV environment** (where Isaac-GR00T
is vendored as a third-party dependency):

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_robocasa

uv run --group gr00t --extra cu128 python third_party/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-DROID \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda
```

For a base-model zero-shot test, replace the model path with
`nvidia/GR00T-N1.7-3B`.

---

## 5) Deploy on CRISP

In `robofab_crisp`:

> **Transport note:** The N1.7 server uses a `msgpack_numpy` ZMQ protocol that
> is incompatible with the legacy N1.5 torch-serialized ZMQ protocol. Use
> `--groot-transport zmq1p7` for N1.7.

```bash
pixi run deploy-gr00t-1p7-fr3-3cams-gamepad -- \
  --groot-transport zmq1p7 \
  --groot-host 127.0.0.1 \
  --groot-port 5555 \
  --repo-id local/fr3_gr00t_1p7_3cams_gamepad_deploy \
  --num-episodes 1 \
  --task "Pick up the red block." \
  --fps 5 \
  --action-chunk-size 4 \
  --async-inference \
  --prefetch-threshold 2 \
  --log-timing \
  --home-config fr3_root_home_lab
```

Recommended progression:

1. First rollout: `--fps 5 --action-chunk-size 1 --dry-run` to verify network
   calls and action decoding without robot motion.
2. Second rollout: `--fps 5 --action-chunk-size 1` live, with e-stop ready.
3. If stable, increase `--action-chunk-size` to `4` or `8`.
4. Only then consider raising `--fps`.

---

## 6) Joint-space deployment (alternative)

If the Cartesian adapter behaves poorly, you can deploy the same checkpoint in
joint-space mode. The N1.7 server still returns `joint_position` as absolute
7-DOF targets; the adapter converts them to joint deltas for CRISP's joint
controller.

Switch the RT PC controller:

```bash
pixi run -e humble ros2 control switch_controllers --deactivate cartesian_impedance_controller --activate joint_impedance_controller
```

Then run the joint pixi task **without the `--` separator** (the task already
bakes in `--joint-control --fps 15 --action-chunk-size 15 --async-inference
--prefetch-threshold 5`):

```bash
pixi run deploy-gr00t-1p7-fr3-3cams-gamepad-joint \
  --groot-transport zmq1p7 \
  --groot-host 127.0.0.1 \
  --groot-port 5555 \
  --repo-id local/fr3_gr00t_1p7_3cams_gamepad_joint_deploy \
  --num-episodes 1 \
  --task "Pick up the red block." \
  --log-timing \
  --home-config fr3_root_home_lab
```

If you want to override the baked-in defaults, pass them after the task name as
well (the last value wins):

```bash
pixi run deploy-gr00t-1p7-fr3-3cams-gamepad-joint \
  --fps 10 \
  --action-chunk-size 8 \
  ...
```

Joint-specific knobs:

| Flag | Default | Meaning |
|---|---|---|
| `--max-joint-step-rad` | 0.2 | Max joint delta sent per 15 Hz step |
| `--joint-target-smoothing-alpha` | 1.0 | Low-pass GR00T joint targets; lower is smoother |
| `--joint-delta-smoothing-alpha` | 1.0 | Low-pass executed joint deltas; lower is smoother |
| `--gripper-flip` | off | Invert open/close mapping in both Cartesian and joint modes |
| `--gripper-threshold` | 0.5 | Threshold for binary gripper decisions |
| `--gripper-hysteresis` | 0.0 | Add a deadband around the gripper threshold |
| `--gripper-debounce-steps` | 1 | Require repeated opposite decisions before switching |
| `--gripper-uncertain-band` | 0.0 | Treat predictions near threshold as uncertain |
| `--gripper-uncertain-policy` | none | Use `hold`, `open`, or `closed` inside the uncertain band |

Recommended progression (no `--` separator):

1. Dry run with single-step joint tracking:
   `--fps 5 --action-chunk-size 1 --prefetch-threshold 0 --dry-run`.
2. Live single-step joint tracking:
   `--fps 5 --action-chunk-size 1 --prefetch-threshold 0`.
3. Match the Cartesian diagnostic timing before trying the baked-in fast path:
   `--fps 5 --action-chunk-size 4 --prefetch-threshold 2`.
4. If joint targets still chatter, reduce `--max-joint-step-rad` and try
   `--joint-target-smoothing-alpha 0.5` or `--joint-delta-smoothing-alpha 0.5`.
5. Only then try higher FPS or the baked-in `15 Hz / 15-step` chunk.

---

## How the adapter works

The new adapter lives in
`robofab_crisp/deployment/gr00t/gr00t_1p7_remote_policy.py`.

### Observation conversion (CRISP → N1.7 DROID)

| N1.7 DROID key | CRISP source |
|---|---|
| `video.exterior_image_1_left` | `observation.images.robot0_agentview_left` |
| `video.wrist_image_left` | `observation.images.robot0_eye_in_hand` |
| `state.eef_9d` | Computed from `observation.state.cartesian` |
| `state.gripper_position` | `observation.state.gripper` (CRISP logs `1 - gripper.value`) |
| `state.joint_position` | `observation.state.joints` |
| `language.annotation.language.language_instruction` | CLI `--task` |

`eef_9d` is `XYZ + rot6d` using the same egocentric frame correction as the
Isaac-GR00T DROID tooling.

### Action conversion (N1.7 DROID → CRISP Cartesian)

The server returns absolute targets:

- `action.eef_9d`
- `action.gripper_position`
- `action.joint_position`

The adapter converts the absolute `eef_9d` target into a Cartesian delta:

```text
T_delta = T_current^{-1} * T_target
position_delta = T_delta[:3, 3]
rotation_delta = Rotation.from_matrix(T_delta[:3, :3]).as_rotvec()
```

The delta is clipped to the configured max step sizes and sent to the CRISP
Cartesian controller. Gripper commands use CRISP's command convention
`0=closed, 1=open`; the adapter binarizes at `gripper_position > 0.5` unless
optional gripper filtering flags are enabled.

### Action conversion (N1.7 DROID → CRISP joint)

For `--joint-control`, the adapter uses the `action.joint_position` and
`action.gripper_position` keys and ignores `action.eef_9d`. At each control step:

```text
current_joint = env.robot.target_joint
delta = clip(target_joint - current_joint, -max_joint_step_rad, +max_joint_step_rad)
action = concatenate([delta, gripper_action])
```

`ManipulatorJointEnv.step()` adds the delta to its current joint target, so the
robot receives the absolute target returned by GR00T.

> **CRISP action semantics:** `crisp_gym`'s joint `env.step()` **always**
> interprets joint actions as relative deltas, regardless of
> `use_relative_actions`. The adapter therefore converts GR00T's absolute joint
> targets into deltas just before stepping.

---

## Known limitations / risks

- **Only two cameras are used.** The right agent-view camera is ignored.
- **Gripper convention may need flipping.** DROID `gripper_position` open/close
  direction may differ from CRISP. Verify on the first live test.
- **Joint ordering.** `observation.state.joints` order in CRISP should match
  DROID's 7-DOF order. If it does not, the `eef_9d`-based control is still
  correct, but the `joint_position` state passed to the model will be
  inconsistent.
- **EE frame convention.** The adapter reuses the exact `compute_eef_9d()`
  function from `Isaac-GR00T/examples/DROID/main_gr00t.py`, including the
  `DROID_EEF_ROTATION_CORRECT` matrix.

---

## Smoke test without ROS

You can test the server with a minimal HTTP client. The expected response keys
are:

- `eef_9d`
- `gripper_position`
- `joint_position`

See `robofab_crisp/deployment/gr00t/README.md` for the generic smoke-test
command.
