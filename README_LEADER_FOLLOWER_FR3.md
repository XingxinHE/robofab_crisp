# FR3 Leader/Follower Runbook (Dual Robot, Same RT PC)

See `README_WORKFLOWS.md` for normalized task aliases and config-composition conventions.

For the 3-camera hardware setup, use `README_LEADER_FOLLOWER_FR3_3CAMS.md`.

This runbook uses two FR3 robots on the same RT PC:
- leader: `left` namespace, IP `172.16.0.33`
- follower: `right` namespace, IP `172.16.0.3`

## What was added

- RT bringup dual launch task in `pixi_franka_ros2`: `franka-dual`
- FR3 leader configs:
  - `config/teleop/fr3_left_leader.yaml`
  - `config/teleop/fr3_right_leader.yaml`
- FR3 follower env configs:
  - `config/envs/fr3_right_leader_follower_teleop.yaml`
  - `config/envs/fr3_right_leader_follower_recording.yaml`
- New tasks in `robofab_crisp`:
  - `teleop-leader-follower-fr3`
  - `record-leader-follower-fr3`
  - `record-leader-follower-fr3-buttons`
  - `record-transition-keyboard`

## 1) Realtime PC bringup (`module_run_on_RT_pc/pixi_franka_ros2`)

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

Switch follower to cartesian controller:

```bash
# terminal 2
pixi run -e humble ros2 control switch_controllers \
  --controller-manager /right/controller_manager \
  --activate cartesian_impedance_controller
```

Optional checks:

```bash
pixi run -e humble ros2 topic list | grep -E '^/(left|right)/'
pixi run -e humble ros2 topic echo /left/current_pose --once
pixi run -e humble ros2 topic echo /right/current_pose --once
```

## 2) Teleop test (`robofab_crisp`)

```bash
# terminal 3 (GPU/ops PC)
pixi run teleop-leader-follower-fr3 -- \
  --leader-config fr3_left_leader \
  --leader-namespace left \
  --follower-config fr3_right_leader_follower_teleop \
  --follower-namespace right \
  --home-on-start \
  --control-frequency 100
```

Notes:
- By default, follower does a one-time startup sync to the leader end-effector pose before delta teleop starts.
- Disable startup sync with `--no-sync-follower-on-start` if needed.
- The script configures the leader controller for teleop.
- Stop with `Ctrl+C`.

### 2b) Manual leader gripper check (prototype target)

Goal: physically operate the left (leader) gripper and verify follower gripper mirrors it.

In a separate terminal, monitor both gripper states:

```bash
pixi run ros2 topic echo /left/gripper/joint_states
pixi run ros2 topic echo /right/gripper/joint_states
```

During leader/follower teleop:
- manually open/close the left FR3 gripper
- verify `/left/gripper/joint_states` changes
- verify right follower gripper physically opens/closes and `/right/gripper/joint_states` changes

If left changes but right does not, stop and check:
- right gripper command topic exists: `/right/gripper/gripper_position_controller/commands`
- right gripper adapter is running in RT bringup (started by `load_gripper:=true`)

## 3) Recording test (`robofab_crisp`)

```bash
pixi run record-leader-follower-fr3 -- \
  --repo-id local/fr3_leader_follower \
  --tasks "pick and place the object" \
  --num-episodes 1
```

The wrapper enforces:
- `--leader-config fr3_left_leader`
- `--leader-namespace left`
- `--follower-config fr3_right_leader_follower_recording`
- `--follower-namespace right`
- `--recording-manager-type keyboard`
- `--fps 15`
- `--no-push-to-hub`

Implementation note:
- This wrapper uses a local patched recorder entrypoint to avoid noisy ROS spin-thread traceback during shutdown in `recording-manager-type ros` flows.

Keyboard recording controls:
- `r`: start/stop episode
- `s`: save episode
- `d`: delete episode
- `q`: quit

## 3b) Recording with FR3 pilot buttons (leader as recording device)

This mode uses ROS recording manager transitions (`record_transition`) and maps leader pilot buttons to recording actions.

### A) Start button bridge (`franka_buttons_ros2`)

In `/home/hex/Documents/github/playground/understand_crisp/franka_buttons_ros2`:

```bash
cp -i .env.template .env
# edit Desk credentials inside .env
docker compose up franka_buttons
```

This should run `franka_pilot_buttons` + `franka_buttons_to_record` and publish to `record_transition`.

### B) Start leader/follower recording in ROS mode (`robofab_crisp`)

```bash
pixi run record-leader-follower-fr3-buttons -- \
  --repo-id local/fr3_leader_follower_buttons \
  --tasks "pick and place the object" \
  --num-episodes 1
```

### C) Keyboard fallback for quit

Keep this in a separate terminal to send `exit` from keyboard (`q`) if needed:

```bash
pixi run record-transition-keyboard
```

Button mapping (expected):
- `circle`: record start/stop
- `check`: save episode
- `cross`: delete episode

Notes:
- FR3 arrow keys (`up/down/left/right`) may not be available while FCI is enabled.
- Therefore, keep keyboard fallback for quit.

## 3c) Post-recording gripper sanity check

After recording 1-2 episodes, inspect gripper-related columns:

```bash
pixi run episode-stats -- \
  --repo-id local/fr3_leader_follower_buttons \
  --column action \
  --column observation.state.gripper
```

Expected:
- `action` last dimension changes when you operate leader gripper
- `observation.state.gripper` changes over the episode

Important convention:
- In `crisp_gym`, `observation.state.gripper` is stored as `1 - gripper.value`.
- So for alignment checks against `action[..., -1]`, compare against `1 - observation.state.gripper`.

Run automated alignment check:

```bash
pixi run gripper-align-check -- \
  --repo-id local/fr3_leader_follower_buttons \
  --episode 0
```

The checker reports best correlation over a lag window and returns `PASS/WARN/FAIL`.

## 4) Safety checklist before moving both robots

- Keep both workspaces clear and non-overlapping.
- Start with small leader motions and verify follower direction/scale.
- Confirm both grippers report state:
  - `/left/gripper/joint_states`
  - `/right/gripper/joint_states`
- If either side behaves oddly, stop teleop and check active controllers per namespace.
