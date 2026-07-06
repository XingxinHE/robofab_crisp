# GR00T Deployment

Run GR00T in the `robofab_robocasa` UV environment and keep CRISP/ROS deployment in
the `robofab_crisp` pixi environment.

## Start the GR00T server with ZMQ

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_robocasa

uv run --group gr00t --extra cu128 python third_party/Isaac-GR00T/scripts/inference_service.py \
  --server \
  --host 127.0.0.1 \
  --port 5555 \
  --model-path /home/hex/Documents/github/playground/understand_crisp/gr00t/use_tool_turn_on_blender/01_real_only_baseline/checkpoint-30000 \
  --data-config panda_omron \
  --embodiment-tag new_embodiment \
  --denoising-steps 4
```

ZMQ is the recommended deployment transport because it matches Isaac-GR00T's
server-client path and avoids JSON serialization overhead for image payloads.

## Start the GR00T server with HTTP

HTTP is useful as a debugging fallback.

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_robocasa

uv run --group gr00t --extra cu128 python third_party/Isaac-GR00T/scripts/inference_service.py \
  --server \
  --http-server \
  --host 127.0.0.1 \
  --port 8000 \
  --model-path /home/hex/Documents/github/playground/understand_crisp/gr00t/close_blender_lid/01_real_only_baseline/checkpoint-30000 \
  --data-config panda_omron \
  --embodiment-tag new_embodiment \
  --denoising-steps 4
```

## Smoke-test the server

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_crisp
pixi run python -m deployment.gr00t.smoke_client \
  --groot-transport zmq \
  --groot-host 127.0.0.1 \
  --groot-port 5555
```

For HTTP:

```bash
pixi run python -m deployment.gr00t.smoke_client \
  --groot-transport http \
  --groot-server http://127.0.0.1:8000
```

## Deploy on CRISP

Start the FR3 and camera ROS nodes first, then:

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_crisp

pixi run deploy-gr00t-fr3-3cams-gamepad -- \
  --groot-transport zmq \
  --groot-host 127.0.0.1 \
  --groot-port 5555 \
  --repo-id local/UseToolTurnOnBlender_gr00t_realonly_casackpt_v2 \
  --num-episodes 10 \
  --task "Turn on the blender by grasping the green block to push the power button." \
  --fps 20 \
  --action-chunk-size 4 \
  --async-inference \
  --prefetch-threshold 2 \
  --log-timing \
  --timing-log-interval 25 \
  --home-config fr3_root_home_robocasa
```

Start with `--fps 5 --action-chunk-size 4 --async-inference`. If behavior is
stable, try `--fps 10 --action-chunk-size 8`. If the model output becomes too
open-loop or delayed, reduce the chunk size before increasing FPS.

## 4. GR00T N1.7 (DROID) zero-shot deployment

GR00T N1.7 uses a different input/output schema from the N1.5 deployment above.
The N1.7 DROID checkpoint expects two cameras, a 17-DOF state, and returns
absolute action targets.

> **Transport note:** The N1.7 server uses a `msgpack_numpy` ZMQ protocol that
> is incompatible with the legacy N1.5 torch-serialized ZMQ protocol. Use
> `--groot-transport zmq1p7` for N1.7. If you run Python directly instead of
> through `pixi run`, make sure `msgpack-numpy` is installed (it is added to
> `pixi.toml`).

### Start the GR00T N1.7 server

The server can run anywhere Isaac-GR00T N1.7 is installed with the correct
Python/CUDA dependencies. Two common options:

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
is already vendored as a third-party dependency):

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_robocasa

uv run --group gr00t --extra cu128 python third_party/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-DROID \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda
```

For the base model zero-shot test, use `--model-path nvidia/GR00T-N1.7-3B`.

### Deploy on CRISP

```bash
cd /home/hex/Documents/github/playground/understand_crisp/robofab_crisp

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

Start with `--fps 5 --action-chunk-size 1`. If behavior is stable, increase
`--action-chunk-size` before raising FPS.

If the DROID server produces stochastic gripper flips, keep the server unchanged
and enable deploy-side filtering first, for example:

```bash
--gripper-hysteresis 0.10 \
--gripper-debounce-steps 2 \
--gripper-uncertain-band 0.10 \
--gripper-uncertain-policy hold
```

> **Note:** The N1.7 DROID checkpoint only ingests two views
> (`exterior_image_1_left` and `wrist_image_left`). The CRISP right agent-view
> camera is ignored for this zero-shot test.

## Benchmark Inference Latency

```bash
pixi run python -m deployment.gr00t.smoke_client \
  --groot-transport zmq \
  --groot-host 127.0.0.1 \
  --groot-port 5555 \
  --warmup-requests 3 \
  --num-requests 20 \
  --no-print-action-shapes
```

If pure GR00T request latency is still around `0.5-0.8s`, the main deployment
fix is action chunking plus async prefetch. You can also benchmark lower server
denoising, for example `--denoising-steps 2`, but that changes policy quality.
