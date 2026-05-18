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
  --model-path /home/hex/Documents/github/playground/understand_crisp/gr00t/close_blender_lid/01_real_only_baseline/checkpoint-30000 \
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
  --repo-id local/CloseBlenderLid_gr00t_realonly_nocasackpt \
  --num-episodes 1 \
  --task "Close the lid blender by securely placing the lid on top." \
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
