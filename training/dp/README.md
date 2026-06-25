# training/dp

Train classic LeRobot Diffusion Policy (`policy.type=diffusion`) on CRISP/LeRobot datasets.

## Quick Start

```bash
pixi run train-dp -- \
  --repo-id local/CloseBlenderLid_LeRobot \
  --batch-size 32
```

Smoke run:

```bash
pixi run train-dp -- \
  --repo-id local/CloseBlenderLid_LeRobot \
  --batch-size 2 \
  --smoke
```

Prepare a fixed-feature dataset first, then train:

```bash
pixi run train-dp -- \
  --prepare-from local/fr3_gamepad_3cams_open \
  --repo-id local/fr3_gamepad_3cams_open_fix_feat \
  --batch-size 8
```

Forward additional LeRobot args after `--`:

```bash
pixi run train-dp -- \
  --repo-id local/CloseBlenderLid_LeRobot \
  --batch-size 32 \
  -- --policy.num_inference_steps=20
```

## Defaults

| Flag | Default | What it does |
|---|---:|---|
| `--steps` | `50000` | Training steps |
| `--batch-size` | `8` | Batch size |
| `--save-freq` | `10000` | Checkpoint save frequency |
| `--log-freq` | `100` | Log frequency |
| `--eval-freq` | `0` | Training-time eval frequency |
| `--n-obs-steps` | `2` | Observation history length |
| `--horizon` | `16` | Diffusion action horizon |
| `--n-action-steps` | `8` | Actions executed per generated chunk |
| `--crop-shape` | `84 84` | Image crop shape passed to LeRobot |
| `--num-inference-steps` | unset | Reverse diffusion steps at inference time |
| `--gpu` | `0` | `CUDA_VISIBLE_DEVICES` |

Notes:

- This module supports classic Diffusion Policy only, not `multi_task_dit`.
- LeRobot Diffusion Policy requires all image observations to have the same shape.
- `horizon` must be compatible with the U-Net downsampling factor from `down_dims`.
- The wrapper keeps ACT-style `--dataset.use_imagenet_stats=true` and `pyav` video backend defaults for existing CRISP datasets.
