# training/pi0

Train a [PI0](https://www.physicalintelligence.company/blog/pi0) policy via lerobot.

PI0 = PaliGemma (vision+language) + Expert Gemma (action) with flow matching.
Heavier than ACT (~24 GB VRAM). Supports language-conditioned tasks.

## Quick start

```bash
# from repo root
python training/pi0/train.py --repo-id local/fr3_leader_follower_3cams_open_fix_feat

# smoke test (2k steps, saves at 1k)
python training/pi0/train.py --smoke

# prepare dataset first, then train
python training/pi0/train.py \
  --prepare-from local/fr3_leader_follower_3cams_open \
  --repo-id local/fr3_leader_follower_3cams_open_fix_feat

# finetune from pretrained PI0 checkpoint
python training/pi0/train.py --policy-path lerobot/pi0 --steps 100000

# forward extra args to lerobot
python training/pi0/train.py -- --policy.push_to_hub=true
```

## SuperPOD Finetuning


```bash
pixi run python training/pi0/train.py \
  --repo-id xingxin-he/pick-throw-paper-with-tray \
  --batch-size 32 \
  --steps 100000 \
  --save-freq 20000 \
  --log-freq 10 \
  --policy-path lerobot/pi0 \
  --attention-implementation fa2 \
  --gpu 0
```

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--repo-id` | `local/fr3_leader_follower_3cams_open_fix_feat` | LeRobot dataset id |
| `--prepare-from` | _(none)_ | Clone+fix source dataset before training |
| `--steps` | `50000` | Training steps |
| `--batch-size` | `8` | Batch size |
| `--save-freq` | `10000` | Checkpoint save frequency |
| `--log-freq` | `100` | Log frequency |
| `--eval-freq` | `0` | Eval frequency (0 = off) |
| `--policy-path` | _(none)_ | Pretrained path (e.g. `lerobot/pi0`); omit to train from scratch |
| `--lr` | `2.5e-5` | Learning rate |
| `--attention-implementation` | `eager` | `eager`, `fa2`, or `flex` |
| `--freeze-vision-encoder` | `true` | Freeze SigLIP vision tower |
| `--train-expert-only` | `false` | Freeze PaliGemma, train expert only |
| `--num-steps` | `10` | Flow matching denoising steps |
| `--gpu` | `0` | CUDA_VISIBLE_DEVICES |
| `--smoke` | `false` | 2000 steps, save every 1000 |

Anything after `--` is forwarded directly to `lerobot.scripts.train`.

## PI0 vs ACT

| | PI0 | ACT |
|---|---|---|
| Architecture | PaliGemma-3B + Expert Gemma-2B | ResNet-18 + Transformer |
| Loss | Flow matching (velocity MSE) | Reconstruction + KL |
| Chunk size | 50 | 100 |
| Image norm | IDENTITY (SigLIP internal) | MEAN_STD |
| Scheduler | CosineDecayWithWarmup (1k/30k) | None |
| Language input | Yes | No |
| VRAM | ~24 GB+ | ~8 GB |

## Requirements

Version pins in `pixi.toml`:

```toml
torch = { version = "==2.8.0", index = "https://download.pytorch.org/whl/cu128" }
torchvision = { version = "==0.23.0", index = "https://download.pytorch.org/whl/cu128" }
transformers = ">=4.50.3"
lerobot = { ..., extras = ["smolvla", "pi0"] }
```

- `torch 2.8.0 + cu128` — pinned for reproducibility.
- `transformers >=4.50.3` — lerobot's minimum; `train.py` includes a `__getattr__` shim to work around a bug in lerobot's PI0 code that accesses `GemmaForCausalLM.embed_tokens` / `.layers` / `.norm` directly (these live on the inner `GemmaModel` in all transformers versions).

## How it works

`train.py` wraps `python -m lerobot.scripts.train` with PI0 defaults:

1. Monkey-patches `GemmaForCausalLM.__getattr__` to delegate to the inner model
2. Optionally prepares dataset via `dataset/00_crisp_to_lerobot_compatible.py`
3. Builds the lerobot CLI command with `--policy.type=pi0` (or `--policy.path`)
4. Runs lerobot training via subprocess

Source: `references_crisp_source_code/lerobot/src/lerobot/policies/pi0/`
