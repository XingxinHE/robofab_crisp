#!/usr/bin/env bash
set -euo pipefail

python -m lerobot.scripts.train \
  --dataset.repo_id=local/fr3_dualcam_streamed \
  --policy.type=act \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=50000 \
  --eval_freq=0 \
  --save_freq=10000 \
  --log_freq=100 \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=pyav
