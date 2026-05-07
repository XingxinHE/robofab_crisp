# Pi0 Deployment

Deploy Pi0 checkpoints with the same robofab homing arguments used by gamepad
teleoperation and ACT deployment.

```bash
pixi run deploy-pi0-fr3-3cams-gamepad -- \
  --repo-id local/with_tray_pi0_deploy \
  --num-episodes 5 \
  --model-path outputs/train/2026-05-03/15-21-46_pi0/checkpoints/000700/pretrained_model \
  --task "pick and throw the paper with the tray" \
  --home-config robots/fr3_root_home_year2.yaml \
  --after-teleop robots/fr3_root_home_year2.yaml
```

Notes:

- `--task` is passed into Pi0 inference as the language prompt. Keep it aligned
  with the task text used during training.
- `--home-config` controls the start and per-episode home pose.
- `--after-teleop` controls the final home after all deployment episodes. If it
  is omitted, deployment falls back to `--home-config`.
- If `--model-path` is omitted, the wrapper picks the most recent local
  `outputs/train/**/pretrained_model` whose `config.json` has `"type": "pi0"`.
