# Diffusion Policy Deployment

Deploy classic LeRobot Diffusion Policy checkpoints with the same robofab homing and recording workflow used by ACT and Pi0 deployment.

```bash
pixi run deploy-dp-fr3-3cams-gamepad -- \
  --repo-id local/close_blender_lid_dp_deploy \
  --num-episodes 5 \
  --model-path outputs/train/<date>/<job>/checkpoints/<step>/pretrained_model \
  --task "close the blender lid" \
  --home-config robots/fr3_root_home_robocasa.yaml \
  --after-teleop robots/fr3_root_home_robocasa.yaml
```

Notes:

- This module supports classic `policy.type=diffusion` checkpoints only.
- `--task` is a dataset recording label. It is not passed into the policy.
- `--num-inference-steps` can override reverse diffusion steps at deployment time.
- `--home-config` controls the start and per-episode home pose.
- `--after-teleop` controls the final home after all deployment episodes. If omitted, deployment falls back to `--home-config`.
- If `--model-path` is omitted, the wrapper picks the most recent local `outputs/train/**/pretrained_model` whose checkpoint type is `diffusion`.
