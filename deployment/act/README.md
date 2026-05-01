# ACT Deployment Module

Canonical deploy entrypoints are in this folder:
- `deploy_profiled.sh`
- `preflight_fr3.py`
- `preflight_fr3_3cams.py`
- `preflight_fr3_3cams_gamepad.py`

Canonical tasks:
- `preflight-deploy-fr3`
- `preflight-deploy-fr3-3cams`
- `preflight-deploy-fr3-3cams-gamepad`
- `deploy-act-fr3`
- `deploy-act-fr3-3cams`
- `deploy-act-fr3-3cams-gamepad`

# Commands

Step 1: Preflight (verify robot is ready)
```
pixi run preflight-deploy-fr3-3cams
```

Step 2: Deploy
```
pixi run deploy-act-fr3-3cams-gamepad -- --repo-id local/with_tray_combined_fix_feat_deploy --num-episodes 5 --model-path outputs/train/2026-05-01/15-34-21_act/checkpoints/001000/pretrained_model
```

With a task/table-specific home:
```
pixi run deploy-act-fr3-3cams-gamepad -- \
  --repo-id local/with_tray_combined_fix_feat_deploy \
  --num-episodes 5 \
  --model-path outputs/train/2026-05-01/15-34-21_act/checkpoints/001000/pretrained_model \
  --home-config robots/fr3_root_home_year2.yaml \
  --after-teleop robots/fr3_root_home_year2.yaml
```
