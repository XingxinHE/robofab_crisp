pixi run train-act -- \
--repo-id local/CloseBlenderLid_LeRobot \
--batch-size 32



close blender lid
/home/hex/Documents/github/data-collection-yr2/robofab_crisp/outputs/train/2026-05-11/20-22-24_act/checkpoints/05000/pretrained_model

open blender lid
/home/hex/Documents/github/data-collection-yr2/robofab_crisp/outputs/train/2026-05-11/21-32-59_act/checkpoints/05000/pretrained_model

hf upload \
  xingxin-he/close-blender-lid-checkpoints \
  /home/hex/Documents/github/data-collection-yr2/robofab_crisp/outputs/train/2026-05-11/20-22-24_act/checkpoints/050000/pretrained_model \
  act_real_robot/v1_2026-05-12 \
  --repo-type model \
  --commit-message "Add ACT real-robot checkpoint v001"
