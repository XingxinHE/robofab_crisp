# CRISP to RoboCasa Conversion Details

This document audits the conversion from the real FR3 CRISP LeRobot dataset to the RoboCasa-like LeRobot dataset used for Isaac GR00T co-training.

Primary converter:

```sh
dataset/01_convert_crisp_to_robocasa.py
```

Core implementation:

```sh
dataset/crisp_to_robocasa.py
```

Default source and output datasets:

```sh
/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema
/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema_robocasa_like
```

## Compatibility Scope

The goal is structural compatibility with RoboCasa `PandaOmron` datasets and GR00T's `panda_omron` data config. This includes state/action dimensions, feature names, modality slices, annotation columns, video metadata, parquet metadata, and loader construction.

This does not prove every semantic choice is ideal for learning. GR00T will not automatically detect a wrong physical convention if the tensor shape and modality metadata still match. Important semantic assumptions are called out below.

Evidence:

```sh
robofab_robocasa/third_party/Isaac-GR00T/gr00t/experiment/data_config.py:640
robofab_robocasa/third_party/Isaac-GR00T/gr00t/experiment/data_config.py:641
robofab_robocasa/third_party/Isaac-GR00T/gr00t/experiment/data_config.py:646
robofab_robocasa/third_party/Isaac-GR00T/gr00t/experiment/data_config.py:653
robofab_robocasa/third_party/Isaac-GR00T/gr00t/experiment/data_config.py:661
```

GR00T `panda_omron` expects:

| Modality | Keys |
| --- | --- |
| video | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand` |
| state | `end_effector_position_relative`, `end_effector_rotation_relative`, `gripper_qpos`, `base_position`, `base_rotation` |
| action | `end_effector_position`, `end_effector_rotation`, `gripper_close`, `base_motion`, `control_mode` |
| language | `annotation.human.task_description` |

## Q1: What the GR00T Smoke Test Proves

The code in `dataset/03_validate_groot_robocasa_compat.py` constructs two `LeRobotSingleDataset` objects and then one `LeRobotMixtureDataset`:

```py
datasets = [
    LeRobotSingleDataset(dataset_path=args.converted_dataset_root, **dataset_kwargs),
    LeRobotSingleDataset(dataset_path=args.robocasa_reference_root, **dataset_kwargs),
]
mixture = LeRobotMixtureDataset(
    data_mixture=[(dataset, 1.0) for dataset in datasets],
    mode="train",
    balance_dataset_weights=True,
    balance_trajectory_weights=True,
    seed=42,
    metadata_config={"percentile_mixing_method": "weighted_average"},
)
sample = mixture[0]
```

If the converted dataset is structurally incompatible with the RoboCasa reference, this code should usually fail. Examples:

| Incompatibility | Expected failure location |
| --- | --- |
| Dataset path missing | `LeRobotSingleDataset` construction |
| Missing `meta/info.json` | `LeRobotSingleDataset` metadata load |
| Missing modality key or parquet column | `LeRobotSingleDataset` indexing / sample fetch |
| Video feature missing `height`, `width`, `channel`, or fps | `LeRobotSingleDataset` video metadata simplification |
| Converted and RoboCasa video modality metadata differ, such as fps mismatch | `LeRobotMixtureDataset` metadata merge |
| State/action modality slices differ | `LeRobotMixtureDataset` metadata merge |
| Video codec unreadable by selected backend | `sample = mixture[0]` |

Code evidence:

```sh
robofab_robocasa/third_party/Isaac-GR00T/gr00t/data/dataset.py:350
robofab_robocasa/third_party/Isaac-GR00T/gr00t/data/dataset.py:361
robofab_robocasa/third_party/Isaac-GR00T/gr00t/data/dataset.py:365
robofab_robocasa/third_party/Isaac-GR00T/gr00t/data/dataset.py:372
robofab_robocasa/third_party/Isaac-GR00T/gr00t/data/dataset.py:1247
robofab_robocasa/third_party/Isaac-GR00T/gr00t/data/dataset.py:1274
```

Important limitation: this smoke test does not prove semantic correctness. It will not necessarily fail if, for example, `gripper_qpos` has the wrong sign convention but remains a 2D float vector, or if the real action scale is physically different from RoboCasa but remains a 12D vector.

## Source CRISP Schema

The converter assumes the CRISP dataset has:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `observation.state` | 20 | CRISP Cartesian state, gripper, joints, target |
| `action` | 7 | Cartesian delta action and gripper target |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:16
dataset/crisp_to_robocasa.py:22
dataset/crisp_to_robocasa.py:63
dataset/crisp_to_robocasa.py:71
```

CRISP `observation.state` input layout:

| CRISP idx | Meaning |
| ---: | --- |
| 0 | x |
| 1 | y |
| 2 | z |
| 3 | roll |
| 4 | pitch |
| 5 | yaw |
| 6 | gripper closedness |
| 7 | joint_0 |
| 8 | joint_1 |
| 9 | joint_2 |
| 10 | joint_3 |
| 11 | joint_4 |
| 12 | joint_5 |
| 13 | joint_6 |
| 14 | target_x |
| 15 | target_y |
| 16 | target_z |
| 17 | target_roll |
| 18 | target_pitch |
| 19 | target_yaw |

CRISP `action` input layout:

| CRISP idx | Meaning |
| ---: | --- |
| 0 | delta x |
| 1 | delta y |
| 2 | delta z |
| 3 | delta roll |
| 4 | delta pitch |
| 5 | delta yaw |
| 6 | gripper target |

CRISP action evidence:

```sh
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:639
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:643
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:653
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:668
robofab_crisp/teleoperations/gamepad/teleop.py:138
robofab_crisp/teleoperations/gamepad/teleop.py:146
```

## Output RoboCasa Schema

The converted output has:

| Field | Shape | dtype |
| --- | ---: | --- |
| `observation.state` | 16 | `float64` |
| `action` | 12 | `float64` |
| `annotation.human.task_description` | 1 | `int64` |
| `annotation.human.task_name` | 1 | `int64` |
| `next.reward` | 1 | `float32` |
| `next.done` | 1 | `bool` |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:47
dataset/crisp_to_robocasa.py:205
dataset/crisp_to_robocasa.py:224
dataset/crisp_to_robocasa.py:341
dataset/crisp_to_robocasa.py:371
dataset/crisp_to_robocasa.py:379
```

## Observation State Conversion

Output `observation.state` layout:

| Output idx | RoboCasa field | Source | Formula |
| ---: | --- | --- | --- |
| 0 | `base_position.x` | fixed base | `0.0` |
| 1 | `base_position.y` | fixed base | `0.0` |
| 2 | `base_position.z` | fixed base | `0.0` |
| 3 | `base_rotation.x` | fixed base | `0.0` |
| 4 | `base_rotation.y` | fixed base | `0.0` |
| 5 | `base_rotation.z` | fixed base | `0.0` |
| 6 | `base_rotation.w` | fixed base | `1.0` |
| 7 | `end_effector_position_relative.x` | CRISP state 0 | `x` |
| 8 | `end_effector_position_relative.y` | CRISP state 1 | `y` |
| 9 | `end_effector_position_relative.z` | CRISP state 2 | `z` |
| 10 | `end_effector_rotation_relative.x` | CRISP state 3:6 | `Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()[0]` |
| 11 | `end_effector_rotation_relative.y` | CRISP state 3:6 | `Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()[1]` |
| 12 | `end_effector_rotation_relative.z` | CRISP state 3:6 | `Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()[2]` |
| 13 | `end_effector_rotation_relative.w` | CRISP state 3:6 | `Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()[3]` |
| 14 | `gripper_qpos[0]` | CRISP state 6 | `(1.0 - gripper_closedness) * 0.08 / 2` |
| 15 | `gripper_qpos[1]` | CRISP state 6 | `-((1.0 - gripper_closedness) * 0.08 / 2)` |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:79
dataset/crisp_to_robocasa.py:89
dataset/crisp_to_robocasa.py:106
dataset/crisp_to_robocasa.py:128
dataset/crisp_to_robocasa.py:135
dataset/crisp_to_robocasa.py:139
dataset/crisp_to_robocasa.py:142
dataset/crisp_to_robocasa.py:143
dataset/crisp_to_robocasa.py:145
dataset/crisp_to_robocasa.py:146
dataset/crisp_to_robocasa.py:147
```

### Gripper State Convention

The converter treats CRISP gripper state as normalized closedness, not physical width and not open fraction.

CRISP evidence:

```sh
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:306
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:307
```

There, CRISP publishes:

```py
gripper_value = 1 - np.array([self.gripper.value])
```

Therefore:

| CRISP state gripper | Meaning | RoboCasa qpos |
| ---: | --- | --- |
| `0.0` | fully open | `[0.04, -0.04]` |
| `1.0` | fully closed | `[0.0, 0.0]` |

The converter clips normalized values to `[0, 1]` before multiplying by `max_width_m=0.08`.

## Action Conversion

Output `action` layout:

| Output idx | RoboCasa field | Source | Formula |
| ---: | --- | --- | --- |
| 0 | `base_motion.x` | fixed base | `0.0` |
| 1 | `base_motion.y` | fixed base | `0.0` |
| 2 | `base_motion.z` | fixed base | `0.0` |
| 3 | `base_motion.rotation` | fixed base | `0.0` |
| 4 | `control_mode` | fixed arm mode | `-1.0` |
| 5 | `end_effector_position.x` | CRISP action 0 | `dx` |
| 6 | `end_effector_position.y` | CRISP action 1 | `dy` |
| 7 | `end_effector_position.z` | CRISP action 2 | `dz` |
| 8 | `end_effector_rotation.x` | CRISP action 3:6 | `Rotation.from_euler("xyz", [droll, dpitch, dyaw]).as_rotvec()[0]` |
| 9 | `end_effector_rotation.y` | CRISP action 3:6 | `Rotation.from_euler("xyz", [droll, dpitch, dyaw]).as_rotvec()[1]` |
| 10 | `end_effector_rotation.z` | CRISP action 3:6 | `Rotation.from_euler("xyz", [droll, dpitch, dyaw]).as_rotvec()[2]` |
| 11 | `gripper_close` | CRISP action 6 | `+1.0 if gripper_action <= 0.5 else -1.0` |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:84
dataset/crisp_to_robocasa.py:116
dataset/crisp_to_robocasa.py:152
dataset/crisp_to_robocasa.py:159
dataset/crisp_to_robocasa.py:163
dataset/crisp_to_robocasa.py:166
dataset/crisp_to_robocasa.py:167
dataset/crisp_to_robocasa.py:168
dataset/crisp_to_robocasa.py:169
dataset/crisp_to_robocasa.py:170
dataset/crisp_to_robocasa.py:171
```

### Gripper Action Convention

The gamepad interface sets:

| Button | CRISP action value | Meaning |
| --- | ---: | --- |
| A | `0.0` | close |
| X | `1.0` | open |

Evidence:

```sh
robofab_crisp/teleoperations/gamepad/gamepad_6dof_interface.py:51
robofab_crisp/teleoperations/gamepad/gamepad_6dof_interface.py:91
robofab_crisp/teleoperations/gamepad/gamepad_6dof_interface.py:92
robofab_crisp/teleoperations/gamepad/gamepad_6dof_interface.py:93
robofab_crisp/teleoperations/gamepad/gamepad_6dof_interface.py:94
robofab_crisp/teleoperations/gamepad/gamepad_6dof_interface.py:95
robofab_crisp/teleoperations/gamepad/teleop.py:136
robofab_crisp/teleoperations/gamepad/teleop.py:146
```

CRISP environment evidence:

```sh
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:336
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:344
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:345
references_crisp_source_code/crisp_gym/crisp_gym/envs/manipulator_env.py:349
```

Therefore the converter maps:

| CRISP action gripper | RoboCasa `gripper_close` |
| ---: | ---: |
| `0.0` | `+1.0` |
| `1.0` | `-1.0` |

## Dropped CRISP-Only Columns

The source dataset contains redundant CRISP sub-state columns. The converted dataset drops them:

```sh
observation.state.cartesian
observation.state.gripper
observation.state.joints
observation.state.target
```

Code evidence:

```sh
dataset/crisp_to_robocasa.py:239
dataset/crisp_to_robocasa.py:349
dataset/crisp_to_robocasa.py:879
```

Reason: GR00T and RoboCasa consume the full `observation.state` vector plus modality slices. Keeping CRISP-only feature metadata risks state dimension confusion in downstream loaders.

## Metadata Conversion

The converter rewrites:

| Metadata | Converted value |
| --- | --- |
| `robot_type` | `PandaOmron` |
| `fps` | source `fps`, default `20` |
| video codec | `h264` |
| video pixel format | `yuv420p` |
| video names | `["height", "width", "channel"]` |
| `embodiment_tag` | `robocasa_panda_omron` |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:47
dataset/crisp_to_robocasa.py:266
dataset/crisp_to_robocasa.py:296
dataset/crisp_to_robocasa.py:303
dataset/crisp_to_robocasa.py:309
dataset/crisp_to_robocasa.py:325
dataset/crisp_to_robocasa.py:391
dataset/crisp_to_robocasa.py:397
dataset/crisp_to_robocasa.py:540
dataset/crisp_to_robocasa.py:549
```

## Video Conversion

The converter transcodes source videos to RoboCasa/GR00T-friendly H.264:

```sh
ffmpeg -y -hide_banner -loglevel error -i <src> -an -c:v libx264 -pix_fmt yuv420p -r 20 <dst>
```

Code evidence:

```sh
dataset/crisp_to_robocasa.py:707
dataset/crisp_to_robocasa.py:717
dataset/crisp_to_robocasa.py:744
dataset/crisp_to_robocasa.py:838
```

Observed output for the converted sample:

```sh
codec_name=h264
width=256
height=256
pix_fmt=yuv420p
avg_frame_rate=20/1
```

Reason: the original CRISP videos were AV1. GR00T with OpenCV backend can fail on AV1 while RoboCasa reference videos are H.264.

## Parquet Column Order and Schema Metadata

Converted parquet column order:

| Order | Column |
| ---: | --- |
| 0 | `annotation.human.task_description` |
| 1 | `annotation.human.task_name` |
| 2 | `observation.state` |
| 3 | `action` |
| 4 | `next.reward` |
| 5 | `next.done` |
| 6 | `timestamp` |
| 7 | `frame_index` |
| 8 | `episode_index` |
| 9 | `index` |
| 10 | `task_index` |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:247
dataset/crisp_to_robocasa.py:900
dataset/crisp_to_robocasa.py:912
dataset/crisp_to_robocasa.py:923
dataset/crisp_to_robocasa.py:930
dataset/crisp_to_robocasa.py:936
dataset/crisp_to_robocasa.py:962
```

The converter rebuilds HuggingFace parquet metadata from the converted feature set. It does not reuse the source parquet metadata, because the source metadata includes CRISP-only feature keys.

Code evidence:

```sh
dataset/crisp_to_robocasa.py:780
dataset/crisp_to_robocasa.py:787
dataset/crisp_to_robocasa.py:962
```

## Annotation and Episode Terminal Fields

Per frame:

| Field | Rule |
| --- | --- |
| `annotation.human.task_description` | copied from `task_index` |
| `annotation.human.task_name` | fixed to `1` by default |
| `next.reward` | `0.0`, except terminal frame |
| `next.done` | `False`, except terminal frame |

Per episode terminal row:

| Field | Value |
| --- | ---: |
| `next.done` | `True` |
| `next.reward` | `1.0` |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:176
dataset/crisp_to_robocasa.py:191
dataset/crisp_to_robocasa.py:197
dataset/crisp_to_robocasa.py:198
dataset/crisp_to_robocasa.py:199
dataset/crisp_to_robocasa.py:200
dataset/crisp_to_robocasa.py:425
dataset/crisp_to_robocasa.py:441
dataset/crisp_to_robocasa.py:886
dataset/crisp_to_robocasa.py:887
dataset/crisp_to_robocasa.py:888
```

For the converted sample, `tasks.jsonl` contains both:

```json
{"task_index": 0, "task": "open the microwave"}
{"task_index": 1, "task": "open the microwave"}
```

This keeps `annotation.human.task_name=1` resolvable if using GR00T's `panda_omron_task_name_lang` config.

## Modality Mapping

`meta/modality.json` slices the flat state/action vectors into GR00T modality keys:

| Modality key | Original key | Slice |
| --- | --- | --- |
| `state.base_position` | `observation.state` | `[0:3]` |
| `state.base_rotation` | `observation.state` | `[3:7]` |
| `state.end_effector_position_relative` | `observation.state` | `[7:10]` |
| `state.end_effector_rotation_relative` | `observation.state` | `[10:14]` |
| `state.gripper_qpos` | `observation.state` | `[14:16]` |
| `action.base_motion` | `action` | `[0:4]` |
| `action.control_mode` | `action` | `[4:5]` |
| `action.end_effector_position` | `action` | `[5:8]` |
| `action.end_effector_rotation` | `action` | `[8:11]` |
| `action.gripper_close` | `action` | `[11:12]` |
| `video.robot0_eye_in_hand` | `observation.images.robot0_eye_in_hand` | full video |
| `video.robot0_agentview_left` | `observation.images.robot0_agentview_left` | full video |
| `video.robot0_agentview_right` | `observation.images.robot0_agentview_right` | full video |
| `annotation.human.task_description` | `annotation.human.task_description` | scalar |

Code evidence:

```sh
dataset/crisp_to_robocasa.py:454
dataset/crisp_to_robocasa.py:457
dataset/crisp_to_robocasa.py:484
dataset/crisp_to_robocasa.py:511
dataset/crisp_to_robocasa.py:522
```

## Validation Scripts

Structural RoboCasa validation:

```sh
pixi run python dataset/02_validate_matched_robocasa.py \
  --robocasa-reference-root /data/robocasa/dataset/v1.0/pretrain/atomic/OpenMicrowave/20250819/lerobot
```

This checks state/action shape, dtype, video metadata, absence of CRISP-only metadata, parquet metadata, terminal fields, task index validity, and first-frame gripper qpos.

Code evidence:

```sh
dataset/02_validate_matched_robocasa.py:100
dataset/02_validate_matched_robocasa.py:109
dataset/02_validate_matched_robocasa.py:126
dataset/02_validate_matched_robocasa.py:144
dataset/02_validate_matched_robocasa.py:181
dataset/02_validate_matched_robocasa.py:191
dataset/02_validate_matched_robocasa.py:220
dataset/02_validate_matched_robocasa.py:241
dataset/02_validate_matched_robocasa.py:255
dataset/02_validate_matched_robocasa.py:259
```

GR00T loader validation:

```sh
uv run --group gr00t --extra cu128 python \
  /home/hex/Documents/github/playground/understand_crisp/robofab_crisp/dataset/03_validate_groot_robocasa_compat.py
```

Observed passing output:

```sh
GR00T compatibility PASSED
- converted dataset length: 117
- RoboCasa dataset length: 26017
- mixture length: 26134
- sample keys: ['action', 'action_mask', 'eagle_content', 'embodiment_id', 'has_real_action', 'segmentation_target', 'segmentation_target_mask', 'state', 'state_mask']
```

## Current Verified Sample Values

For the current converted sample:

| Check | Value |
| --- | --- |
| `robot_type` | `PandaOmron` |
| `fps` | `20` |
| first-frame `gripper_qpos` | `[0.03999494016170502, -0.03999494016170502]` |
| final `next.done` | `True` |
| final `next.reward` | `1.0` |
| task indices | `[0, 1]` |
| video codec | `h264` |
| video fps | `20/1` |

## Known Semantic Risks

These are not loader blockers, but they matter for co-training quality.

| Risk | Why it matters | Current status |
| --- | --- | --- |
| Action scale mismatch | RoboCasa actions are often normalized policy commands, while CRISP actions are physical delta commands. GR00T can normalize by stats, but cross-dataset scale still affects learning. | Not solved by structural conversion. Needs empirical check or action rescaling experiment. |
| Frame convention | We assume CRISP EE pose is already relative to the robot base, matching RoboCasa relative EE fields. | Implemented as direct passthrough. Should be verified with robot/base frame calibration. |
| Euler convention | We use SciPy `Rotation.from_euler("xyz", ...)`. | Matches the current CRISP orientation representation assumption. |
| Quaternion sign discontinuity | Quaternions can flip sign for equivalent rotations. | GR00T converts rotations to `rotation_6d`, so equivalent sign flips should not be a structural issue. |
| Task semantics | The real sample says "open the microwave"; the user-provided RoboCasa target path is `TurnOnMicrowave`. | For semantic co-training, prefer RoboCasa `OpenMicrowave` as the reference/mix partner. |

## Reproduction Commands

Regenerate converted dataset:

```sh
cd /home/hex/Documents/github/playground/understand_crisp/robofab_crisp
pixi run python dataset/01_convert_crisp_to_robocasa.py
```

Run focused tests:

```sh
cd /home/hex/Documents/github/playground/understand_crisp/robofab_crisp
pixi run pytest tests/dataset/test_crisp_to_robocasa.py \
  tests/dataset/test_crisp_to_robocasa_modality_and_annotation.py -q
```

Run structural validation:

```sh
cd /home/hex/Documents/github/playground/understand_crisp/robofab_crisp
pixi run python dataset/02_validate_matched_robocasa.py \
  --robocasa-reference-root /data/robocasa/dataset/v1.0/pretrain/atomic/OpenMicrowave/20250819/lerobot
```

Run GR00T compatibility validation:

```sh
cd /home/hex/Documents/github/playground/understand_crisp/robofab_robocasa
uv run --group gr00t --extra cu128 python \
  /home/hex/Documents/github/playground/understand_crisp/robofab_crisp/dataset/03_validate_groot_robocasa_compat.py
```
