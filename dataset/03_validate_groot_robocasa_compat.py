from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import tyro


DEFAULT_CONVERTED_ROOT = Path(
    "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema_robocasa_like"
)
DEFAULT_ROBOCASA_ROOT = Path(
    "/data/robocasa/dataset/v1.0/pretrain/atomic/OpenMicrowave/20250819/lerobot"
)


@dataclass
class Args:
    converted_dataset_root: Path = DEFAULT_CONVERTED_ROOT
    robocasa_reference_root: Path = DEFAULT_ROBOCASA_ROOT
    isaac_groot_root: Path = Path(
        "/home/hex/Documents/github/playground/understand_crisp/robofab_robocasa/third_party/Isaac-GR00T"
    )
    data_config: str = "panda_omron"
    video_backend: str = "opencv"


def main(args: Args) -> None:
    if args.isaac_groot_root.exists():
        sys.path.insert(0, str(args.isaac_groot_root))

    from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    cfg = DATA_CONFIG_MAP[args.data_config]
    dataset_kwargs = {
        "modality_configs": cfg.modality_config(),
        "transforms": cfg.transform(),
        "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        "video_backend": args.video_backend,
        "filter_key": None,
    }

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

    print("GR00T compatibility PASSED")
    print(f"- converted dataset length: {len(datasets[0])}")
    print(f"- RoboCasa dataset length: {len(datasets[1])}")
    print(f"- mixture length: {len(mixture)}")
    print(f"- sample keys: {sorted(sample.keys())}")


if __name__ == "__main__":
    main(tyro.cli(Args))
