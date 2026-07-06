from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crisp_gym.util.gripper_mode import GripperMode  # noqa: E402
from deployment.gr00t.gripper_postprocessing import (  # noqa: E402
    GripperPostprocessConfig,
    GripperPostprocessor,
)
from deployment.gr00t.gr00t_1p7_remote_joint_policy import (  # noqa: E402
    Gr00t1p7RemoteJointPolicy,
)


def test_gripper_postprocessor_preserves_raw_default() -> None:
    processor = GripperPostprocessor()

    assert processor.process(0.2) == 0.2
    assert processor.process(0.2, force_binary=True) == 0.0
    assert processor.process(0.8, force_binary=True) == 1.0


def test_gripper_postprocessor_flip_before_threshold() -> None:
    processor = GripperPostprocessor(GripperPostprocessConfig(flip=True))

    assert processor.process(0.2, force_binary=True) == 1.0
    assert processor.process(0.8, force_binary=True) == 0.0


def test_gripper_postprocessor_hysteresis_holds_state_near_threshold() -> None:
    processor = GripperPostprocessor(
        GripperPostprocessConfig(threshold=0.5, hysteresis=0.1)
    )

    assert processor.process(0.6) == 1.0
    assert processor.process(0.45) == 1.0
    assert processor.process(0.39) == 0.0


def test_gripper_postprocessor_debounce_requires_consecutive_switches() -> None:
    processor = GripperPostprocessor(
        GripperPostprocessConfig(threshold=0.5, debounce_steps=2)
    )

    assert processor.process(0.2) == 0.0
    assert processor.process(0.9) == 0.0
    assert processor.process(0.9) == 1.0


def test_gripper_postprocessor_uncertain_band_policies() -> None:
    hold_processor = GripperPostprocessor(
        GripperPostprocessConfig(
            threshold=0.5,
            uncertain_band=0.1,
            uncertain_policy="hold",
        )
    )
    assert hold_processor.process(0.8) == 1.0
    assert hold_processor.process(0.52) == 1.0

    closed_processor = GripperPostprocessor(
        GripperPostprocessConfig(
            threshold=0.5,
            uncertain_band=0.1,
            uncertain_policy="closed",
        )
    )
    assert closed_processor.process(0.52) == 0.0


def _fake_joint_policy(
    *,
    target_joint: np.ndarray,
    gripper_position: float,
    max_joint_step_rad: float = 0.2,
) -> Gr00t1p7RemoteJointPolicy:
    policy = object.__new__(Gr00t1p7RemoteJointPolicy)
    policy.max_joint_step_rad = max_joint_step_rad
    policy.joint_target_smoothing_alpha = 1.0
    policy.joint_delta_smoothing_alpha = 1.0
    policy._last_smoothed_target_joint = None
    policy._last_smoothed_joint_delta = None
    policy.gripper_filter = GripperPostprocessor()
    policy._action_queue = deque(
        [
            {
                "target_joint": target_joint.astype(np.float32),
                "gripper_position": gripper_position,
            }
        ]
    )
    current_joint = np.zeros(7, dtype=np.float32)
    policy.env = SimpleNamespace(
        num_joints=7,
        robot=SimpleNamespace(target_joint=current_joint),
        config=SimpleNamespace(gripper_mode=GripperMode.ABSOLUTE_CONTINUOUS),
        action_space=SimpleNamespace(
            shape=(8,),
            low=np.full(8, -10.0, dtype=np.float32),
            high=np.full(8, 10.0, dtype=np.float32),
        ),
    )
    return policy


def test_joint_policy_converts_absolute_target_to_clipped_delta() -> None:
    target_joint = np.array([0.1, -0.3, 0.05, 0.0, 0.4, -0.05, 0.2], dtype=np.float32)
    policy = _fake_joint_policy(target_joint=target_joint, gripper_position=0.7)

    action = policy._dequeue_action()

    np.testing.assert_allclose(
        action[:7],
        np.array([0.1, -0.2, 0.05, 0.0, 0.2, -0.05, 0.2], dtype=np.float32),
    )
    np.testing.assert_allclose(action[7], 0.7)
