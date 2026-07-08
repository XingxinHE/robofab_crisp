from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deployment.act.deploy_policy import install_gripper_state_clamp


class DummyEnv:
    def __init__(self) -> None:
        self.calls = 0

    def get_obs(self) -> dict:
        self.calls += 1
        return {
            "observation.state.cartesian": np.ones(6, dtype=np.float32),
            "observation.state.gripper": np.float32(5.0e-4),
            "observation.state.joints": np.arange(7, dtype=np.float32),
            "observation.state": np.concatenate(
                [
                    np.ones(6, dtype=np.float32),
                    np.array([5.0e-4], dtype=np.float32),
                    np.arange(7, dtype=np.float32),
                ]
            ),
        }


def test_install_gripper_state_clamp_sets_split_and_concatenated_state() -> None:
    env = DummyEnv()

    install_gripper_state_clamp(env, value=0.0)
    obs = env.get_obs()

    assert env.calls == 1
    assert obs["observation.state.gripper"] == np.float32(0.0)
    assert obs["observation.state"][6] == np.float32(0.0)
    assert np.allclose(obs["observation.state.cartesian"], np.ones(6, dtype=np.float32))
    assert np.allclose(obs["observation.state"][7:], np.arange(7, dtype=np.float32))
