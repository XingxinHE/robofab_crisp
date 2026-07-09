from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deployment.act.deploy_policy import (
    install_gripper_state_clamp,
    make_policy_data_fn,
    override_action_gripper,
)


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


def test_override_action_gripper_returns_applied_copy() -> None:
    raw_action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.25], dtype=np.float32)

    applied_action = override_action_gripper(raw_action, value=1.0)

    assert applied_action[-1] == np.float32(1.0)
    assert raw_action[-1] == np.float32(0.25)


class DummyActionTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def squeeze(self, dim: int) -> "DummyActionTensor":
        assert dim == 0
        return DummyActionTensor(np.squeeze(self.array, axis=dim))

    def to(self, device: str) -> "DummyActionTensor":
        assert device == "cpu"
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class DummyConn:
    def __init__(self, action: np.ndarray) -> None:
        self.action = DummyActionTensor(action[None, :])
        self.sent = []

    def send(self, value: object) -> None:
        self.sent.append(value)

    def recv(self) -> DummyActionTensor:
        return self.action


class DummyPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.env = DummyEnvWithStep()
        self.parent_conn = DummyConn(action)

    def make_data_fn(self):  # noqa: ANN201
        raise AssertionError("override path should build an explicit data fn")


class DummyEnvWithStep(DummyEnv):
    def __init__(self) -> None:
        super().__init__()
        self.stepped_actions = []

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs["observation.state.gripper"] = np.array([5.0e-4], dtype=np.float32)
        return obs

    def step(self, action: np.ndarray, block: bool = False):  # noqa: ANN201
        self.stepped_actions.append((action.copy(), block))
        return {}, 0.0, False, False, {}


def test_make_policy_data_fn_records_and_steps_with_overridden_action() -> None:
    raw_action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.25], dtype=np.float32)
    policy = DummyPolicy(raw_action)

    data_fn = make_policy_data_fn(policy, override_gripper=1.0)
    _obs, applied_action = data_fn()

    assert policy.env.stepped_actions[0][0][-1] == np.float32(1.0)
    assert policy.env.stepped_actions[0][1] is False
    assert applied_action[-1] == np.float32(1.0)
    assert raw_action[-1] == np.float32(0.25)
    assert policy.parent_conn.sent[0]["observation.state"].shape == (14,)
