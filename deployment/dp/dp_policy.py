"""Classic LeRobot Diffusion Policy wrapper for CRISP deployment."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.util.lerobot_features import (
    concatenate_state_features,
    numpy_obs_to_torch,
)
from crisp_gym.util.setup_logger import setup_logging
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class

try:
    from lerobot.policies.factory import make_pre_post_processors

    USE_LEROBOT_PROCESSORS = True
except ImportError:
    make_pre_post_processors = None  # type: ignore[assignment]
    USE_LEROBOT_PROCESSORS = False


logger = logging.getLogger(__name__)
Observation = dict[str, Any]
Action = np.ndarray


def _prepare_diffusion_obs(obs_raw: Observation) -> Observation:
    obs = dict(obs_raw)
    obs["observation.state"] = concatenate_state_features(obs)
    return obs


class DiffusionLerobotPolicy:
    """LeRobot Diffusion Policy adapter with subprocess inference."""

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict[str, Any] | None = None,
        warmup_steps: int = 5,
        startup_timeout_sec: float = 120.0,
        action_timeout_sec: float = 30.0,
    ) -> None:
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.action_timeout_sec = action_timeout_sec
        self.overrides = overrides if overrides is not None else {}

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env": env,
                "overrides": self.overrides,
                "warmup_steps": warmup_steps,
            },
            daemon=True,
        )
        self.inf_proc.start()
        self._wait_until_ready(startup_timeout_sec)

    def _wait_until_ready(self, startup_timeout_sec: float) -> None:
        if not self.parent_conn.poll(startup_timeout_sec):
            self.inf_proc.terminate()
            self.inf_proc.join(timeout=5.0)
            raise TimeoutError(
                "Diffusion inference worker did not become ready within "
                f"{startup_timeout_sec:.1f}s."
            )

        message = self.parent_conn.recv()
        if not isinstance(message, dict) or message.get("type") != "ready":
            self.shutdown()
            raise RuntimeError(f"Diffusion inference worker failed to start: {message}")

    def make_data_fn(self) -> Callable[[], tuple[Observation, Action]]:
        def _fn() -> tuple[Observation, Action]:
            obs_raw = _prepare_diffusion_obs(self.env.get_obs())

            self.parent_conn.send(obs_raw)
            if not self.parent_conn.poll(self.action_timeout_sec):
                raise TimeoutError(
                    "Timed out waiting for Diffusion Policy action after "
                    f"{self.action_timeout_sec:.1f}s."
                )

            message = self.parent_conn.recv()
            if isinstance(message, dict) and message.get("type") == "error":
                raise RuntimeError(message.get("message", "Diffusion inference failed."))

            action = _action_to_numpy(message)
            try:
                self.env.step(action, block=False)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error during environment step: %s", exc)

            return obs_raw, action

        return _fn

    def reset(self) -> None:
        self.parent_conn.send("reset")

    def shutdown(self) -> None:
        try:
            if self.inf_proc.is_alive():
                self.parent_conn.send(None)
        except Exception:  # noqa: BLE001
            pass

        self.inf_proc.join(timeout=10.0)
        if self.inf_proc.is_alive():
            self.inf_proc.terminate()
            self.inf_proc.join(timeout=5.0)


def _action_to_numpy(action: Any) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        return action.squeeze(0).detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).squeeze(0)


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
    overrides: dict[str, Any] | None = None,
    warmup_steps: int = 5,
) -> None:
    setup_logging()
    worker_logger = logging.getLogger(__name__)

    preprocessor = None
    postprocessor = None

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        worker_logger.warning(
            "[DP Inference] Could not import LeRobot third-party plugin registry."
        )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        worker_logger.info("[DP Inference] Using device: %s", device)

        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
        if train_config.policy is None:
            raise ValueError(f"Policy config missing in {pretrained_path}.")
        if train_config.policy.type != "diffusion":
            raise ValueError(
                "Expected a diffusion checkpoint, got "
                f"policy.type={train_config.policy.type!r}."
            )

        _check_dataset_metadata_if_available(train_config, env, worker_logger)

        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)

        for override_key, override_value in (overrides or {}).items():
            if not hasattr(policy.config, override_key):
                raise ValueError(f"Unknown Diffusion Policy config override: {override_key}")
            old_value = getattr(policy.config, override_key)
            worker_logger.warning(
                "[DP Inference] Overriding policy config: %s = %s -> %s",
                override_key,
                old_value,
                override_value,
            )
            setattr(policy.config, override_key, override_value)

        if hasattr(policy.config, "device"):
            policy.config.device = str(device)

        policy.reset()
        policy.to(device).eval()
        worker_logger.info("[DP Inference] Loaded policy from %s.", pretrained_path)

        if USE_LEROBOT_PROCESSORS and make_pre_post_processors is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=pretrained_path,
            )

        _warmup_policy(
            policy=policy,
            env=env,
            preprocessor=preprocessor,
            warmup_steps=warmup_steps,
            worker_logger=worker_logger,
        )
        policy.reset()
        if preprocessor is not None:
            preprocessor.reset()
        if postprocessor is not None:
            postprocessor.reset()

        conn.send({"type": "ready"})

        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                worker_logger.info("[DP Inference] Resetting policy.")
                policy.reset()
                if preprocessor is not None:
                    preprocessor.reset()
                if postprocessor is not None:
                    postprocessor.reset()
                continue

            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                if preprocessor is not None:
                    obs = preprocessor(obs)
                action = policy.select_action(obs)
                if postprocessor is not None:
                    action = postprocessor(action)

            conn.send(
                action.detach().cpu() if isinstance(action, torch.Tensor) else action
            )

    except Exception as exc:  # noqa: BLE001
        worker_logger.exception("[DP Inference] Exception: %s", exc)
        try:
            conn.send({"type": "error", "message": repr(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()
        worker_logger.info("[DP Inference] Worker shutting down.")


def _warmup_policy(
    policy: Any,
    env: ManipulatorBaseEnv,
    preprocessor: Any,
    warmup_steps: int,
    worker_logger: logging.Logger,
) -> None:
    if warmup_steps <= 0:
        return

    warmup_obs_raw = _prepare_diffusion_obs(env.observation_space.sample())
    warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
    if preprocessor is not None:
        warmup_obs = preprocessor(warmup_obs)

    elapsed_list: list[float] = []
    worker_logger.info("[DP Inference] Warming up policy for %d steps.", warmup_steps)
    with torch.inference_mode():
        for _ in range(warmup_steps):
            start = time.time()
            _ = policy.select_action(warmup_obs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_list.append(time.time() - start)

    policy.reset()
    worker_logger.info(
        "[DP Inference] Warmup timing: avg=%.2fms, max=%.2fms, min=%.2fms",
        float(np.mean(elapsed_list)) * 1000.0,
        max(elapsed_list) * 1000.0,
        min(elapsed_list) * 1000.0,
    )


def _check_dataset_metadata_if_available(
    train_config: TrainPipelineConfig,
    env: ManipulatorBaseEnv,
    worker_logger: logging.Logger,
) -> None:
    try:
        from crisp_gym.policy.lerobot_policy import _check_dataset_metadata
    except Exception as exc:  # noqa: BLE001
        worker_logger.warning(
            "[DP Inference] Could not import CRISP metadata checker: %s", exc
        )
        return

    _check_dataset_metadata(train_config, env, worker_logger)
