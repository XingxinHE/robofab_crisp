"""Pi0-specific LeRobot policy wrapper for CRISP deployment.

The upstream CRISP LeRobot wrapper is action-only and does not inject language
tasks into inference batches. Pi0 requires ``batch["task"]`` for every
``select_action`` call, including warmup, so deployment uses this isolated
wrapper instead of changing the generic ACT path.
"""

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


def apply_transformers_gemma_shim() -> None:
    """Apply the Gemma compatibility shim used by Pi0 training.

    lerobot's Pi0 code accesses attributes such as ``embed_tokens`` and
    ``layers`` directly on ``GemmaForCausalLM``. In recent transformers these
    live on the inner ``GemmaModel`` at ``self.model``.
    """
    try:
        import torch.nn as nn
        import transformers
    except Exception:  # noqa: BLE001
        logger.warning("Could not import transformers for Gemma shim.", exc_info=True)
        return

    gemma_cls = getattr(transformers, "GemmaForCausalLM", None)
    if gemma_cls is None or getattr(gemma_cls, "_robofab_pi0_compat", False):
        return

    nn_getattr = nn.Module.__getattr__

    def _compat(self: Any, name: str) -> Any:
        try:
            return nn_getattr(self, name)
        except AttributeError:
            model = self._modules.get("model")
            if model is not None and hasattr(model, name):
                return getattr(model, name)
            raise

    gemma_cls.__getattr__ = _compat
    gemma_cls._robofab_pi0_compat = True


def _prepare_pi0_obs(obs_raw: Observation, task: str) -> Observation:
    obs = dict(obs_raw)
    obs["observation.state"] = concatenate_state_features(obs)
    obs["task"] = [task]
    return obs


class Pi0LerobotPolicy:
    """LeRobot Pi0 policy with language task injection."""

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        task: str,
        overrides: dict[str, Any] | None = None,
        warmup_steps: int = 5,
        startup_timeout_sec: float = 900.0,
        action_timeout_sec: float = 120.0,
    ) -> None:
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.task = task
        self.action_timeout_sec = action_timeout_sec
        self.overrides = overrides if overrides is not None else {}

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env": env,
                "task": task,
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
                "Pi0 inference worker did not become ready within "
                f"{startup_timeout_sec:.1f}s."
            )

        message = self.parent_conn.recv()
        if not isinstance(message, dict) or message.get("type") != "ready":
            self.shutdown()
            raise RuntimeError(f"Pi0 inference worker failed to start: {message}")

    def make_data_fn(self) -> Callable[[], tuple[Observation, Action]]:
        def _fn() -> tuple[Observation, Action]:
            obs_raw = _prepare_pi0_obs(self.env.get_obs(), self.task)

            self.parent_conn.send(obs_raw)
            if not self.parent_conn.poll(self.action_timeout_sec):
                raise TimeoutError(
                    "Timed out waiting for Pi0 action after "
                    f"{self.action_timeout_sec:.1f}s."
                )

            message = self.parent_conn.recv()
            if isinstance(message, dict) and message.get("type") == "error":
                raise RuntimeError(message.get("message", "Pi0 inference failed."))

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
    task: str,
    overrides: dict[str, Any] | None = None,
    warmup_steps: int = 5,
) -> None:
    setup_logging()
    worker_logger = logging.getLogger(__name__)
    apply_transformers_gemma_shim()

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        worker_logger.warning(
            "[Pi0 Inference] Could not import LeRobot third-party plugin registry."
        )

    preprocessor = None
    postprocessor = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        worker_logger.info("[Pi0 Inference] Using device: %s", device)

        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
        if train_config.policy is None:
            raise ValueError(f"Policy config missing in {pretrained_path}.")
        if train_config.policy.type != "pi0":
            raise ValueError(
                f"Expected a pi0 checkpoint, got policy.type={train_config.policy.type!r}."
            )

        _check_dataset_metadata_if_available(train_config, env, worker_logger)

        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)

        for override_key, override_value in (overrides or {}).items():
            old_value = getattr(policy.config, override_key)
            worker_logger.warning(
                "[Pi0 Inference] Overriding policy config: %s = %s -> %s",
                override_key,
                old_value,
                override_value,
            )
            setattr(policy.config, override_key, override_value)

        if hasattr(policy.config, "device"):
            policy.config.device = str(device)

        policy.reset()
        policy.to(device).eval()
        worker_logger.info("[Pi0 Inference] Loaded policy from %s.", pretrained_path)

        if USE_LEROBOT_PROCESSORS and make_pre_post_processors is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=pretrained_path,
            )

        _warmup_policy(
            policy=policy,
            env=env,
            task=task,
            device=device,
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
                worker_logger.info("[Pi0 Inference] Resetting policy.")
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
        worker_logger.exception("[Pi0 Inference] Exception: %s", exc)
        try:
            conn.send({"type": "error", "message": repr(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()
        worker_logger.info("[Pi0 Inference] Worker shutting down.")


def _warmup_policy(
    policy: Any,
    env: ManipulatorBaseEnv,
    task: str,
    device: torch.device,
    preprocessor: Any,
    warmup_steps: int,
    worker_logger: logging.Logger,
) -> None:
    if warmup_steps <= 0:
        return

    warmup_obs_raw = _prepare_pi0_obs(env.observation_space.sample(), task)
    warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
    if preprocessor is not None:
        warmup_obs = preprocessor(warmup_obs)

    elapsed_list: list[float] = []
    worker_logger.info("[Pi0 Inference] Warming up policy for %d steps.", warmup_steps)
    with torch.inference_mode():
        for _ in range(warmup_steps):
            start = time.time()
            _ = policy.select_action(warmup_obs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_list.append(time.time() - start)

    worker_logger.info(
        "[Pi0 Inference] Warmup timing: avg=%.2fms, max=%.2fms, min=%.2fms",
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
            "[Pi0 Inference] Could not import CRISP metadata checker: %s", exc
        )
        return

    _check_dataset_metadata(train_config, env, worker_logger)
