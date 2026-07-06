"""Shared GR00T N1.7 gripper postprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


UncertainPolicy = Literal["none", "hold", "open", "closed"]


@dataclass(frozen=True)
class GripperPostprocessConfig:
    """Configuration for optional deploy-side gripper stabilization.

    Defaults preserve the raw GR00T output except for callers that explicitly ask
    for binary output, such as the Cartesian adapter's legacy thresholding.
    """

    flip: bool = False
    threshold: float = 0.5
    hysteresis: float = 0.0
    debounce_steps: int = 1
    uncertain_band: float = 0.0
    uncertain_policy: UncertainPolicy = "none"

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("gripper threshold must be in [0, 1]")
        if self.hysteresis < 0.0:
            raise ValueError("gripper hysteresis must be >= 0")
        if self.uncertain_band < 0.0:
            raise ValueError("gripper uncertain band must be >= 0")
        if self.debounce_steps < 1:
            raise ValueError("gripper debounce steps must be >= 1")
        if self.uncertain_policy not in {"none", "hold", "open", "closed"}:
            raise ValueError(
                "gripper uncertain policy must be one of: none, hold, open, closed"
            )

    @property
    def stabilizing(self) -> bool:
        return (
            self.hysteresis > 0.0
            or self.debounce_steps > 1
            or self.uncertain_policy != "none"
        )


class GripperPostprocessor:
    """Stateful postprocessor for GR00T gripper predictions."""

    def __init__(self, config: GripperPostprocessConfig | None = None) -> None:
        self.config = config or GripperPostprocessConfig()
        self.reset()

    def reset(self) -> None:
        self._state: bool | None = None
        self._pending_state: bool | None = None
        self._pending_count = 0

    @property
    def state(self) -> bool | None:
        return self._state

    def process(self, value: float, *, force_binary: bool = False) -> float:
        """Return a command target in [0, 1].

        When no stabilization is enabled, this preserves the raw continuous
        target unless ``force_binary`` is requested. When stabilization is
        enabled, output is binary so hysteresis/debounce has an unambiguous
        command to hold.
        """
        target = float(np.clip(value, 0.0, 1.0))
        if self.config.flip:
            target = 1.0 - target

        if not self.config.stabilizing:
            if force_binary:
                return 1.0 if target >= self.config.threshold else 0.0
            return target

        candidate = self._candidate_state(target)
        state = self._debounced_state(candidate)
        return 1.0 if state else 0.0

    def _candidate_state(self, target: float) -> bool:
        lower = self.config.threshold - self.config.uncertain_band
        upper = self.config.threshold + self.config.uncertain_band
        if self.config.uncertain_band > 0.0 and lower <= target <= upper:
            if self.config.uncertain_policy == "hold" and self._state is not None:
                return self._state
            if self.config.uncertain_policy == "open":
                return True
            if self.config.uncertain_policy == "closed":
                return False

        if self._state is None or self.config.hysteresis == 0.0:
            return target >= self.config.threshold

        if self._state:
            return target >= self.config.threshold - self.config.hysteresis
        return target > self.config.threshold + self.config.hysteresis

    def _debounced_state(self, candidate: bool) -> bool:
        if self._state is None:
            self._state = candidate
            self._pending_state = None
            self._pending_count = 0
            return self._state

        if candidate == self._state:
            self._pending_state = None
            self._pending_count = 0
            return self._state

        if self.config.debounce_steps <= 1:
            self._state = candidate
            self._pending_state = None
            self._pending_count = 0
            return self._state

        if self._pending_state == candidate:
            self._pending_count += 1
        else:
            self._pending_state = candidate
            self._pending_count = 1

        if self._pending_count >= self.config.debounce_steps:
            self._state = candidate
            self._pending_state = None
            self._pending_count = 0

        return self._state
