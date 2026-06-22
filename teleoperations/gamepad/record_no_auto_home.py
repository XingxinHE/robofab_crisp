#!/usr/bin/env python3
"""Record LeRobot data with direct Xbox gamepad teleop and no automatic homing."""

from __future__ import annotations

from teleoperations.gamepad.record import RecordingLifecycle, main


NO_AUTO_HOME_LIFECYCLE = RecordingLifecycle(
    name="no_auto_home",
    home_on_startup=False,
    open_gripper_on_startup=False,
    reset_env_on_episode_start=True,
    open_gripper_on_episode_start=False,
    home_on_episode_end=False,
    open_gripper_on_episode_end=False,
    home_on_exit=False,
)


if __name__ == "__main__":
    raise SystemExit(main(lifecycle=NO_AUTO_HOME_LIFECYCLE))
