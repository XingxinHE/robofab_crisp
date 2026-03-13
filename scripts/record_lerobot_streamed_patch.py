#!/usr/bin/env python3
"""Compatibility launcher for CRISP streamed teleop recording.

This patches crisp_gym's streamed teleop function to use TeleopStreamedPose.last_gripper
instead of the non-existent leader.gripper attribute.
"""

from __future__ import annotations

import numpy as np

from crisp_gym.record import record_functions as record_functions_module


def _patched_make_teleop_streamer_fn(env, leader):
    """Create a teleop function for streamed pose + gripper leader topics."""
    prev_pose = leader.last_pose
    first_step = True

    def _fn():
        nonlocal prev_pose, first_step
        if first_step:
            first_step = False
            prev_pose = leader.last_pose
            return None, None

        pose = leader.last_pose
        action_pose = pose - prev_pose
        prev_pose = pose

        # Streamed leader exposes `last_gripper`; default to follower value until first message.
        try:
            leader_gripper = leader.last_gripper
        except Exception:  # noqa: BLE001
            leader_gripper = env.gripper.value if env.gripper is not None else 0.0

        follower_gripper = env.gripper.value if env.gripper is not None else 0.0
        gripper_action = record_functions_module._leader_gripper_to_action(
            leader_value=leader_gripper,
            follower_value=follower_gripper,
            control_mode=env.config.gripper_mode,
        )

        action_pose_vector = action_pose.to_array(env.config.orientation_representation)
        action = np.concatenate([action_pose_vector, [gripper_action]])
        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def main() -> None:
    # Patch module function before importing the script entrypoint.
    record_functions_module.make_teleop_streamer_fn = _patched_make_teleop_streamer_fn

    import crisp_gym.scripts.record_lerobot_format_leader_follower as entry

    # Entry module imported function by name; patch it as well.
    entry.make_teleop_streamer_fn = _patched_make_teleop_streamer_fn
    entry.main()


if __name__ == "__main__":
    main()
