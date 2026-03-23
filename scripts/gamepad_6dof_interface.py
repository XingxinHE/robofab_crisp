#!/usr/bin/env python3
"""Reusable Xbox gamepad 6DoF interface for FR3 teleop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame


def _apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def _trigger_to_01(axis_value: float) -> float:
    return float(np.clip((axis_value + 1.0) * 0.5, 0.0, 1.0))


@dataclass
class Gamepad6DofConfig:
    controller_index: int = 0
    deadzone: float = 0.10
    linear_step: float = 0.003
    yaw_step: float = 0.03
    roll_pitch_step: float = 0.02
    fine_scale: float = 0.4
    enable_roll_pitch: bool = False


@dataclass
class GamepadCommand:
    dx: float
    dy: float
    dz: float
    roll: float
    pitch: float
    yaw: float
    gripper_target: float
    should_quit: bool
    sync_requested: bool
    coarse_mode: bool
    roll_pitch_enabled: bool


class XboxGamepad6Dof:
    def __init__(self, cfg: Gamepad6DofConfig):
        self.cfg = cfg
        self.joystick: pygame.joystick.Joystick | None = None
        self.gripper_target = 1.0
        self.coarse_mode = True
        self.roll_pitch_enabled = bool(cfg.enable_roll_pitch)

    def start(self) -> None:
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count <= self.cfg.controller_index:
            raise RuntimeError(
                f"No joystick at index {self.cfg.controller_index}. Detected count={count}."
            )

        self.joystick = pygame.joystick.Joystick(self.cfg.controller_index)
        self.joystick.init()

    def stop(self) -> None:
        if self.joystick is not None:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()

    def get_name(self) -> str:
        if self.joystick is None:
            return "<not connected>"
        return self.joystick.get_name()

    def poll(self) -> GamepadCommand:
        if self.joystick is None:
            raise RuntimeError("Gamepad not initialized. Call start() first.")

        should_quit = False
        sync_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            elif event.type == pygame.JOYBUTTONDOWN:
                # Xbox mapping from pygame: A=0, B=1, X=2, Y=3, LB=4, RB=5, Back=6, Start=7
                if event.button == 0:  # A -> close
                    self.gripper_target = 0.0
                elif event.button == 2:  # X -> open
                    self.gripper_target = 1.0
                elif event.button == 1:  # B -> quit
                    should_quit = True
                elif event.button == 3:  # Y -> sync
                    sync_requested = True
                elif event.button == 7:  # Start -> coarse/fine toggle
                    self.coarse_mode = not self.coarse_mode
                elif event.button == 6:  # Back -> roll/pitch toggle
                    self.roll_pitch_enabled = not self.roll_pitch_enabled

        linear_step = self.cfg.linear_step
        yaw_step = self.cfg.yaw_step
        roll_pitch_step = self.cfg.roll_pitch_step
        if not self.coarse_mode:
            linear_step *= self.cfg.fine_scale
            yaw_step *= self.cfg.fine_scale
            roll_pitch_step *= self.cfg.fine_scale

        # Left stick -> XY
        left_x = _apply_deadzone(float(self.joystick.get_axis(0)), self.cfg.deadzone)
        left_y = _apply_deadzone(float(self.joystick.get_axis(1)), self.cfg.deadzone)

        # Triggers -> Z
        lt = _trigger_to_01(float(self.joystick.get_axis(2)))
        rt = _trigger_to_01(float(self.joystick.get_axis(5)))

        # Bumpers -> yaw
        lb = 1.0 if self.joystick.get_button(4) else 0.0
        rb = 1.0 if self.joystick.get_button(5) else 0.0

        # Right stick -> roll/pitch (optional)
        right_x = _apply_deadzone(float(self.joystick.get_axis(3)), self.cfg.deadzone)
        right_y = _apply_deadzone(float(self.joystick.get_axis(4)), self.cfg.deadzone)

        dx = -left_y * linear_step
        dy = -left_x * linear_step
        dz = (rt - lt) * linear_step

        if self.roll_pitch_enabled:
            roll = right_y * roll_pitch_step
            pitch = right_x * roll_pitch_step
        else:
            roll = 0.0
            pitch = 0.0

        yaw = (lb - rb) * yaw_step

        return GamepadCommand(
            dx=dx,
            dy=dy,
            dz=dz,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gripper_target=self.gripper_target,
            should_quit=should_quit,
            sync_requested=sync_requested,
            coarse_mode=self.coarse_mode,
            roll_pitch_enabled=self.roll_pitch_enabled,
        )
