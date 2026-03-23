from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf

from crisp_py.config.path import find_config
from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose


def get_description_name(robot_type: str) -> str:
    if robot_type in ["fr3", "panda"]:
        return "panda_description"
    if robot_type in ["iiwa", "iiwa14"]:
        return "iiwa14_description"
    return f"{robot_type}_description"


def should_add_gripper_to_config(robot_type: str) -> bool:
    return robot_type in ["fr3", "panda"]


robot_type: Literal["fr3", "panda", "iiwa14"] = os.getenv("CRISP_VISER_ROBOT", "fr3")  # type: ignore[assignment]
home_time = float(os.getenv("CRISP_VISER_HOME_TIME", "2.0"))
viser_port = int(os.getenv("CRISP_VISER_PORT", "8080"))
controller_config = "control/default_cartesian_impedance.yaml"

robot = make_robot(robot_type)
robot.wait_until_ready()

robot.config.time_to_home = home_time
robot.home()
start_pose = robot.end_effector_pose

robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
param_file = find_config(controller_config)
if param_file is None:
    raise FileNotFoundError(
        f"Could not find {controller_config} in CRISP config paths."
    )
robot.cartesian_controller_parameters_client.load_param_config(file_path=param_file)

server = viser.ViserServer(port=viser_port)

urdf = load_robot_description(get_description_name(robot_type))
viser_urdf = ViserUrdf(
    server,
    urdf_or_path=urdf,
    load_meshes=True,
    load_collision_meshes=False,
    collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
)

with server.gui.add_folder("Visibility"):
    show_meshes_cb = server.gui.add_checkbox("Show meshes", viser_urdf.show_visual)
    show_collision_meshes_cb = server.gui.add_checkbox(
        "Show collision meshes", viser_urdf.show_collision
    )


@show_meshes_cb.on_update
def _(_):
    viser_urdf.show_visual = show_meshes_cb.value


@show_collision_meshes_cb.on_update
def _(_):
    viser_urdf.show_collision = show_collision_meshes_cb.value


show_meshes_cb.visible = True
show_collision_meshes_cb.visible = False

actuation = (
    np.array([*robot.joint_values, 0.0])
    if should_add_gripper_to_config(robot_type)
    else np.array(robot.joint_values)
)
viser_urdf.update_cfg(actuation)

trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
server.scene.add_grid(
    "/grid",
    width=2,
    height=2,
    position=(
        0.0,
        0.0,
        trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
    ),
)

transform_handle = server.scene.add_transform_controls(
    "/end_effector_target",
    position=start_pose.position,
    wxyz=start_pose.orientation.as_quat(scalar_first=True),
    scale=0.3,
    line_width=3.0,
)


@transform_handle.on_update
def update_robot_target(handle: viser.TransformControlsEvent) -> None:
    rot = Rotation.from_quat(handle.target.wxyz, scalar_first=True)
    pose = Pose(position=handle.target.position, orientation=rot)
    robot.set_target(pose=pose)


print(f"Viser teleop is running at http://localhost:{viser_port}")

while True:
    actuation = (
        np.array([*robot.joint_values, 0.0])
        if should_add_gripper_to_config(robot_type)
        else np.array(robot.joint_values)
    )
    viser_urdf.update_cfg(actuation)
    time.sleep(0.01)