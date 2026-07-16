#!/usr/bin/env bash

# Pixi does not auto-load .env files, so load the workspace-local one here.
script_dir="${BASH_SOURCE[0]%/*}"
project_root="${script_dir%/*}"
env_file="$project_root/.env"

[ -f "$env_file" ] && . "$env_file"

# Safe local default. For multi-machine DDS, set this in .env to the LAN/Wi-Fi NIC.
export ROS_NETWORK_INTERFACE="${ROS_NETWORK_INTERFACE:-lo}"


# Keep ROS logs in a writable directory.
if [ -z "${ROS_LOG_DIR:-}" ]; then
  export ROS_LOG_DIR="${TMPDIR:-/tmp}/ros-log"
fi

mkdir -p "${ROS_LOG_DIR}"
