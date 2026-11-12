#!/bin/bash
# 启动 brain 节点（前台阻塞）。
#
# 用法：
#   ./scripts/start_brain.sh [launch 参数...]
#
# 示例：
#   实机单台：./scripts/start_brain.sh real_robot:=true team_id:=29 player_id:=1
#   仿真多机：./scripts/start_brain.sh robot_name:=robot_03 team_id:=29 player_id:=3
#
# 参数说明见 `ros2 launch brain launch.py --show-args` 或 README。
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
if [ ! -f "$ROS_SETUP" ]; then
  echo "[ERR] ROS 2 setup not found: $ROS_SETUP" >&2
  echo "      Set ROS_DISTRO or ROS_SETUP to your ROS 2 installation." >&2
  exit 1
fi
source "$ROS_SETUP"
source "$ROOT/install/setup.bash"

exec ros2 launch brain launch.py "$@"
