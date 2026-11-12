#!/bin/bash
# 启动 vision 节点（前台阻塞；无 ROS 环境依赖之外的额外脚本）。
#
# 用法：
#   ./scripts/start_vision.sh [launch 参数...]
#
# 默认读取 install/vision/share/vision/config/vision.yaml 中的模型路径；
# NO_CUDA 构建必须用 .onnx 模型，例如：
#   ./scripts/start_vision.sh sim:=true save_data:=false save_depth:=false \
#     detection_model_path:="$PWD/src/vision/model/sim_data_det_0126.onnx" \
#     segmentation_model_path:="$PWD/src/vision/model/sim_data_seg_0126.onnx"
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

exec ros2 launch vision launch.py "$@"
