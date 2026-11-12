#!/bin/bash
# 构建 robocup_brain_demo（brain + vision + 消息接口）。
#
# 只构建本目录内的包：
#   src/brain、src/vision
#   src/interface/{game_controller_interface,vision_interface}
#   third_party/zl_interfaces/src/{robot_interfaces,robot_imu/yesense_interface}
#
# 实机（aarch64 + CUDA/TensorRT）：./build.sh
# 仿真 / 无 CUDA（x86_64）：      ./build_no_cuda.sh（需要 ONNX Runtime，
#                                 见 third_party/onnxruntime/install_onnxruntime.sh）
#
# 若当前 shell 已 source 过其他含同名消息包的工作区，可通过
# STRIP_PREFIX_PATTERNS 剔除对应前缀（见下方说明）。
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
if [ ! -f "$ROS_SETUP" ]; then
  echo "[ERR] ROS 2 setup not found: $ROS_SETUP" >&2
  echo "      Set ROS_DISTRO or ROS_SETUP to your ROS 2 installation." >&2
  exit 1
fi
source "$ROS_SETUP"

# 可选：剔除当前 shell 中其他工作区的 install 前缀，避免同名消息包混入。
# 取值是冒号分隔的 shell 通配符，例如：
#   export STRIP_PREFIX_PATTERNS='*/old_ws/install*:/data/other_ws/install*'
STRIP_PREFIX_PATTERNS="${STRIP_PREFIX_PATTERNS:-}"

_strip_prefix() {
  local var="$1" out="" p pat
  local -a parts patterns
  IFS=':' read -r -a parts <<< "${!var:-}"
  IFS=':' read -r -a patterns <<< "$STRIP_PREFIX_PATTERNS"
  for p in "${parts[@]}"; do
    for pat in "${patterns[@]}"; do
      [ -n "$pat" ] || continue
      case "$p" in
        $pat) continue 2 ;;
      esac
    done
    out="${out:+$out:}$p"
  done
  printf -v "$var" '%s' "$out"
  export "$var"
}
for v in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH PYTHONPATH; do
  _strip_prefix "$v"
done
unset -f _strip_prefix

export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-g -rdynamic")
if [ "${NO_CUDA_BUILD:-0}" = "1" ]; then
  CMAKE_ARGS+=(-DNO_CUDA=ON)
fi

colcon --log-base log build --symlink-install \
  --base-paths src third_party/zl_interfaces/src \
  --build-base build --install-base install \
  --cmake-args "${CMAKE_ARGS[@]}" \
  "$@"

echo "Build complete: source $ROOT/install/setup.bash"
