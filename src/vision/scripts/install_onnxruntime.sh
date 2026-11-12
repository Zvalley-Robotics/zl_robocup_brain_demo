#!/bin/bash
# 兼容入口：ONNX Runtime 安装已统一到仓库根的离线脚本。
# 实际实现见 third_party/onnxruntime/install_onnxruntime.sh（按架构选择 tarball）。
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/../../../third_party/onnxruntime/install_onnxruntime.sh" "$@"
