#!/bin/bash
# 无 CUDA / 仿真构建：vision 使用 ONNX Runtime 推理（x86_64 开发机常用）。
#
# 依赖：ONNX Runtime C++ 库（third_party/onnxruntime/install_onnxruntime.sh）。
# 实机 aarch64 + TensorRT 请改用 ./build.sh。
set -e

cd "$(dirname "$0")"
NO_CUDA_BUILD=1 ./build.sh "$@"
