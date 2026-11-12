#!/bin/bash
# 安装 ONNX Runtime C++ 库到 /usr/local（vision 的 NO_CUDA 构建需要）。
#
# 按当前架构选择随包 tarball：
#   x86_64  -> onnxruntime-linux-x64-1.23.2.tgz
#   aarch64 -> onnxruntime-linux-aarch64-1.23.2.tgz
#
# 用法：./install_onnxruntime.sh
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
ARCH="$(uname -m)"

case "$ARCH" in
  x86_64)  PKG="onnxruntime-linux-x64-1.23.2" ;;
  aarch64) PKG="onnxruntime-linux-aarch64-1.23.2" ;;
  *) echo "[ERR] 不支持的架构: $ARCH" >&2; exit 1 ;;
esac

TARBALL="$HERE/$PKG.tgz"
[ -f "$TARBALL" ] || { echo "[ERR] 缺少 $TARBALL" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "解压 $PKG ..."
tar -xzf "$TARBALL" -C "$TMP"

echo "安装到 /usr/local ..."
sudo cp -r "$TMP/$PKG/include/"* /usr/local/include/
sudo cp -r "$TMP/$PKG/lib/"* /usr/local/lib/
sudo ldconfig

echo "[OK] 已安装："
ls -lh /usr/local/include/onnxruntime_cxx_api.h
ls -lh /usr/local/lib/libonnxruntime.so*
echo "现在可用 ./build_no_cuda.sh 构建（x86_64 仿真 / 无 CUDA 环境）。"
