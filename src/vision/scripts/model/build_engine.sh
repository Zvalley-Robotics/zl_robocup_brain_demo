#!/bin/bash
# 将 Ultralytics .pt 权重转换为 TensorRT engine。
#
# 依赖：
#   - 已用 ./build.sh 构建好的本仓库（需要 CUDA/TensorRT，NO_CUDA 构建不含转换工具）
#   - conda（默认环境名 trt；可用 CONDA_ENV / CONDA_BASE 覆盖）
#
# 用法：
#   MODEL_TYPE=detect ./build_engine.sh path/to/model.pt
#   MODEL_TYPE=seg    ./build_engine.sh path/to/model-seg.pt
#
# 产物：$EXPORTED_MODEL_DIR/model.engine（默认 ./exported_model）
set -e

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
pt_file="$1"
model_type="${MODEL_TYPE:-detect}"

if [ -z "$pt_file" ] || [ ! -f "$pt_file" ]; then
    echo "用法: MODEL_TYPE=detect|seg $0 <model.pt>" >&2
    exit 1
fi

case "$model_type" in
    detect) exe=yolov8_det ;;
    seg)    exe=yolov8_seg ;;
    *) echo "不支持的 MODEL_TYPE: $model_type（可用 detect / seg）" >&2; exit 1 ;;
esac

real_pt_file=$(realpath "$pt_file")
exported_model_path="${EXPORTED_MODEL_DIR:-$PWD/exported_model}"
mkdir -p "$exported_model_path"

# conda 环境：默认找 $CONDA_BASE/etc/profile.d/conda.sh，再回退到常见安装位置。
CONDA_ENV="${CONDA_ENV:-trt}"
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
for candidate in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -n "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "[ERR] 找不到 conda，请设置 CONDA_BASE（conda 安装根目录）" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -q "^${CONDA_ENV}[[:space:]]"; then
    echo "Conda environment '${CONDA_ENV}' exists."
else
    echo "Creating conda environment '${CONDA_ENV}' from requirements.txt..."
    conda create -y -n "$CONDA_ENV" --file "$ROOT/src/vision/scripts/model/requirements.txt" \
        -c pytorch -c conda-forge
fi
conda activate "$CONDA_ENV"

echo "start wts converting"
python3 "$ROOT/src/vision/scripts/model/gen_wts.py" \
    -w "$real_pt_file" -o "$exported_model_path/model.wts" -t "$model_type"
echo "finish wts converting"

# shellcheck disable=SC1091
source "$ROOT/install/setup.bash"
ros2 run vision "$exe" -s "$exported_model_path/model.wts" \
    "$exported_model_path/model.engine" s
echo "engine written to $exported_model_path/model.engine"
