# 模型目录

本目录随仓库提供两个 TensorRT 序列化模型：

| 文件 | 用途 | 说明 |
|---|---|---|
| `det_261019_fp16.engine` | 目标检测 | 输入 640×640；类别见 `vision.yaml` 的 `detection_model.classnames` |
| `seg_261027_fp16.engine` | 场地线分割 | 输入 640×640；单类场地线模型（`field_line_class_id: -1`） |

## 重要限制

- **TensorRT engine 与硬件/版本绑定。** engine 由特定 TensorRT、CUDA 和 GPU 组合序列化
  得到，换 JetPack 版本、换 GPU 型号或换 TensorRT 版本后通常无法加载，需要重新生成。
- **仓库不包含 ONNX 模型。** `./build_no_cuda.sh` 只负责编译 ONNX Runtime 推理后端；
  运行 vision 时需通过 launch 参数提供你自己的 `.onnx` 文件：
  ```bash
  ./scripts/start_vision.sh save_data:=false \
    detection_model_path:=/abs/path/det.onnx \
    segmentation_model_path:=/abs/path/seg.onnx
  ```
- **模型权重的再分发许可需自行确认。** 本仓库只包含转换后的 engine；如果要在公开
  平台分发，请确认模型训练数据、权重和网络结构的许可证允许再分发。

## 重新生成 engine

在已安装 CUDA/TensorRT 的机器上先构建本仓库（`./build.sh`），然后：

```bash
cd /path/to/robocup_brain_demo
MODEL_TYPE=detect ./src/vision/scripts/model/build_engine.sh /path/to/det.pt
MODEL_TYPE=seg    ./src/vision/scripts/model/build_engine.sh /path/to/seg.pt
```

脚本使用 conda 环境准备 PyTorch/Ultralytics（默认环境名 `trt`，可用 `CONDA_ENV`
覆盖），依次执行 `.pt -> .wts -> .engine`。产物默认写入当前目录下的
`exported_model/`。

`scripts/model/requirements.txt` 是导出的 **linux-64** conda 环境清单；如果转换
脚本运行在 aarch64 上，请自行安装 `torch`、`ultralytics` 等依赖，而不是直接使用该文件。

## 输入预处理

模型输入为 640×640、BGR、letterbox 缩放；置信度和 NMS 阈值在
`src/vision/config/vision.yaml` 的 `detection_model` / `segmentation_model`
中配置，可在运行时用 ROS 参数覆盖。
