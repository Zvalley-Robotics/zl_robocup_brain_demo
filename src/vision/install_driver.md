# 相机驱动安装

本仓库不包含相机驱动或 ROS 2 wrapper。vision 节点只订阅标准
`sensor_msgs/Image`、`sensor_msgs/CompressedImage` 和 `sensor_msgs/CameraInfo`，
因此任一能发布这些话题的相机方案都可以接入。下面是两种常见相机的安装提示。

## ZED

在 **JetPack 6.0** 的 Orin 设备上安装 ZED SDK：

```bash
wget https://download.stereolabs.com/zedsdk/5.0/l4t36.3/jetsons \
  -O ZED_SDK_Tegra_L4T36.3_v5.0.0.zstd.run
chmod +x ZED_SDK_Tegra_L4T36.3_v5.0.0.zstd.run
./ZED_SDK_Tegra_L4T36.3_v5.0.0.zstd.run -- silent skip_cuda  # 需要输入密码
```

ROS 2 wrapper 见 [stereolabs/zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper)。

## RealSense

**不要通过 apt 安装 librealsense**：apt 版本在部分设备上无法正确枚举相机。
请参考 [librealsense](https://github.com/IntelRealSense/librealsense)
的源码构建说明，以及
[realsense-ros](https://github.com/IntelRealSense/realsense-ros) 的 ROS 2 wrapper。

驱动就绪后，在 `src/vision/config/vision_local.yaml` 中填写实际话题名
（模板见 `src/vision/config/vision_local.yaml.example`）。
