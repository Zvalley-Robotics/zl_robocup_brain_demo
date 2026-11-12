#!/usr/bin/env python3
"""vision 链路本机测试图像源（无真实相机环境的冒烟工具）

循环发布「图像 + 相机内参 + 头姿」三类输入，供 vision_node 在没有相机时
验证完整推理链路：图像回调 → 数据同步 → ONNX 模型加载/推理 →
/soccer/detection、/ball、/line_segments 输出。

配套用法（先 source ROS 与 robocup_brain_demo 工作区；sim:=false）：

  # 终端 1：图像源（png/jpg）
  python3 scripts/vision_test_feed.py --image /path/to/test.png

  # 终端 2：vision（cd 到 robocup_brain_demo 根目录；no_cuda 构建必须用 .onnx）
  cd robocup_brain_demo && source install/setup.bash
  ros2 launch vision launch.py sim:=false save_data:=false \
    color_topic:=/test/color_image intrin_topic:=/test/camera_info depth_topic:=/test/depth \
    detection_model_path:=$PWD/src/vision/model/sim_data_det_0126.onnx \
    segmentation_model_path:=$PWD/src/vision/model/sim_data_seg_0126.onnx

  # 终端 3：验证
  ros2 topic hz --no-daemon /soccer/detection
  ros2 topic echo --no-daemon --once /soccer/detection

参数说明：
  --image FILE        必填；png/jpg，循环发布
  --freq HZ           图像频率（默认 2.0；CPU 单线程推理较慢，不建议拉高）
  --color-topic T     默认 /test/color_image；话题名含 "compressed" 时改发
                      CompressedImage(JPEG)（对应 vision 的 compressed 分支）
  --info-topic T      默认 /test/camera_info
  --pose-topic T      默认 /robot/head_to_base（vision 与 brain 都订阅）
  --head-height M     头姿高度（默认 0.45，姿态取水平）
  --fx / --fy         焦距（像素）；缺省 fx=fy=0.47×宽（宽 FOV 近似）
  --no-info / --no-pose  调试用：关闭对应发布

注意：发布的是固定头姿；与仿真/bridge 同时运行时 /robot/head_to_base 会互相污染，
仅用于独立冒烟场景。
"""

import argparse
import io
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class VisionTestFeed(Node):
    def __init__(self, args):
        super().__init__("vision_test_feed")
        self.args = args

        rgb = np.asarray(PILImage.open(args.image).convert("RGB"))
        self.height, self.width = rgb.shape[:2]
        self.bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR
        self.jpeg_bytes = None
        if "compressed" in args.color_topic:
            buf = io.BytesIO()
            PILImage.fromarray(rgb).save(buf, format="JPEG", quality=90)
            self.jpeg_bytes = buf.getvalue()

        fx = args.fx if args.fx > 0 else float(round(self.width * 0.47))
        fy = args.fy if args.fy > 0 else fx
        self.fx, self.fy = fx, fy
        self.cx, self.cy = self.width / 2.0, self.height / 2.0

        if "compressed" in args.color_topic:
            self.color_pub = self.create_publisher(CompressedImage, args.color_topic, 1)
        else:
            self.color_pub = self.create_publisher(Image, args.color_topic, 1)
        self.info_pub = None
        if not args.no_info:
            self.info_pub = self.create_publisher(CameraInfo, args.info_topic, 1)
        self.pose_pub = None
        if not args.no_pose:
            self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 1)

        self.n_img = 0
        self.create_timer(1.0 / max(args.freq, 0.1), self.pub_image)
        if self.info_pub is not None:
            self.create_timer(1.0, self.pub_info)
        if self.pose_pub is not None:
            self.create_timer(0.02, self.pub_pose)

        self.get_logger().info(
            f"test feed: {args.image} {self.width}x{self.height} "
            f"color={args.color_topic} info={args.info_topic} pose={args.pose_topic} "
            f"freq={args.freq}Hz fx={self.fx:.1f} head_z={args.head_height}"
        )

    def pub_image(self):
        stamp = self.get_clock().now().to_msg()
        if self.jpeg_bytes is not None:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = self.jpeg_bytes
        else:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            msg.height = self.height
            msg.width = self.width
            msg.encoding = "bgr8"
            msg.is_bigendian = False
            msg.step = self.width * 3
            msg.data = self.bgr.tobytes()
        self.color_pub.publish(msg)
        self.n_img += 1
        if self.n_img == 1 or self.n_img % 50 == 0:
            self.get_logger().info(f"image frames published: {self.n_img}")

    def pub_info(self):
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(msg)

    def pub_pose(self):
        # 2026-11-11 起 /robot/head_to_base 为 geometry_msgs/PoseStamped，
        # vision 与 brain 都订阅该话题；话题名/类型不一致会导致收不到。
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = self.args.head_height
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)


def parse_args(argv):
    p = argparse.ArgumentParser(description="vision 链路测试图像源", add_help=True)
    p.add_argument("--image", required=True, help="循环发布的图像文件（png/jpg）")
    p.add_argument("--freq", type=float, default=2.0, help="图像频率 Hz（默认 2.0）")
    p.add_argument("--color-topic", default="/test/color_image")
    p.add_argument("--info-topic", default="/test/camera_info")
    p.add_argument("--pose-topic", default="/robot/head_to_base")
    p.add_argument("--head-height", type=float, default=0.45)
    p.add_argument("--fx", type=float, default=0.0)
    p.add_argument("--fy", type=float, default=0.0)
    p.add_argument("--no-info", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init(args=None)
    node = VisionTestFeed(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[vision_test_feed] stopped", flush=True)


if __name__ == "__main__":
    main()
