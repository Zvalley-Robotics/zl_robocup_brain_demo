# robocup_brain_demo

RoboCup 人形足球机器人的 **决策（brain）** 与 **视觉（vision）** 工作区，基于 ROS 2 Humble。
仓库从完整比赛工作区中抽取为可独立构建的最小集合，适用于三类场景：

- 在仿真中运行 brain 的决策、行为树与定位；
- 单独运行 vision 节点，验证相机、模型和感知链路；
- 部署到 aarch64 + JetPack/TensorRT 实机。

本仓库**不包含**机器人底层驱动、运动控制固件或仿真器。brain 通过 `zl_io` 约定的话题
与本体/仿真桥交互；vision 通过标准 `sensor_msgs` 与相机驱动交互。

## 快速检查清单

```bash
# 1. 安装 ONNX Runtime 并构建（x86_64 / 无 CUDA）
./third_party/onnxruntime/install_onnxruntime.sh
./build_no_cuda.sh
source install/setup.bash

# 2. 跑纯算法单元测试（不需要 ROS 网络）
./build/brain/test_one_m_candidate_selection
./build/brain/test_split_localization_schedule

# 3. 确认 brain 能启动（没有本体数据时只会空转）
timeout 5 ros2 launch brain launch.py disable_com:=true
```

完整闭环还需要仿真器（见 5.4）或实机 zl 本体栈；视觉推理需要自备模型（见第 12 节）。

---

## 1. 架构与数据流

```
                         ┌──────────────────────────────┐
 GameController ──UDP──> │                              │
 /soccer/game_controller │           brain_node         │──> /cmd_vel
                         │  behavior tree @ 100 Hz      │──> /robot/input/intent
 /soccer/detection   ───>│  localization / strategy     │──> /robot/external/head_control
 /soccer/ball        ───>│  obstacle avoidance          │──> /soccer/robot_pose
 /robot/head_to_base ───>│                              │──> /soccer/ball_position
                         └──────────────────────────────┘
                                      ▲
                                      │ perception
                         ┌──────────────────────────────┐
 Camera ───────────────> │          vision_node         │──> /soccer/detection
 (color/depth/info)      │  TensorRT 或 ONNX 推理        │──> /soccer/ball
                         │  检测 + 分割 + 位姿估计        │──> /soccer/line_segments
                         └──────────────────────────────┘
```

- **brain**：100 Hz tick 行为树；订阅视觉检测、机器人状态、IMU、里程计和裁判信息；
  输出速度、意图、头部控制和调试可视化。
- **vision**：订阅图像/深度/相机内参和 `/robot/head_to_base` 头姿，输出球、检测框和
  场地线；模型推理后端在编译期二选一（TensorRT 或 ONNX Runtime）。
- **队内通信**：UDP 发现端口 `20000 + team_id`，单播端口
  `30000 + team_id + player_id`（`comm.multi_instance: true` 时每人独立端口）。
- **裁判通信**：UDP 3939 回传，目标地址由 `game_control_ip` 指定。

---

## 2. 目录结构

```
src/brain/                    # 决策节点、行为树、配置、定位算法、单元测试
  behavior_trees/game.xml     # 行为树入口（比赛状态机）
  behavior_trees/subtrees/    # 可复用的子树
  config/config.yaml          # 全机共用的基线参数
  config/config_local.yaml.example  # 现场覆盖层模板（复制为 config_local.yaml）
  launch/launch.py            # brain 启动入口
  src/robot_io/               # zl 原生 I/O、球门位姿合成、球类型转换
  test/                       # 纯算法单元测试（不需要 ROS 网络）

src/vision/                   # 视觉节点、配置、模型、第三方依赖
  config/vision.yaml          # 全机共用的基线参数
  config/vision_local.yaml.example  # 相机/标定覆盖层模板
  config/field.yaml           # 场地标志点坐标（adult_size）
  model/                      # TensorRT engine + 模型说明
  launch/launch.py            # vision 启动入口
  install_driver.md           # 相机驱动安装提示（ZED / RealSense）
  scripts/                    # ONNX Runtime 安装、engine 生成工具

src/interface/
  game_controller_interface/  # 裁判盒消息
  vision_interface/           # 视觉消息

third_party/
  zl_interfaces/src/          # robot_interfaces / yesense_interface 冻结快照
  onnxruntime/                # x86_64 / aarch64 ONNX Runtime 1.23.2 离线包

scripts/
  start_brain.sh              # brain 前台启动包装
  start_vision.sh             # vision 前台启动包装
  vision_test_feed.py         # 无相机时的测试图像源

build.sh                      # aarch64 + CUDA/TensorRT 构建
build_no_cuda.sh              # x86_64 / 仿真，ONNX Runtime 构建
```

---

## 3. 环境要求

### 硬件 / 系统组合

| 场景 | 平台 | 推理后端 |
|---|---|---|
| 仿真 / 桌面开发 | x86_64 Ubuntu 22.04 | ONNX Runtime（CPU） |
| 实机 | aarch64 Jetson + JetPack 6 | TensorRT（CUDA） |

ROS 2 版本：**Humble**。构建脚本默认读取
`/opt/ros/${ROS_DISTRO:-humble}/setup.bash`，也可用 `ROS_SETUP=/path/to/setup.bash` 覆盖。

### ROS 2 与系统依赖

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-behaviortree-cpp \
  ros-humble-backward-ros \
  ros-humble-image-transport \
  ros-humble-tf2-ros ros-humble-tf2-geometry-msgs \
  ros-humble-rosidl-default-generators ros-humble-rosidl-default-runtime \
  libopencv-dev libpcl-dev libyaml-cpp-dev libeigen3-dev libboost-all-dev \
  python3-colcon-common-extensions python3-numpy python3-pil python3-matplotlib
```

- `behaviortree_cpp` 使用 4.x（行为树 XML 声明 `BTCPP_format="4"`）。
- 实机额外需要 JetPack 自带的 CUDA、TensorRT 和 `nvcc`。
- x86 的 `NO_CUDA` 构建需要 ONNX Runtime C++ 库，见下一节。
- 相机驱动不在本仓库内，见 [src/vision/install_driver.md](src/vision/install_driver.md)。

---

## 4. 构建

### 4.1 x86_64 / 仿真（NO_CUDA + ONNX Runtime）

```bash
# 首次：把随包的 ONNX Runtime 安装到 /usr/local（需要 sudo）
./third_party/onnxruntime/install_onnxruntime.sh

./build_no_cuda.sh
source install/setup.bash
```

> `build_no_cuda.sh` 会编译 vision 的 ONNX 推理后端，但**仓库不附带 `.onnx` 模型**。
> 运行 vision 时需要自行提供 `.onnx` 文件并通过 launch 参数指定（见 5.2）。
> 只运行 brain 时不需要 ONNX 模型。

### 4.2 aarch64 / 实机（CUDA + TensorRT）

```bash
./build.sh
source install/setup.bash
```

默认路径：

- CUDA：`/usr/local/cuda`
- TensorRT（仅 x86_64）：`/usr/local/TensorRT-8.6.1.6`

非默认安装可用 CMake 变量覆盖：

```bash
colcon build --base-paths src third_party/zl_interfaces/src \
  --cmake-args -DCUDA_ROOT=/opt/cuda -DTENSORRT_ROOT=/opt/TensorRT
```

### 4.3 构建细节

- 构建采用 `--symlink-install`，修改 `src/*/config` 或 `behavior_trees` 后通常无需重新编译。
- 若当前 shell 还 source 过其他含同名消息包的工作区，可用
  `STRIP_PREFIX_PATTERNS` 剔除其前缀，避免同名包混入：
  ```bash
  export STRIP_PREFIX_PATTERNS='*/old_ws/install*:/data/other_ws/install*'
  ```
- 只重建某个包：
  ```bash
  colcon build --base-paths src third_party/zl_interfaces/src --packages-select brain
  ```

---

## 5. 运行

### 5.1 brain

```bash
# 实机单台
./scripts/start_brain.sh real_robot:=true team_id:=29 player_id:=1 role:=striker

# 仿真多机中的一台
./scripts/start_brain.sh robot_name:=robot_03 team_id:=29 player_id:=3

# 查看全部 launch 参数
ros2 launch brain launch.py --show-args
```

常用 launch 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `real_robot` | `false` | 实机安全配置：关闭真值注入、无话题后缀、`use_sim_time=false` |
| `robot_name` | 空 | 多机命名空间，如 `robot_03`；会设置 `zl_io.topic_ns=/robot_03`、`cmd_vel_topic=/robot_03/cmd_vel` |
| `team_id` / `player_id` / `role` | 空 | 覆盖 `config.yaml` 中的队伍、号码和角色 |
| `num_players` | `1` | 实机同队上场人数；多台同队时必须传实际人数 |
| `tree` | `game.xml` | 行为树文件名，位于 `src/brain/behavior_trees/` |
| `sim` | `false` | 仿真标记；注意 MuJoCo 不发布 `/clock`，此模式下仍关闭 `use_sim_time` |
| `sim_profile` | 空 | 可选仿真画像 YAML；不得与 `real_robot:=true` 同时使用 |
| `enable_auto_visual_kick` | `false` | 实机是否允许自动视觉踢球 |
| `enable_avoidance` | 空 | 显式 `true/false` 覆盖 `obstacle_avoidance.avoid_during_chase` |
| `goal_synth` | `true` | 实机是否发布真实球门位姿 |
| `disable_com` / `agent_mode` | `false` | 关闭队内通信 / APP 接管模式 |
| `odom_source` | `legacy` | `legacy` 或 `autoodom` |

### 5.2 vision

```bash
# 实机：使用 vision.yaml 中的 TensorRT engine
./scripts/start_vision.sh save_data:=false

# NO_CUDA / 仿真：必须提供 .onnx 模型（路径可为绝对路径）
./scripts/start_vision.sh sim:=false save_data:=false save_depth:=false \
  detection_model_path:=/abs/path/det.onnx \
  segmentation_model_path:=/abs/path/seg.onnx \
  color_topic:=/robot_03/camera/color/image_raw \
  depth_topic:=/robot_03/camera/depth/image_rect_raw \
  intrin_topic:=/robot_03/camera/color/camera_info
```

常用 launch 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `save_data` | `true` | 保存图像/位姿到 `$HOME/Workspace/vision_log/<时间戳>`；调试时建议 `false` |
| `save_depth` | `true` | 同时保存深度图（依赖 `save_data`） |
| `save_fps` | `2` | 每秒保存帧数 |
| `show_det` / `show_seg` | `false` | OpenCV 窗口显示；无显示器时用 `publish_det_img` |
| `publish_det_img` | `false` | 发布 `/soccer/debug/detection_img`（JPEG CompressedImage） |
| `sim` | `false` | 是否使用仿真时钟；MuJoCo 场景保持 `false` |
| `color_topic` / `depth_topic` / `intrin_topic` | 空 | 覆盖 `vision.yaml` 中的相机话题 |
| `detection_model_path` / `segmentation_model_path` | 空 | 覆盖模型路径，支持绝对路径 |
| `vision_config_path` | 空 | 自定义 `vision.yaml`/`vision_local.yaml` 所在目录 |

多机 vision 通过 `vision.yaml`（或 `vision_local.yaml`）里的 `robot_name` 设置话题后缀，
例如 `robot_name: "robot_03"` 会发布 `/soccer/detection/robot_03` 等话题。

> **实机标定**：aarch64 构建会额外生成 `calibration_node`（x86 构建不包含）。手眼/外参
> 标定可运行 `ros2 run vision calibration_node handeye <vision.yaml>` 并按提示操作；
> 标定结果默认询问是否写入 `/opt/robot/vision.yaml`，需要写权限。日志在
> `$HOME/Workspace/calibration_log/`。

### 5.3 无相机冒烟测试

`vision_test_feed.py` 循环发布图像、相机内参和固定头姿，适合验证完整推理链路：

```bash
# 终端 1：测试图像源
python3 scripts/vision_test_feed.py --image /path/to/test.png

# 终端 2：vision，订阅测试图像源
./scripts/start_vision.sh sim:=false save_data:=false \
  color_topic:=/test/color_image \
  intrin_topic:=/test/camera_info \
  depth_topic:=/test/depth \
  detection_model_path:=/abs/path/det.onnx \
  segmentation_model_path:=/abs/path/seg.onnx

# 终端 3：验证输出
ros2 topic hz /soccer/detection
ros2 topic echo --once /soccer/detection
```

> 测试源只发布图像、内参和头姿，不发布深度图；需要验证深度分支时，在
> `vision_local.yaml` 中覆盖 `use_depth` / `ball_pose_estimator.use_depth`，
> 或自行补充深度发布者。

### 5.4 与 robocup_sim 配合（可选）

本仓库不包含仿真器。若使用配套的 `robocup_sim` 工作区，把其 `inventory.yaml` 中的
`brain_ws` 指向本仓库：

```yaml
brain_ws: /path/to/robocup_brain_demo
```

然后由仿真 CLI 启动 brain：

```bash
cd /path/to/robocup_sim
./robocup_sim up --role brain
```

仿真默认用真值/桥接提供球和目标感知，不启动 vision。需要验证视觉时，按 5.2 单独启动
vision 节点，并把相机话题指向仿真桥实际发布的话题。仿真相机分辨率通常很低，仅适合
验证链路，模型效果请用实机图像或 `vision_test_feed.py`。

---

## 6. 配置体系

### 6.1 brain 参数分层

按优先级从低到高：

1. `src/brain/config/config.yaml` —— 全机共用基线；
2. `src/brain/config/config_local.yaml` —— 本机/现场覆盖层（已 gitignore）；
3. `~/agents/soccer/brain.yaml` —— 用户级覆盖层（可选）；
4. launch 内联参数 —— 最高优先级。

模板：`src/brain/config/config_local.yaml.example`。

```bash
cp src/brain/config/config_local.yaml.example src/brain/config/config_local.yaml
```

### 6.2 vision 参数分层

1. `src/vision/config/vision.yaml` —— 全机共用基线；
2. `src/vision/config/vision_local.yaml` —— 本机相机/标定覆盖层（已 gitignore）；
3. `~/agents/soccer/vision.yaml` —— 用户级覆盖层（可选）。

模板：`src/vision/config/vision_local.yaml.example`。

```bash
cp src/vision/config/vision_local.yaml.example src/vision/config/vision_local.yaml
```

`vision.yaml` 中的相机内外参只是**示例标定值**。换相机、换镜头或换安装后必须重新标定，
建议把结果只写进 `vision_local.yaml`。

### 6.3 常用参数入口

| 想修改 | 位置 |
|---|---|
| 队伍、号码、角色、上场人数 | `config.yaml` 的 `game.*`，或 launch 参数 |
| 速度上限、步态补偿、头部限位 | `config.yaml` 的 `robot.*` |
| 定位粒子数、漂移阈值、调度开关 | `config.yaml` 的 `locator.*` |
| 避障栅格、绕行距离 | `config.yaml` 的 `obstacle_avoidance.*` |
| 视觉踢球触发距离/角度 | `config.yaml` / `config_local.yaml` 的 `RLVisionKick.*` |
| 球记忆时间、队友球位共享 | `config.yaml` 的 `strategy.*` |
| 裁判机 IP | `config_local.yaml` 的 `game_control_ip` |
| 相机话题、内外参、模型路径 | `vision.yaml` 或 `vision_local.yaml` |
| 行为树结构 | `src/brain/behavior_trees/game.xml` 与 `subtrees/` |

核对运行中的实际参数（不要凭 YAML 推断）：

```bash
ros2 param get /brain_node game.team_id
ros2 param dump /brain_node
```

---

## 7. 话题与接口

### brain 主要输入

| 话题 | 类型 | 说明 |
|---|---|---|
| `/soccer/game_controller` | `game_controller_interface/GameControlData` | 裁判盒状态 |
| `/soccer/detection` | `vision_interface/Detections` | 视觉检测 |
| `/soccer/ball` | `vision_interface/Ball` | 视觉球位 |
| `/soccer/line_segments` | `vision_interface/LineSegments` | 场地线 |
| `/robot/head_to_base` | `geometry_msgs/PoseStamped` | 头部相对基座位姿 |
| `/robot/state` | `robot_interfaces/RobotState` | 本体状态 |
| `/robot/driver/joint_states` | `robot_interfaces/JointStates` | 关节状态 |
| `/robot/imu/nav_all` | `yesense_interface/NavAll` | IMU/姿态 |
| `/robot/odometry/odom` | `nav_msgs/Odometry` | 里程计 |
| `/soccer/ground_truth_pose` | `geometry_msgs/Pose2D` | 可选真值位姿 |
| `/soccer/ground_truth_ball` | `geometry_msgs/Point` | 可选真值球位 |

### brain 主要输出

| 话题 | 类型 | 说明 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | 速度指令（BEST_EFFORT） |
| `/robot/input/intent` | `robot_interfaces/RobotIntent` | 意图/模式切换（RELIABLE + TRANSIENT_LOCAL） |
| `/robot/external/head_control` | `robot_interfaces/HeadControl` | 头部控制 |
| `/soccer/robot_pose` | `geometry_msgs/Pose2D` | 定位输出 |
| `/soccer/ball_position` | `geometry_msgs/Point` | 球位估计 |
| `/soccer/teammates_poses` | `std_msgs/Float64MultiArray` | 队友位姿 |
| `/soccer/brain_strategy` | `std_msgs/String` | 当前策略 |
| `/soccer/visualization_markers` 等 | `visualization_msgs` | 调试可视化 |

多机时 `robot_name` 非空，`/soccer/*` 话题会追加 `/<robot_name>` 后缀；`zl_io` 的
命名空间和 `cmd_vel` 话题由 launch 自动改写。

### vision 输出

| 话题 | 类型 | 说明 |
|---|---|---|
| `/soccer/detection` | `vision_interface/Detections` | 检测框与类别 |
| `/soccer/ball` | `vision_interface/Ball` | 球位置 |
| `/soccer/line_segments` | `vision_interface/LineSegments` | 场地线分割 |
| `/soccer/debug/detection_img` | `sensor_msgs/CompressedImage` | 可选调试图（`publish_det_img:=true`） |

---

## 8. 行为树

- 入口：`src/brain/behavior_trees/game.xml`，根节点按
  `control_state`、`gc_game_state`、`gc_game_sub_state_type` 等黑板变量切换分支。
- 子树：`src/brain/behavior_trees/subtrees/`，包括找球、追踪球、定位、前锋/守门员
  常规与任意球。
- 自定义节点（`SimpleChase`、`Kick`、`MoveHead`、`CamScanField`、`SelfLocateEnterField`
  等）的注册和实现见 `src/brain/src/brain_tree.cpp` 与 `src/brain/include/brain_tree.h`。
- 启动时用 `tree:=<文件名>` 选择其他入口树；文件必须位于 `behavior_trees/` 下。
- 修改 XML 后无需重新编译（symlink-install），重启 brain 即可。

---

## 9. 测试与调试

```bash
# 纯算法单元测试（推荐，不依赖 ROS 网络）
./build/brain/test_one_m_candidate_selection
./build/brain/test_split_localization_schedule

# 完整 ctest（同时会跑 uncrustify / xmllint 等风格检查）
colcon test --base-paths src third_party/zl_interfaces/src --packages-select brain
colcon test-result --verbose
```

> `colcon test` 会额外执行代码风格检查；当前快照中 uncrustify / xmllint 有已知失败，
> 不代表功能回归。只关心算法正确性时，直接运行上面两个测试二进制即可。

定位日志分析：

```bash
python3 src/brain/scripts/analyze_brain_loc.py --log /path/to/brain.log
```

常用调试话题：

```bash
ros2 topic echo /soccer/robot_pose
ros2 topic echo /soccer/ball_position
ros2 topic echo /soccer/brain_strategy
ros2 topic hz /soccer/detection
ros2 topic echo /soccer/debug/detection_img --once
```

---

## 10. 新手阅读路线

第一次接触本仓库，建议按以下顺序阅读：

1. 本文档第 1 节的架构图，先建立“感知 → 决策 → 本体”的数据流概念；
2. `src/brain/config/config.yaml`，了解可调参数和机器人假设；
3. `src/brain/behavior_trees/game.xml`，理解比赛状态机；
4. `src/brain/include/brain_tree.h` 与 `src/brain/src/brain_tree.cpp`，看自定义行为树节点；
5. `src/brain/src/brain.cpp` 的 `init()` 和 topic 订阅/发布，理解输入输出；
6. `src/vision/config/vision.yaml` 与 `src/vision/src/vision_node.cpp`，理解视觉配置和推理流程；
7. `src/interface/*/msg/` 与 `third_party/zl_interfaces/src/`，确认消息契约；
8. 跑一遍第 9 节的测试，再用 `vision_test_feed.py` 做一次无相机冒烟。

---

## 11. 常见问题

**Q：仿真里 brain 不动 / 时钟不前进？**
MuJoCo 不发布 `/clock`。使用 `sim:=true` 时 launch 会强制 `use_sim_time=false`；
如果自行启动节点，务必保持 `use_sim_time=false`。

**Q：vision 收不到图像？**
用 `ros2 topic info -v <topic>` 检查发布端 QoS。深度话题的 QoS 必须与相机发布端一致
（`vision.yaml` 的 `depth_qos`，`reliable` / `best_effort`）；话题名包含 `compressed`
时 vision 会走 `CompressedImage` 分支。

**Q：没有 `/soccer/detection` 输出？**
确认 vision 订阅的图像、内参、`/robot/head_to_base` 三类输入时间戳能同步；
`save_data` 默认开启时日志在 `$HOME/Workspace/vision_log/<时间戳>`，可先用
`save_data:=false` 关闭。NO_CUDA 构建必须传入 `.onnx` 模型，不能用 `.engine`。

**Q：TensorRT engine 加载失败？**
engine 与 TensorRT/CUDA/GPU 版本绑定。换 JetPack 或换机器后需重新生成，见
[src/vision/model/README.md](src/vision/model/README.md)。

**Q：编译时提示 ONNX Runtime 找不到？**
先执行 `./third_party/onnxruntime/install_onnxruntime.sh`，确认
`/usr/local/include/onnxruntime_cxx_api.h` 和 `/usr/local/lib/libonnxruntime.so` 存在；
或把安装路径加入 `CMAKE_PREFIX_PATH`。

**Q：编译时出现同名消息包冲突？**
当前 shell 可能 source 了其他工作区。用 `STRIP_PREFIX_PATTERNS` 剔除其 install 前缀
（见 4.3），或新开一个只 source ROS 和本仓库的 shell。

**Q：实机上自动视觉踢球不生效？**
`real_robot:=true` 下 launch 会用 `enable_auto_visual_kick`（默认 `false`）覆盖配置文件；
需要显式传 `enable_auto_visual_kick:=true`，并确认门/球感知话题有数据。

---

## 12. 已知限制

- **TensorRT engine 不可跨机器直接复用。** 随包的 `det_261112_fp16.engine` /
  `seg_261112_fp16.engine` 只保证在生成它们的软硬件组合上可加载。
- **仓库不附带 ONNX 模型。** x86/NO_CUDA 构建可以编译，但运行 vision 需要用户自备
  `.onnx` 文件。
- **示例标定值不可直接用于新机器人。** `vision.yaml` 的相机内外参来自一台实机，
  必须重新标定。
- **仓库包含较大的二进制文件**（两个 engine 约 50 MB、ONNX Runtime 离线包约 15 MB、
  Ceres 库约 23 MB）。发布到有单文件/仓库体积限制的平台时，可考虑 Git LFS 或将
  大文件放到 Release 资产。
- **第三方消息包的许可证信息不完整。** `third_party/zl_interfaces` 的
  `package.xml` 仍为 `TODO`，公开分发前请补全。
- **模型权重的再分发许可需自行确认。** 见 `src/vision/model/README.md`。

---

## 13. 来源与许可

本仓库是 RoboCup 足球机器人工作区的独立抽取版本。顶层 `LICENSE` 为 Apache-2.0；
`src/vision` 包声明 BSD；各消息包和第三方组件的许可见对应目录。

| 组件 | 位置 | 许可证 |
|---|---|---|
| ONNX Runtime | `third_party/onnxruntime/` | MIT |
| Ceres Solver（含二进制） | `src/vision/thirdparty/` | BSD-3-Clause |
| GoogleTest | `src/vision/thirdparty/googletest/` | BSD-3-Clause |
| zl_interfaces | `third_party/zl_interfaces/` | 待补充（`package.xml` 仍为 TODO） |
| 感知模型权重 / engine | `src/vision/model/` | 需自行确认再分发权限 |

使用或再分发前，请同时确认模型权重、第三方二进制和 vendored 接口包的许可证。

---

## 14. 维护说明

- 不要把 `config_local.yaml`、`vision_local.yaml` 或 `~/agents/soccer/*.yaml`
  提交到仓库；仓库只保留 `.example` 模板。
- 发布前用 `git log -S'<敏感值>'` 检查历史提交；如果历史中包含旧的现场 IP、绝对路径
  或标定值，需要重写历史（例如 `git filter-repo`）或从干净历史重新初始化仓库。
- 修改模型或配置默认值后，同步更新本文档的“已知限制”和参数表。
