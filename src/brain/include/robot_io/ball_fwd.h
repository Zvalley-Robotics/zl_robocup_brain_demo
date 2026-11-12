#pragma once

// 球检测类型转换（原 brain_zl_bridge 的 ball_type_adapter 节点，2026-11-03 并入 brain）。
//
// vision 发布 vision_interface/msg/Ball，zl_robot 的 rl_manager 订阅
// robot_interfaces/msg/Ball —— 同名字段但包名不同，ROS 2 类型不匹配，domain_bridge
// 也不做类型转换。这里完成一次字段直拷：
//   /soccer/ball (vision_interface/Ball)
//     -> /soccer/ball_robot (robot_interfaces/Ball)
// 实机形态再由板侧 domain_bridge remap 回 /soccer/ball（域 10）；
// 六机自包含形态可用参数直接把输出指向 zl 契约名。
//
// 两种使用方式（同一实现）：
//   * Brain 进程内：RobotClient 用 brain 节点构造（随 brain 启停）；
//   * 独立进程：brain 包的 ball_fwd_node（nx_ctl.sh ball 模式，只发球位置、不起 brain）。

#include <rclcpp/rclcpp.hpp>

#include <robot_interfaces/msg/ball.hpp>
#include <vision_interface/msg/ball.hpp>

namespace robot_io
{

class BallFwd
{
public:
  explicit BallFwd(rclcpp::Node * node);

  void init();

private:
  void onBall(const vision_interface::msg::Ball & msg);

  rclcpp::Node * node_;
  rclcpp::Publisher<robot_interfaces::msg::Ball>::SharedPtr publisher_;
  rclcpp::Subscription<vision_interface::msg::Ball>::SharedPtr subscription_;
  bool logged_first_ = false;
};

}  // namespace robot_io
