#include "robot_io/ball_fwd.h"

#include <functional>
#include <memory>
#include <string>

namespace
{
template <typename T>
T paramOrDefault(rclcpp::Node * node, const std::string & name, const T & def)
{
  if (node->has_parameter(name)) {
    return node->get_parameter(name).get_value<T>();
  }
  return node->declare_parameter<T>(name, def);
}
}  // namespace

namespace robot_io
{

BallFwd::BallFwd(rclcpp::Node * node)
: node_(node)
{
}

void BallFwd::init()
{
  if (!paramOrDefault<bool>(node_, "zl_io.ball_fwd_enable", true)) {
    RCLCPP_INFO(node_->get_logger(), "ball_fwd 关闭（zl_io.ball_fwd_enable=false）");
    return;
  }
  const auto input_topic =
    paramOrDefault<std::string>(node_, "zl_io.ball_in_topic", "/soccer/ball");
  const auto output_topic =
    paramOrDefault<std::string>(node_, "zl_io.ball_out_topic", "/soccer/ball_robot");

  // reliable d10：下游 rl_manager 订阅为 reliable；depth 取 10（vision 源 30Hz，
  // depth=1 时跨机实测掉到 7~8Hz，无法满足球历史窗口）。
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  publisher_ = node_->create_publisher<robot_interfaces::msg::Ball>(output_topic, qos);
  subscription_ = node_->create_subscription<vision_interface::msg::Ball>(
    input_topic, qos, std::bind(&BallFwd::onBall, this, std::placeholders::_1));

  RCLCPP_INFO(
    node_->get_logger(), "ball_fwd: %s (vision_interface/Ball) -> %s (robot_interfaces/Ball)",
    input_topic.c_str(), output_topic.c_str());
}

void BallFwd::onBall(const vision_interface::msg::Ball & msg)
{
  robot_interfaces::msg::Ball out;
  out.header = msg.header;
  out.x = msg.x;
  out.y = msg.y;
  out.confidence = msg.confidence;
  publisher_->publish(out);

  if (!logged_first_) {
    logged_first_ = true;
    RCLCPP_INFO(
      node_->get_logger(), "首帧球已转发: x=%.3f y=%.3f confidence=%.1f frame_id=%s", out.x,
      out.y, out.confidence, out.header.frame_id.c_str());
  }
}

}  // namespace robot_io
