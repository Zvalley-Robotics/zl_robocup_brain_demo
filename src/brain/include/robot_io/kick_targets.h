#pragma once

// 真实球门位姿合成（原 brain_zl_bridge 的 goal_pose_synth 节点，2026-11-03 并入 brain）。
//
// 用「场地几何 + Brain 场地系位姿」算出基座系门心与门外法线，喂给 zl 侧 t4_kicking：
//   /robot/perception/target_destination(Point) / target_goal_normal(Vector3)
// 实机上经 NX 域100 的 /soccer/goal_*_robot 话题 + 板侧 domain_bridge remap 落地；
// 六机自包含形态（域内无 domain_bridge）可用参数直接把输出指向 zl 契约名。
//
// 语义约束（与 policy/context.py 成对，勿单改一面）：
//   * 门点发**原始基座系**三维量（消费端会再用 IMU 四元数变换后取 xy）；
//   * 法线发门口**外法线**（门内指向场内），消费端 goal_dir = atan2(-ny, -nx)；
//   * 位姿无效/超时/越界/重定位跳变时**停止发布**（消费端零阶保持上一帧）。

#include <map>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include "robot_io/zl_io.h"

class Brain;

namespace robot_io
{

class KickTargets
{
public:
  explicit KickTargets(Brain * brain);

  void init();

  // 由 RobotClient::tick() 每拍调用；pose 为 Brain 场地系位姿。
  void update(
    double pose_x, double pose_y, double pose_theta, double pose_stamp_sec,
    const ZlIo::NavSnapshot & nav, double now_sec);

  // 门位姿是否在稳定发布（供视觉踢交棒门；enable=false 时不拦，保持旧兜底行为）。
  bool active(double now_sec) const;

private:
  void publishTick(
    double pose_x, double pose_y, double pose_theta, const ZlIo::NavSnapshot & nav,
    double now_sec);

  void warnThrottled(const char * key, const std::string & msg);
  Brain * brain_;

  // ---- 场地几何（Brain 场地系：+x 指向对方球门）----
  double goal_line_x_ = 7.0;
  double goal_line_y_ = 0.0;
  double attack_sign_ = 1.0;
  double goal_z_ = -0.68;

  // ---- 门控参数 ----
  double publish_hz_ = 50.0;
  double pose_timeout_sec_ = 0.3;
  double nav_all_timeout_sec_ = 0.3;
  double field_bound_x_ = 8.0;
  double field_bound_y_ = 5.0;
  double stable_required_sec_ = 1.0;
  double jump_reset_m_ = 1.0;
  double jump_reset_rad_ = 0.5;
  bool use_nav_all_tilt_ = true;
  double max_tilt_deg_ = 25.0;

  // ---- 状态 ----
  bool has_prev_ = false;
  bool stable_ = false;
  bool enable_logged_ = false;
  double prev_x_ = 0.0;
  double prev_y_ = 0.0;
  double prev_theta_ = 0.0;
  double stable_since_ = 0.0;
  double last_publish_sec_ = 0.0;
  std::map<std::string, double> warn_last_;

  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr destination_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr normal_pub_;
};

}  // namespace robot_io
