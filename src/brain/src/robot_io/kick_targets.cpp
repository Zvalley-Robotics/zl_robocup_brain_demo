#include "robot_io/kick_targets.h"

#include <cmath>
#include <string>

#include "brain.h"
#include "utils/print.h"

namespace
{
constexpr double kDeg2Rad = M_PI / 180.0;
constexpr double kQuatNormEps = 1e-6;
constexpr double kWarnPeriodSec = 5.0;

double wrapToPi(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

struct Vec3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

// 旋转矩阵（行主序）：v_dst = R * v_src。
struct Mat3
{
  double r[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
};

// 四元数 [w,x,y,z] -> 基座到世界（ROS 右手系，+x 前 +y 左 +z 上）。
Mat3 rotationFromQuat(double w, double x, double y, double z)
{
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (norm < kQuatNormEps) {
    return Mat3{};
  }
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;
  Mat3 out;
  out.r[0][0] = 1.0 - 2.0 * (y * y + z * z);
  out.r[0][1] = 2.0 * (x * y - z * w);
  out.r[0][2] = 2.0 * (x * z + y * w);
  out.r[1][0] = 2.0 * (x * y + z * w);
  out.r[1][1] = 1.0 - 2.0 * (x * x + z * z);
  out.r[1][2] = 2.0 * (y * z - x * w);
  out.r[2][0] = 2.0 * (x * z - y * w);
  out.r[2][1] = 2.0 * (y * z + x * w);
  out.r[2][2] = 1.0 - 2.0 * (x * x + y * y);
  return out;
}

// 欧拉角（度，ZYX）-> R = Rz(yaw) * Ry(pitch) * Rx(roll)；四元数无效时的兜底。
Mat3 rotationFromEulerDeg(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll * kDeg2Rad), sr = std::sin(roll * kDeg2Rad);
  const double cp = std::cos(pitch * kDeg2Rad), sp = std::sin(pitch * kDeg2Rad);
  const double cy = std::cos(yaw * kDeg2Rad), sy = std::sin(yaw * kDeg2Rad);
  Mat3 out;
  out.r[0][0] = cy * cp;
  out.r[0][1] = cy * sp * sr - sy * cr;
  out.r[0][2] = cy * sp * cr + sy * sr;
  out.r[1][0] = sy * cp;
  out.r[1][1] = sy * sp * sr + cy * cr;
  out.r[1][2] = sy * sp * cr - cy * sr;
  out.r[2][0] = -sp;
  out.r[2][1] = cp * sr;
  out.r[2][2] = cp * cr;
  return out;
}

// v_base = R^T * v_world。
Vec3 applyTranspose(const Mat3 & r, const Vec3 & v)
{
  return Vec3{
    r.r[0][0] * v.x + r.r[1][0] * v.y + r.r[2][0] * v.z,
    r.r[0][1] * v.x + r.r[1][1] * v.y + r.r[2][1] * v.z,
    r.r[0][2] * v.x + r.r[1][2] * v.y + r.r[2][2] * v.z,
  };
}

// 绕 z 轴旋转（右手，逆时针为正）。
void rotateZ(double angle, double & x, double & y)
{
  const double c = std::cos(angle), s = std::sin(angle);
  const double nx = c * x - s * y;
  const double ny = s * x + c * y;
  x = nx;
  y = ny;
}

template <typename T>
T paramOrDefault(Brain * node, const std::string & name, const T & def)
{
  if (node->has_parameter(name)) {
    return node->get_parameter(name).get_value<T>();
  }
  return node->declare_parameter<T>(name, def);
}
}  // namespace

namespace robot_io
{

KickTargets::KickTargets(Brain * brain)
: brain_(brain)
{
}

void KickTargets::init()
{
  // 总开关每拍读取（运行期 ros2 param set 即时生效，与原 goal_pose_synth 一致）。
  paramOrDefault<bool>(brain_, "zl_io.kick_targets_enable", true);

  goal_line_x_ = paramOrDefault<double>(brain_, "zl_io.goal_line_x", 7.0);
  goal_line_y_ = paramOrDefault<double>(brain_, "zl_io.goal_line_y", 0.0);
  attack_sign_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_attack_sign", 1.0);
  goal_z_ = paramOrDefault<double>(brain_, "zl_io.goal_z", -0.68);

  publish_hz_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_publish_hz", 50.0);
  pose_timeout_sec_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_pose_timeout_sec", 0.3);
  nav_all_timeout_sec_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_nav_timeout_sec", 0.3);
  field_bound_x_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_field_bound_x", 8.0);
  field_bound_y_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_field_bound_y", 5.0);
  stable_required_sec_ =
    paramOrDefault<double>(brain_, "zl_io.kick_targets_stable_required_sec", 1.0);
  jump_reset_m_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_jump_reset_m", 1.0);
  jump_reset_rad_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_jump_reset_rad", 0.5);
  use_nav_all_tilt_ = paramOrDefault<bool>(brain_, "zl_io.kick_targets_use_nav_tilt", true);
  max_tilt_deg_ = paramOrDefault<double>(brain_, "zl_io.kick_targets_max_tilt_deg", 25.0);

  const auto dest_topic = paramOrDefault<std::string>(
    brain_, "zl_io.goal_dest_topic", "/soccer/goal_destination_robot");
  const auto normal_topic = paramOrDefault<std::string>(
    brain_, "zl_io.goal_normal_topic", "/soccer/goal_normal_robot");

  // reliable d10：下游 rl_manager 订阅为 reliable，best_effort 不兼容。
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  destination_pub_ = brain_->create_publisher<geometry_msgs::msg::Point>(dest_topic, qos);
  normal_pub_ = brain_->create_publisher<geometry_msgs::msg::Vector3>(normal_topic, qos);

  brain_->log->log(
    "zl_io/kick_targets",
    format(
      "门位姿合成就绪: 门=(%.2f,%.2f) sign=%.0f z=%.2f -> %s + %s（enable=%s, tilt=%s）",
      goal_line_x_, goal_line_y_, attack_sign_, goal_z_, dest_topic.c_str(), normal_topic.c_str(),
      brain_->get_parameter("zl_io.kick_targets_enable").as_bool() ? "true" : "false",
      use_nav_all_tilt_ ? "IMU" : "yaw-only"));
}

void KickTargets::update(
  double pose_x, double pose_y, double pose_theta, double pose_stamp_sec,
  const ZlIo::NavSnapshot & nav, double now_sec)
{
  const bool enable = brain_->get_parameter("zl_io.kick_targets_enable").as_bool();
  if (enable != enable_logged_) {
    enable_logged_ = enable;
    brain_->log->log(
      "zl_io/kick_targets",
      enable ? "门位姿发布 已开启" : "门位姿发布 已关闭（消费端保持上一帧/假门）");
  }
  if (!enable) {
    // 关闭期间不累计稳定时长：重新开启需要重新收敛 1s，避免用旧位姿立刻发门。
    has_prev_ = false;
    stable_ = false;
    return;
  }

  const bool finite =
    std::isfinite(pose_x) && std::isfinite(pose_y) && std::isfinite(pose_theta);
  const bool inside =
    std::fabs(pose_x) <= field_bound_x_ && std::fabs(pose_y) <= field_bound_y_;
  if (!finite || !inside) {
    if (has_prev_) {
      warnThrottled(
        "pose_invalid",
        format("机器人位姿无效或越界（x=%.2f y=%.2f），暂停发布门位姿", pose_x, pose_y));
    }
    has_prev_ = false;
    stable_ = false;
    return;
  }

  if ((now_sec - pose_stamp_sec) > pose_timeout_sec_) {
    warnThrottled(
      "pose_stale", format("机器人位姿超时 %.2fs，不发布门位姿", now_sec - pose_stamp_sec));
    return;
  }

  if (has_prev_) {
    const double jump_m = std::hypot(pose_x - prev_x_, pose_y - prev_y_);
    const double jump_rad = std::fabs(wrapToPi(pose_theta - prev_theta_));
    if (jump_m > jump_reset_m_ || jump_rad > jump_reset_rad_) {
      warnThrottled(
        "pose_jump",
        format("位姿跳变 %.2f m / %.2f rad（重定位？），重新累计稳定时长", jump_m, jump_rad));
      stable_ = false;
      stable_since_ = now_sec;
    }
  } else {
    // 首帧（或中断后重新开始）：从这里计稳定时长。
    stable_since_ = now_sec;
  }

  prev_x_ = pose_x;
  prev_y_ = pose_y;
  prev_theta_ = pose_theta;
  has_prev_ = true;

  if (!stable_ && (now_sec - stable_since_) >= stable_required_sec_) {
    stable_ = true;
    brain_->log->log(
      "zl_io/kick_targets",
      format("位姿稳定，开始发布门位姿: x=%.2f y=%.2f theta=%.2f", pose_x, pose_y, pose_theta));
  }
  if (!stable_) {
    warnThrottled(
      "pose_unstable", format("机器人位姿尚未连续稳定 %.1fs，不发布门位姿", stable_required_sec_));
    return;
  }

  const double period = publish_hz_ > 1.0 ? 1.0 / publish_hz_ : 0.02;
  if ((now_sec - last_publish_sec_) < period) {
    return;
  }
  last_publish_sec_ = now_sec;

  publishTick(pose_x, pose_y, pose_theta, nav, now_sec);
}

bool KickTargets::active(double now_sec) const
{
  // 关闭时保持旧行为（消费端用兜底假门），不把“未启用”判成“不可踢”。
  if (!brain_->get_parameter("zl_io.kick_targets_enable").as_bool()) {
    return true;
  }
  if (!stable_ || last_publish_sec_ <= 0.0) {
    return false;
  }
  return (now_sec - last_publish_sec_) <= pose_timeout_sec_ + 0.10;
}

void KickTargets::publishTick(
  double pose_x, double pose_y, double pose_theta, const ZlIo::NavSnapshot & nav,
  double now_sec)
{
  // 场地系下门心相对机器人的向量（z 用门心与基座的高度差）。
  const double gx = attack_sign_ * goal_line_x_ - pose_x;
  const double gy = attack_sign_ * goal_line_y_ - pose_y;
  // 门外法线（场地系）：从门内指向场内，攻击门在 +x 时即 -x。
  const double nx_field = -attack_sign_;
  const double ny_field = 0.0;

  Vec3 dest;
  Vec3 normal;
  bool precise = false;

  if (use_nav_all_tilt_ && nav.has_nav && (now_sec - nav.stamp_sec) <= nav_all_timeout_sec_) {
    Mat3 r_base_world;
    double yaw_nav = 0.0;
    const double qnorm = std::sqrt(
      nav.q[0] * nav.q[0] + nav.q[1] * nav.q[1] + nav.q[2] * nav.q[2] + nav.q[3] * nav.q[3]);
    if (qnorm > kQuatNormEps) {
      r_base_world = rotationFromQuat(nav.q[0], nav.q[1], nav.q[2], nav.q[3]);
      yaw_nav = std::atan2(r_base_world.r[1][0], r_base_world.r[0][0]);
    } else {
      r_base_world = rotationFromEulerDeg(nav.roll_deg, nav.pitch_deg, nav.yaw_deg);
      yaw_nav = nav.yaw_deg * kDeg2Rad;
    }
    const double tilt = std::acos(std::fmin(1.0, std::fmax(-1.0, r_base_world.r[2][2])));
    if (tilt <= max_tilt_deg_ * kDeg2Rad) {
      // 场地系与导航系只差一个 yaw：先把场地系向量转到导航水平系，再进基座系。
      const double delta = pose_theta - yaw_nav;
      double lx = gx, ly = gy;
      rotateZ(-delta, lx, ly);
      dest = applyTranspose(r_base_world, Vec3{lx, ly, goal_z_});
      double lx_n = nx_field, ly_n = ny_field;
      rotateZ(-delta, lx_n, ly_n);
      normal = applyTranspose(r_base_world, Vec3{lx_n, ly_n, 0.0});
      precise = true;
    }
  }
  if (!precise) {
    // yaw-only：基座系与场地系只差 theta，直接水平旋转。
    double lx = gx, ly = gy;
    rotateZ(-pose_theta, lx, ly);
    dest = Vec3{lx, ly, goal_z_};
    double lx_n = nx_field, ly_n = ny_field;
    rotateZ(-pose_theta, lx_n, ly_n);
    normal = Vec3{lx_n, ly_n, 0.0};
  }

  geometry_msgs::msg::Point dest_msg;
  dest_msg.x = dest.x;
  dest_msg.y = dest.y;
  dest_msg.z = dest.z;
  destination_pub_->publish(dest_msg);

  geometry_msgs::msg::Vector3 normal_msg;
  normal_msg.x = normal.x;
  normal_msg.y = normal.y;
  normal_msg.z = normal.z;
  normal_pub_->publish(normal_msg);
}

// 同一 key 的日志按 kWarnPeriodSec 限频，避免 100Hz 刷屏。
void KickTargets::warnThrottled(const char * key, const std::string & msg)
{
  const double now = brain_->now().seconds();
  auto it = warn_last_.find(key);
  if (it != warn_last_.end() && (now - it->second) < kWarnPeriodSec) {
    return;
  }
  warn_last_[key] = now;
  brain_->log->warn("zl_io/kick_targets", msg);
}

}  // namespace robot_io
