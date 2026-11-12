#pragma once

// zl_robot 原生 I/O（厂商接口，2026-11-03 起替代 brain_zl_bridge）。
//
// 职责：RobotClient 的唯一下行出口与上行入口，直接说 zl_robot 的语言：
//   下行  /cmd_vel、/robot/input/intent、/robot/external/head_control
//   上行  /robot/odometry/odom、/robot/driver/joint_states、/robot/imu/nav_all、
//         /robot/recovery_state、/robot/state、/robot/rl_manager/status、
//         /robot/router/active_intent
//
// 语义与旧桥一一对应（kMove/kRotateHead/kChangeMode/kGetUp/kVisualKick 以及
// odometer_state/low_state/fall_down_recovery_state 三条原厂接口话题），但不再经过
// JSON-RPC 信封、原厂接口包消息与独立桥进程。
//
// 线程模型：Brain 有 100Hz tick 线程与单线程 executor。下行来自 tick 线程
// （行为树 → RobotClient），上行回调在 executor 线程；因此本类内部状态用一把 mutex
// 保护（旧桥"全在单线程 executor 所以无锁"的前提在进程内已不成立）。

#include <cstdint>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

#include <robot_interfaces/msg/head_control.hpp>
#include <robot_interfaces/msg/joint_states.hpp>
#include <robot_interfaces/msg/rl_manager_status.hpp>
#include <robot_interfaces/msg/robot_intent.hpp>
#include <robot_interfaces/msg/robot_state.hpp>
#include <yesense_interface/msg/nav_all.hpp>

class Brain;

namespace robot_io
{

// zl_robot RobotState.state（robot_interfaces/msg/RobotState.msg）
enum ZlState : uint8_t
{
  ZL_BOOTING = 0,
  ZL_INIT = 1,
  ZL_IDLE = 2,
  ZL_DO_READY = 3,
  ZL_READY = 4,
  ZL_RL_STANDBY = 5,
  ZL_RL_ACTIVE = 6,
  ZL_RULE_STANDBY = 7,
  ZL_RULE_ACTIVE = 8,
  ZL_TRANSITIONING = 9,
  ZL_FAULT = 10,
  ZL_DAMP = 11,
  ZL_SHUTDOWN = 12,
};

// 状态推进目标：由 robocupWalk()/standUp()（Walk）与 enterDamping()（Damp）设置。
enum class SeqTarget
{
  None,
  Walk,  // 目标 RL_ACTIVE（可消费 /cmd_vel）
  Damp,  // 阻尼
};

class ZlIo
{
public:
  explicit ZlIo(Brain * brain);

  // 声明参数、创建收发接口。须在 Brain 配置加载完成后调用。
  void init();

  // 状态推进序列器 + 踢球交接状态机；由 Brain::tick() 以 100Hz 驱动。
  void tick();

  // ---------------- 下行 ----------------
  void sendTwist(double vx, double vy, double vyaw);
  void sendHead(double pitch, double yaw);  // 弧度，相机光轴语义
  void requestWalk();
  void requestDamp();
  void startVisualKick();
  void stopMotion();

  // ---------------- 状态查询 ----------------
  uint8_t zlState() const;
  bool kickActive() const;

  // IMU 姿态快照（供门位姿合成做基座倾角补偿）与里程计时间戳（判位姿新鲜度）。
  struct NavSnapshot
  {
    bool has_nav = false;
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
    double q[4] = {1.0, 0.0, 0.0, 0.0};  // q0=w, q1=x, q2=y, q3=z
    double stamp_sec = 0.0;
  };
  NavSnapshot navSnapshot() const;
  double lastOdomSec() const;
  // consistent 来源失效时禁止基于过期位姿继续运动；legacy 沿用原行为。
  bool odometryAllowsMotion() const;

private:
  // ---------------- 上行回调 ----------------
  void onZlState(const robot_interfaces::msg::RobotState & msg);
  void onJointStates(const robot_interfaces::msg::JointStates & msg);
  void onNavAll(const yesense_interface::msg::NavAll & msg);
  void onOdom(const nav_msgs::msg::Odometry & msg);
  void onRecovery(const std_msgs::msg::UInt8MultiArray & msg);
  void onRlStatus(const robot_interfaces::msg::RlManagerStatus & msg);
  void onActiveIntent(const robot_interfaces::msg::RobotIntent & msg);

  // ---------------- 内部（调用方须持有 mtx_）----------------
  void sendIntent(
    const char * intent_type, const char * command, const char * action = "",
    const std::string & source_override = "", int priority_override = -1);
  void sendKickIntent(const char * intent_type, const char * command, const char * action = "");
  void enterKickMode();
  void finishKickMode(const std::string & reason);
  void tickKickMode();
  void stepSequencer();
  double nowSec() const;
  bool isZlActive() const;

  Brain * brain_;

  // tick 线程（下行/序列器）与 executor 线程（上行回调）共享的状态。
  mutable std::mutex mtx_;

  // ---- 参数 ----
  double head_yaw_sign_ = 1.0;
  double head_pitch_sign_ = 1.0;
  double head_yaw_offset_deg_ = 0.0;
  double head_pitch_offset_deg_ = -36.0;
  double fb_head_yaw_sign_ = 1.0;
  double fb_head_pitch_sign_ = 1.0;
  std::string intent_source_ = "api";
  int intent_priority_ = 50;
  double intent_interval_sec_ = 0.8;
  bool auto_match_intent_priority_ = true;
  std::string switch_model_intent_source_;
  int switch_model_intent_priority_ = -1;
  std::string kick_model_name_ = "t4_kicking";
  double kick_rl_timeout_sec_ = 8.0;
  int restore_mix_attempts_ = 1;
  double restore_mix_interval_sec_ = 0.8;
  std::string active_intent_topic_ = "/robot/router/active_intent";
  std::string rl_status_topic_ = "/robot/rl_manager/status";
  // 话题命名（多机仿真用）：topic_ns_ 同时作为 state/joint/nav/odom/recovery/intent/head
  // 的前缀（单机 /robot，多机 /robot_0N），cmd_vel 单独给（单机 /cmd_vel）。
  std::string topic_ns_ = "/robot";
  std::string cmd_vel_topic_ = "/cmd_vel";
  bool odom_consistent_ = false;
  double consistent_odom_scale_ = 1.0;

  // ---- 接口 ----
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<robot_interfaces::msg::RobotIntent>::SharedPtr intent_pub_;
  rclcpp::Publisher<robot_interfaces::msg::HeadControl>::SharedPtr head_pub_;
  rclcpp::Subscription<robot_interfaces::msg::RobotState>::SharedPtr zl_state_sub_;
  rclcpp::Subscription<robot_interfaces::msg::JointStates>::SharedPtr joint_states_sub_;
  rclcpp::Subscription<yesense_interface::msg::NavAll>::SharedPtr nav_all_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  // 默认关闭的接入核验出口，仅记录Brain实际接受的位姿，不参与控制或TF。
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr accepted_odom_debug_pub_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr recovery_sub_;
  rclcpp::Subscription<robot_interfaces::msg::RlManagerStatus>::SharedPtr rl_status_sub_;
  rclcpp::Subscription<robot_interfaces::msg::RobotIntent>::SharedPtr active_intent_sub_;

  // ---- zl 反馈缓存 ----
  uint8_t zl_state_ = 0;
  bool zl_state_seen_ = false;
  double head_yaw_rad_ = 0.0;
  double head_pitch_rad_ = 0.0;
  bool has_nav_ = false;
  double euler_roll_deg_ = 0.0;
  double euler_pitch_deg_ = 0.0;
  double euler_yaw_deg_ = 0.0;
  double nav_quat_[4] = {1.0, 0.0, 0.0, 0.0};
  double nav_stamp_sec_ = 0.0;
  double odom_stamp_sec_ = 0.0;

  // ---- 意图优先级自适应 ----
  bool has_active_intent_ = false;
  bool active_intent_warned_ = false;
  std::string active_intent_source_;
  uint8_t active_intent_priority_ = 0;

  // ---- 下行状态 ----
  SeqTarget seq_target_ = SeqTarget::None;
  bool damped_by_brain_ = false;
  double last_intent_time_sec_ = 0.0;

  // ---- 踢球交接（RL <-> MIX）----
  bool kick_rl_active_ = false;
  bool kick_model_seen_ = false;
  bool restore_pending_ = false;
  int restore_attempts_ = 0;
  double kick_started_sec_ = 0.0;
  double kick_model_last_seen_sec_ = 0.0;
  double restore_last_send_sec_ = 0.0;

  // ---- 速度出口保护 ----
  // 踢球结束后短暂抑制速度：击球瞬间本体可能仍在侧倾，立即要速度会把机身压垮。
  // 0 = 关闭；默认 0.8 s，可用 zl_io.kick_settle_suppress_sec 调。
  double kick_settle_suppress_sec_ = 0.8;
  double velocity_suppress_until_sec_ = 0.0;
  double last_suppress_log_sec_ = 0.0;
};

}  // namespace robot_io
