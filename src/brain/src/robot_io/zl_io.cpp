#include "robot_io/zl_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "brain.h"
#include "utils/math.h"
#include "utils/print.h"

namespace robot_io
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRad2Deg = 180.0 / kPi;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr double kConsistentOdomMaxAgeSec = 0.15;
constexpr double kOdomFutureToleranceSec = 0.02;

// 速度抑制窗用单调时钟：ROS 时间回退/跳变（本项目已多次出现）不能让保护窗变成
// “卡住不发速度”或“提前失效”。
double monoNowSec()
{
  return std::chrono::duration<double>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

// 参数读取：优先取 yaml 覆盖值，未声明过则以默认值声明。
template <typename T>
T paramOrDefault(Brain * node, const std::string & name, const T & def)
{
  if (node->has_parameter(name)) {
    return node->get_parameter(name).get_value<T>();
  }
  return node->declare_parameter<T>(name, def);
}
}  // namespace

ZlIo::ZlIo(Brain * brain)
: brain_(brain)
{
}

void ZlIo::init()
{
  // ------------------------------------------------ 参数 ----
  head_yaw_sign_ = paramOrDefault<double>(brain_, "zl_io.head_yaw_sign", 1.0);
  head_pitch_sign_ = paramOrDefault<double>(brain_, "zl_io.head_pitch_sign", 1.0);
  head_yaw_offset_deg_ = paramOrDefault<double>(brain_, "zl_io.head_yaw_offset_deg", 0.0);
  head_pitch_offset_deg_ = paramOrDefault<double>(brain_, "zl_io.head_pitch_offset_deg", -36.0);
  fb_head_yaw_sign_ = paramOrDefault<double>(brain_, "zl_io.fb_head_yaw_sign", 1.0);
  fb_head_pitch_sign_ = paramOrDefault<double>(brain_, "zl_io.fb_head_pitch_sign", 1.0);
  intent_source_ = paramOrDefault<std::string>(brain_, "zl_io.intent_source", "api");
  intent_priority_ = paramOrDefault<int>(brain_, "zl_io.intent_priority", 50);
  intent_interval_sec_ = paramOrDefault<double>(brain_, "zl_io.intent_interval_sec", 0.8);
  auto_match_intent_priority_ =
    paramOrDefault<bool>(brain_, "zl_io.auto_match_intent_priority", true);
  switch_model_intent_source_ =
    paramOrDefault<std::string>(brain_, "zl_io.switch_model_intent_source", "");
  switch_model_intent_priority_ =
    paramOrDefault<int>(brain_, "zl_io.switch_model_intent_priority", -1);
  kick_model_name_ = paramOrDefault<std::string>(brain_, "zl_io.kick_model_name", "t4_kicking");
  kick_rl_timeout_sec_ = paramOrDefault<double>(brain_, "zl_io.kick_rl_timeout_sec", 8.0);
  restore_mix_attempts_ = paramOrDefault<int>(brain_, "zl_io.restore_mix_attempts", 1);
  restore_mix_interval_sec_ =
    paramOrDefault<double>(brain_, "zl_io.restore_mix_interval_sec", 0.8);
  // 踢球结束后速度抑制窗（秒）：0 = 关闭。依据 2026-11-09 39 号机实测，motion ended 后
  // brain 在 0.4 s 就下发 vx=0.67，而本体仍在击球侧倾中，是翻倒的加速因素。
  kick_settle_suppress_sec_ =
    paramOrDefault<double>(brain_, "zl_io.kick_settle_suppress_sec", 0.8);

  // 话题命名：单机取实机契约名（/robot/* + /cmd_vel）；多机仿真每台一个命名空间
  // （如 /robot_01/* + /robot_01/cmd_vel，由 launch 按 robot_name 注入）。
  topic_ns_ = paramOrDefault<std::string>(brain_, "zl_io.topic_ns", "/robot");
  while (!topic_ns_.empty() && topic_ns_.back() == '/') {
    topic_ns_.pop_back();
  }
  cmd_vel_topic_ = paramOrDefault<std::string>(brain_, "zl_io.cmd_vel_topic", "/cmd_vel");
  const std::string state_topic = topic_ns_ + "/state";
  const std::string joint_topic = topic_ns_ + "/driver/joint_states";
  const std::string nav_topic = topic_ns_ + "/imu/nav_all";
  const std::string odom_topic = paramOrDefault<std::string>(
    brain_, "zl_io.odom_topic", topic_ns_ + "/odometry/odom");
  const auto odom_convention = paramOrDefault<std::string>(
    brain_, "zl_io.odom_convention", "legacy_world_position");
  if (odom_convention != "legacy_world_position" && odom_convention != "consistent") {
    throw std::invalid_argument("zl_io.odom_convention 只接受 legacy_world_position 或 consistent");
  }
  odom_consistent_ = odom_convention == "consistent";
  consistent_odom_scale_ = paramOrDefault<double>(brain_, "zl_io.consistent_odom_scale", 1.0);
  if (!std::isfinite(consistent_odom_scale_) || consistent_odom_scale_ <= 0.0) {
    throw std::invalid_argument("zl_io.consistent_odom_scale 必须为正有限数");
  }
  const auto accepted_debug_topic = paramOrDefault<std::string>(
    brain_, "zl_io.accepted_odom_debug_topic", "");
  if (!accepted_debug_topic.empty()) {
    accepted_odom_debug_pub_ = brain_->create_publisher<nav_msgs::msg::Odometry>(
      accepted_debug_topic, rclcpp::QoS(10));
  }
  const std::string recovery_topic = topic_ns_ + "/recovery_state";
  const std::string intent_topic = topic_ns_ + "/input/intent";
  const std::string head_topic = topic_ns_ + "/external/head_control";
  active_intent_topic_ = paramOrDefault<std::string>(
    brain_, "zl_io.active_intent_topic", topic_ns_ + "/router/active_intent");
  rl_status_topic_ =
    paramOrDefault<std::string>(brain_, "zl_io.rl_status_topic", topic_ns_ + "/rl_manager/status");

  // ------------------------------------------------ 下行接口 ----
  // QoS 与 zl_robot 原生契约一致（rl_manager 的 /cmd_vel 订阅为 BEST_EFFORT depth1，
  // intent/head_control 为 RELIABLE depth10）。
  cmd_vel_pub_ = brain_->create_publisher<geometry_msgs::msg::Twist>(
    cmd_vel_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).best_effort());
  // 意图发布者必须用 TRANSIENT_LOCAL：板端 domain_bridge 在域 100 侧订阅
  // /robot/input/intent 用的是 TL（2026-11-10 40 号机实测：100 侧不兼容告警
  // "Last incompatible policy: DURABILITY_QOS_POLICY"，域10 转发端
  // robot_external_bridge_10 也是 TL）。DDS 下 volatile pub → TL sub 不兼容
  // （一条都收不到），而 TL pub 同时兼容 volatile sub（router/teleop 侧）与 TL sub，
  // 所以只能往 TL 侧靠。depth=1：晚加入的订阅者最多拿一条旧意图，brain 0.8 s 重发。
  intent_pub_ = brain_->create_publisher<robot_interfaces::msg::RobotIntent>(
    intent_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  head_pub_ = brain_->create_publisher<robot_interfaces::msg::HeadControl>(
    head_topic, rclcpp::QoS(10));

  // ------------------------------------------------ 上行接口 ----
  // 全部 BEST_EFFORT depth5：与 3588 侧 zl 发布端 / domain_bridge 白名单一致。
  const auto be_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
  zl_state_sub_ = brain_->create_subscription<robot_interfaces::msg::RobotState>(
    state_topic, be_qos, bind(&ZlIo::onZlState, this, std::placeholders::_1));
  joint_states_sub_ = brain_->create_subscription<robot_interfaces::msg::JointStates>(
    joint_topic, be_qos, bind(&ZlIo::onJointStates, this, std::placeholders::_1));
  nav_all_sub_ = brain_->create_subscription<yesense_interface::msg::NavAll>(
    nav_topic, be_qos, bind(&ZlIo::onNavAll, this, std::placeholders::_1));
  odom_sub_ = brain_->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic, be_qos, bind(&ZlIo::onOdom, this, std::placeholders::_1));
  recovery_sub_ = brain_->create_subscription<std_msgs::msg::UInt8MultiArray>(
    recovery_topic, be_qos, bind(&ZlIo::onRecovery, this, std::placeholders::_1));
  rl_status_sub_ = brain_->create_subscription<robot_interfaces::msg::RlManagerStatus>(
    rl_status_topic_, be_qos, bind(&ZlIo::onRlStatus, this, std::placeholders::_1));
  active_intent_sub_ = brain_->create_subscription<robot_interfaces::msg::RobotIntent>(
    active_intent_topic_, be_qos, bind(&ZlIo::onActiveIntent, this, std::placeholders::_1));

  brain_->log->log(
    "zl_io/init",
    format(
      "zl_robot 原生 I/O 就绪：ns=%s cmd_vel=%s intent=%s/%d interval=%.2fs kick_model=%s "
      "head_pitch_offset=%.1fdeg",
      topic_ns_.c_str(), cmd_vel_topic_.c_str(), intent_source_.c_str(), intent_priority_,
      intent_interval_sec_, kick_model_name_.c_str(), head_pitch_offset_deg_));
}

// ------------------------------------------------------------ 下行 ----

void ZlIo::sendTwist(double vx, double vy, double vyaw)
{
  std::lock_guard<std::mutex> lk(mtx_);

  if (damped_by_brain_) {
    // 明确的阻尼指令之后忽略运动，直到 brain 重新请求行走。
    return;
  }

  // 速度出口保护：
  //   ① 本体倒地（等 RobotState.fallen 就位后生效）→ 持续零速，等本体自行起身；
  //   ② 踢球结束抑制窗（zl_io.kick_settle_suppress_sec，默认 0.8 s）→ 窗内零速。
  const bool fallen = brain_->data->recoveryState == RobotRecoveryState::HAS_FALLEN;
  const double now_sec = monoNowSec();
  if (fallen || now_sec < velocity_suppress_until_sec_) {
    geometry_msgs::msg::Twist zero;
    cmd_vel_pub_->publish(zero);
    if (now_sec - last_suppress_log_sec_ >= 0.5) {
      last_suppress_log_sec_ = now_sec;
      brain_->log->log(
        "zl_io/suppress",
        fallen ? "本体倒地，强制零速"
               : format("踢球结束抑制窗，剩余 %.2fs，强制零速",
                        velocity_suppress_until_sec_ - now_sec));
    }
    return;
  }

  // 自愈：比赛流程中 brain 只持续下发速度、不周期发模式。
  // 若机器人被重置/冷启动后停在待命态，速度会被 rl_manager 丢弃而"站着不动"；
  // 此时以速度到达为信号请求一次行走序列（IDLE/DO_READY/READY -> RL_ACTIVE）。
  if (seq_target_ == SeqTarget::None) {
    const uint8_t s = zl_state_;
    if (s == ZL_IDLE || s == ZL_DO_READY || s == ZL_READY) {
      seq_target_ = SeqTarget::Walk;
      brain_->log->log(
        "zl_io/seq",
        format("速度到达但 zl 未在行走态(state=%u)，自动请求行走序列", static_cast<unsigned>(s)));
    }
  }

  geometry_msgs::msg::Twist twist;
  twist.linear.x = vx;
  twist.linear.y = vy;
  twist.angular.z = vyaw;
  cmd_vel_pub_->publish(twist);
}

void ZlIo::sendHead(double pitch, double yaw)
{
  std::lock_guard<std::mutex> lk(mtx_);
  // brain 侧 pitch/yaw 为相机光轴语义（弧度）；external_node 期望关节角（度）。
  robot_interfaces::msg::HeadControl out;
  out.yaw = static_cast<float>(yaw * kRad2Deg * head_yaw_sign_ + head_yaw_offset_deg_);
  out.pitch = static_cast<float>(pitch * kRad2Deg * head_pitch_sign_ + head_pitch_offset_deg_);
  out.is_relative = false;
  head_pub_->publish(out);
}

void ZlIo::requestWalk()
{
  std::lock_guard<std::mutex> lk(mtx_);
  damped_by_brain_ = false;
  seq_target_ = SeqTarget::Walk;
  brain_->log->log("zl_io/seq", "请求行走序列（ready -> walk）");
}

void ZlIo::requestDamp()
{
  std::lock_guard<std::mutex> lk(mtx_);
  if (kick_rl_active_) {
    finishKickMode("收到阻尼指令，踢球被打断");
  }
  damped_by_brain_ = true;
  seq_target_ = SeqTarget::Damp;

  geometry_msgs::msg::Twist zero;
  cmd_vel_pub_->publish(zero);
  brain_->log->log("zl_io/seq", "请求阻尼（damping）");
}

void ZlIo::stopMotion()
{
  std::lock_guard<std::mutex> lk(mtx_);
  geometry_msgs::msg::Twist zero;
  cmd_vel_pub_->publish(zero);
}

void ZlIo::startVisualKick()
{
  std::lock_guard<std::mutex> lk(mtx_);
  if (kick_rl_active_) {
    return;
  }
  // 先切 RL（特技模式，头部交回策略低头看球），再切踢球动作模型。
  enterKickMode();
  sendKickIntent("model", "switch_model", kick_model_name_.c_str());
  brain_->log->log(
    "zl_io/kick",
    format("视觉踢球：已切 RL 并请求模型 %s（超时 %.1fs）", kick_model_name_.c_str(),
           kick_rl_timeout_sec_));
}

// ------------------------------------------------------------ 状态查询 ----

uint8_t ZlIo::zlState() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return zl_state_;
}

bool ZlIo::kickActive() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return kick_rl_active_;
}

// ------------------------------------------------------------ 序列器 ----

void ZlIo::tick()
{
  std::lock_guard<std::mutex> lk(mtx_);
  tickKickMode();
  stepSequencer();
}

void ZlIo::stepSequencer()
{
  if (seq_target_ == SeqTarget::None) {
    return;
  }
  if (nowSec() - last_intent_time_sec_ < intent_interval_sec_) {
    return;
  }

  const uint8_t s = zl_state_;
  switch (seq_target_) {
    case SeqTarget::Walk:
      if (isZlActive()) {
        seq_target_ = SeqTarget::None;
        return;
      }
      if (s == ZL_FAULT) {
        sendIntent("reset", "reset_fault");
      } else if (s == ZL_DAMP) {
        sendIntent("status", "reinit");
      } else if (s == ZL_IDLE) {
        sendIntent("status", "ready");
      } else if (s == ZL_DO_READY || s == ZL_READY) {
        sendIntent("status", "walk");
      }
      // RL_STANDBY / TRANSITIONING / BOOTING / INIT：等待状态机自动推进。
      break;

    case SeqTarget::Damp:
      if (s == ZL_DAMP) {
        seq_target_ = SeqTarget::None;
        return;
      }
      if (s != ZL_FAULT && s != ZL_SHUTDOWN) {
        sendIntent("status", "damp");
      }
      break;

    case SeqTarget::None:
    default:
      break;
  }
}

// ------------------------------------------------------------ 踢球交接 ----

void ZlIo::enterKickMode()
{
  sendKickIntent("config", "mode", "rl");
  kick_rl_active_ = true;
  kick_started_sec_ = nowSec();
  kick_model_seen_ = false;
  kick_model_last_seen_sec_ = 0.0;
  restore_pending_ = false;
  restore_attempts_ = 0;
  // 新一轮踢球开始，清掉上一轮残留的速度抑制窗。
  velocity_suppress_until_sec_ = 0.0;
}

void ZlIo::finishKickMode(const std::string & reason)
{
  kick_rl_active_ = false;
  // 击球瞬间本体可能仍在侧倾：11-09 实测 motion ended 后 1 s 内侧倾 17°→81°。
  // 结束后开一个短抑制窗，避免 brain 立即追球/加速把机身压翻。
  velocity_suppress_until_sec_ = monoNowSec() + std::max(0.0, kick_settle_suppress_sec_);
  sendKickIntent("config", "mode", "mix");
  restore_pending_ = true;
  restore_attempts_ = 1;
  restore_last_send_sec_ = nowSec();
  brain_->log->log(
    "zl_io/kick",
    format("踢球结束（%s）：切回 MIX（第 %d/%d 次），速度抑制 %.2fs", reason.c_str(),
           restore_attempts_, restore_mix_attempts_, kick_settle_suppress_sec_));
}

void ZlIo::tickKickMode()
{
  const double now_sec = nowSec();
  if (kick_rl_active_ && (now_sec - kick_started_sec_) > kick_rl_timeout_sec_) {
    finishKickMode("超时兜底（未见模型回切）");
    return;
  }
  if (!restore_pending_) {
    return;
  }
  if (restore_attempts_ >= restore_mix_attempts_) {
    restore_pending_ = false;
    if (restore_mix_attempts_ > 1) {
      brain_->log->warn(
        "zl_io/kick",
        format("切回 MIX 已重发 %d 次仍未确认；若头部不跟 brain，请执行 board_ctl.sh prep",
               restore_mix_attempts_));
    }
    return;
  }
  if ((now_sec - restore_last_send_sec_) >= restore_mix_interval_sec_) {
    sendKickIntent("config", "mode", "mix");
    restore_last_send_sec_ = now_sec;
    ++restore_attempts_;
  }
}

void ZlIo::sendKickIntent(const char * intent_type, const char * command, const char * action)
{
  std::string src = switch_model_intent_source_;
  int pri = switch_model_intent_priority_;
  if (src.empty() && pri < 0 && auto_match_intent_priority_ && has_active_intent_ &&
    !active_intent_source_.empty())
  {
    // 跟随板侧 router 当前会话：router 只拒 candidate > active，同值可通过。
    src = active_intent_source_;
    pri = static_cast<int>(active_intent_priority_);
  } else if (src.empty() && pri < 0 && auto_match_intent_priority_ && !has_active_intent_ &&
    !active_intent_warned_)
  {
    active_intent_warned_ = true;
    brain_->log->warn(
      "zl_io/kick",
      format(
        "未收到 %s：踢球交接按 intent_source=%s priority=%d 发送；若会话被 teleop(0) 锁存会被拒",
        active_intent_topic_.c_str(), intent_source_.c_str(), intent_priority_));
  }
  sendIntent(intent_type, command, action, src, pri);
}

void ZlIo::sendIntent(
  const char * intent_type, const char * command, const char * action,
  const std::string & source_override, int priority_override)
{
  robot_interfaces::msg::RobotIntent intent;
  intent.header.stamp = brain_->now();
  intent.source = source_override.empty() ? intent_source_ : source_override;
  intent.intent_type = intent_type;
  intent.command = command;
  intent.action = action;
  const int priority = priority_override >= 0 ? priority_override : intent_priority_;
  intent.priority = static_cast<uint8_t>(std::clamp(priority, 0, 255));
  intent_pub_->publish(intent);
  last_intent_time_sec_ = nowSec();
  brain_->log->log(
    "zl_io/intent",
    format(
      "%s/%s %s (source=%s priority=%d zl_state=%u)", intent_type, command, action,
      intent.source.c_str(), priority, static_cast<unsigned>(zl_state_)));
}

// ------------------------------------------------------------ 上行回调 ----

void ZlIo::onZlState(const robot_interfaces::msg::RobotState & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);
  if (!zl_state_seen_) {
    zl_state_seen_ = true;
    brain_->log->log(
      "zl_io/state", format("收到 zl_robot 状态: %s(%u)", msg.state_name.c_str(),
                            static_cast<unsigned>(msg.state)));
  }
  zl_state_ = msg.state;
}

void ZlIo::onJointStates(const robot_interfaces::msg::JointStates & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);
  for (const auto & state : msg.states) {
    if (state.joint_name == "head_yaw") {
      head_yaw_rad_ = state.position;
    } else if (state.joint_name == "head_pitch") {
      head_pitch_rad_ = state.position;
    }
  }
  // 与旧桥 publishLowState 等价：brain 语义下 headYaw/headPitch 是相机光轴姿态，
  // 反馈侧按 head_pitch_offset_deg 反向换算，必须与命令侧成对。
  brain_->data->headYaw = fb_head_yaw_sign_ * head_yaw_rad_;
  brain_->data->headPitch =
    fb_head_pitch_sign_ * (head_pitch_rad_ - head_pitch_offset_deg_ * kDeg2Rad);
}

void ZlIo::onNavAll(const yesense_interface::msg::NavAll & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);
  euler_roll_deg_ = msg.euler.roll;
  euler_pitch_deg_ = msg.euler.pitch;
  euler_yaw_deg_ = msg.euler.yaw;
  nav_quat_[0] = msg.quat.q0;
  nav_quat_[1] = msg.quat.q1;
  nav_quat_[2] = msg.quat.q2;
  nav_quat_[3] = msg.quat.q3;
  nav_stamp_sec_ = nowSec();
  has_nav_ = true;
}

void ZlIo::onOdom(const nav_msgs::msg::Odometry & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);

  // odom 的 position 是世界轴累计位移，orientation 是相对启动朝向的相对角，两者不同系
  // （robot_odometry 的已知契约）。用本体 IMU（nav_all）绝对朝向自算
  // yaw_origin = yaw_world - theta_rel，把 position 旋回启动 odom 系，使其与 theta_rel 同系。
  const auto & q = msg.pose.pose.orientation;
  const double theta_rel = std::atan2(
    2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));

  const double px = msg.pose.pose.position.x;
  const double py = msg.pose.pose.position.y;
  if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(theta_rel)) {
    return;
  }
  // 自洽坐标的新来源必须携带新鲜采样时间，避免旧位姿让 Brain 持续误认为有效。
  const double receive_sec = nowSec();
  double accepted_stamp_sec = receive_sec;
  if (odom_consistent_) {
    const double sample_sec = rclcpp::Time(msg.header.stamp).seconds();
    const double age = receive_sec - sample_sec;
    if (sample_sec <= 0.0 || age < -kOdomFutureToleranceSec || age > kConsistentOdomMaxAgeSec) {
      return;
    }
    // 同一时钟纪元拒绝重复和乱序；真正换纪元后允许新鲜采样重新建立时间。
    const bool clock_reset = receive_sec < odom_stamp_sec_ - kOdomFutureToleranceSec;
    if (!clock_reset && sample_sec <= odom_stamp_sec_) {
      return;
    }
    accepted_stamp_sec = sample_sec;
  }
  // 新来源保留采样时刻，不能将接收时间作为新的有效期起点。
  odom_stamp_sec_ = accepted_stamp_sec;

  double ox = px, oy = py;
  if (!odom_consistent_ && has_nav_) {
    const double yaw_origin = euler_yaw_deg_ * kDeg2Rad - theta_rel;
    const double c = std::cos(yaw_origin);
    const double s = std::sin(yaw_origin);
    ox = c * px + s * py;
    oy = -s * px + c * py;
  }
  // nav_all 尚未到达时透传原始 odom（约首帧），随后自动收敛到正确系。

  // 新来源位置与朝向已经同系，不再重复旋转或套用旧模型的尺度修正。
  const double odom_scale = odom_consistent_ ? consistent_odom_scale_ :
    brain_->config->get_robot_odom_factor();
  brain_->data->robotPoseToOdom.x = ox * odom_scale;
  brain_->data->robotPoseToOdom.y = oy * odom_scale;
  brain_->data->robotPoseToOdom.theta = theta_rel;

  transCoord(
    brain_->data->robotPoseToOdom.x, brain_->data->robotPoseToOdom.y,
    brain_->data->robotPoseToOdom.theta,
    brain_->data->odomToField.x, brain_->data->odomToField.y, brain_->data->odomToField.theta,
    brain_->data->robotPoseToField.x, brain_->data->robotPoseToField.y,
    brain_->data->robotPoseToField.theta);
  if (accepted_odom_debug_pub_) {
    // consistent路径保留输入的精确采样时间，不经double秒数往返转换。
    // 这是Brain局部位姿的接收取证，不是独立真值或新的里程计来源。
    nav_msgs::msg::Odometry accepted;
    if (odom_consistent_) accepted.header.stamp = msg.header.stamp;
    else accepted.header.stamp = brain_->now();
    accepted.header.frame_id = "odom";
    accepted.child_frame_id = "torso_link";
    accepted.pose.pose.position.x = brain_->data->robotPoseToOdom.x;
    accepted.pose.pose.position.y = brain_->data->robotPoseToOdom.y;
    accepted.pose.pose.orientation.z = std::sin(theta_rel * 0.5);
    accepted.pose.pose.orientation.w = std::cos(theta_rel * 0.5);
    accepted.pose.covariance[0] = accepted.pose.covariance[7] =
      accepted.pose.covariance[35] = 1e6;
    accepted_odom_debug_pub_->publish(accepted);
  }
}

void ZlIo::onRecovery(const std_msgs::msg::UInt8MultiArray & msg)
{
  // 契约：3 字节 {state, is_recovery_available, planner_index}
  // （见 docs/02_interface/zl_robot_required_interfaces.md）。
  if (msg.data.size() < 3) {
    brain_->log->warn(
      "zl_io/recovery", format("recovery_state 数据不足 3 字节(%zu)，丢弃", msg.data.size()));
    return;
  }
  if (msg.data[0] > 3) {
    brain_->log->warn(
      "zl_io/recovery", format("未知 recovery state=%u，丢弃", static_cast<unsigned>(msg.data[0])));
    return;
  }

  std::lock_guard<std::mutex> lk(mtx_);
  RobotRecoveryStateData recoveryState;
  recoveryState.state = msg.data[0];
  recoveryState.is_recovery_available = msg.data[1];
  recoveryState.current_planner_index = msg.data[2];

  const RobotRecoveryState recoveryStateMap[] = {
    RobotRecoveryState::IS_READY,
    RobotRecoveryState::IS_FALLING,
    RobotRecoveryState::HAS_FALLEN,
    RobotRecoveryState::IS_GETTING_UP,
  };
  brain_->data->recoveryState = recoveryStateMap[static_cast<int>(recoveryState.state)];
  brain_->data->isRecoveryAvailable = static_cast<bool>(recoveryState.is_recovery_available);
  brain_->data->currentRobotModeIndex = static_cast<int>(recoveryState.current_planner_index);
}

void ZlIo::onRlStatus(const robot_interfaces::msg::RlManagerStatus & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);
  if (!kick_rl_active_) {
    return;
  }
  const double now_sec = nowSec();
  if (msg.active_model == kick_model_name_) {
    if (!kick_model_seen_) {
      kick_model_seen_ = true;
      brain_->log->log("zl_io/kick", format("踢球模型已激活: %s", msg.active_model.c_str()));
    }
    kick_model_last_seen_sec_ = now_sec;
    return;
  }
  // 已确认切到踢球模型、且已回切到其它模型并稳定一小段（避开切换瞬时抖动）。
  constexpr double kBackDebounceSec = 0.3;
  if (kick_model_seen_ && (now_sec - kick_model_last_seen_sec_) >= kBackDebounceSec) {
    finishKickMode("回合结束，模型已回切: " + msg.active_model);
  }
}

void ZlIo::onActiveIntent(const robot_interfaces::msg::RobotIntent & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);
  active_intent_source_ = msg.source;
  active_intent_priority_ = msg.priority;
  if (!has_active_intent_) {
    has_active_intent_ = true;
    brain_->log->log(
      "zl_io/priority",
      format("会话意图来源已同步: source=%s priority=%u（踢球交接跟随该来源/优先级）",
             active_intent_source_.c_str(), static_cast<unsigned>(active_intent_priority_)));
  }
}

// ------------------------------------------------------------ 工具 ----

double ZlIo::nowSec() const
{
  return brain_->now().seconds();
}

ZlIo::NavSnapshot ZlIo::navSnapshot() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  NavSnapshot out;
  out.has_nav = has_nav_;
  out.roll_deg = euler_roll_deg_;
  out.pitch_deg = euler_pitch_deg_;
  out.yaw_deg = euler_yaw_deg_;
  for (int i = 0; i < 4; ++i) {
    out.q[i] = nav_quat_[i];
  }
  out.stamp_sec = nav_stamp_sec_;
  return out;
}

double ZlIo::lastOdomSec() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return odom_stamp_sec_;
}

bool ZlIo::odometryAllowsMotion() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  if (!odom_consistent_) {
    return true;
  }
  const double age = nowSec() - odom_stamp_sec_;
  return odom_stamp_sec_ > 0.0 && age >= -kOdomFutureToleranceSec &&
         age <= kConsistentOdomMaxAgeSec;
}

bool ZlIo::isZlActive() const
{
  return zl_state_ == ZL_RL_ACTIVE || zl_state_ == ZL_RULE_ACTIVE;
}

}  // namespace robot_io
