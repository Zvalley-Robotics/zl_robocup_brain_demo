#include <iostream>
#include <string>
#include <fstream> 
#include <algorithm>
#include <yaml-cpp/yaml.h>

#include "brain.h"
#include "utils/print.h"
#include "utils/math.h"
#include "utils/misc.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std;
using std::placeholders::_1;

#define SUB_STATE_QUEUE_SIZE 1

Brain::Brain() : rclcpp::Node("brain_node")
{
    // Initialize TF broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(
        *this,
        rclcpp::QoS(10).transient_local());

    // need to declare parameters before accessing them, and use dot notation for nested parameters
    
    declare_parameter<int>("game.team_id", 0);
    declare_parameter<bool>(GAME_AGENT_MODE_PARAM, false);
    // player_id parameter validation
    rcl_interfaces::msg::ParameterDescriptor player_id_desc;
    player_id_desc.description = "player_id must be an integer between 1 and 11";
    player_id_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
    player_id_desc.integer_range.resize(1);
    player_id_desc.integer_range[0].from_value = 1;
    player_id_desc.integer_range[0].to_value = HL_MAX_NUM_PLAYERS;
    player_id_desc.integer_range[0].step = 1;
    declare_parameter<int>("game.player_id", 29);
    declare_parameter<string>("game.field_type", "");

    // player_role parameter validation
    rcl_interfaces::msg::ParameterDescriptor player_role_desc;
    player_role_desc.description = "Player role must be either 'striker' or 'goal_keeper'";
    player_role_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
    declare_parameter<string>("game.player_role", "", player_role_desc);
    declare_parameter<bool>("game.treat_person_as_robot", false);
    declare_parameter<int>("game.number_of_players", 2);

    declare_parameter<string>("robot.robot_name", "");
    declare_parameter<double>("robot.robot_height", 1.0);
    declare_parameter<double>("robot.odom_factor", 1.0);
    declare_parameter<double>("robot.vx_factor", 0.5);
    declare_parameter<double>("robot.yaw_offset", 0.0);
    declare_parameter<double>("robot.vx_limit", 1.0);
    declare_parameter<double>("robot.vy_limit", 0.4);
    declare_parameter<double>("robot.vtheta_limit", 1.0);
    // 后退（vx<0）单独上限：vx_limit 正负对称，各节点都能要求 1.0 m/s 后退而双足后退不稳。
    // setVelocity 出口统一夹住（覆盖 moveToPoseOnField* 的 -vxLimit 与 READY 的 -0.4）；
    // <=0 或 >= vx_limit 时等于关闭。Adjust 另有 XML backoff_speed 更紧。
    declare_parameter<double>("robot.vx_limit_backward", 0.25);
    declare_parameter<double>("robot.min_vx", 0.4);
    declare_parameter<double>("robot.min_vy", 0.3);
    declare_parameter<double>("robot.min_vtheta", 0.2);
    // 本体对「合速度模 < 0.1 m/s」不响应（不是逐轴）。min_lin_speed 是合模抬升门限
    // （0.12 = 死区 0.1 + 20% 余量）：按比例放大模长、保持方向，后退方向不抬升。
    // min_vx/min_vy 仍保留，行为树侧用它们判断“某分量是否可表达”。
    declare_parameter<double>("robot.min_lin_speed", 0.12);
    // 2026-11-05：3588 侧取消速度映射缩减后，本体 1:1 执行下发值（旧版实际约折半）。
    // speed_scale 是“下发值→本体”的统一缩放（1.0 = 新真实速度，0.5 = 回退旧手感）；
    // yaw_track_* 收口原本硬编码的 *4.0 球方位跟踪增益（SimpleChase / 门将 / Assist 三处）。
    declare_parameter<double>("robot.speed_scale", 1.0);
    declare_parameter<double>("robot.yaw_track_gain", 2.0);
    declare_parameter<double>("robot.yaw_track_deadband", 0.08);
    // 2026-11-06：转向纯 P 在执行滞后+增益翻倍下形成极限环（球静止时机器人来回摆）。
    // yaw_track_damping 是 D 项增益，作用于 odom 实测角速度；<=0 退化为原纯 P（回退开关）。
    // yaw_rate_lpf_hz 抑制步态振动带进来的角速度噪声。
    declare_parameter<double>("robot.yaw_track_damping", 0.20);
    declare_parameter<double>("robot.yaw_rate_lpf_hz", 8.0);
    // 下发出口统一的斜率（加速度）限幅，替代各阶段各自打补丁：一拍内全额反向本质是阶跃
    // 指令，限速后变斜坡。板端 rl_manager 已有 ≈2.5 m/s² 平滑，本层须更小才有增量意义。
    // <=0 关闭，三轴可独立回退。
    declare_parameter<double>("robot.accel_limit_vx", 2.0);
    declare_parameter<double>("robot.accel_limit_vy", 1.2);
    declare_parameter<double>("robot.accel_limit_vtheta", 8.0);
    // 减速（刹车）斜率：|指令| 变小或换向时生效，<=0 回退到对应加速度值
    declare_parameter<double>("robot.decel_limit_vx", 0.0);
    declare_parameter<double>("robot.decel_limit_vy", 0.0);
    declare_parameter<double>("robot.decel_limit_vtheta", 0.0);
    // 距离-速度刹车模型：v = −a·τ + √((a·τ)² + 2a·d)
    declare_parameter<double>("robot.brake_decel", 2.5);
    declare_parameter<double>("robot.brake_latency", 0.20);
    // 定位振荡取证：>0 时按该频率打印 dbg/track（pose / 球机体系 / 球场地系 / 决策），0 关闭
    declare_parameter<double>("debug.track_log_hz", 0.0);
    // 头部关节限位（rad）。pitch 语义是「相机光轴下俯角」，由 bridge 的
    // head_pitch_offset_deg=-36 换算成 head_pitch 关节角；新头部相机相对 head_pitch_link
    // 固定下俯 36°、pitch 关节限位 ±25°，故光轴可达 11°~61° 下俯，越界会被夹死。
    declare_parameter<double>("robot.head_yaw_limit_left", 1.1);
    declare_parameter<double>("robot.head_yaw_limit_right", -1.1);
    declare_parameter<double>("robot.head_pitch_limit_up", 0.20);
    declare_parameter<double>("robot.head_pitch_limit_down", 1.06);
    // 头部单一出口仲裁（2026-11-17，见 RobotClient::requestHead）：丢球时 CamFindBall(700ms)
    // 与 CamFastScan(550ms) 同拍各发一套 6 点目标，头部在两套端点间反复反向（剧烈晃头）。
    declare_parameter<double>("robot.head_arb_stale_ms", 200.0);   // 请求多久不刷新算“不再请求”
    declare_parameter<double>("robot.head_arb_dedup_rad", 0.01);   // 与上一条目标相差小于此不重发
    declare_parameter<double>("robot.head_sweep_interval_ms", 550.0);  // 扫场每点驻留（夹 ≥500）

    declare_parameter<double>("strategy.ball_confidence_threshold", 50.0);   
    declare_parameter<double>("strategy.ball_memory_timeout", 3.0);
    declare_parameter<double>("strategy.tm_ball_dist_threshold", 3.0);
    // 队友球位共享（P0）：包新鲜度上限、与本机检测的冲突门限、贴身 hint 的队友距离、
    // 来源切换后的踢球保护窗。
    declare_parameter<double>("strategy.tm_ball_lag_max_ms", 300.0);
    declare_parameter<double>("strategy.tm_ball_conflict_dist", 1.0);
    declare_parameter<double>("strategy.tm_ball_hint_peer_sep", 1.5);
    declare_parameter<double>("strategy.ball_src_guard_ms", 150.0);
    // 球速估计（2026-11-16，门将拦截第 1 档）：用 0.35 s 窗口最小二乘拟合场系球速。
    // window_s 越大越平滑但滞后越多；max_speed 之外视为检测跳变并重置；reset_ms 无检测
    // 即清空（旧速度不得用于外推）；min_samples 不足时不输出速度。
    declare_parameter<double>("strategy.ball_est_window_s", 0.35);
    declare_parameter<double>("strategy.ball_est_max_speed", 6.0);
    declare_parameter<double>("strategy.ball_est_reset_ms", 500.0);
    declare_parameter<int>("strategy.ball_est_min_samples", 3);
    // 门将球位新鲜度：超过 stale_ms 无新检测即不驱动站位（回 hold 守门），
    // 预测只用 ≤ pred_max_age_ms 的球位（再旧的话外推误差大于收益）。
    declare_parameter<double>("strategy.gk_ball_stale_ms", 500.0);
    declare_parameter<double>("strategy.gk_ball_pred_max_age_ms", 400.0);
    // 门将拦截几何（2026-11-17，第 2 档）：
    //   gk_pred_min_speed   球速超过该值才做“过线点预判”（m/s，原硬编码 0.35 漏慢球）
    //   gk_pred_max_tti_s   预判时间窗（s，原硬编码 2.0；放宽到 3.5 补地滚球）
    //   gk_target_track_mps 站位目标点速率上限（m/s），抗定位跳变（实测每 10 s 跳 >0.3 m）
    declare_parameter<double>("strategy.gk_pred_min_speed", 0.25);
    declare_parameter<double>("strategy.gk_pred_max_tti_s", 3.5);
    declare_parameter<double>("strategy.gk_target_track_mps", 0.6);
    // 门将 y 轴加速度上限（m/s²）：全局 accel_limit_vy=1.2 从 0 到 0.6 要 0.5 s，
    // 在 1 s 级扑救窗口里吃掉一半时间；取 2.5 与本体自身平滑值对齐（板端 rl_manager ≈2.5）。
    // <=0 回退到全局值。
    declare_parameter<double>("strategy.gk_accel_vy", 2.5);
    declare_parameter<bool>("strategy.limit_near_ball_speed", true);
    declare_parameter<double>("strategy.near_ball_speed_limit", 0.3);
    declare_parameter<double>("strategy.near_ball_range", 4.0);
    declare_parameter<bool>("strategy.abort_kick_when_ball_moved", false);
    declare_parameter<bool>("strategy.enable_bypass", false);
    declare_parameter<bool>("strategy.enable_shoot", false);
    declare_parameter<bool>("strategy.enable_directional_kick", false);

    declare_parameter<bool>("strategy.use_squat_block", false);
    declare_parameter<double>("strategy.squat_block_msecs", 2000.0);
    declare_parameter<bool>("strategy.use_move_block", true);
    declare_parameter<double>("strategy.move_block_msecs", 2000.0);
    declare_parameter<bool>("strategy.enable_auto_visual_kick", false);
    declare_parameter<bool>("strategy.enable_auto_visual_defend", false);

    declare_parameter<bool>("strategy.power_shoot.enable", false);
    declare_parameter<bool>("strategy.power_shoot.use_for_kickoff", false);
    declare_parameter<double>("strategy.power_shoot.xmin", 0.5);
    declare_parameter<double>("strategy.power_shoot.xmax", 1.0);
    declare_parameter<double>("strategy.power_shoot.ymin", -0.5);
    declare_parameter<double>("strategy.power_shoot.ymax", 0.5);
    declare_parameter<double>("strategy.shoot.threat_threshold", 0.0);
    declare_parameter<double>("strategy.shoot.xmin", 0.5);
    declare_parameter<double>("strategy.shoot.xmax", 1.0);
    declare_parameter<double>("strategy.shoot.ymin", -0.5);
    declare_parameter<double>("strategy.shoot.ymax", 0.5);
    declare_parameter<bool>("strategy.cooperation.enable_role_switch", true);
    declare_parameter<double>("strategy.cooperation.ball_control_cost_threshold", 10.0);

    declare_parameter<int>("obstacle_avoidance.depth_sample_step", 16);
    declare_parameter<double>("obstacle_avoidance.obstacle_min_height", 0.15);
    declare_parameter<double>("obstacle_avoidance.grid_size", 0.2);
    declare_parameter<double>("obstacle_avoidance.max_x", 0.2);
    declare_parameter<double>("obstacle_avoidance.max_y", 0.2);
    declare_parameter<double>("obstacle_avoidance.exclusion_x", 0.25);
    declare_parameter<double>("obstacle_avoidance.exclusion_y", 0.4);
    declare_parameter<double>("obstacle_avoidance.ball_exclusion_radius", 0.3);
    declare_parameter<double>("obstacle_avoidance.ball_exclusion_height", 0.3);
    declare_parameter<double>("obstacle_avoidance.occupancy_threshold", 5.0);
    declare_parameter<double>("obstacle_avoidance.collision_threshold", 0.5);
    declare_parameter<double>("obstacle_avoidance.safe_distance", 2.0);
    declare_parameter<double>("obstacle_avoidance.avoid_secs", 3.0);
    declare_parameter<bool>("obstacle_avoidance.enable_freekick_avoid", false);
    declare_parameter<double>("obstacle_avoidance.freekick_start_placing_safe_distance", 0.5);
    declare_parameter<double>("obstacle_avoidance.freekick_start_placing_avoid_secs", 1.5);
    declare_parameter<double>("obstacle_avoidance.obstacle_memory_msecs", 500.0);
    declare_parameter<bool>("obstacle_avoidance.avoid_during_chase", false);
    declare_parameter<double>("obstacle_avoidance.chase_ao_safe_dist", 2.0);
    // 绕行角上限（度）：栅格只覆盖相机 FOV（半角约 40°），超出为盲区，必须限幅
    declare_parameter<double>("obstacle_avoidance.avoid_max_offset_deg", 30.0);
    // 贴身区（m）：障碍近于此距离时不再做小角度绕行（走廊模型会误判“已绕开”），改由调用方停车
    declare_parameter<double>("obstacle_avoidance.avoid_min_clearance_m", 0.8);
    // 绕行分支基准线速度（Chase/Assist 共用；原为两处硬编码 0.5）
    declare_parameter<double>("obstacle_avoidance.avoid_speed", 0.5);
    // ---- 速度包络（2026-11-17）：按“最紧清距”限制行进速度，只降不升 ----
    declare_parameter<bool>("obstacle_avoidance.speed_envelope_enable", true);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_decel", 2.5);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_latency", 0.20);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_floor", 0.7);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_ceiling", 2.2);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_margin", 0.35);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_tm_margin", 0.70);
    declare_parameter<double>("obstacle_avoidance.speed_envelope_field_margin", 0.60);
    declare_parameter<bool>("obstacle_avoidance.avoid_during_kick", false);
    declare_parameter<double>("obstacle_avoidance.kick_ao_safe_dist", 1.0);
    // base 原点（torso_link）-> 地面的 +z 偏置，必须与 zl 侧 head_to_base 原点离地高度一致；
    // 深度点云转 base 系与 obstacle_min_height 的基准都依赖它。
    declare_parameter<double>("obstacle_avoidance.base_height_m", 0.83);
    // 每帧发布深度点云/障碍栅格可视化（开销大，仅排查时开）
    declare_parameter<bool>("obstacle_avoidance.publish_pointcloud", false);
    
    declare_parameter<bool>("RLVisionKick.enableAutoVisualKick", true);
    declare_parameter<double>("RLVisionKick.autoVisualKickEnableDistMin", 0.5);
    declare_parameter<double>("RLVisionKick.autoVisualKickEnableDistMax", 1.5);
    declare_parameter<double>("RLVisionKick.autoVisualKickEnableAngle", 0.5);
    
    declare_parameter<int>("locator.min_marker_count", 5);
    declare_parameter<double>("locator.max_residual", 0.3);
    // 定位实验参数：默认保持现有 Brain 的调度、搜索及 1M 行为。
    declare_parameter<int>("locator.num_particles", 200);
    declare_parameter<double>("locator.trust_yaw_half_width_deg", 1.0);
    declare_parameter<bool>("locator.split_schedule", false);
    declare_parameter<double>("locator.split_retry_ms", 500.0);
    declare_parameter<double>("locator.split_min_drift_m", 0.2);
    declare_parameter<double>("locator.split_max_drift_m", 2.0);
    declare_parameter<bool>("locator.split_unbounded", false);
    declare_parameter<double>("locator.max_drift_speed", 0.1);
    declare_parameter<bool>("locator.one_m_multi_candidate", false);
    declare_parameter<bool>("locator.one_m_candidate_diagnostics", false);
    declare_parameter<double>("locator.one_m_score_margin", 0.02);
    declare_parameter<double>("locator.one_m_ambiguity_separation_m", 0.2);
    declare_parameter<bool>("locator.diagnostics", false);
    // 轨迹诊断默认关闭；间隔单位为秒。
    declare_parameter<bool>("locator.trace_enabled", false);
    declare_parameter<double>("locator.trace_interval_sec", 0.1);

    declare_parameter<bool>("enable_com", true);
    // Multi-instance: several teammates run on one host (same IP). The teammate table is
    // keyed by playerId and every player gets its own unicast port (30000 + teamId + playerId).
    declare_parameter<bool>("comm.multi_instance", false);

    // 相机话题/内参以 vision.yaml 为准；以下仅为 vision.yaml 缺失时的兜底值（实机 RealSense 命名）
    declare_parameter<string>("vision.image_camera_info_topic", "/camera/camera/color/camera_info");
    declare_parameter<string>("vision.depth_image_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    declare_parameter<string>("vision.depth_camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info");
    // 深度订阅 QoS：reliable | best_effort，必须与相机发布端一致，否则静默收不到
    declare_parameter<string>("vision.depth_qos", "reliable");


    declare_parameter<string>("game_control_ip", "0.0.0.0");

    declare_parameter<string>("tree_file_path", "");
    declare_parameter<string>("vision_config_path", "");
    declare_parameter<string>("vision_config_local_path", "");

    declare_parameter<int>("recovery.retry_max_count", 2);

    // Ground truth localization bypass (mocap/simulation), default off
    declare_parameter<bool>("ground_truth.enable_pose", false);
    declare_parameter<bool>("ground_truth.enable_ball", false);
    declare_parameter<double>("ground_truth.ball_timeout_sec", 1.0);
}

Brain::~Brain()
{

}

void Brain::init()
{
    
    config = std::make_shared<BrainConfig>(this);
    loadConfig();

    data = std::make_shared<BrainData>();
    locator = std::make_shared<Locator>();
    log = std::make_shared<BrainLog>(this);
    tree = std::make_shared<BrainTree>(this);
    client = std::make_shared<RobotClient>(this);
    communication = std::make_shared<BrainCommunication>(this);
    visualizer = std::make_shared<VisualizationPublisher>(this);

    locator->init(config->fieldDimensions, config->get_min_marker_count(), config->get_max_residual());

   
    tree->init();

   
    client->init(config->get_robot_name());

    
    communication->initCommunication();

    data->lastSuccessfulLocalizeTime = get_clock()->now();
    data->timeLastDet = get_clock()->now();
    data->timeLastDepth = get_clock()->now();
    data->timeLastLineDet = get_clock()->now();
    data->timeLastGamecontrolMsg = get_clock()->now();
    data->ball.timePoint = get_clock()->now();

    
    auto now = get_clock()->now();
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        data->tmStatus[i].isAlive = false;
        data->tmStatus[i].timeLastCom = now;
    }
    data->tmLastCmdChangeTime = now;


    string topic_suffix = "";
    if(config->get_robot_name() != "") {
        RCLCPP_INFO(this->get_logger(), "Robot name set to: %s", config->get_robot_name().c_str());
        topic_suffix = "/" + config->get_robot_name();
    }
    detectionsSubscription = create_subscription<vision_interface::msg::Detections>("/soccer/detection" + topic_suffix, SUB_STATE_QUEUE_SIZE, bind(&Brain::detectionsCallback, this, _1));
    subFieldLine = create_subscription<vision_interface::msg::LineSegments>("/soccer/line_segments" + topic_suffix, SUB_STATE_QUEUE_SIZE, bind(&Brain::fieldLineCallback, this, _1));
    // zl 侧反馈（里程计 / 头部关节 / 跌倒恢复 / 状态 / rl_manager / 路由会话）已由
    // RobotClient → ZlIo 直接订阅并写入 BrainData，不再经过原厂话题。
    headPoseSubscription = create_subscription<geometry_msgs::msg::PoseStamped>("/robot/head_to_base" + topic_suffix, SUB_STATE_QUEUE_SIZE, bind(&Brain::headPoseCallback, this, _1));

    // Ground truth bypass (mocap/simulation localization); subscriptions are only created when enabled
    if (config->get_gt_pose_enabled()) {
        gtPoseSubscription = create_subscription<geometry_msgs::msg::Pose2D>(
            "/soccer/ground_truth_pose" + topic_suffix, rclcpp::QoS(10),
            bind(&Brain::gtPoseCallback, this, _1));
    }
    if (config->get_gt_ball_enabled()) {
        gtBallSubscription = create_subscription<geometry_msgs::msg::Point>(
            "/soccer/ground_truth_ball" + topic_suffix, rclcpp::QoS(10),
            bind(&Brain::gtBallCallback, this, _1));
    }

    imageCameraInfoSubscription = create_subscription<sensor_msgs::msg::CameraInfo>(
        config->get_image_camera_info_topic(), SUB_STATE_QUEUE_SIZE, bind(&Brain::imageCameraInfoCallback, this, _1));

    depthCameraInfoSubscription = create_subscription<sensor_msgs::msg::CameraInfo>(
        config->get_depth_camera_info_topic(), SUB_STATE_QUEUE_SIZE, bind(&Brain::depthCameraInfoCallback, this, _1));

    // create publisher for field dimensions
    pubFieldDimensions = create_publisher<std_msgs::msg::Float64MultiArray>("/soccer/field_dimensions" + topic_suffix, rclcpp::QoS(1).transient_local());
    
    // create publisher for robot pose
    pubRobotPose = create_publisher<geometry_msgs::msg::Pose2D>("/soccer/robot_pose" + topic_suffix, 10);
    pubBallPosition = create_publisher<geometry_msgs::msg::Point>("/soccer/ball_position" + topic_suffix, 10);
    pubTeammatesPoses = create_publisher<std_msgs::msg::Float64MultiArray>("/soccer/teammates_poses" + topic_suffix, 10);
    pubBrainStrategy = create_publisher<std_msgs::msg::String>("/soccer/brain_strategy" + topic_suffix, 10);
    pubKickBall = create_publisher<brain::msg::Kick>("/kick_ball", 10);

    // subscribe to depth image topic
    // QoS 必须与相机发布端一致：RealSense 默认 SYSTEM_DEFAULT（通常 reliable），
    // 而 robot_bringup/camera_params.yaml 那套是 SENSOR_DATA（best_effort）；
    // 错配时订阅端静默收不到（不报错，只是永不进回调）。
    const string depthQosStr = config->get_depth_qos();
    rclcpp::QoS depthQos = (depthQosStr == "best_effort")
                               ? rclcpp::QoS(rclcpp::SensorDataQoS())
                               : rclcpp::QoS(SUB_STATE_QUEUE_SIZE);
    string depthTopic = config->get_depth_image_topic();
    if (depthTopic.find("compressed") != std::string::npos) {
        compressedDepthImageSubscription = create_subscription<sensor_msgs::msg::CompressedImage>(
            depthTopic, depthQos, bind(&Brain::compressedDepthImageCallback, this, _1));
    } else {
        depthImageSubscription = create_subscription<sensor_msgs::msg::Image>(
            depthTopic, depthQos, bind(&Brain::depthImageCallback, this, _1));
    }
    RCLCPP_INFO(get_logger(), "[depth] subscribe %s qos=%s", depthTopic.c_str(), depthQosStr.c_str());


    // agent soccer demo without referee, run specialized settings
    if (get_parameter(GAME_AGENT_MODE_PARAM).as_bool()) {
        RCLCPP_INFO(get_logger(), "Running in agent mode, subscribing game state");

        RCLCPP_INFO(get_logger(), "Running in agent mode, subscribing team_id changes");
        param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);

        auto team_id_cb = [this](const rclcpp::Parameter &p) {
            RCLCPP_INFO(this->get_logger(), "[team_id_subscripter] team_id changed to %ld", p.as_int());
            if (!this->communication->isTeamChanged()) {
                RCLCPP_INFO(this->get_logger(), "[team_id_subscripter] team_id not changed, no need to re-init communication");
                return;
            }
            this->communication.reset();
            RCLCPP_INFO(this->get_logger(), "[team_id_subscripter] communication reset success");
            // Recreate communication object
            this->communication = std::make_shared<BrainCommunication>(this);
            this->communication->initCommunication();
            // Reset teammates communication status
            auto now = get_clock()->now();
            for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
                this->data->tmStatus[i].isAlive = false;
                this->data->tmStatus[i].timeLastCom = now;
            }
            RCLCPP_INFO(this->get_logger(), "[team_id_subscripter] communication re-init success");
        };

        auto player_role_cb = [this](const rclcpp::Parameter &p) {
            RCLCPP_INFO(this->get_logger(), "[player_role_subscripter] role from %s changed to %s",
                        this->tree->getEntry<string>("player_role").c_str(),
                        p.as_string().c_str());
            tree->setEntry<string>("player_role", config->get_player_role());
        };

        team_id_handle_ = param_subscriber_->add_parameter_callback("game.team_id", team_id_cb);
        player_role_handle_ = param_subscriber_->add_parameter_callback("game.player_role", player_role_cb);

        // agent mode does not have referee, force clear penalty status to allow normal communication between teammates
        std::fill(std::begin(this->data->penalty), std::end(this->data->penalty), PENALTY_NONE);
    };

    // Publish field dimensions information (called after publisher creation)
    publishFieldDimensions();
}

void Brain::loadConfig()
{
    // 相机话题与内参以 vision.yaml 为准（实机单一来源）：brain 只读不写，
    // 未提供时回退到 vision.* 参数，保证不部署视觉时仍能启动。
    string visionConfigPath, visionConfigLocalPath;
    get_parameter("vision_config_path", visionConfigPath);
    get_parameter("vision_config_local_path", visionConfigLocalPath);
    if (!filesystem::exists(visionConfigPath)) {
        // vision.yaml 缺失时优雅降级（仿真/bridge 模式可能不部署视觉）。
        // 注意：外参与此无关——/robot/head_to_base 自带 eye(optical)->base，不再需要 camera.extrin。
        RCLCPP_WARN(get_logger(),
                    "vision_config_path %s not exists, camera topics/intrinsics fall back to brain params",
                    visionConfigPath.c_str());
        config->handle();
        std::ostringstream oss;
        config->print(oss);
        prtDebug(oss.str());
        return;
    }
    // else
    YAML::Node vConfig = YAML::LoadFile(visionConfigPath);
    if (filesystem::exists(visionConfigLocalPath)) {
        YAML::Node vConfigLocal = YAML::LoadFile(visionConfigLocalPath);
        MergeYAML(vConfig, vConfigLocal);
    }

    // 从 vision.yaml 解析相机参数：话题 / 内参初值 /（不再解析 extrin）
    YAML::Node camera = vConfig["camera"];
    const bool cameraOk = camera && camera.IsMap();
    auto readCameraStr = [&](const char *key, string &out) {
        if (cameraOk && camera[key] && camera[key].IsScalar()) out = camera[key].as<string>();
    };
    auto deriveCameraInfoTopic = [](const string &imageTopic) -> string {
        // RealSense 约定：<前缀>/image_raw -> <前缀>/camera_info。
        // aligned_depth_to_color 的 camera_info 由驱动随图像发布，其内参已对齐到彩色相机。
        const string suffixes[] = {"/image_raw", "/image_rect_raw"};
        for (const auto &s : suffixes) {
            if (imageTopic.size() > s.size() &&
                imageTopic.compare(imageTopic.size() - s.size(), s.size(), s) == 0) {
                return imageTopic.substr(0, imageTopic.size() - s.size()) + "/camera_info";
            }
        }
        return "";
    };

    readCameraStr("depth_topic", config->depthImageTopicOverride);
    readCameraStr("intrin_topic", config->imageCameraInfoTopicOverride); // 彩色相机内参

    string depthInfoOverride;
    readCameraStr("depth_intrin_topic", depthInfoOverride); // 可选：显式指定深度内参话题
    if (!depthInfoOverride.empty()) {
        config->depthCameraInfoTopicOverride = depthInfoOverride;
    } else if (!config->depthImageTopicOverride.empty()) {
        config->depthCameraInfoTopicOverride = deriveCameraInfoTopic(config->depthImageTopicOverride);
    }

    if (cameraOk) {
        YAML::Node intrin = camera["intrin"];
        if (intrin && intrin.IsMap()) {
            auto readDouble = [&](const char *key, double fallback) -> double {
                if (intrin[key] && intrin[key].IsScalar()) return intrin[key].as<double>();
                return fallback;
            };
            double fx = readDouble("fx", 0.0);
            double fy = readDouble("fy", 0.0);
            double cx = readDouble("cx", 0.0);
            double cy = readDouble("cy", 0.0);
            if (fx > 1e-6 && fy > 1e-6) {
                config->depthCameraFx = fx;
                config->depthCameraFy = fy;
                config->depthCameraCx = cx;
                config->depthCameraCy = cy;
                // FOV 仅用于障碍表清理的视野判据，收到 camera_info 后会被更新得更准确
                config->depthCameraFovX = 2.0 * atan(640.0 / (2.0 * fx));
                config->depthCameraFovY = 2.0 * atan(480.0 / (2.0 * fy));
            }
        }
    }

    // 外参契约（2026-10）：/robot/head_to_base 就是 eye(optical)->base（躯干原点、去 yaw），
    // 旋转已含相机安装俯仰，点云直接左乘即得 base 系，不需要再做任何平移。
    // camera.extrin 已被 vision 标为 legacy，这里明确不读，避免"既乘头部位姿又乘外参"的重复变换。
    RCLCPP_INFO(get_logger(),
                "[vision] depth_topic=%s depth_info=%s color_info=%s intrinsics=%s(fx=%.1f fy=%.1f) base_height=%.2f "
                "(camera.extrin ignored: /robot/head_to_base provides eye->base)",
                config->get_depth_image_topic().c_str(),
                config->get_depth_camera_info_topic().c_str(),
                config->get_image_camera_info_topic().c_str(),
                config->depthCameraFx > 0 ? "vision.yaml" : "none",
                config->depthCameraFx, config->depthCameraFy,
                config->get_base_height_m());

    config->handle();

    // playerRole [striker, goal_keeper]
    string _playerRole = config->get_player_role();
    if (_playerRole != "striker" && _playerRole != "goal_keeper") {
        throw invalid_argument("player_role must be one of [striker, goal_keeper]. Got: " + _playerRole);
    }


    ostringstream oss;
    config->print(oss);
    prtDebug(oss.str());
}


void Brain::tick()
{
    // zl 状态推进序列器 + 踢球交接状态机（与行为树同节拍 100Hz，替代原桥接的 100ms timer）
    client->tick();

    // Output debug & log related information
    logDebugInfo();
    logLags();
    logStatusToConsole();
    logObstacleStatus();
    
    // Publish visualization markers
    publishVisualizationMarkers();
    
    // Publish odom to map TF transform
    publishOdomToMapTF();

    // Ground truth ball timeout (mutual exclusion: when gt ball is enabled, vision
    // ball is fully disabled — each run uses only one channel).
    // On timeout clear gt state; set lose_ball 2s after the loss.
    if (config->get_gt_ball_enabled()) {
        if (gtBallActive && !gtBallFresh()) {
            gtBallActive = false;
            data->ballDetected = false;
            tree->setEntry<bool>("ball_location_known", false);
            gtBallLostTime = get_clock()->now();
        }
        if (gtBallLostTime.seconds() > 0.0 && !data->ballDetected &&
            (get_clock()->now() - gtBallLostTime).seconds() > 2.0) {
            data->lose_ball = true;
        }
    }

    
    // Publish position information
    publishRobotPose();
    publishBallPosition();
    publishTeammatesPoses();
    
    // Publish kick message
    pubKickMsg();
    
    updateMemory();
    handleSpecialStates();
    handleCooperation();
    // 球速估计必须在 handleCooperation 之后：球源 LOCAL↔TM 切换、本机确认锁都在那一步落定，
    // 这里才拿得到本拍最终的 data->ball。
    updateBallEstimate();

    tree->tick();

    // 上报本 tick 实际执行的行为树策略链（UI 展示用；变更时发布 + 2s 心跳）
    publishBrainStrategy();
}

void Brain::handleSpecialStates() {

    const double KICKOFF_DURATION = 10.0; 
    string gameState = tree->getEntry<string>("gc_game_state");
    bool isKickoffSide = tree->getEntry<bool>("gc_is_kickoff_side");
    string gameSubStateType = tree->getEntry<string>("gc_game_sub_state_type");
    string gameSubState = tree->getEntry<string>("gc_game_sub_state");
    bool isFreekickKickoffSide = tree->getEntry<bool>("gc_is_sub_state_kickoff_side");
    auto now = get_clock()->now();

    if (gameState == "SET" && isKickoffSide) {
        data->isKickingOff = true;
        data->kickoffStartTime = now;
    } else if (msecsSince(data->kickoffStartTime) > KICKOFF_DURATION * 1000) {
        data->isKickingOff = false;
    }

    if (gameState == "PLAY" && gameSubStateType == "FREE_KICK" && isFreekickKickoffSide) {
        data->isFreekickKickingOff = true;
        data->freekickKickoffStartTime = now;
    } else if (msecsSince(data->freekickKickoffStartTime) > KICKOFF_DURATION * 1000) {
        data->isFreekickKickingOff = false;
        data->isDirectShoot = false;
    }

    static int lastScore = 0;
    if (data->score > lastScore) {
        tree->setEntry<bool>("we_just_scored", true);
        lastScore = data->score;
    }
    if (gameState == "SET") {
        tree->setEntry<bool>("we_just_scored", false);
    }
}

void Brain::handleCooperation() {
    auto log_ = [=](string msg) {
        log->debug("handleCooperation", msg);
    };
    log_("handle cooperation");

    const int CMD_COOLDOWN = 2000; // Minimum interval between commands in milliseconds to prevent oscillation
    const int COM_TIMEOUT = 5000; // Teammate information timeout in milliseconds, if no information is received within this time, the teammate is considered offline
    // lead 仲裁死区：cost 差在该范围内按 player_id 小者优先（跨机同结论）。原 0.1 在实机
    // 上几乎不起作用（cost 抖动幅度可达数个单位），放宽后需要挑战者明显更省才夺权。
    constexpr double LEAD_COST_DEADBAND = 0.8;

    int selfId = config->get_player_id();
    int selfIdx = selfId - 1;
    int numOfPlayers = config->get_num_of_players();
    string selfRole = config->get_player_role();

    vector<int> aliveTmIdxs = {}; // Indices of all alive teammates, excluding self

    // Update own status
    data->tmImAlive = 
        (data->penalty[selfIdx] == PENALTY_NONE) // Am I online, i.e., not penalized
        && tree->getEntry<bool>("odom_calibrated"); // Before localization is successful, consider own information unreliable, not alive
    updateCostToKick(); // Current cost to kick
    log_(format("ImAlive: %d, myCost: %.1f", data->tmImAlive, data->tmMyCost));

    
    // Determine the number of alive teammates on the field based on referee data and log it
    int gcAliveCount = 0; 
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++)
    {
        if (data->penalty[i] == PENALTY_NONE) {
            gcAliveCount++;
        }
    }
    log_(format("gcAliveCnt: %d", gcAliveCount));

    // Process teammate information. If no information is received from a teammate for a long time, or the teammate is in a penalty state according to the referee, the teammate is considered offline. Then get the indices of all online teammates.
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx) continue; // Skip self

        if (
            data->penalty[i] != PENALTY_NONE // Penalized by referee
            || msecsSince(data->tmStatus[i].timeLastCom) > COM_TIMEOUT // Or communication timeout
        ) {
            data->tmStatus[i].isAlive = false;
            data->tmStatus[i].isLead = false;
        }
        
        if (data->tmStatus[i].isAlive) {
            aliveTmIdxs.push_back(i);
            log->log_scalar("tm_status",format("tm_alive_scalar_%d", i + 1), data->tmStatus[i].cost);
            log->log_scalar("tm_status",format("tm_lead_scalar_%d", i + 1), data->tmStatus[i].isLead ? 1 : 0);
        }
    }
    log_(format("alive TM Count: %d", aliveTmIdxs.size()));

    // Log information of currently alive teammates
    log_(format("Self: cost: %.1f, isLead: %d", data->tmMyCost, data->tmImLead));


    // Process ball position information reported by teammates
    // 2026-11-10 P0：来源状态机 + 冲突止血 + 贴身 hint。
    // 原实现只看 ballDetected 就采信，可能拿到队友 ball_memory_timeout 记忆窗里的旧值；
    // 本机一恢复检测又无条件覆盖回来。两机场系本身差 0.7~1.4 m（定位退化），表现为球位
    // 在两机之间乒乓（实测 1.2 次/分钟、同刻球位差中位数 0.73 m）。这里加三道闸：
    //   1) 只有「队友本帧检测到 + 包新鲜(≤tm_ball_lag_max_ms) + 队友自己也认为已知球位」
    //      才作可靠源；队友自己在用记忆的包不能当检测值用。
    //   2) 本机有新鲜检测时以本机为准（本机检测更新鲜），与队友值差 >tm_ball_conflict_dist
    //      记冲突并拒绝采信。
    //   3) 球位离我 <tm_ball_dist_threshold 或队友自己在用记忆时，只降级为 hint：
    //      仅驱动转头（CamTrackBall），绝不驱动追球/踢球。
    static rclcpp::Time lastTmBallPosTime = get_clock()->now();
    const double TM_BALL_TIMEOUT = 1000.;        // reliable 保持时长
    const double TM_BALL_HINT_TIMEOUT = 300.;    // hint 保持时长
    const double TM_LAG_MAX = get_parameter_or("strategy.tm_ball_lag_max_ms", 300.0);
    const double CONFLICT_DIST = get_parameter_or("strategy.tm_ball_conflict_dist", 1.0);
    const double HINT_PEER_SEP = get_parameter_or("strategy.tm_ball_hint_peer_sep", 1.5);
    const double SRC_GUARD_MS = get_parameter_or("strategy.ball_src_guard_ms", 150.0);
    const double RANGE_THRESHOLD = config->get_tm_ball_dist_threshold(); 
    int trustedTMIdx = -1;
    int hintTMIdx = -1;
    double minRange = 1e6;
    double hintMinRange = 1e6;
    double trustedTmBallDist = 0.;
    double trustedTmBallLag = 0.;
    int conflictCnt = 0;
    const bool localBallFresh = data->ballDetected && msecsSince(data->ball.timePoint) <= kBallFreshMs;

    // 最近 alive 队友距离（贴身 hint 的准入条件：两机不近时，“离我 <2 m 的球”更可能是误检）
    double nearestPeerDist = 1e9;
    for (int i = 0; i < aliveTmIdxs.size(); i++) {
        const auto &s = data->tmStatus[aliveTmIdxs[i]];
        double d = norm(s.robotPoseToField.x - data->robotPoseToField.x, s.robotPoseToField.y - data->robotPoseToField.y);
        if (d < nearestPeerDist) nearestPeerDist = d;
    }

    log_(format("Find ball info among %d alive TMs", aliveTmIdxs.size()));
    for (int i = 0; i < aliveTmIdxs.size(); i++) {
        int tmIdx = aliveTmIdxs[i];
        auto status = data->tmStatus[tmIdx];
        const double lag = msecsSince(status.timeLastCom);
        log_(format("TM %d, ballDetected: %d, ballRange: %.1f, lag: %.0fms", tmIdx + 1, status.ballDetected, status.ballRange, lag));
        // 只认“本帧检测 + 包新鲜”的球位
        if (!status.ballDetected || lag > TM_LAG_MAX) continue;

        double dist = norm(status.ballPosToField.x - data->robotPoseToField.x, status.ballPosToField.y - data->robotPoseToField.y);
        if (localBallFresh) {
            // 本机检测更新鲜：以本机为准，只做一致性记录
            double diff = norm(status.ballPosToField.x - data->ball.posToField.x, status.ballPosToField.y - data->ball.posToField.y);
            if (diff > CONFLICT_DIST) {
                conflictCnt++;
                log_(format("tm ball conflict: TM %d diff %.2f > %.2f, keep LOCAL", tmIdx + 1, diff, CONFLICT_DIST));
                continue;
            }
        }
        if (status.ballLocationKnown && dist > RANGE_THRESHOLD) {
            if (status.ballRange < minRange) {
                minRange = status.ballRange;
                trustedTMIdx = tmIdx;
                trustedTmBallDist = dist;
                trustedTmBallLag = lag;
            }
        } else if (status.ballRange < hintMinRange) {
            // 队友球位离我太近（可能是我视野死角里的误检）或队友自己也在用记忆：
            // 只作转头提示，绝不驱动追球/踢球
            hintMinRange = status.ballRange;
            hintTMIdx = tmIdx;
        }
    }
    if (localBallFresh) hintTMIdx = -1;                               // 本机看得到就不需要 hint
    if (hintTMIdx >= 0 && nearestPeerDist > HINT_PEER_SEP) hintTMIdx = -1;
    if (trustedTMIdx >= 0) { // Teammate saw the ball
        log_(format("Reliable tm ball found. PlayerID = %d, dist %.2f, lag %.0fms", trustedTMIdx + 1, trustedTmBallDist, trustedTmBallLag));
        data->tmBall.posToField = data->tmStatus[trustedTMIdx].ballPosToField;
        updateRelativePos(data->tmBall);

        tree->setEntry<bool>("tm_ball_pos_reliable", true);
        lastTmBallPosTime = get_clock()->now();
        if (!tree->getEntry<bool>("ball_location_known")) { // If I have forgotten, but the teammate's information is reliable, update my memory of the ball's position with the teammate's information.
            log_("update ball.posToField");
            data->ball.posToField = data->tmBall.posToField;
            updateRelativePos(data->ball);
            if (!ballPosFromTeammate) { // 来源切换：进入保护窗（窗内禁踢）
                ballPosFromTeammate = true;
                ballPosHandoffTime = get_clock()->now();
            }
        } else if (ballPosFromTeammate) {
            ballPosFromTeammate = false;
            ballPosHandoffTime = get_clock()->now();
        }
    } else {
        log_("TM reported NO BALL or can not be trusted");
        if (msecsSince(lastTmBallPosTime) > TM_BALL_TIMEOUT) {
            log_("TM ball timeout reached");
            tree->setEntry<bool>("tm_ball_pos_reliable", false);
        }
        if (ballPosFromTeammate && localBallFresh) { // 本机重新看到球：切回本地源
            ballPosFromTeammate = false;
            ballPosHandoffTime = get_clock()->now();
        }
    }

    // 贴身 hint：只更新 tmBall（供 CamTrackBall 转头），绝不碰 data->ball
    if (hintTMIdx >= 0) {
        data->tmBall.posToField = data->tmStatus[hintTMIdx].ballPosToField;
        updateRelativePos(data->tmBall);
        lastTmBallHintTime = get_clock()->now();
        log_(format("tm ball hint from P%d (range %.1f, nearest peer %.1f)", hintTMIdx + 1, hintMinRange, nearestPeerDist));
    }
    const bool tmBallHintAlive = lastTmBallHintTime.nanoseconds() > 0 &&
        msecsSince(lastTmBallHintTime) <= TM_BALL_HINT_TIMEOUT;
    tree->setEntry<bool>("tm_ball_pos_hint", tmBallHintAlive);

    // 来源交接保护窗：LOCAL/TM 切换后的短时间内禁止触发踢球（球位在两套场系间可能跳 0.7~1.4 m）
    const bool ballSrcGuard = ballPosHandoffTime.nanoseconds() > 0 && msecsSince(ballPosHandoffTime) < SRC_GUARD_MS;
    tree->setEntry<bool>("ball_src_guard", ballSrcGuard);

    // 1 Hz INFO：现场区分“没共享”/“共享了但被拒”/“冲突”
    static rclcpp::Time lastTmBallLog = get_clock()->now();
    if (msecsSince(lastTmBallLog) > 1000) {
        lastTmBallLog = get_clock()->now();
        RCLCPP_INFO(get_logger(), "[tmball] src=%s tm=%d lag=%.0fms dist=%.2f range=%.2f hint=%d conflict=%d localFresh=%d peerSep=%.2f guard=%d",
            ballPosFromTeammate ? "TM" : "LOCAL",
            trustedTMIdx >= 0 ? trustedTMIdx + 1 : (hintTMIdx >= 0 ? hintTMIdx + 1 : 0),
            trustedTmBallLag, trustedTmBallDist, trustedTMIdx >= 0 ? minRange : hintMinRange,
            tmBallHintAlive ? 1 : 0, conflictCnt, localBallFresh ? 1 : 0, nearestPeerDist, ballSrcGuard ? 1 : 0);
    }

    bool isAgentMode = get_parameter(GAME_AGENT_MODE_PARAM).as_bool();
    // Based on the number of players on the field (referee data is more accurate), decide whether the goal_keeper should switch to striker
    bool switchRole = config->get_enable_role_switch();

    // In agent mode, there is no referee, so do not switch roles
    if (!isAgentMode && switchRole) {
        if (data->penalty[selfIdx] == PENALTY_NONE) { // I am not penalized
            if (gcAliveCount < numOfPlayers) { // And the field is not full
                log_("Not full team. I must be Striker");
                tree->setEntry<string>("player_role", "striker"); // Then I must be the striker
            }
        } else { // I am penalized
            if (gcAliveCount == numOfPlayers - 1) { // And there is exactly one player missing on the field
                log_("I am only on under penalty, I must be goal keeper");
                tree->setEntry<string>("player_role", "goal_keeper"); // Then I must be the goal keeper. Because I am the last one on the field, the last one is the goal keeper.
            }
    
        }
    }
    // If there is no strategy during halftime, switch the role back to the initial setting at the start of the second half
    if (tree->getEntry<string>("gc_game_state") == "INITIAL") {
        tree->setEntry<string>("player_role", selfRole);
    }

    // New lead logic, no longer sending commands, but judging by itself 
    // 追球权（lead）只在前锋之间竞争，守门员的 cost 不参与：
    // 否则球在己方半场时守在门口的守门员往往是全场最低 cost，会把所有前锋压成非 lead
    // （集体进 assist 慢速站位），形成“守门员一人远处追球、其他人看戏”。
    // 追球权互斥：cost 唯一最低的存活前锋当 lead；cost 差在 LEAD_COST_DEADBAND 内按
    // player_id 小者优先（两机独立计算，用同一组 cost 值，结论必然一致）。原逻辑只在
    // “最低 cost 队友 < 阈值”时让位，两台前锋 cost 都 ≥ 阈值（球远、丢球、倒地恢复期、
    // 互挡加成）时会同时判 lead，双 chase 奔向同一球后点且 Chase 无队友让位 → 两机挤在一起。
    double tmMinCost = 1e5;
    int myCostRank = 0;
    int myStrikerIDRank = 0;
    bool teammateBetter = false;
    for (int i = 0; i < aliveTmIdxs.size(); i++) {
        int tmIdx = aliveTmIdxs[i];
        auto tmStatus = data->tmStatus[tmIdx];
        if (tmStatus.role == "striker") {
            if (tmStatus.cost < tmMinCost) tmMinCost = tmStatus.cost;
            if (tmStatus.cost < data->tmMyCost) myCostRank++;
            if (tmStatus.cost < data->tmMyCost - LEAD_COST_DEADBAND) teammateBetter = true; // 队友明显更优
            else if (fabs(tmStatus.cost - data->tmMyCost) <= LEAD_COST_DEADBAND && tmIdx + 1 < selfId) teammateBetter = true; // 接近时 id 小者优先
        }
        if (tmIdx < selfIdx && tmStatus.role == "striker") myStrikerIDRank++;
    }
    data->tmMyCostRank = myCostRank;
    data->myStrikerIDRank = myStrikerIDRank;

    // 迟滞（2026-11-02）：唯一性判据本身保证“同一组 cost 下恰有一台 lead”，但两台机比较的
    // 是“自己的实时 cost”与“对方 30~90 ms 前广播的 cost”，而 cost 里的丢球/摔倒惩罚在跳变
    // 瞬间会短时改变排序（实机实测 P1 cost 单采样跳变 >5 共 307 次）。逐拍比较会出现互不
    // 一致的结论：双 LEAD 与双无 LEAD 各占约 3%，lead 平均 3.4 s 易手一次，两机在 chase 与
    // assist 之间来回切，最终挤在球附近。候选结论连续保持 LEAD_SWITCH_HOLD_MS 才真正换手，
    // 把单拍瞬态滤掉；输入持续偏离时仍然会在一拍内收敛，不改变正常换手速度。
    constexpr double LEAD_SWITCH_HOLD_MS = 200.0;
    const bool desiredLead = !teammateBetter;
    if (desiredLead != data->tmImLead) {
        if (!leadSwitchPending || leadSwitchTarget != desiredLead) {
            leadSwitchPending = true;
            leadSwitchTarget = desiredLead;
            leadSwitchStartTime = get_clock()->now();
        } else if (msecsSince(leadSwitchStartTime) >= LEAD_SWITCH_HOLD_MS) {
            data->tmImLead = desiredLead;
            leadSwitchPending = false;
            log_(data->tmImLead ? "I am Lead" : "I am not lead");
        }
    } else {
        leadSwitchPending = false;
    }
    tree->setEntry<bool>("is_lead", data->tmImLead);
    log_(format("tmMinCost: %.1f, myCost: %.1f, myCostRank: %d, myStrikerIDRank: %d, desiredLead: %d, leadPending: %d",
        tmMinCost, data->tmMyCost, myCostRank, myStrikerIDRank, desiredLead ? 1 : 0, leadSwitchPending ? 1 : 0));

    // Publish command: Goalkeeper should attack, instruct to switch goalkeeper
    // 门控：换门将协商也受 enable_role_switch 控制（默认 false = 角色静态，守门员不外抛门将角色）
    if (
        !isAgentMode
        && switchRole // 仅显式打开角色切换时才允许门将协商换岗
        && data->tmImAlive // I am online
        && tree->getEntry<string>("player_role") == "goal_keeper" // I am the goalkeeper
        && data->tmImLead // I am the lead
        && msecsSince(data->tmLastCmdChangeTime) > CMD_COOLDOWN // Enough time has passed since the last command.
    ) {
        auto distToGoal = [=](Pose2D pose) {
            return norm(pose.x + config->fieldDimensions.length / 2.0, pose.y);
        };
        double maxDist = 0.0;
        double minDist = 1e6;
        int minIndex = -1; // Index of the teammate closest to the goal
        double myDist = distToGoal(data->robotPoseToField);
        for (int i = 0; i < aliveTmIdxs.size(); i++) {
            int tmIdx = aliveTmIdxs[i];
            auto tmPose = data->tmStatus[tmIdx].robotPoseToField;
            double dist = distToGoal(tmPose);
            if (dist > maxDist) maxDist = dist;
            if (dist < minDist) {
                minIndex = tmIdx;
                minDist = dist;
            }
        }
        if (minIndex >= 0 && myDist > maxDist) {
            data->tmLastCmdChangeTime = get_clock()->now();
            data->tmMyCmd = 10 + minIndex + 1; // Command is 10 + teammate id, e.g., 10 + 1 = 11, means player 1 becomes the goalkeeper
            data->tmCmdId += 1; // Increase command ID to make the command unique
            data->tmMyCmdId = data->tmCmdId;
            tree->setEntry<string>("player_role", "striker");
            log_(format("goalie: i am too far from goal, i ask player %d to attack", minIndex + 1));
        } else {
            log_(format("goalie: i am close enough to goal, no need to attack, my dist: %.2f", myDist));
        }
    }

    // Handle received commands
    auto cmd = data->tmReceivedCmd;
    if (cmd != 0) { // Teammate wants to take control
        log_(format("received cmd %d from teammate", cmd));
        if (cmd == 100) { // Teammate wants to take control
            data->tmImLead = false; // I am no longer the lead
            tree->setEntry<bool>("is_lead", false);
            log_("teammate wants to take lead, i'll assist");
        } else if (cmd > 10 && cmd < 20) { // Goalkeeper wants to attack
            if (!switchRole) { // 角色静态：忽略门将换岗命令，保持原角色
                log_("role switch disabled, ignore goalie-attack cmd");
            } else {
                log_("goalie wants to attack");
                int newGoalieId = cmd - 10;
                if (newGoalieId == selfId) { // I am the new goalkeeper
                    log_("i become goalie");
                    tree->setEntry<string>("player_role", "goal_keeper");
                } else { // I am not the new goalkeeper
                    log_(format("teammate %d becomes goalie", newGoalieId));
                }
            }
        } else {
            log_(format("unknown cmd %d from teammate", cmd));
        }

        data->tmReceivedCmd = 0; // Clear the received command, so it is executed only once.
    }

    tree->setEntry<bool>("is_lead", data->tmImLead);

    // In the READY state, based on the initial settings and the number of players on the field, reassign roles.
    if (
        (tree->getEntry<string>("gc_game_state") == "READY" || tree->getEntry<string>("gc_game_sub_state") == "GET_READY") 
        && gcAliveCount == numOfPlayers
    ) {
        // READY state, and all players are on the field, return to initial roles
        tree->setEntry<string>("player_role", selfRole);
        log_(format("all teammates on field. Back to initial role: %s", selfRole.c_str()));
    }

    return;
}

void Brain::updateMemory()
{
    updateBallMemory();
    updateRobotMemory();
    updateObstacleMemory();
    updateKickoffMemory();
}

void Brain::updateObstacleMemory() {
   
    auto obstacles = data->getObstacles();
    vector<GameObject> obs_new = {};

    const double OBS_EXPIRE_TIME = config->get_obstacle_memory_msecs();
    for (int i = 0; i < obstacles.size(); i++) {
        auto obs = obstacles[i];
        // 小写 "ball" 是本节点自己插入的球（data->ball.label），大写 "Ball" 是视觉标签；
        // 两者都要剔除，否则 READY 期间球会在障碍表里逐拍累积。
        if (obs.label == "Ball" || obs.label == "ball") continue; 
        if (msecsSince(obs.timePoint) > OBS_EXPIRE_TIME)  continue; 


        updateRelativePos(obs);
        obs_new.push_back(obs);
    }


    if (
        (config->get_enable_obstacle_avoidance() && isFreekickStartPlacing())
        || tree->getEntry<string>("gc_game_state") == "READY"
    ) {
        // READY/罚球摆位时不许碰球，把球当障碍。但球的 confidence 是 0-100 的检测分，
        // 与深度栅格“单帧落格点数”不是一套量纲，这里抬到阈值之上，避免被格点阈值误杀。
        auto ballObs = data->ball;
        ballObs.confidence = max(ballObs.confidence, config->get_occupancy_threshold() + 1.0);
        obs_new.push_back(ballObs);
    }

    data->setObstacles(obs_new);
}

void Brain::updateBallMemory()
{

    double secs = msecsSince(data->ball.timePoint) / 1000;
    
    double ballMemTimeout = config->get_ball_memory_timeout();

    if (secs > ballMemTimeout) 
    { 
        tree->setEntry<bool>("ball_location_known", false);
        tree->setEntry<bool>("ball_out", false); 

        // 位置记忆过期：把 confidence 清零作为“此球位已失效”的显式标记。
        // 位置字段保留原值（清成 0 会被当作场地中心，置 NaN 会污染 posToField
        // 参与的坐标链与可视化），消费端一律用 ballDetected + kBallFreshMs 判断。
        data->ball.confidence = 0.0;
    }

    
    updateRelativePos(data->ball);
    updateRelativePos(data->tmBall);
    tree->setEntry<double>("ball_range", data->ball.range);
}

void Brain::updateBallEstimate()
{
    // 球位来源切换（LOCAL↔TM，两套场系差 0.7~1.4 m）时整段重置：
    // 否则交接瞬间的位置跳变会被拟合成几 m/s 的假速度，门将据此算出毫无意义的过线点。
    if (tree->getEntry<bool>("ball_src_guard")) ballEstimator.reset();

    const double ageMs = msecsSince(data->ball.timePoint);
    ballEstimator.update(data->ball, data->ballDetected, ageMs,
                         get_parameter_or("strategy.ball_est_window_s", 0.35),
                         get_parameter_or("strategy.ball_est_max_speed", 6.0),
                         get_parameter_or("strategy.ball_est_min_samples", 3),
                         get_parameter_or("strategy.ball_est_reset_ms", 500.0));

    const auto &est = ballEstimator.state();
    // 速度只在“球位仍在 kBallFreshMs 内”时对外有效：过了新鲜窗位置本身就是旧值，再叠加速度
    // 只会把误差做大，消费端应退回保守站位（见 GoToGoalBlockingPosition）。
    // 不另判 data->ballDetected：视觉 30 Hz 的单帧掉帧不应让速度逐拍失效，球龄才是真判据。
    const bool valid = est.valid && ageMs <= kBallFreshMs;
    tree->setEntry<double>("ball_vel_x", valid ? est.vx : 0.0);
    tree->setEntry<double>("ball_vel_y", valid ? est.vy : 0.0);
    tree->setEntry<double>("ball_speed", valid ? est.speed : 0.0);
    tree->setEntry<double>("ball_age_ms", ageMs);
    tree->setEntry<bool>("ball_est_valid", valid);
}

void Brain::updateRobotMemory() {
    auto robots = data->getRobots();
    vector<GameObject> newRobots = {};

    for (int i = 0; i < robots.size(); i++) {
        auto r = robots[i];


        if (msecsSince(r.timePoint) > 1000)  continue;


        updateRelativePos(r);
        newRobots.push_back(r);
    }

    data->setRobots(newRobots);
}

void Brain::updateKickoffMemory() {
    
    static Point ballPos;
    const double BALL_MOVE_THRESHOLD_FACTOR = 0.15; 
    const double BALL_MOVE_THRESHOLD_MIN = 0.3; 
    auto ballMoved = [=]() {
        if (!data->ballDetected) return false; 
        double range = data->ball.range;
        double threshold = max(range * BALL_MOVE_THRESHOLD_FACTOR, BALL_MOVE_THRESHOLD_MIN);
        double posChange = norm(data->ball.posToRobot.x - ballPos.x, data->ball.posToRobot.y - ballPos.y);
        return posChange > threshold;
    };
    static rclcpp::Time kickOffTime;
    const double TIMEOUT = 1000 * 10; 
    auto timeReached = [=]() {
        return msecsSince(kickOffTime) > TIMEOUT;
    };
    bool isWaitingForKickoff = (
        (tree->getEntry<string>("gc_game_state") == "SET"  || tree->getEntry<string>("gc_game_state") == "READY")
        && !tree->getEntry<bool>("gc_is_kickoff_side")
    );
    bool isWaitingForFreekickKickoff = (
        (tree->getEntry<string>("gc_game_sub_state") == "SET" || tree->getEntry<string>("gc_game_sub_state") == "GET_READY")
        && !tree->getEntry<bool>("gc_is_sub_state_kickoff_side")
    );
    if ( isWaitingForFreekickKickoff || isWaitingForKickoff) {
        ballPos = data->ball.posToRobot;
        kickOffTime = get_clock()->now();
        tree->setEntry<bool>("wait_for_opponent_kickoff", true);
    } else if (tree->getEntry<bool>("wait_for_opponent_kickoff")) {
        if (ballMoved() || timeReached()) {
            tree->setEntry<bool>("wait_for_opponent_kickoff", false);
        }
    }
}

vector<double> Brain::getGoalPostAngles(const double margin)
{
    double leftX, leftY, rightX, rightY; 

    leftX = config->fieldDimensions.length / 2;
    leftY = config->fieldDimensions.goalWidth / 2;
    rightX = config->fieldDimensions.length / 2;
    rightY = -config->fieldDimensions.goalWidth / 2;


    auto goalposts = data->getGoalposts();
    for (int i = 0; i < goalposts.size(); i++)
    {
        auto post = goalposts[i];
        if (post.name == "OL")
        {
            leftX = post.posToField.x;
            leftY = post.posToField.y;
        }
        else if (post.name == "OR")
        {
            rightX = post.posToField.x;
            rightY = post.posToField.y;
        }
    }

    const double theta_l = atan2(leftY - margin - data->ball.posToField.y, leftX - data->ball.posToField.x);
    const double theta_r = atan2(rightY + margin - data->ball.posToField.y, rightX - data->ball.posToField.x);

    vector<double> vec = {theta_l, theta_r};
    return vec;
}

double Brain::calcKickDir(double goalPostMargin) {
    double dir_rb_f = data->robotBallAngleToField; 
    auto goalPostAngles = getGoalPostAngles(goalPostMargin);
    double theta_l = goalPostAngles[0]; 
    double theta_r = goalPostAngles[1];
    
    if (isAngleGood(goalPostMargin)) return dir_rb_f;

    double delta_l = fabs(toPInPI(theta_l - dir_rb_f));
    double delta_r = fabs(toPInPI(theta_r - dir_rb_f));
    if (delta_l < delta_r) return theta_l;
    // else 
    return theta_r;
}

void Brain::updateCostToKick() {
    auto log_ = [=](string msg) {
        log->debug("updateCostToKick", msg);
    };
    double cost = 0.;

    // ball not detected
    double secsSinceBallDet = msecsSince(data->ball.timePoint) / 1000;
    cost += secsSinceBallDet;
    log_(format("ball not dectect cost: %.1f", secsSinceBallDet));

    // Ball lost（1 s 内 0→5 线性爬升，替代原 0/5 台阶）
    if (!tree->getEntry<bool>("ball_location_known")) {
        // 原实现是台阶：球一离开视野 cost 单拍 +5。丢球时刻在两条机上不同步，且
        // 比较方看到的是对方滞后的 cost，于是这台报“我已丢球”而对面还按旧值算，
        // lead 在这一步上就会抢/让一次。改成随丢球时长爬升，单拍幅度降到 ~1。
        const double lostRamp = min(1.0, secsSinceBallDet / 1.0);
        cost += 5.0 * lostRamp;
        log_(format("ball lost cost: %.1f (ramp %.2f)", 5.0 * lostRamp, lostRamp));
    }

    // cost of chasing the ball
    cost += data->ball.range;
    log_(format("ball range cost: %.1f", data->ball.range));
    
    
    // cost of obstacles on the way to the ball
    if (distToObstacle(data->ball.yawToRobot) < 1.5) {
        log_(format("obstacle cost: %.1f", 2.0));
        cost += 0.5;
    }

    // cost of turning towards the ball
    cost += fabs(data->ball.yawToRobot) / 1.0; 
    log_(format("ball yaw cost: %.1f", fabs(data->ball.yawToRobot) / 1.0));


    // cost of bumping into teammates
    int selfIdx = config->get_player_id() - 1;
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx) continue; // Skip self

        auto status = data->tmStatus[i]; // Teammate status
        if (!status.isAlive) continue; // Skip offline teammates

        double theta_tm2ball = atan2(status.ballPosToField.y - status.robotPoseToField.y, status.ballPosToField.x - status.robotPoseToField.x);
        double range_tm2ball = norm(status.ballPosToField.y - status.robotPoseToField.y, status.ballPosToField.x - status.robotPoseToField.x);
        double theta_me2ball = data->robotBallAngleToField;
        double range_me2ball = data->ball.range;
        double deltaTheta = fabs(toPInPI(theta_tm2ball - theta_me2ball));

        const double BUMP_DIST = 1.0;
        if (range_tm2ball < range_me2ball && sin(deltaTheta) * range_tm2ball < BUMP_DIST) {
            cost += 2.0;
            log_(format("bump cost: %.1f", 2.0));  
        }
    }

    // cost of adjusting
    cost += fabs(toPInPI(data->kickDir - data->robotBallAngleToField)) * 0.4 / 0.3; // 0.4 is the approximate distance to turn around the ball, 0.3 is the approximate tangential speed to turn around the ball
    log_(format("adjust cost: %.1f", fabs(toPInPI(data->kickDir - data->robotBallAngleToField)) * 0.4 / 0.3));
    

    // cost increase after falling（0→15 在 1.5 s 内爬升，替代原 0/15 台阶）
    if (data->recoveryState == RobotRecoveryState::HAS_FALLEN) {
        if (!fallTimerActive) {
            fallTimerActive = true;
            fallStartTime = get_clock()->now();
        }
        const double fallRamp = min(1.0, msecsSince(fallStartTime) / 1500.0);
        cost += 15.0 * fallRamp;
        log_(format("fall cost: %.1f (ramp %.2f)", 15.0 * fallRamp, fallRamp));
    } else {
        fallTimerActive = false;
    }

    
    // cost increase if localization fails
    if (!tree->getEntry<bool>("odom_calibrated")) {
        cost += 100;
        log_(format("localization cost: %.1f", 100.0));  

    }
    

    // smoothing + 中值滤波：EMA 只削弱单拍跳变，突变仍会在 1~2 拍内传导到 tmMyCost，
    // 而 lead 是逐拍比较的，1~2 拍的偏差就足以让两台机得出相反结论。取最近 5 拍（≈60 ms）
    // 的中位数，单拍尖尖被完全剔除；两台机用同一窗口、同一公式，比较仍然对称。
    double lastCost = data->tmMyCost;
    const double smoothCost = lastCost * 0.8 + cost * 0.2;
    costHistory[costHistoryIdx] = smoothCost;
    costHistoryIdx = (costHistoryIdx + 1) % COST_HISTORY_LEN;
    if (costHistoryFilled < COST_HISTORY_LEN) costHistoryFilled++;
    double sortedCost[COST_HISTORY_LEN];
    std::copy(costHistory, costHistory + costHistoryFilled, sortedCost);
    std::sort(sortedCost, sortedCost + costHistoryFilled);
    data->tmMyCost = sortedCost[costHistoryFilled / 2];
    log_(format("lastCost: %.1f, newCost: %.1f, smoothCost: %.1f, medianCost: %.1f", lastCost, cost, smoothCost, data->tmMyCost));

    return;
}

bool Brain::isAngleGood(double goalPostMargin, string type) {
    double angle = 0;
    if (type == "kick") angle = data->robotBallAngleToField; // type=="kick" robot to ball, direction in field coordinate system
    if (type == "shoot") angle = data->robotPoseToField.theta; // type=="shoot" robot orientation
    

    auto goalPostAngles = getGoalPostAngles(goalPostMargin);
    double theta_l = goalPostAngles[0]; 
    double theta_r = goalPostAngles[1]; 
    
    if (theta_l - theta_r < M_PI / 3 * 2) { 
        goalPostAngles = getGoalPostAngles(0.5);
        theta_l = goalPostAngles[0]; 
        theta_r = goalPostAngles[1]; 
    }

    return (theta_l > angle && theta_r < angle);
}

bool Brain::isBallOut(double locCompareDist, double lineCompareDist)
{
    auto ball = data->ball;
    auto fd = config->fieldDimensions;

    if (fabs(ball.posToField.x) > fd.length / 2 + locCompareDist)
        return true;
    if (fabs(ball.posToField.y) > fd.width / 2 + locCompareDist)
        return true;
    
    auto fieldLines = data->getFieldLines();
    for (int i = 0; i < fieldLines.size(); i++) {
        auto line = fieldLines[i];
        if (
            (line.type == LineType::TouchLine || line.type == LineType::GoalLine)
            && line.confidence > 1.0
         ) {
            Point2D p = {ball.posToField.x, ball.posToField.y};
            // prtWarn(format("Ball: %.2f, %.2f PerpDist: %.2f", ball.posToField.x, ball.posToField.y, pointPerpDistToLine(p, line.posToField)));
            if (pointPerpDistToLine(p, line.posToField) > lineCompareDist) return true;
        }
    }
    return false;
}

void Brain::updateBallOut() {
    bool lastBallOut = tree->getEntry<bool>("ball_out");
    double range = lastBallOut ? 4.0 : 2.5;
    double threshold = config->get_ball_out_threshold();
    threshold += (data->isFreekickKickingOff ? 1.0 : 0.0); // If a free kick is being taken, relax the out-of-bounds judgment
    threshold *= (lastBallOut ? 1.0 : 1.5); // Prevent oscillation. If the last judgment was out-of-bounds, relax the out-of-bounds judgment
    tree->setEntry<bool>("ball_out", isBallOut(threshold, 10.0) && data->ball.range < range); // Strictly determine out-of-bounds through localization
}

double Brain::distToBorder() {
    vector<Line> borders;
    auto fd = config->fieldDimensions;
    borders.push_back({fd.length / 2, fd.width / 2, -fd.length / 2, fd.width / 2});
    borders.push_back({fd.length / 2, -fd.width / 2, -fd.length / 2, -fd.width / 2});
    borders.push_back({fd.length / 2, fd.width / 2, fd.length / 2, -fd.width / 2});
    borders.push_back({-fd.length / 2, fd.width / 2, -fd.length / 2, -fd.width / 2});
    double maxDist = -100;
    Point2D robot = {data->robotPoseToField.x, data->robotPoseToField.y};
    for (int i = 0; i < borders.size(); i++) {
        auto line = borders[i];
        double dist = pointPerpDistToLine(robot, line);
        if (dist > maxDist) maxDist = dist;
    }
    return maxDist;
}

bool Brain::isBoundingBoxInCenter(BoundingBox boundingBox, double xRatio, double yRatio) {
    double x = (boundingBox.xmin + boundingBox.xmax) / 2.0;
    double y = (boundingBox.ymin + boundingBox.ymax) / 2.0;

    return (x  > config->cameraImageWidth * (1 - xRatio) / 2)
        && (x < config->cameraImageWidth * (1 + xRatio) / 2)
        && (y > config->cameraImageHeight * (1 - yRatio) / 2)
        && (y < config->cameraImageHeight * (1 + yRatio) / 2);
}

bool Brain::isDefensing() {
    bool isFreeKick = tree->getEntry<string>("gc_game_sub_state_type") == "FREE_KICK";
    bool isKickoffSide = tree->getEntry<bool>("gc_is_sub_state_kickoff_side");
    
    return isFreeKick && (!isKickoffSide);
}

void Brain::calibrateOdom(double x, double y, double theta)
{

    double x_or, y_or, theta_or; // or = odom to robot
    x_or = -cos(data->robotPoseToOdom.theta) * data->robotPoseToOdom.x - sin(data->robotPoseToOdom.theta) * data->robotPoseToOdom.y;
    y_or = sin(data->robotPoseToOdom.theta) * data->robotPoseToOdom.x - cos(data->robotPoseToOdom.theta) * data->robotPoseToOdom.y;
    theta_or = -data->robotPoseToOdom.theta;

    
    transCoord(x_or, y_or, theta_or,
               x, y, theta,
               data->odomToField.x, data->odomToField.y, data->odomToField.theta);


    transCoord(
        data->robotPoseToOdom.x, data->robotPoseToOdom.y, data->robotPoseToOdom.theta,
        data->odomToField.x, data->odomToField.y, data->odomToField.theta,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta);


    double placeHolder;
    // ball
    transCoord(
        data->ball.posToRobot.x, data->ball.posToRobot.y, 0,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
        data->ball.posToField.x, data->ball.posToField.y, placeHolder 
    );

    // robots
    auto robots = data->getRobots();
    for (int i = 0; i < robots.size(); i++) {
        updateFieldPos(robots[i]);
    }
    data->setRobots(robots);

    // goalposts
    auto goalposts = data->getGoalposts();
    for (int i = 0; i < goalposts.size(); i++) {
        updateFieldPos(goalposts[i]);
    }
    
    // markers
    auto markings = data->getMarkings();
    for (int i = 0; i < markings.size(); i++) {
        updateFieldPos(markings[i]);
    }

    // relog
    vector<GameObject> gameObjects = {};
    if(data->ballDetected) gameObjects.push_back(data->ball);
    for (int i = 0; i < markings.size(); i++) gameObjects.push_back(markings[i]);
    for (int i = 0; i < robots.size(); i++) gameObjects.push_back(robots[i]);
    for (int i = 0; i < goalposts.size(); i++) gameObjects.push_back(goalposts[i]);
}


void Brain::pubKickMsg() {
    if (!pubKickBall) return;
    if (!data->ballDetected) return;
    brain::msg::Kick kickMsg;
    kickMsg.header.stamp = get_clock()->now();
    kickMsg.x = data->ball.posToRobot.x;
    kickMsg.y = data->ball.posToRobot.y;
    kickMsg.dir = toPInPI(data->kickDir - data->robotPoseToField.theta);

    double goal_x = config->fieldDimensions.length / 2;
    double goal_y = 0.0;
    Pose2D goalPose;
    goalPose.x = goal_x;
    goalPose.y = goal_y;
    double ball_x = data->ball.posToField.x;
    double ball_y = data->ball.posToField.y;
    double dist = std::sqrt((goal_x - ball_x) * (goal_x - ball_x) + (goal_y - ball_y) * (goal_y - ball_y));
    dist = std::abs(dist);
    double power = 0.0;

    if (dist > 6.0) {
        power = 1.5;
    } else {
        power = 6.0;
    }
    kickMsg.power = power;

    auto goalPose_r = data->field2robot(goalPose);
    kickMsg.goal_x = goalPose_r.x;  
    kickMsg.goal_y = goalPose_r.y;

    kickMsg.robot_theta_to_field = data->robotPoseToField.theta;

    pubKickBall->publish(kickMsg);
}

double Brain::msecsSince(rclcpp::Time time)
{
    auto now = this->get_clock()->now();
    if (time.get_clock_type() != now.get_clock_type()) return 1e18;
    return (now - time).nanoseconds() / 1e6;
}

rclcpp::Time Brain::timePointFromHeader(std_msgs::msg::Header header) {
    auto stamp = header.stamp;
    // NOTE It seems that regardless of whether use_sim_time is true or false, ROS_TIME is used
    int32_t sec = stamp.sec;
    uint32_t nanosec = stamp.nanosec;
    if (sec < 0) {
        prtErr(format("Negative time: sec: %d nanosec: %u", sec, nanosec));
        sec = 0; // Prevent crash, but likely needs better handling
        nanosec = 1;
    }
    return rclcpp::Time(sec, nanosec, RCL_ROS_TIME); // should not crash
}


void Brain::gameControlCallback(const game_controller_interface::msg::GameControlData &msg)
{
    if (get_parameter(GAME_AGENT_MODE_PARAM).as_bool()) {
        prtWarn("Agent mode, ignore game control");
        return;
    };
    data->timeLastGamecontrolMsg = get_clock()->now();

    // Handle primary game state
    auto lastGameState = tree->getEntry<string>("gc_game_state"); // Primary game state
    vector<string> gameStateMap = {
        "INITIAL", // Initialization state, players prepare off the field
        "READY",   // Ready state, players enter the field and move to their starting positions
        "SET",     // Stop action, wait for the referee to start the game
        "PLAY",    // Normal gameplay
        "END"      // Game over
    };

    int teamId = config->get_team_id();
    int playerId = config->get_player_id();
    string gameState = gameStateMap[static_cast<int>(msg.state)];
    tree->setEntry<string>("gc_game_state", gameState);
    bool isKickOffSide = (msg.kick_off_team == teamId); // Whether our team is the kickoff side
    tree->setEntry<bool>("gc_is_kickoff_side", isKickOffSide);

    // Handle secondary game state
    string gameSubStateType;
    switch (static_cast<int>(msg.secondary_state)) {
        case 0:
            gameSubStateType = "NONE";
            data->realGameSubState = "NONE";
            break;
        case 3:
            gameSubStateType = "TIMEOUT"; // Includes both team timeouts and referee timeout
            data->realGameSubState = "TIMEOUT";
            break;
        // Temporarily do not handle other states, except TIMEOUT, all are treated as FREE_KICK
        case 4:
            // gameSubStateType = "DIRECT_FREEKICK";
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "DIRECT_FREEKICK";
            data->isDirectShoot = true;
            break;
        case 5:
            // gameSubStateType = "INDIRECT_FREEKICK";
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "INDIRECT_FREEKICK";
            break;
        case 6:
            // gameSubStateType = "PENALTY_KICK";
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "PENALTY_KICK";
            data->isDirectShoot = true;
            break;
        case 7:
            // gameSubStateType = "CORNER_KICK";
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "CORNER_KICK";
            break;
        case 8:
            // gameSubStateType = "GOAL_KICK";
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "GOAL_KICK";
            data->isDirectShoot = true;
            break;
        case 9:
            // gameSubStateType = "THROW_IN";
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "THROW_IN";
            break;
        default:
            gameSubStateType = "FREE_KICK";
            break;
    }
    vector<string> gameSubStateMap = {"STOP", "GET_READY", "SET"};                               // STOP: Stop; -> GET_READY: Move to offensive or defensive position; -> SET: Stand still
    string gameSubState = gameSubStateMap[static_cast<int>(msg.secondary_state_info[1])];
    tree->setEntry<string>("gc_game_sub_state_type", gameSubStateType);
    tree->setEntry<string>("gc_game_sub_state", gameSubState);
    bool isSubStateKickOffSide = (static_cast<int>(msg.secondary_state_info[0]) == teamId); // In the secondary state, whether our team is the kickoff side. For example, if the current secondary state is a free kick, whether our team is taking the free kick
    tree->setEntry<bool>("gc_is_sub_state_kickoff_side", isSubStateKickOffSide);

    // cout << "game state: " << gameState << " game sub state type: " << gameSubStateType << endl;
    // Find team information
    game_controller_interface::msg::TeamInfo myTeamInfo;
    game_controller_interface::msg::TeamInfo oppoTeamInfo;
    if (msg.teams[0].team_number == teamId) {
        myTeamInfo = msg.teams[0];
        oppoTeamInfo = msg.teams[1];
    } else if (msg.teams[1].team_number == teamId) {
        myTeamInfo = msg.teams[1];
        oppoTeamInfo = msg.teams[0];
    } else {
        // Our team is not included in the data packet, should not process further
        prtErr(format("received invalid game controller message team0 %d, team1 %d, teamId %d",
            msg.teams[0].team_number, msg.teams[1].team_number,teamId));
        return;
    }

    int liveCount = 0;
    int oppoLiveCount = 0;
    // Handle penalty state. penalty[playerId - 1] represents whether our player is in a penalty state. Handling penalty state means the player cannot move.
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        data->penalty[i] = static_cast<int>(myTeamInfo.players[i].penalty);
        
        if (static_cast<int>(myTeamInfo.players[i].red_card_count) > 0) {
            data->penalty[i] = PENALTY_SUBSTITUTE;
        }

        if (data->penalty[i] == PENALTY_NONE) liveCount++;
        data->oppoPenalty[i] = static_cast<int>(oppoTeamInfo.players[i].penalty);

        if (static_cast<int>(oppoTeamInfo.players[i].red_card_count) > 0) {
            data->oppoPenalty[i] = PENALTY_SUBSTITUTE;
        }

        if (data->oppoPenalty[i] == PENALTY_NONE) oppoLiveCount++;
    }
    data->liveCount = liveCount;
    data->oppoLiveCount = oppoLiveCount;

    bool lastIsUnderPenalty = tree->getEntry<bool>("gc_is_under_penalty");
    bool isUnderPenalty = (data->penalty[playerId - 1] != PENALTY_NONE); // Whether the current robot is under penalty
    tree->setEntry<bool>("gc_is_under_penalty", isUnderPenalty);
    if (isUnderPenalty && !lastIsUnderPenalty) tree->setEntry<bool>("odom_calibrated", false); // If penalized, need to re-enter the field, thus need to re-localize

    // FOR FUN Handle goal celebration waving logic
    data->score = static_cast<int>(myTeamInfo.score);
    data->oppoScore = static_cast<int>(oppoTeamInfo.score);
    data->secsRemaining = static_cast<int>(msg.secs_remaining);
}

void Brain::detectionsCallback(const vision_interface::msg::Detections &msg)
{
    // std::lock_guard<std::mutex> guard(data->brainMutex);
    
    // auto detection_time_stamp = msg.header.stamp;
    // rclcpp::Time timePoint(detection_time_stamp.sec, detection_time_stamp.nanosec);
    data->camConnected = true;
    auto timePoint = timePointFromHeader(msg.header);

    auto now = get_clock()->now();
    data->timeLastDet = timePoint; // Used to output delay information during debugging

    auto gameObjects = getGameObjects(msg);

    // Group the detected objects
    vector<GameObject> balls, goalposts, persons, robots, obstacles, markings;
    for (int i = 0; i < gameObjects.size(); i++)
    {
        const auto &obj = gameObjects[i];
        if (obj.label == "Ball")
            balls.push_back(obj);
        if (obj.label == "Goalpost")
            goalposts.push_back(obj);
        if (obj.label == "Person")
        {
            persons.push_back(obj);

            // For debugging purposes, you can set treat_person_as_robot in the config to treat Person as Robot
            if (config->get_treat_person_as_robot())
                robots.push_back(obj);
        }
        if (obj.label == "Opponent")
            robots.push_back(obj);
        if (obj.label == "LCross" || obj.label == "TCross" || obj.label == "XCross" || obj.label == "PenaltyPoint")
            markings.push_back(obj);
    }

    // Process the grouped objects separately
    detectProcessBalls(balls);
    detectProcessGoalposts(goalposts);
    detectProcessMarkings(markings);
    detectProcessRobots(robots);

    // Handle and record vision information
    detectProcessVisionBox(msg);

}

void Brain::updateLinePosToField(FieldLine& line) {
    double __; // __ is a placeholder for transformations
    transCoord(
        line.posToRobot.x0, line.posToRobot.y0, 0,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
        line.posToField.x0, line.posToField.y0, __
    );
    transCoord(
        line.posToRobot.x1, line.posToRobot.y1, 0,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
        line.posToField.x1, line.posToField.y1, __
    );
}

void Brain::fieldLineCallback(const vision_interface::msg::LineSegments &msg)
{
    auto timePoint = timePointFromHeader(msg.header);

    auto now = get_clock()->now();
    data->timeLastLineDet = timePoint; // Used to output delay information during debugging

    vector<FieldLine> lines = {};
    FieldLine line;

    double x0, y0, x1, y1, __; // __ is a placeholder for transformations 
    for (int i = 0; i < msg.coordinates.size() / 4; i++) {
        int index = i * 4;
        line.posToRobot.x0 = msg.coordinates[index]; line.posOnCam.x0 = msg.coordinates_uv[index];
        line.posToRobot.y0 = msg.coordinates[index + 1]; line.posOnCam.y0 = msg.coordinates_uv[index + 1];
        line.posToRobot.x1 = msg.coordinates[index + 2]; line.posOnCam.x1 = msg.coordinates_uv[index + 2];
        line.posToRobot.y1 = msg.coordinates[index + 3]; line.posOnCam.y1 = msg.coordinates_uv[index + 3];
        updateLinePosToField(line);
        line.timePoint = timePoint;

        lines.push_back(line);
    }
    lines = processFieldLines(lines);
    data->setFieldLines(lines);

    

    
}

bool Brain::gtBallFresh()
{
    if (lastGtBallTime.seconds() <= 0.0) {
        return false; // no ground truth ball received yet
    }
    return (get_clock()->now() - lastGtBallTime).seconds() < config->get_gt_ball_timeout_sec();
}

void Brain::gtPoseCallback(const geometry_msgs::msg::Pose2D &msg)
{
    if (!tree) {
        return; // guard against messages arriving before init completes
    }

    // Ground truth pose injection: calibrate with the exact field pose, which also
    // short-circuits SelfLocate (the particle filter no longer runs).
    calibrateOdom(msg.x, msg.y, msg.theta);
    tree->setEntry<bool>("odom_calibrated", true);
    data->lastSuccessfulLocalizeTime = get_clock()->now();

    log->debug("gt_pose_callback", format("x: %.2f, y: %.2f, theta: %.2f", msg.x, msg.y, msg.theta));
}

void Brain::gtBallCallback(const geometry_msgs::msg::Point &msg)
{
    if (!tree) {
        return;
    }

    // Ground truth ball: write the field position directly and derive the robot-frame
    // position with the inverse transform (same convention as getGameObjects).
    data->ball.label = "ball";
    data->ball.confidence = 100;
    data->ball.posToField.x = msg.x;
    data->ball.posToField.y = msg.y;
    data->ball.posToField.z = 0;

    vector<double> ballToRobot = trans(
        msg.x, msg.y, 0,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta, "forth");
    data->ball.posToRobot.x = ballToRobot[0];
    data->ball.posToRobot.y = ballToRobot[1];
    data->ball.posToRobot.z = 0;
    data->ball.range = hypot(ballToRobot[0], ballToRobot[1]);
    data->ball.timePoint = get_clock()->now();

    data->ballDetected = true;
    data->lose_ball = false;
    tree->setEntry<bool>("ball_location_known", true);
    updateBallOut();
    data->robotBallAngleToField = atan2(
        data->ball.posToField.y - data->robotPoseToField.y,
        data->ball.posToField.x - data->robotPoseToField.x);

    lastGtBallTime = get_clock()->now();
    gtBallActive = true;
    gtBallLostTime = rclcpp::Time(0);

    log->debug("gt_ball_callback", format("field: (%.2f, %.2f), robot: (%.2f, %.2f)", msg.x, msg.y, ballToRobot[0], ballToRobot[1]));
}

void Brain::imageCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    config->cameraImageWidth = msg->width;
    config->cameraImageHeight = msg->height;
}

void Brain::depthCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    // 内参优先用相机实测值；空/零内参（未标定或占位消息）不得覆盖 vision.yaml 初值。
    double fx = static_cast<double>(msg->k[0]);
    double fy = static_cast<double>(msg->k[4]);
    if (fx <= 1e-6 || fy <= 1e-6) {
        RCLCPP_WARN(get_logger(), "[depth] camera_info has invalid intrinsics (k[0]=%.3f k[4]=%.3f), ignored", fx, fy);
        return;
    }
    config->depthCameraFx = fx;
    config->depthCameraCx = static_cast<double>(msg->k[2]);
    config->depthCameraFy = fy;
    config->depthCameraCy = static_cast<double>(msg->k[5]);
    double w = static_cast<double>(msg->width);
    double h = static_cast<double>(msg->height);
    config->depthCameraFovX = 2.0 * atan(w / (2.0 * config->depthCameraFx));
    config->depthCameraFovY = 2.0 * atan(h / (2.0 * config->depthCameraFy));
    if (!config->depthIntrinsicsFromCameraInfo) {
        config->depthIntrinsicsFromCameraInfo = true;
        RCLCPP_INFO(get_logger(), "[depth] intrinsics from camera_info: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f fovx=%.1fdeg",
                    msg->width, msg->height, fx, fy,
                    config->depthCameraCx, config->depthCameraCy, rad2deg(config->depthCameraFovX));
    }
}

void Brain::headPoseCallback(const geometry_msgs::msg::PoseStamped& msg)
{
    // /robot/head_to_base 语义（zl 侧 head_pose.py）：eye(optical) -> 躯干系（torso_link 原点），
    // 去航向 yaw、只保留 roll/pitch，**不含 base_height**。旋转已含相机安装俯仰与光学系轴变换，
    // 因此点云直接左乘即可转到 base 系，不需要再减高度、也不能再乘 camera.extrin。
    //
    // 消息类型自 2026-11-11 起为 PoseStamped（zl 侧新增 header.stamp、frame_id 留空），
    // 位姿字段改在 msg.pose 下；header.stamp 暂不参与计算（跨机时钟不同源，见 vision PoseCallBack）。
    //
    // 不要退回 /robot/head_pose：那是骨盆系（过 waist_yaw），点云会随腰部扭转，
    // 自车排除框（正前方）会随腰角失效。
    Eigen::Matrix4d camToBase = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z);
    if (q.norm() > 1e-6) {
        camToBase.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    }
    camToBase.block<3, 1>(0, 3) = Eigen::Vector3d(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);

    data->camToRobot = camToBase;
    data->camToRobotValid = true;
}

int Brain::markCntOnFieldLine(const string markType, const FieldLine line, const double margin) {
    int cnt = 0;
    auto markings = data->getMarkings();
    for (int i = 0; i < markings.size(); i++) {
        auto marking = markings[i];
        if (marking.label == markType) {
            Point2D point = {marking.posToField.x, marking.posToField.y};
            if (fabs(pointPerpDistToLine(point, line.posToField)) < margin) {
                cnt += 1;
            }
        }
    }
    return cnt;
}

int Brain::goalpostCntOnFieldLine(const FieldLine line, const double margin) {
    int cnt = 0;
    auto goalposts = data->getGoalposts();
    for (int i = 0; i < goalposts.size(); i++) {
        auto post = goalposts[i];
        Point2D point = {post.posToField.x, post.posToField.y};
        if (pointMinDistToLine(point, line.posToField) < margin) {
            cnt += 1;
        }
    }
    return cnt;
}

bool Brain::isBallOnFieldLine(const FieldLine line, const double margin) {
    auto ballPos = data->ball.posToField;
    Point2D point = {ballPos.x, ballPos.y}; 
    return fabs(pointPerpDistToLine(point, line.posToField)) < margin;
}

void Brain::identifyFieldLine(FieldLine& line) {
    auto mapLines = config->mapLines;
    FieldLine mapLine;
    double confidence;
    line.type = LineType::NA;

    double bestConfidence = 0;
    double secondBestConfidence = 0;
    int bestIndex = -1;
    for (int i = 0; i < mapLines.size(); i++) {
        mapLine = mapLines[i];
        confidence = line.dir == mapLine.dir ? 
            probPartOfLine(line.posToField, mapLine.posToField)
            : 0.0;

        // Boost confidence with other features
        if (mapLine.type == LineType::GoalLine) { 
            confidence += 0.3 * markCntOnFieldLine("TCross", line, 0.2);
            confidence += 0.5 * goalpostCntOnFieldLine(line, 0.2);
            if (
                isBallOnFieldLine(line)
                && (tree->getEntry<string>("gc_game_sub_state") == "GET_READY" || tree->getEntry<string>("gc_game_sub_state") == "SET")
                && (data->realGameSubState == "CORNER_KICK")
            ) confidence += 0.3; // During corner kick, the ball is on the goal line
        }
        if (mapLine.type == LineType::MiddleLine) {
            confidence += 0.3 * markCntOnFieldLine("XCross", line, 0.2);
            if (
                isBallOnFieldLine(line)
                && (tree->getEntry<string>("gc_game_sub_state") == "GET_READY" || tree->getEntry<string>("gc_game_sub_state") == "SET")
                && (data->realGameSubState == "GOAL_KICK")
            ) confidence += 0.3; // During goal kick, the ball is on the middle line
        }
        if (mapLine.type == LineType::TouchLine) {
            if (
                isBallOnFieldLine(line)
                && (tree->getEntry<string>("gc_game_sub_state") == "GET_READY" || tree->getEntry<string>("gc_game_sub_state") == "SET")
                && (data->realGameSubState == "GOAL_KICK" || data->realGameSubState == "CORNER_KICK" || data->realGameSubState == "THROW_IN")
            ) confidence += 0.3; // During goal kick, corner kick, and throw-in, the ball is on the touch line
        }
        
        // Prevent misidentifying goal area line as goal line
        auto fd = config->fieldDimensions;
        if (
            mapLine.type == LineType::GoalLine
            && fabs(line.posToField.y0) < fd.goalAreaWidth / 2 + 0.5
            && fabs(line.posToField.y1) < fd.goalAreaWidth / 2 + 0.5
        ) confidence -= 0.3;

        // Prevent misidentifying penalty area as touchline
        if (
            mapLine.type == LineType::TouchLine
            && min(fabs(line.posToField.x0), fabs(line.posToField.x1)) > fd.length / 2.0 -  fd.penaltyAreaLength - 0.5
            && line.posToField.x0 * line.posToField.x1 > 0
        ) confidence -= 0.3;

        double length = norm(line.posToField.x0 - line.posToField.x1, line.posToField.y0 - line.posToField.y1);
        if (length < 0.5) confidence -= 0.5;
        else if (length < 1.0) confidence -= 0.1;
        
        if (confidence > bestConfidence) {
            secondBestConfidence = bestConfidence;
            bestConfidence = confidence;
            bestIndex = i;
        }
    }

    if (bestConfidence - secondBestConfidence < 0.5) bestConfidence -= 0.5;



    if (bestIndex >= 0 && bestIndex < mapLines.size()) {
        line.type = mapLines[bestIndex].type;
        line.half = mapLines[bestIndex].half;
        line.side = mapLines[bestIndex].side;
        line.confidence = bestConfidence;
        return;
    }

    // else 
    line.type = LineType::NA;
    line.half = LineHalf::NA;
    line.side = LineSide::NA;
    line.confidence = 0.0;
    return;
}

void Brain::identifyMarking(GameObject& marking) {
    double minDist = 100;
    double secMinDist = 100;
    int mmIndex = -1;
    for (int i = 0; i < config->mapMarkings.size(); i++) {
       auto mm = config->mapMarkings[i];
       
       if (mm.type != marking.label) continue;

       double dist = norm(marking.posToField.x - mm.x, marking.posToField.y - mm.y);

       if (dist < minDist) {
           secMinDist = minDist;
           minDist = dist;
           mmIndex = i;
       } else if (dist < secMinDist) {
           secMinDist = dist; 
       }
    }

    auto fd = config->fieldDimensions;
    if (
        mmIndex >=0 && mmIndex < config->mapMarkings.size()
        && minDist < 1.5 * 14 / fd.length // 1.0 for adultsize
        && secMinDist - minDist > 1.5 * 14 / fd.length // 2.0 for adultsize
    ) {
        marking.id = mmIndex;
        marking.name = config->mapMarkings[mmIndex].name;
        marking.idConfidence = 1.0;
    } else {
        marking.id = -1;
        marking.name = "NA";
        marking.idConfidence = 0.0;
    }
}


void Brain::identifyGoalpost(GameObject& goalpost) {
    string side = "NA";
    string half = "NA";
    if (goalpost.posToField.x > 0) half = "O";
    else half = "S";

    if (goalpost.posToField.y > 0) side = "L";
    else side = "R";
    
    goalpost.id = 0;
    goalpost.name = half + side;
    goalpost.idConfidence = 1.0;
}

vector<FieldLine> Brain::processFieldLines(vector<FieldLine>& fieldLines) {
    vector<FieldLine> original = fieldLines;
    vector<FieldLine> res;
    

    int sizeBefore = original.size();
    // merge lines that are actually the same line
    for (int i = 0; i < original.size(); i++) {
        for (int j = i + 1; j < original.size(); j++) {
            auto line1 = original[i].posToField;
            auto line2 = original[j].posToField;
            if (isSameLine(line1, line2, 0.1, 1.0)) {
                FieldLine mergedLine;
                mergedLine.posToField = mergeLines(line1, line2);
                mergedLine.posToRobot = mergeLines(original[i].posToRobot, original[j].posToRobot);
                mergedLine.posOnCam = mergeLines(original[i].posOnCam, original[j].posOnCam);
                mergedLine.timePoint = original[i].timePoint;

                // replace first line in original with merged line and remove second line
                original[i] = mergedLine;
                original.erase(original.begin() + j);
                j--;
            }
        }
    }
    int sizeAfter = original.size();
    // filter out lines that are too short and infer direction while ditch lines whose dir cannot be inferred
    double valve = 0.2;
    for (int i = 0; i < original.size(); i++) {
        auto line = original[i];
        auto lineDir = atan2(line.posToField.y1 - line.posToField.y0, line.posToField.x1 - line.posToField.x0);

        if (fabs(toPInPI(lineDir - M_PI)) < 0.1 || fabs(lineDir) < 0.1) line.dir = LineDir::Vertical;
        else if (fabs(toPInPI(lineDir - M_PI/2)) < 0.1 || fabs(toPInPI(lineDir + M_PI/2)) < 0.1) line.dir = LineDir::Horizontal;
        else continue;

        // if line is direction can be verified, check if it is long enough
        if (lineLength(line.posToField) > valve) {
            res.push_back(line);
        }
    }

    // identify each line 
    for (int i = 0; i < res.size(); i++) {
        identifyFieldLine(res[i]);
    }
    return res;
}


vector<GameObject> Brain::getGameObjects(const vision_interface::msg::Detections &detections)
{
    vector<GameObject> res;

    // auto timestamp = detections.header.stamp;
    // rclcpp::Time timePoint(timestamp.sec, timestamp.nanosec);
    auto timePoint = timePointFromHeader(detections.header);

    for (int i = 0; i < detections.detected_objects.size(); i++)
    {
        auto obj = detections.detected_objects[i];
        GameObject gObj;

        gObj.timePoint = timePoint;
        gObj.label = obj.label;
        gObj.color = obj.color;

        if (obj.target_uv.size() == 2)
        { // Precise pixel position information of ground markings
            gObj.precisePixelPoint.x = static_cast<double>(obj.target_uv[0]);
            gObj.precisePixelPoint.y = static_cast<double>(obj.target_uv[1]);
        }

        gObj.boundingBox.xmax = obj.xmax;
        gObj.boundingBox.xmin = obj.xmin;
        gObj.boundingBox.ymax = obj.ymax;
        gObj.boundingBox.ymin = obj.ymin;
        gObj.confidence = obj.confidence;

        // Do not use depth measurement, directly use projection distance
        gObj.posToRobot.x = obj.position_projection[0];
        gObj.posToRobot.y = obj.position_projection[1];

        // Calculate angles
        gObj.range = norm(gObj.posToRobot.x, gObj.posToRobot.y);
        gObj.yawToRobot = atan2(gObj.posToRobot.y, gObj.posToRobot.x);
        gObj.pitchToRobot = atan2(config->get_robot_height(), gObj.range); // Note: this is an approximate value

        // Calculate the position of the object in the field coordinate system
        transCoord(
            gObj.posToRobot.x, gObj.posToRobot.y, 0,
            data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
            gObj.posToField.x, gObj.posToField.y, gObj.posToField.z // Note: z is not used elsewhere, here it is just a placeholder
        );

        res.push_back(gObj);
    }

    return res;
}

void Brain::detectProcessBalls(const vector<GameObject> &ballObjs)
{
    // Mutual exclusion: when gt ball is enabled, vision ball is fully disabled.
    // Each run uses only one channel (simulation → gt, real robot → vision).
    if (config->get_gt_ball_enabled()) {
        return;
    }

    static rclcpp::Time lastSeenRealBallTime; 
    double bestConfidence = 0;
    int indexRealBall = -1;  // Which ball is considered real, -1 means no ball detected

    // Find the most likely real ball
    for (int i = 0; i < ballObjs.size(); i++)
    {
        auto ballObj = ballObjs[i];
        auto oldBall = data->ball;

        // Prevent misidentifying lights in the sky as balls
        if (ballObj.posToRobot.x < -0.5 || ballObj.posToRobot.x > 15.0)
            continue;

        // If the confidence is too low, consider it a false detection
        if (ballObj.confidence < config->get_ball_confidence_threshold())
            continue;


        // Find the ball with the highest confidence among the remaining ones
        if (ballObj.confidence > bestConfidence)
        {
            bestConfidence = ballObj.confidence;
            indexRealBall = i;
        }
    }

    auto now = this->get_clock()->now(); 

    if (indexRealBall >= 0)
    { // Ball detected
        data->ballDetected = true;

        data->ball = ballObjs[indexRealBall];
        data->ball.confidence = bestConfidence;

        tree->setEntry<bool>("ball_location_known", true);
        updateBallOut();
        
        lastSeenRealBallTime = now;
        data->lose_ball = false;
    }
    else
    { // No ball detected
        data->ballDetected = false;
        data->ball.boundingBox.xmin = 0;
        data->ball.boundingBox.xmax = 0;
        data->ball.boundingBox.ymin = 0;
        data->ball.boundingBox.ymax = 0;
        // 置信度清零：标记当前球位不再来自本帧检测（位置与 timePoint 保留，
        // 球龄由 timePoint 计算；消费端不得再用它驱动动作或上报）。
        data->ball.confidence = 0.0;

        if (lastSeenRealBallTime.seconds() > 0.0)
        {
            double msecs = (now - lastSeenRealBallTime).nanoseconds() / 1e6;
            data->lose_ball = (msecs > 2000.0);
        }
        else
        {
            data->lose_ball = false;
        }
    }

    // Calculate the vector from the robot to the ball in the field coordinate system
    data->robotBallAngleToField = atan2(data->ball.posToField.y - data->robotPoseToField.y, data->ball.posToField.x - data->robotPoseToField.x);
}

void Brain::detectProcessMarkings(const vector<GameObject> &markingObjs)
{
    // 2026-10-14: 50 → 30。llvmpipe 软渲染下标记成像质量偏低：入场区
    // （真实 x≈2.35, y≈5.14）实测 LCross 的 confidence 在 30~50 之间抖动，
    // 阈值 50 时单帧最多只有 1 个标记（TCross），SelfLocateEnterField 的
    // minMarkerCnt 永不满足、入场定位无法收敛。降至 30 后单帧典型得到
    // LCross+TCross 两个标记（45s 探针：48/191 帧出现该组合），配合
    // locator.min_marker_count=2 可完成入场定位。
    const double confidenceValve = 30; // Exclude markings with confidence below this threshold
    vector<GameObject> markings = {};
    for (int i = 0; i < markingObjs.size(); i++)
    {
        auto marking = markingObjs[i];

        // If the confidence is too low, consider it a false detection
        if (marking.confidence < confidenceValve)
            continue;

        // Exclude markings misidentified in the sky
        if (marking.posToRobot.x < -0.5 || marking.posToRobot.x > 15.0)
            continue;

        // If the marking passes all checks, record it in the brain
        identifyMarking(marking);
        markings.push_back(marking);
    }
    data->setMarkings(markings);
}

void Brain::detectProcessGoalposts(const vector<GameObject> &goalpostObjs)
{
    const double confidenceValve = 50; // Exclude goalposts with confidence below this threshold
    vector<GameObject> goalposts = {};

    for (int i = 0; i < goalpostObjs.size(); i++) {
        auto goalpost = goalpostObjs[i];

        // If the confidence is too low, consider it a false detection
        if (goalpost.confidence < confidenceValve)
            continue;

        identifyGoalpost(goalpost);
        goalposts.push_back(goalpost);
    }
    data->setGoalposts(goalposts);

}


void Brain::detectProcessRobots(const vector<GameObject> &robotObjs) {

    vector<GameObject> robots = {};
    for (int i = 0; i < robotObjs.size(); i++) {
        auto rbt = robotObjs[i];
        if (rbt.confidence < 50) continue;
        
        // else
        robots.push_back(rbt);
    }

    data->setRobots(robots);
}


void Brain::detectProcessVisionBox(const vision_interface::msg::Detections &msg) {
    // 防护：corner_pos 不足 8 个元素时无法构建 VisionBox，直接跳过
    if (msg.corner_pos.size() < 8) return;

    auto timePoint = timePointFromHeader(msg.header);

    // Process and record vision information
    VisionBox vbox;
    vbox.timePoint = timePoint;
    for (int i = 0; i < msg.corner_pos.size(); i++) vbox.posToRobot.push_back(msg.corner_pos[i]);

    // Handle the case where the top-left and top-right points have x < 0, indicating an infinitely distant scene
    const double VISION_LIMIT = 20.0;
    vector<vector<double>> v = {};
    for (int i = 0; i < 4; i++) {
        int start = i; int end = (i + 1) % 4;
        v.push_back({vbox.posToRobot[end * 2] - vbox.posToRobot[start * 2], vbox.posToRobot[end * 2 + 1] - vbox.posToRobot[start * 2 + 1]});
        v.push_back({-vbox.posToRobot[end * 2] + vbox.posToRobot[start * 2], -vbox.posToRobot[end * 2 + 1] + vbox.posToRobot[start * 2 + 1]});
    }

    for (int i = 0; i < 2; i++) {
        double ox = vbox.posToRobot[2* i]; double oy = vbox.posToRobot[2 * i + 1];
        if (
            (i == 0 && crossProduct(v[5], v[6]) < 0)
            || (i == 1 && crossProduct(v[3], v[4]) < 0)
        ){
            vbox.posToRobot[2 * i] = -ox / fabs(ox) * VISION_LIMIT;
            vbox.posToRobot[2 * i + 1] = -oy / fabs(oy) * VISION_LIMIT;
        }
    }

    // transform to field coordinate system
    for (int i = 0; i < 5; i++) {
        double xr, yr, xf, yf, __;
        xr = vbox.posToRobot[2 * i];
        yr = vbox.posToRobot[2 * i + 1];
        transCoord(
            xr, yr, 0,
            data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
            xf, yf, __
        );
        vbox.posToField.push_back(xf);
        vbox.posToField.push_back(yf);
    }
    
    data->visionBox = vbox;
}

void Brain::logDepth(int grid_x_count, int grid_y_count, vector<vector<int>> &grid_occupied, vector<std::array<float, 3>> &points_robot) {
    // 点云与障碍栅格可视化是每帧（30 Hz）发布的重操作，只在排查时开
    if (!config->get_publish_pointcloud()) return;
    // time is set on the outside
    const double grid_size = config->get_grid_size();  // Grid size
    const double x_min = 0.0, x_max = config->get_max_x();
    const double y_min = -config->get_max_y();
    const double y_max = -y_min;

    // Publish point cloud to ROS2 topic
    std::vector<std::tuple<float, float, float>> cloud_points;
    cloud_points.reserve(points_robot.size());
    for (const auto &point : points_robot) {
        cloud_points.emplace_back(point[0], point[1], point[2]);
    }
    visualizer->publishPointCloud(cloud_points, "odom");
    
    // Publish obstacle grid to ROS2 topic
    // ROS OccupancyGrid uses row-major format: index = y * width + x
    // width corresponds to the number of cells in the X direction, height corresponds to the number of cells in the Y direction
    std::vector<int8_t> grid_data(grid_x_count * grid_y_count, -1);  // -1 represents unknown
    for (int j = 0; j < grid_y_count; j++) {        // Y direction (rows)
        for (int i = 0; i < grid_x_count; i++) {    // X direction (columns)
            int index = j * grid_x_count + i;        // row-major: y * width + x
            if (grid_occupied[i][j] > 0) {
                // Map occupancy count to the range 0-100
                int occupancy = std::min(100, static_cast<int>(grid_occupied[i][j] * 10));
                grid_data[index] = static_cast<int8_t>(occupancy);
            } else {
                grid_data[index] = 0;  // Free
            }
        }
    }
    visualizer->publishObstacleGrid(grid_data, grid_x_count, grid_y_count, 
                                   grid_size, x_min, y_min, "odom");

    // Log ball exclusion box
    double r = config->get_ball_exclusion_radius();
    double h = config->get_ball_exclusion_height();
    log->debug(
        "depth/ball_exclusion_box",
        format("Ball exclusion box at (%.2f, %.2f) with radius %.2f", data->ball.posToRobot.x, data->ball.posToRobot.y, r)
    );
}

void Brain::logDebugInfo() {
    auto log_ = [=](string msg) {
        log->debug("brain_tick", msg);
    };
    string gameState = tree->getEntry<string>("gc_game_state");
    string gameSubState = tree->getEntry<string>("gc_game_sub_state");
    string gameSubStateType = tree->getEntry<string>("gc_game_sub_state_type");
    string isLead = data->tmImLead ? "ON" : "OFF";
    string ballOut = tree->getEntry<bool>("ball_out") ? "YES" : "NO";
    string ballDetected = data->ballDetected ? "YES" : "NO";
    string decision = tree->getEntry<string>("decision");
    string freeKickKickingOff = data->isFreekickKickingOff ? "YES" : "NO";
    string directShoot = data->isDirectShoot ? "YES" : "NO";
    int myStrikerIDRank = data->myStrikerIDRank;
    log_(format("Game State: %s, SubState: %s, SubStateType: %s, Lead: %s, Decision: %s, FreeKickKickingOff: %s, DirectShoot: %s, PrimaryStriker: %d",
        gameState.c_str(), gameSubState.c_str(), gameSubStateType.c_str(), isLead.c_str(), decision.c_str(), freeKickKickingOff.c_str(), directShoot.c_str(), myStrikerIDRank));

    log->debug("my_cost_scalar", format("cost: %.2f", data->tmMyCost));
    log->debug("my_lead_scalar", format("lead: %d", data->tmImLead));
}

void Brain::updateRelativePos(GameObject &obj) {
    Pose2D pf;
    pf.x = obj.posToField.x;
    pf.y = obj.posToField.y;
    pf.theta = 0;
    Pose2D pr = data->field2robot(pf);
    obj.posToRobot.x = pr.x;
    obj.posToRobot.y = pr.y;
    obj.range = norm(obj.posToRobot.x, obj.posToRobot.y);
    obj.yawToRobot = atan2(obj.posToRobot.y, obj.posToRobot.x);
    // range < robot_height 时 ratio>1，asin 返回 NaN（球贴脚时必现）；
    // 该值经 CamTrackBall 传到 moveHead 后会被当作“非有限值”丢弃，
    // 表现为头部 pitch 停在原地、不再跟球（实测 528 条 WARN）。夹到 [-1,1] 取极值姿态。
    obj.pitchToRobot = asin(cap(config->get_robot_height() / max(obj.range, 1e-3), 1.0, -1.0));
}

void Brain::updateFieldPos(GameObject &obj) {
    Pose2D pr;
    pr.x = obj.posToRobot.x;
    pr.y = obj.posToRobot.y;
    pr.theta = 0;
    Pose2D pf = data->robot2field(pr);
    obj.posToField.x = pf.x;
    obj.posToField.y = pf.y;
    obj.range = norm(obj.posToRobot.x, obj.posToRobot.y);
    obj.yawToRobot = atan2(obj.posToRobot.y, obj.posToRobot.x);
    // range < robot_height 时 ratio>1，asin 返回 NaN（球贴脚时必现）；
    // 该值经 CamTrackBall 传到 moveHead 后会被当作“非有限值”丢弃，
    // 表现为头部 pitch 停在原地、不再跟球（实测 528 条 WARN）。夹到 [-1,1] 取极值姿态。
    obj.pitchToRobot = asin(cap(config->get_robot_height() / max(obj.range, 1e-3), 1.0, -1.0));
}

void Brain::compressedDepthImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    data->timeLastDepth = get_clock()->now();
    try {
        // Decode compressed image
        cv::Mat compressed_data = cv::Mat(msg->data);
        cv::Mat depth_decoded = cv::imdecode(compressed_data, cv::IMREAD_ANYDEPTH);
        
        if (depth_decoded.empty()) {
            RCLCPP_ERROR(get_logger(), "Failed to decode compressed depth image");
            return;
        }

        // Convert to floating-point format
        cv::Mat depthFloat;
        if (depth_decoded.type() == CV_16UC1) {
            depth_decoded.convertTo(depthFloat, CV_32FC1, 1.0 / 1000.0);
        } else if (depth_decoded.type() == CV_32FC1) {
            depthFloat = depth_decoded;
        } else {
            RCLCPP_ERROR(get_logger(), "Unsupported decoded depth image type: %d", depth_decoded.type());
            return;
        }
        
        // Call the unified depth image processing function
        processDepthImage(depthFloat, depth_decoded.cols, depth_decoded.rows, msg->header);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(get_logger(), "Error in compressedDepthImageCallback: %s", e.what());
    }
}

void Brain::depthImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    data->timeLastDepth = get_clock()->now();
    try {
        // Check if the image data is valid
        if (msg->data.empty() || msg->height == 0 || msg->width == 0) {
            RCLCPP_WARN(get_logger(), "Received empty depth image");
            return;
        }

        // Create depth image and conversion
        cv::Mat depthFloat;
        // Process based on image encoding
        if (msg->encoding == "16UC1" || msg->encoding == "mono16") {
            size_t expected = (size_t)msg->width * msg->height * sizeof(uint16_t);
            if (msg->data.size() < expected) {
                RCLCPP_ERROR(get_logger(), "Depth mono16 size mismatch");
                return;
            }
            cv::Mat depthRaw(msg->height, msg->width, CV_16UC1, const_cast<uint8_t*>(msg->data.data()));
            depthRaw.convertTo(depthFloat, CV_32FC1, 1.0 / 1000.0); // If actual depth unit is mm
        } else if (msg->encoding == "32FC1") {
            // Check if data size is correct
            size_t expected_size = msg->height * msg->width * sizeof(float);
            if (msg->data.size() != expected_size) {
                RCLCPP_ERROR(get_logger(), "Depth image size mismatch: expected %zu, got %zu", 
                    expected_size, msg->data.size());
                return;
            }

            // Directly create 32-bit floating-point depth image
            depthFloat = cv::Mat(msg->height, msg->width, CV_32FC1, 
                const_cast<float*>(reinterpret_cast<const float*>(msg->data.data()))).clone();
            
        } else {
            RCLCPP_ERROR(get_logger(), "Unsupported depth image encoding: %s", msg->encoding.c_str());
            return;
        }

        // Call the unified depth image processing function
        processDepthImage(depthFloat, msg->width, msg->height, msg->header);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Exception in depth image callback: %s", e.what());
    }
}

void Brain::processDepthImage(const cv::Mat &depthFloat, int width, int height, const std_msgs::msg::Header &header)
{
    // 前置校验：没有有效外参/内参时算出的点云是错的（会把相机系当 base 系、
    // 或按 643.9 量级内参缩放），宁可不上障碍也不要上假障碍。告警限频 1 s。
    if (!data->camToRobotValid) {
        static rclcpp::Time lastWarn = get_clock()->now();
        if (msecsSince(lastWarn) > 1000) {
            lastWarn = get_clock()->now();
            RCLCPP_WARN(get_logger(), "[depth] skip: no /robot/head_to_base yet (camToRobot invalid)");
        }
        return;
    }
    if (config->depthCameraFx <= 1e-6 || config->depthCameraFy <= 1e-6) {
        static rclcpp::Time lastWarn = get_clock()->now();
        if (msecsSince(lastWarn) > 1000) {
            lastWarn = get_clock()->now();
            RCLCPP_WARN(get_logger(), "[depth] skip: no valid depth intrinsics (camera_info / vision.yaml both empty)");
        }
        return;
    }
    try {
        vector<std::array<float, 3>> points_robot;  // for log

        const double fx = config->depthCameraFx;
        const double fy = config->depthCameraFy;
        const double cx = config->depthCameraCx;
        const double cy = config->depthCameraCy;
        
        // Define grid parameters
        const double grid_size = config->get_grid_size();  // Grid size
        const double x_min = 0.0, x_max = config->get_max_x();
        const double y_min = -config->get_max_y();
        const double y_max = -y_min;
        const int grid_x_count = static_cast<int>((x_max - x_min) / grid_size);
        const int grid_y_count = static_cast<int>((y_max - y_min) / grid_size);
        
        // Create grid occupancy array
        vector<vector<int>> grid_occupied(grid_x_count, vector<int>(grid_y_count, 0));
        
        // Process depth image points
        const int sampleStep = config->get_depth_sample_step();
        for (int y = 0; y < height; y += sampleStep)
        {
            for (int x = 0; x < width; x += sampleStep)
            {
                float depth = depthFloat.at<float>(y, x);
                if (depth > 0)
                {
                    // Convert to camera coordinate system
                    double x_cam = (x - cx) * depth / fx;
                    double y_cam = (y - cy) * depth / fy;
                    double z_cam = depth;

                    // Convert to robot coordinate system
                    Eigen::Vector4d point_cam(x_cam, y_cam, z_cam, 1.0);
                    Eigen::Vector4d point_robot = data->camToRobot * point_cam;
                    
                    // Record points for visualization
                    points_robot.push_back({static_cast<float>(point_robot(0)), static_cast<float>(point_robot(1)), static_cast<float>(point_robot(2))});
                    
                    // Update grid occupancy
                    // obstacle_min_height 的语义是“离地高度”，而 point_robot 已是 base 系
                    // （地面在 z = -base_height），所以阈值必须随 base_height 平移。
                    const double Z_THRESHOLD = config->get_obstacle_min_height() - config->get_base_height_m();
                    const double EXCLUDE_MAX_X = config->get_exclusion_x(); // Exclude robot's own body
                    const double EXCLUDE_MIN_X = -EXCLUDE_MAX_X;
                    const double EXCLUDE_MAX_Y = config->get_exclusion_y(); // Exclude robot's own body
                    const double EXCLUDE_MIN_Y = -EXCLUDE_MAX_Y;

                    auto isInRange = [&]() {
                        return point_robot(0) >= x_min && point_robot(0) < x_max
                            && point_robot(1) >= y_min && point_robot(1) < y_max;
                    };
                    auto isSelfBody = [&]() {
                        return point_robot(0) >= EXCLUDE_MIN_X && point_robot(0) <= EXCLUDE_MAX_X
                            && point_robot(1) >= EXCLUDE_MIN_Y && point_robot(1) <= EXCLUDE_MAX_Y;
                    };
                    auto isBall = [&]() {
                        double r = config->get_ball_exclusion_radius();
                        // ball_exclusion_height 同为离地高度，与 Z_THRESHOLD 同一基准
                        double h = config->get_ball_exclusion_height() - config->get_base_height_m();
                        return fabs(point_robot(0) - data->ball.posToRobot.x) < r 
                            && fabs(point_robot(1) - data->ball.posToRobot.y) < r
                            && point_robot(2) < h;
                    };

                    if (
                        point_robot(2) > Z_THRESHOLD 
                        && isInRange()
                        &&!isSelfBody() 
                        &&!isBall()
                    )
                    {
                        int grid_x = static_cast<int>((point_robot(0) - x_min) / grid_size);
                        int grid_y = static_cast<int>((point_robot(1) - y_min) / grid_size);
                        
                        // Add boundary check to prevent out-of-bounds
                        if (grid_x >= 0 && grid_x < grid_x_count && grid_y >= 0 && grid_y < grid_y_count) {
                            grid_occupied[grid_x][grid_y] += 1;
                        }
                    }
                }
            }
        }

        auto obs_old = data->getObstacles();
        vector<GameObject> obs_new = {};

        // Record newly seen obstacles
        for (int i = 0; i < grid_x_count; i++) {
            for (int j = 0; j < grid_y_count; j++) {
                if (grid_occupied[i][j] > 0) {
                    GameObject obj;
                    obj.label = "Obstacle";
                    obj.timePoint = get_clock()->now();
                    obj.posToRobot.x = x_min + (i + 0.5) * grid_size;
                    obj.posToRobot.y = y_min + (j + 0.5) * grid_size;
                    obj.confidence = grid_occupied[i][j];
                    updateFieldPos(obj);
                    obs_new.push_back(obj);
                }
            }
        }

        // Clean up old obstacles
        for (int i = 0; i < obs_old.size(); i++) {
           // First, clear old obstacles within the current field of view. Note that the angle is only roughly calculated, and the range is appropriately expanded using an offset.
            double visionLeft = data->headYaw + config->depthCameraFovX / 2;
            double visionRight = data->headYaw - config->depthCameraFovX / 2;
            auto obs = obs_old[i];
            const double offset = 0.20;
            double obsYawLeft = atan2(obs.posToRobot.y - offset, obs.posToRobot.x + offset);
            double obsYawRight = atan2(obs.posToRobot.y + offset, obs.posToRobot.x + offset);
            if (obsYawLeft < visionLeft && obsYawRight > visionRight) continue; 

            // If the old obstacle is too close to the new obstacle, it is considered no longer present to prevent accumulation at the boundaries.
            bool found = false;
            for (int j = 0; j < obs_new.size(); j++) {
                auto obs_n = obs_new[j];
                double dist = norm(obs.posToRobot.x - obs_n.posToRobot.x, obs.posToRobot.y - obs_n.posToRobot.y);
                if (dist < 0.5 * grid_size) {
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // else
            obs_new.push_back(obs);
        }

        
        data->setObstacles(obs_new); // note: Old obstacles that have timed out are not cleared here, but in the tick function
        logDepth(grid_x_count, grid_y_count, grid_occupied, points_robot);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Exception in depth image processing: %s", e.what());
    }
}

double Brain::distToObstacle(double angle) {
    auto obs = data->getObstacles();
    double minDist = 1e9;
    double obstacleThreshold = config->get_occupancy_threshold();
    double collisionThreshold = config->get_collision_threshold();

    for (int i = 0; i < obs.size(); i++) {
        if (obs[i].confidence < obstacleThreshold) continue;

        auto o = obs[i];
        Line line = {
            0, 0,
            cos(angle) * 100, sin(angle) * 100
        };
        double perpDist = fabs(pointPerpDistToLine(Point2D{o.posToRobot.x, o.posToRobot.y}, line));
        if (perpDist < collisionThreshold) {
            double dist = innerProduct(vector<double>{o.posToRobot.x, o.posToRobot.y}, vector<double>{cos(angle), sin(angle)});
            if (dist > 0 && dist < minDist) {
                minDist = dist;
            }
        }
    }
    return minDist;
}

double Brain::brakeSpeed(double distance, double decel, double latency) {
    if (!std::isfinite(distance)) return 1e9;
    if (distance <= 0.0) return 0.0;
    const double a = decel > 0.0 ? decel : config->get_brake_decel();
    const double tau = latency >= 0.0 ? latency : config->get_brake_latency();
    if (a <= 0.0) return 1e9; // 未配置 = 不约束
    const double at = a * max(0.0, tau);
    return -at + sqrt(at * at + 2.0 * a * distance);
}

double Brain::speedEnvelopeCap(double headingRad, const char *tag) {
    const double vMax = config->get_speed_envelope_ceiling() > 0.0
                            ? config->get_speed_envelope_ceiling()
                            : config->get_vx_limit();
    if (!config->get_speed_envelope_enable() || vMax <= 0.0) return vMax;
    const double a = config->get_speed_envelope_decel();
    const double tau = config->get_speed_envelope_latency();
    if (a <= 0.0) return vMax;

    double capAll = vMax;

    // 1) 障碍：沿该方向的走廊清距。栅格量程外/无障时恒为 1e9，等于不约束。
    const double obsDist = distToObstacle(headingRad);
    if (obsDist < 1e8) {
        capAll = min(capAll, brakeSpeed(obsDist - config->get_speed_envelope_margin(), a, tau));
    }

    // 2) 队友：最近存活队友的直线距离（让位仲裁另有一套，这里只做减速不干预方向）
    double tmDist = 1e9;
    const int selfIdx = config->get_player_id() - 1;
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx || !data->tmStatus[i].isAlive) continue;
        const double d = norm(data->robotPoseToField.x - data->tmStatus[i].robotPoseToField.x,
                              data->robotPoseToField.y - data->tmStatus[i].robotPoseToField.y);
        if (d < tmDist) tmDist = d;
    }
    if (tmDist < 1e8) {
        capAll = min(capAll, brakeSpeed(tmDist - config->get_speed_envelope_tm_margin(), a, tau));
    }

    // 3) 场地边线：沿行进方向（场系）到最近边界的距离，已越界视为 0。
    // 射线与矩形求交：x/y 两个方向各算一次，取先打到的那条。
    const double fieldHeading = data->robotPoseToField.theta + headingRad;
    const double dx = cos(fieldHeading), dy = sin(fieldHeading);
    const double halfL = config->fieldDimensions.length / 2.0;
    const double halfW = config->fieldDimensions.width / 2.0;
    const double px = data->robotPoseToField.x, py = data->robotPoseToField.y;
    double edgeDist = 1e9;
    if (fabs(dx) > 1e-9) edgeDist = min(edgeDist, max(0.0, ((dx > 0 ? halfL : -halfL) - px) / dx));
    if (fabs(dy) > 1e-9) edgeDist = min(edgeDist, max(0.0, ((dy > 0 ? halfW : -halfW) - py) / dy));
    if (edgeDist < 1e8) {
        capAll = min(capAll, brakeSpeed(edgeDist - config->get_speed_envelope_field_margin(), a, tau));
    }

    // 下限兜底（先夹到不超过上限，否则 cap() 的参数顺序会反转）
    const double floorV = min(config->get_speed_envelope_floor(), vMax);
    capAll = cap(capAll, vMax, floorV);

    // 1 Hz 观测：实机上区分“是包络在限速”与“节点/加速度限制在限速”。
    if (capAll < vMax - 1e-3) {
        static rclcpp::Time lastEnvLog = get_clock()->now();
        if (msecsSince(lastEnvLog) > 1000) {
            lastEnvLog = get_clock()->now();
            log->log(string("env/") + (tag ? tag : "all"),
                     format("cap=%.2f obs=%.2f tm=%.2f edge=%.2f vmax=%.2f a=%.1f tau=%.2f",
                            capAll, obsDist, tmDist, edgeDist, vMax, a, tau));
        }
    }
    return capAll;
}

vector<double> Brain::findSafeDirections(double startAngle, double safeDist, double step) {
    double safeAngleLeft = startAngle;
    double safeAngleRight = startAngle;
    double leftFound = 0;
    double rightFound = 0;
    for (double angle = startAngle; angle < startAngle + M_PI; angle += step) {
        if (distToObstacle(angle) > safeDist) {
            safeAngleLeft = angle;
            leftFound = 1;
            break;
        }
    }
    for (double angle = startAngle; angle > startAngle - M_PI; angle -= step) {
        if (distToObstacle(angle) > safeDist) {
            safeAngleRight = angle;
            rightFound = 1;
            break;
        }
    }

    return vector<double>{leftFound, toPInPI(safeAngleLeft), rightFound, toPInPI(safeAngleRight)};
}

double Brain::calcAvoidDir(double startAngle, double safeDist) {
    auto res = findSafeDirections(startAngle, safeDist);
    bool leftFound = res[0] > 0.5;
    bool rightFound = res[2] > 0.5;
    double angleLeft = res[1];
    double angleRight = res[3]; 
    double determinedAngle = 0;
    if (leftFound && rightFound) {
        determinedAngle = fabs(angleLeft) < fabs(angleRight) ? angleLeft : angleRight;
    } else if (leftFound) {
        determinedAngle = angleLeft;
    } else if (rightFound) {
        determinedAngle = angleRight;
    } else {
        return 0;
    }
    return toPInPI(determinedAngle);
}

bool Brain::calcSafeAvoidDir(double startAngle, double safeDist, double maxOffsetRad, double &outDir) {
    // 贴身区不做绕行：走廊模型下，小角度偏转会把障碍推出 collision_threshold，
    // distToObstacle 随即返回 1e9，回查会把 0.6 m 的障碍误判成“已绕开”
    // （实测 0.62 m 时算出的绕行角塜缩到 -2.9°）。这一段交给调用方停车。
    if (distToObstacle(startAngle) < config->get_avoid_min_clearance_m()) return false;

    // 栅格只覆盖相机 FOV（水平半角约 40°），FOV 外 distToObstacle 恒返 1e9，
    // 会被 findSafeDirections 当成“安全方向”。因此这里先限幅，再回查限幅后的方向是否真的可行。
    double dir = calcAvoidDir(startAngle, safeDist);
    dir = cap(dir, maxOffsetRad, -maxOffsetRad);
    if (distToObstacle(dir) < safeDist * 0.5) return false;
    outDir = toPInPI(dir);
    return true;
}

bool Brain::calcTeammateSeparationVelocity(double &vx, double &vy, double &vtheta, double rSep, double speed) {
    auto robotPose = data->robotPoseToField;
    auto ballPos = data->ball.posToField;
    int selfIdx = config->get_player_id() - 1;

    // 找最近的存活队友（不跳过门将/lead：这类队友拥有优先权）
    double sepDist = 1e9;
    double sepDX = 0, sepDY = 0;
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx) continue;
        auto tmStatus = data->tmStatus[i];
        if (!tmStatus.isAlive) continue;
        double dx = robotPose.x - tmStatus.robotPoseToField.x;
        double dy = robotPose.y - tmStatus.robotPoseToField.y;
        double d = norm(dx, dy);
        if (d < sepDist) {
            sepDist = d;
            sepDX = dx;
            sepDY = dy;
        }
    }
    if (sepDist >= rSep) return false;

    double dirX = sepDX, dirY = sepDY;
    if (sepDist < 0.05) { // 位置几乎重合，无法定向，按球在场地哪一侧决定退让方向
        dirX = ballPos.y > 0 ? -1.0 : 1.0;
        dirY = 0.0;
    }
    if (distToBorder() > - 1.0) { // 已贴边，改沿切线退开，避免被顶出场外
        double txDir = -dirY, tyDir = dirX;
        if (txDir * robotPose.x + tyDir * robotPose.y > 0) {
            txDir = -txDir;
            tyDir = -tyDir;
        }
        dirX = txDir;
        dirY = tyDir;
    }
    Pose2D sepDir;
    sepDir.x = dirX;
    sepDir.y = dirY;
    sepDir.theta = 0;
    auto sepDir_r = data->field2robot(sepDir);
    double sepNorm = norm(sepDir_r.x, sepDir_r.y);
    vx = speed * sepDir_r.x / sepNorm;
    vy = speed * sepDir_r.y / sepNorm;
    vtheta = data->ball.yawToRobot * 4.0;
    log->debug("TeammateSeparation", format("teammate too close (%.2f m), yield vx: %.2f vy: %.2f", sepDist, vx, vy));
    return true;
}

bool Brain::calcChaseYieldVelocity(double &vx, double &vy, double &vtheta, double minBallRange) {
    // 追球/绕球阶段（Chase、Adjust）的单侧队友让位。与 Assist 的直接让位不同：
    // Chase 里两台同时判 lead 的瞬态会同时冲向同一球后点，如果两边都无条件退开，
    // 会变成互相退让、球无人跟；这里先用 (cost, player_id) 仲裁决定谁让：
    //   对方明显更省（差 > 0.1）或同档内 id 更小 → 我退，对方不退，因此不会双退。
    // 球已进入踢球距离时不退，避免把正在起脚/绕球对准的一台拽离球。
    if (data->ball.range < minBallRange) return false;

    auto robotPose = data->robotPoseToField;
    int selfIdx = config->get_player_id() - 1;
    double sepDist = 1e9;
    int nearIdx = -1;
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx) continue;
        auto &tmStatus = data->tmStatus[i];
        if (!tmStatus.isAlive) continue;
        double d = norm(robotPose.x - tmStatus.robotPoseToField.x, robotPose.y - tmStatus.robotPoseToField.y);
        if (d < sepDist) {
            sepDist = d;
            nearIdx = i;
        }
    }
    if (nearIdx < 0 || sepDist >= 0.7) return false;

    auto &nearTm = data->tmStatus[nearIdx];
    bool iYield = (nearTm.role == "goal_keeper"); // 门将有优先权，贴身时直接让
    if (!iYield) {
        constexpr double YIELD_DEADBAND = 0.1;
        if (nearTm.cost < data->tmMyCost - YIELD_DEADBAND) iYield = true;
        else if (fabs(nearTm.cost - data->tmMyCost) <= YIELD_DEADBAND) iYield = (nearIdx + 1) > config->get_player_id();
    }
    if (!iYield) return false;

    if (!calcTeammateSeparationVelocity(vx, vy, vtheta, 0.7, 0.4)) return false;
    // 1 Hz INFO：实机上 debug 级日志默认不落 stdout（/tmp/brain.log 里看不到），
    // 需要一条能直接查的让位证据。判据：双机贴身时只有较差的那台打印 yield。
    static rclcpp::Time lastYieldLog = get_clock()->now();
    if (msecsSince(lastYieldLog) > 1000) {
        lastYieldLog = get_clock()->now();
        RCLCPP_INFO(get_logger(), "[yield] yield to P%d sep=%.2f m ball=%.2f m tmCost=%.1f myCost=%.1f",
            nearIdx + 1, sepDist, data->ball.range, nearTm.cost, data->tmMyCost);
    }
    log->debug("TeammateSeparation", format("yield in chase (sep %.2f m, ball %.2f m, tmCost %.1f, myCost %.1f)",
        sepDist, data->ball.range, nearTm.cost, data->tmMyCost));
    return true;
}


// ------------------------------------------------------ Debug log related ------------------------------------------------------
void Brain::logLags() {
    
    double detLag = msecsSince(data->timeLastDet);

    double MAX_LAG_LENGTH = config->cameraImageWidth;
    log->log_scalar(
        "performance",
        "detection_lag",
        detLag
    );
    

    // log fieldline detection delay
    double lineLag = msecsSince(data->timeLastLineDet);


    log->log_scalar(
        "performance",
        "fieldline_detection_lag",
        lineLag
    );

    // log game control delay
    double gcLag = msecsSince(data->timeLastGamecontrolMsg);

    log->log_scalar(
        "performance",
        "gamecontrol_lag",
        gcLag
    );
}

void Brain::logObstacleStatus() {
    // 深度链是避障的唯一输入源，但它没有心跳——没有这条日志就无法区分
    // "没有障碍"与"根本没有深度数据"。1 Hz 输出，量很小。
    static rclcpp::Time lastLog = get_clock()->now();
    if (msecsSince(lastLog) < 1000) return;
    lastLog = get_clock()->now();

    const double depthLag = msecsSince(data->timeLastDepth);
    auto obs = data->getObstacles();
    double nearest = 1e9;
    double nearestYawDeg = 0.0;
    double maxConf = 0.0;
    for (const auto &o : obs) {
        double d = norm(o.posToRobot.x, o.posToRobot.y);
        if (d < nearest) {
            nearest = d;
            nearestYawDeg = rad2deg(atan2(o.posToRobot.y, o.posToRobot.x));
        }
        if (o.confidence > maxConf) maxConf = o.confidence;
    }

    // nyaw/maxconf 用来区分“正前方真没东西”（fwd=1e9 但 nyaw 偏大）与
    // “有障碍却被 occupancy_threshold 滤掉”（maxconf < 阈值）。
    RCLCPP_INFO(get_logger(),
                "[obstacle] n=%zu nearest=%.2f nyaw=%+.0f maxconf=%.0f fwd=%.2f depth_lag=%.0fms intrinsics=%s extrin=ignored",
                obs.size(), nearest, nearestYawDeg, maxConf, distToObstacle(0.0), depthLag,
                config->depthIntrinsicsFromCameraInfo ? "camera_info" : "vision.yaml");

    if (depthLag > 1000) {
        static rclcpp::Time lastWarn = get_clock()->now();
        if (msecsSince(lastWarn) > 10000) {
            lastWarn = get_clock()->now();
            RCLCPP_WARN(get_logger(),
                        "[depth] no image for %.1fs (topic=%s), obstacle avoidance degraded",
                        depthLag / 1000.0, config->get_depth_image_topic().c_str());
        }
    }
}

void Brain::logStatusToConsole() {
    static int cnt = 0;
    const int LOG_INTERVAL = 30;
    cnt++;
    if (cnt % LOG_INTERVAL == 0) {
        string msg = "";
        string gameState = tree->getEntry<string>("gc_game_state");
        gameState = gameState == "" ? "-----" : gameState;
        string gameSubType = tree->getEntry<string>("gc_game_sub_state_type");
        gameSubType = gameSubType == "" ? "-----" : gameSubType;
        string gameSubState = tree->getEntry<string>("gc_game_sub_state");
        gameSubState = gameSubState == "" ? "-----" : gameSubState;

        msg += format(
            "ROBOT:\n\tTeamID: %d\tPlayerID: %d\tNumberOfPlayers: %d\tRole: %s\tStartRole: %s\n\n",
            config->get_team_id(),
            config->get_player_id(),
            config->get_num_of_players(),
            tree->getEntry<string>("player_role").c_str(),
            config->get_player_role().c_str()
        );
        msg += format(
            "GAME:\n\tState: %s\tKickOffSide: %s\tisKickingOff: %s(%s)\n\tSubType: %s\tSubState: %s\tSubKickOffSide: %s\tisKickingOff: %s(%s)\n\tScore: %s\tJustScored: %s\n\tLiveCount: %d\tOppoLiveCount: %d\n\n", 
            gameState.c_str(), 
            tree->getEntry<bool>("gc_is_kickoff_side") ? "YES" : "NO",
            data->isKickingOff ? "YES" : "NO",
            msecsSince(data->kickoffStartTime)/1000 > 100 ? "--" : to_string(msecsSince(data->kickoffStartTime)/1000).c_str(),
            gameSubType.c_str(),
            gameSubState.c_str(),
            tree->getEntry<bool>("gc_is_sub_state_kickoff_side") ? "YES" : "NO",
            data->isFreekickKickingOff ? "YES" : "NO",
            msecsSince(data->freekickKickoffStartTime)/1000 > 100 ? "--" : to_string(msecsSince(data->freekickKickoffStartTime)/1000).c_str(),
            format("%d:%d", data->score, data->oppoScore).c_str(),
            tree->getEntry<bool>("we_just_scored") ? "YES" : "NO",
            data->liveCount,
            data->oppoLiveCount
        );
        
        msg += getComLogString();

        msg += format(
            "DEBUG:\n\tcom: %s\tvxFactor: %.2f\tyawOffset: %.2f\n\tControlState: %d\n\tTickTime: %.0fms",
            config->get_enable_com() ? "YES" : "NO",
            config->get_vx_factor(),
            config->get_yaw_offset(),
            tree->getEntry<int>("control_state"),
            msecsSince(data->lastTick)
        );
        prtDebug(msg);
    }

    // 2026-11-05 定位振荡取证：同一拍打印"机器人在场地图的位姿 + 球在机器人系 + 球在场地图 + 决策"。
    // 用途：区分"目标（球位/场地坐标）本身在跳"与"执行层在摆"——只看 setVelocity_out 无法区分。
    // debug.track_log_hz <= 0 时关闭（默认）。
    {
        const double trackHz = config->get_track_log_hz();
        if (trackHz > 0)
        {
            static rclcpp::Time lastTrackLog;
            if (lastTrackLog.nanoseconds() == 0 || msecsSince(lastTrackLog) >= 1000.0 / trackHz)
            {
                lastTrackLog = get_clock()->now();
                log->log("dbg/track", format(
                    "pose=(%.2f,%.2f,%.2f) ballR=(%.2f,%.2f) ballF=(%.2f,%.2f) range=%.2f yaw=%.2f det=%d age=%.0fms kickDir=%.2f dec=%s",
                    data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
                    data->ball.posToRobot.x, data->ball.posToRobot.y,
                    data->ball.posToField.x, data->ball.posToField.y,
                    data->ball.range, data->ball.yawToRobot,
                    data->ballDetected ? 1 : 0, msecsSince(data->ball.timePoint),
                    data->kickDir, tree->getEntry<string>("decision").c_str()));
            }
        }
    }
    data->lastTick = get_clock()->now();
}

string Brain::getComLogString() {
    stringstream ss;
    int onFieldCnt = 0;
    int aliveCnt = 0;
    int selfIdx = config->get_player_id() - 1;
    vector<int> onFieldIdxs = {};
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx) continue;

        if (data->penalty[i] == PENALTY_NONE) {
            onFieldCnt += 1;
            onFieldIdxs.push_back(i);
        }

        if (data->tmStatus[i].isAlive) aliveCnt += 1;
    }
    ss << CYAN_CODE << "COM: " << "\n";
    ss << "Teammates: OnField: " << onFieldCnt << "[";
    for (int i = 0; i < onFieldIdxs.size(); i++) {
        int idx = onFieldIdxs[i];
        ss << " P" << idx + 1 << " ";
    }
    ss << "]";
    ss << "  Alive: " << aliveCnt << "  TMCMDID: " << data->tmCmdId << "  ReceivedDMD: " << data->tmReceivedCmd << "\n";

    // Self info
    ss << "Self\tCost: " << format("%.1f", data->tmMyCost) << "\tLead: ";
    if (data->tmImLead)
        ss << GREEN_CODE << "YES" << CYAN_CODE;
    else
        ss << RED_CODE << "NO" << CYAN_CODE;
    ss << "    TMCMD: " << data->tmCmdId << format("\tCMD: [%d]%d", data->tmMyCmdId, data->tmMyCmd);
    ss << "\n";

    // Teammates info
    for (int i = 0; i < onFieldIdxs.size(); i++) {
        int idx = onFieldIdxs[i];
        auto status = data->tmStatus[idx];
        ss << "P" << idx + 1 << "[";
        if (status.isAlive)
            ss << GREEN_CODE << "★" << CYAN_CODE;
        else 
            ss << RED_CODE << "☆" << CYAN_CODE;
        ss << "]\tCost: " << format("%.1f", status.cost);
        ss << "\tLead: ";
        if (status.isLead)
            ss << GREEN_CODE << "YES" << CYAN_CODE;
        else
            ss << RED_CODE << "NO" << CYAN_CODE;
        ss << "\tCMD: " << format("[%d]%d", status.cmdId, status.cmd);
        ss << "\tLag: " << format("%.0f", msecsSince(status.timeLastCom)) << "ms" << "\n";
    }
    ss << "\n";
    
    return ss.str();
}

bool Brain::isFreekickStartPlacing() {
    return (tree->getEntry<string>("gc_game_sub_state_type") == "FREE_KICK" && tree->getEntry<string>("gc_game_state") == "PLAY" && tree->getEntry<string>("gc_game_sub_state") == "GET_READY");
}


void Brain::agentCommandCallback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Received agent command: %s", msg->data.c_str());

    data->timeLastGamecontrolMsg = get_clock()->now();



    // agent_command    control_state   gc_game_state
    // locate           2               INITIAL
    // ready            3               READY
    // set              3               SET
    // play             3               PLAY
    // stop             3               END
    string game_state;
    int control_state;

    // Set control_state and gc_game_state based on msg->data
    if (msg->data == "locate") {
        control_state = 2;
        game_state = "INITIAL";
    } else if (msg->data == "ready") {
        control_state = 3;
        game_state = "READY";
    } else if (msg->data == "set") {
        control_state = 3;
        game_state = "SET";

        // Same as play: no real referee here, treat us as the kickoff side so
        // updateKickoffMemory() does not set wait_for_opponent_kickoff and the
        // following PLAY keeps the normal-play branch.
        tree->setEntry<bool>("gc_is_kickoff_side", true);
        tree->setEntry<bool>("gc_is_sub_state_kickoff_side", true);
    } else if (msg->data == "play") {
        control_state = 3;
        game_state = "PLAY";

        // In agent mode, start the kickoff directly
        tree->setEntry<bool>("gc_is_kickoff_side", true);
        tree->setEntry<bool>("gc_is_sub_state_kickoff_side", true);
    } else if (msg->data == "stop") {
        control_state = 3;
        game_state = "END";
    } else {
        RCLCPP_WARN(get_logger(), "Unknown agent command: %s", msg->data.c_str());
        return;
    }

    tree->setEntry<int>("control_state", control_state);
    tree->setEntry<string>("gc_game_state", game_state);
    // Currently, playerRole is automatically modified during runtime in the following situations:
    // handleCooperation
    //   1. If there is no strategy during halftime, the role will be switched back to the initial setting at the start of the second half, requiring gc_game_state to be INITIAL corresponding to the soccer_agent's positioning state.
    //   2. If role_switch is enabled, the role will be automatically switched when the number of players on the field is insufficient after a penalty.
    //   3. In the ready state, the role will be set according to the player_role parameter only when the field is full.
    //   4. When the goalkeeper goes out, the role will be automatically switched.
    // Remote control switches playerRole
    //
    // After switching the control state, set the player_role to ensure it takes effect
    tree->setEntry<string>("player_role", config->get_player_role());
}

void Brain::publishVisualizationMarkers()
{
    visualization_msgs::msg::MarkerArray marker_array;

    // 1. Publish field map (fixed)
    auto &fd = config->fieldDimensions;
    
    // Center line
    marker_array.markers.push_back(
        visualizer->createFieldCenterLineMarker(fd.width, "map"));
    
    // Center circle
    marker_array.markers.push_back(
        visualizer->createFieldCenterCircleMarker(fd.circleRadius, "map"));
    
    // Field boundary
    marker_array.markers.push_back(
        visualizer->createFieldBoundaryMarker(fd.length, fd.width, "map"));
    
    // Our goal area
    marker_array.markers.push_back(
        visualizer->createGoalAreaMarker(true, fd.length, fd.goalAreaLength, fd.goalAreaWidth, "map"));
    
    // Opponent's goal area
    marker_array.markers.push_back(
        visualizer->createGoalAreaMarker(false, fd.length, fd.goalAreaLength, fd.goalAreaWidth, "map"));
    
    // Our penalty area (large penalty area)
    marker_array.markers.push_back(
        visualizer->createPenaltyAreaMarker(true, fd.length, fd.penaltyAreaLength, fd.penaltyAreaWidth, "map"));
    
    // Opponent's penalty area (large penalty area)
    marker_array.markers.push_back(
        visualizer->createPenaltyAreaMarker(false, fd.length, fd.penaltyAreaLength, fd.penaltyAreaWidth, "map"));
    
    // Our penalty point
    marker_array.markers.push_back(
        visualizer->createPenaltyPointMarker(true, fd.length, fd.penaltyDist, "map"));
    
    // Opponent's penalty point
    marker_array.markers.push_back(
        visualizer->createPenaltyPointMarker(false, fd.length, fd.penaltyDist, "map"));

    // 2. Publish robot position - with orientation arrow
    auto robot_marker = visualizer->createRobotMarker(
        data->robotPoseToField.x,
        data->robotPoseToField.y,
        data->robotPoseToField.theta,
        "map");
    marker_array.markers.push_back(robot_marker);

    // 3. Publish ball position
    auto ball_marker = visualizer->createBallMarker(
        data->ball.posToField.x,
        data->ball.posToField.y,
        0.11, // ball radius is 0.11
        "map");
    marker_array.markers.push_back(ball_marker);

    // 4. Publish observed Mark points (dynamic)
    std::vector<std::tuple<double, double, char>> mark_points;
    auto markings = data->getMarkings();
    for (const auto &marking : markings)
    {
        char type = marking.label == "LCross" ? 'L' : 
                   (marking.label == "TCross" ? 'T' : 
                   (marking.label == "XCross" ? 'X' : 'P'));
        mark_points.push_back({marking.posToField.x, marking.posToField.y, type});
    }
    
    if (!mark_points.empty())
    {
        auto mark_markers = visualizer->createObservedMarkPointMarkers(mark_points, "map");
        marker_array.markers.insert(marker_array.markers.end(), 
            mark_markers.begin(), mark_markers.end());
    }

    // 5. Publish observed field lines (dynamic)
    std::vector<std::vector<geometry_msgs::msg::Point>> field_lines;
    auto lines = data->getFieldLines();
    for (const auto &line : lines)
    {
        std::vector<geometry_msgs::msg::Point> line_points;
        geometry_msgs::msg::Point p1, p2;
        p1.x = line.posToField.x0;
        p1.y = line.posToField.y0;
        p1.z = 0.0;
        p2.x = line.posToField.x1;
        p2.y = line.posToField.y1;
        p2.z = 0.0;
        line_points.push_back(p1);
        line_points.push_back(p2);
        field_lines.push_back(line_points);
    }
    
    if (!field_lines.empty())
    {
        auto line_markers = visualizer->createObservedFieldLineMarkers(field_lines, "map");
        marker_array.markers.insert(marker_array.markers.end(), 
            line_markers.begin(), line_markers.end());
    }

    // 6. Publish GameController information (game state, score, etc.)
    string gameState = tree->getEntry<string>("gc_game_state");
    string gameSubState = tree->getEntry<string>("gc_game_sub_state");
    int myScore = data->score;
    int oppoScore = data->oppoScore;
    int remainingTime = data->secsRemaining;
    
    auto gc_marker = visualizer->createGameControllerInfoMarker(
        myScore,
        oppoScore,
        remainingTime,
        "map");
    marker_array.markers.push_back(gc_marker);
    
    auto gc_state_marker = visualizer->createGameControllerStateMarker(
        gameState,
        gameSubState,
        "map");
    marker_array.markers.push_back(gc_state_marker);

    visualizer->publishMarkers(marker_array);
}

void Brain::publishOdomToMapTF()
{
    // Create transform message
    geometry_msgs::msg::TransformStamped transformStamped;
    
    // Set timestamp
    transformStamped.header.stamp = this->now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "odom";
    
    // Set translation (robot position in the field coordinate system)
    transformStamped.transform.translation.x = data->robotPoseToField.x;
    transformStamped.transform.translation.y = data->robotPoseToField.y;
    transformStamped.transform.translation.z = 0.0;
    
    // Set rotation (robot orientation in the field coordinate system)
    tf2::Quaternion q;
    q.setRPY(0, 0, data->robotPoseToField.theta);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();
    
    // Publish tf transform
    tf_broadcaster_->sendTransform(transformStamped);
}

void Brain::publishFieldDimensions()
{
    auto msg = std_msgs::msg::Float64MultiArray();
    auto &fd = config->fieldDimensions;
    
    // Pack field dimensions into an array
    // Order: length, width, penaltyDist, goalWidth, circleRadius, 
    //       penaltyAreaLength, penaltyAreaWidth, goalAreaLength, goalAreaWidth
    msg.data = {
        fd.length,
        fd.width,
        fd.penaltyDist,
        fd.goalWidth,
        fd.circleRadius,
        fd.penaltyAreaLength,
        fd.penaltyAreaWidth,
        fd.goalAreaLength,
        fd.goalAreaWidth
    };
    
    pubFieldDimensions->publish(msg);
    RCLCPP_INFO(get_logger(), "Published field dimensions: length=%.2f, width=%.2f", fd.length, fd.width);
}

void Brain::publishRobotPose()
{
    auto msg = geometry_msgs::msg::Pose2D();
    msg.x = data->robotPoseToField.x;
    msg.y = data->robotPoseToField.y;
    msg.theta = data->robotPoseToField.theta;
    
    pubRobotPose->publish(msg);
}

void Brain::publishBallPosition()
{
    // 只在“本帧检测到球且位置新鲜”时上报。失球后继续发最后位置会让 UI /
    // 裁判机长期显示幽灵球；裁判机又按“位姿包最新”在多机上报之间选球，
    // 过期位置还会让球点在不同机器人之间跳动。
    if (!data->ballDetected || msecsSince(data->ball.timePoint) > kBallFreshMs)
    {
        return;
    }

    auto msg = geometry_msgs::msg::Point();
    msg.x = data->ball.posToField.x;
    msg.y = data->ball.posToField.y;
    msg.z = 0.0;
    
    pubBallPosition->publish(msg);
}

void Brain::publishTeammatesPoses()
{
    auto msg = std_msgs::msg::Float64MultiArray();
    
    // Data format: every 3 numbers represent a group (x, y, theta)
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (data->tmStatus[i].isAlive && i != (config->get_player_id() - 1)) {
            msg.data.push_back(data->tmStatus[i].robotPoseToField.x);
            msg.data.push_back(data->tmStatus[i].robotPoseToField.y);
            msg.data.push_back(data->tmStatus[i].robotPoseToField.theta);
        }
    }
    
    pubTeammatesPoses->publish(msg);
}

void Brain::publishBrainStrategy()
{
    // 组装 JSON（不含 "t"；body 相同即视为无变更）。字符串仅需转义引号与反斜杠。
    static const auto esc = [](const string &s) {
        string out;
        out.reserve(s.size() + 8);
        for (char c : s) {
            if (c == '"' || c == '\\') out.push_back('\\');
            out.push_back(c);
        }
        return out;
    };
    static const auto jstr = [](const string &s) { return "\"" + esc(s) + "\""; };
    static const auto jbool = [](bool b) { return string(b ? "1" : "0"); };

    string body = "{\"robot\":" + jstr(config->get_robot_name());
    body += ",\"role\":" + jstr(tree->getEntry<string>("player_role"));
    body += ",\"ctl\":" + to_string(tree->getEntry<int>("control_state"));
    body += ",\"gc\":" + jstr(tree->getEntry<string>("gc_game_state"));
    body += ",\"sub\":" + jstr(tree->getEntry<string>("gc_game_sub_state"));
    body += ",\"subst\":" + jstr(tree->getEntry<string>("gc_game_sub_state_type"));
    body += ",\"dec\":" + jstr(tree->getEntry<string>("decision"));
    body += ",\"pen\":" + jbool(tree->getEntry<bool>("gc_is_under_penalty"));
    body += ",\"wfk\":" + jbool(tree->getEntry<bool>("wait_for_opponent_kickoff"));
    body += ",\"out\":" + jbool(tree->getEntry<bool>("ball_out"));

    // 主链（从根到最深执行节点的 XML name 序列）与本轮执行过的全部具名/动作节点
    const auto &path = tree->getStrategyPath();
    body += ",\"path\":[";
    for (size_t i = 0; i < path.size(); i++) {
        if (i > 0) body += ",";
        body += jstr(path[i]);
    }
    body += "]";
    const auto &active = tree->getStrategyActive();
    body += ",\"active\":[";
    for (size_t i = 0; i < active.size(); i++) {
        if (i > 0) body += ",";
        body += jstr(active[i]);
    }
    body += "]}";

    auto now = get_clock()->now();
    if (lastStrategyPubTime.get_clock_type() != now.get_clock_type()) {
        lastStrategyPubTime = now;  // 首次：对齐时钟源并立即发布
    }
    bool changed = (body != lastStrategyBody);
    bool minGap = msecsSince(lastStrategyPubTime) > 50;   // 限速 20 Hz
    bool heartbeat = msecsSince(lastStrategyPubTime) > 2000;
    if ((changed && minGap) || heartbeat) {
        lastStrategyBody = body;
        lastStrategyPubTime = now;
        auto msg = std_msgs::msg::String();
        // 前端用于新鲜度判定；body 前插入墙钟时间戳（秒，浮点）
        msg.data = "{\"t\":" + to_string(now.seconds()) + "," + body.substr(1);
        pubBrainStrategy->publish(msg);
    }
}