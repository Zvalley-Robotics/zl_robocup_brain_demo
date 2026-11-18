#pragma once

#include <string>
#include <ostream>
#include <rclcpp/rclcpp.hpp>

#include "types.h"
#include "utils/math.h"
#include "RoboCupGameControlData.h"

const std::string GAME_AGENT_MODE_PARAM = "game.agent_mode";

class Brain; // forward declaration

using namespace std;

/**
 * Stores configuration values required by `Brain`. These values should be
 * determined at initialization and treated as read-only during decision-making.
 * Values that change during runtime should be placed in `BrainData`.
 * Notes:
 * 1) Configuration is read from `config/config.yaml` (全机一致的基线).
 * 2) `config/config_local.yaml` 与 `~/agents/soccer/brain.yaml` 是可选覆盖层，
 *    存在才加载（launch.py 逐层判断），按顺序逐键覆盖前一层。
 * 3) launch 内联参数优先级最高（`real_robot:=true` 等分支会重写部分键），
 *    生效值用 `ros2 param get /brain_node <参数名>` 或 `ros2 param dump /brain_node`
 *    核对，勿凭 yaml 推断。
 */
class BrainConfig
{
public:
    // Pointer to ROS2 node used to access parameters
    BrainConfig(Brain *argBrain);

    int get_player_id();
    int get_team_id();
    string get_player_role();
    string get_field_type();   
    int get_num_of_players();
    bool get_treat_person_as_robot();
    string get_game_control_ip();

    string get_robot_name();
    double get_robot_height();        
    double get_robot_odom_factor();
    double get_vx_factor();
    double get_yaw_offset();
    double get_vx_limit();
    double get_vy_limit();
    double get_vx_limit_backward();// 后退（vx<0）单独上限；<=0 或 >= vx_limit 时等于关闭
    double get_vtheta_limit();
    // 减速（刹车）斜率：|指令| 变小或换向时生效；<=0 时回退到对应的加速度值。
    double get_decel_limit_vx();
    double get_decel_limit_vy();
    double get_decel_limit_vtheta();
    // 距离-速度刹车模型参数（Brain::brakeSpeed）
    double get_brake_decel();
    double get_brake_latency();
    double get_head_yaw_limit_left();
    double get_head_yaw_limit_right();
    double get_head_pitch_limit_up();
    double get_head_pitch_limit_down();
    double get_min_vx();
    double get_min_vy();
    double get_min_vtheta();
    double get_min_lin_speed();     // 线速度合模抬升门限（本体对 <0.1 m/s 不响应）
    double get_speed_scale();       // 下发值→本体的统一缩放（1.0=新真实速度，0.5=回退旧手感）
    double get_yaw_track_gain();    // 球方位→角速度的跟踪增益（原硬编码 *4.0）
    double get_yaw_track_deadband();// 球方位跟踪死区（rad），内直接归零
    double get_yaw_track_damping(); // 转向阻尼增益（D 项）；<=0 时退化为纯 P
    double get_yaw_rate_lpf_hz();   // 实测角速度（odom yaw 差分）低通截止频率
    double get_accel_limit_vx();    // 下发前 vx 斜率限幅（m/s²），<=0 关闭
    double get_accel_limit_vy();    // 下发前 vy 斜率限幅（m/s²），<=0 关闭
    double get_accel_limit_vtheta();// 下发前 vtheta 斜率限幅（rad/s²），<=0 关闭
    double get_track_log_hz();     // dbg/track 打印频率（Hz），<=0 关闭
    int get_cmd_log_interval_ms(); // setVelocity 日志节流间隔(ms)，<=0 关闭

    double get_ball_confidence_threshold();
    double get_ball_memory_timeout();
    double get_tm_ball_dist_threshold();
    bool get_limit_near_ball_speed();
    double get_near_ball_speed_limit();
    double get_near_ball_range();
    double get_ball_out_threshold();
    bool get_abort_kick_when_ball_moved();
    bool get_enable_role_switch();
    double get_ball_control_cost_threshold();

    bool get_enable_auto_visual_kick();
    double get_auto_visual_kick_enable_dist_min();
    double get_auto_visual_kick_enable_dist_max();
    double get_auto_visual_kick_enable_angle();

    int get_min_marker_count();
    double get_max_residual();

    bool get_gt_pose_enabled();
    bool get_gt_ball_enabled();
    double get_gt_ball_timeout_sec();


    bool get_enable_com();
    bool get_multi_instance();

    string get_tree_file_path();

    int get_depth_sample_step();
    double get_obstacle_min_height();
    double get_grid_size();
    double get_max_x();
    double get_max_y();
    double get_exclusion_x();
    double get_exclusion_y();
    double get_ball_exclusion_radius();
    double get_ball_exclusion_height();
    double get_occupancy_threshold();
    bool get_enable_obstacle_avoidance();
    double get_freekick_start_placing_safe_distance();
    double get_freekick_start_placing_avoid_secs();
    double get_obstacle_memory_msecs();
    bool get_avoid_during_kick();
    double get_kick_ao_safe_dist();
    bool get_avoid_during_chase();
    double get_chase_ao_safe_dist();
    // 绕行角上限（度）：栅格只覆盖相机 FOV，超出部分为盲区，必须限幅
    double get_avoid_max_offset_deg();
    // 贴身区（m）：障碍近于此距离时不再尝试小角度绕行，直接降速/停车。
    // 原因：走廊模型下小偏转会把障碍推出 collision_threshold，distToObstacle 返回 1e9，
    // 回查会把近距障碍误判为“已绕开”。
    double get_avoid_min_clearance_m();
    // 绕行分支基准线速度（m/s）：Chase/Assist 两处原为硬编码 0.5。
    // 注意它不是限速阀——绕行时实际速度就是它（微偏转分支为 0.7×），与 vx_limit 无关。
    double get_avoid_speed();
    // ---- 速度包络（2026-11-17）：按“最紧清距”给出本拍允许的速度上限，只降不升 ----
    bool get_speed_envelope_enable();
    double get_speed_envelope_decel();      // m/s²：刹车模型的本体可达减速度
    double get_speed_envelope_latency();    // s：感知+决策+执行总滞后
    double get_speed_envelope_floor();      // m/s：下限，避免贴身清距把速度压成爬行
    double get_speed_envelope_ceiling();    // m/s：上限，<=0 时取 robot.vx_limit
    double get_speed_envelope_margin();     // m：障碍清距里扣掉的余量
    double get_speed_envelope_tm_margin();  // m：队友清距里扣掉的余量
    double get_speed_envelope_field_margin(); // m：边线清距里扣掉的余量
    double get_collision_threshold();
    double get_safe_distance();
    double get_avoid_secs();
    // 基座原点到地面（odom 原点）的 +z 偏置，与 zl 侧 head_pose 的 base_height 必须一致。
    // 深度点云转到 base 系时用它做 z 平移；obstacle_min_height 也是离地高度，两者相减得到 base 系阈值。
    double get_base_height_m();
    // 每帧发布深度点云 + 障碍栅格可视化（重操作，仅排查时开）
    bool get_publish_pointcloud();
    // 深度图订阅 QoS：reliable | best_effort（须与相机发布端一致，否则静默收不到）
    string get_depth_qos();


    int get_retry_max_count();

    string get_image_camera_info_topic();
    string get_depth_image_topic();
    string get_depth_camera_info_topic();

    FieldDimensions fieldDimensions; 
    vector<FieldLine> mapLines;       
    vector<MapMarking> mapMarkings;   


    int cameraImageWidth = 1280;
    int cameraImageHeight = 720;


    double depthCameraFovX = deg2rad(90);
    double depthCameraFovY = deg2rad(65);
    // 内参初值来自 vision.yaml 的 camera.intrin（实机 640x480 标定值），
    // 运行期收到 depth camera_info 后以实测值覆盖。0 表示尚无有效内参，深度回调会跳过。
    double depthCameraFx = 0;
    double depthCameraFy = 0;
    double depthCameraCx = 0;
    double depthCameraCy = 0;
    // true = 内参已被 camera_info 覆盖；false = 仍用 vision.yaml 初值（日志据此区分来源）
    bool depthIntrinsicsFromCameraInfo = false;

    // 相机话题以 vision.yaml 为准（实机单一来源）；为空表示 vision.yaml 未提供，回退 vision.* 参数。
    string depthImageTopicOverride;
    string depthCameraInfoTopicOverride;
    string imageCameraInfoTopicOverride;

    // 已废弃：/robot/head_to_base 直接提供 eye(optical)->base，不再需要 camera.extrin。
    // 保留成员仅为避免连带改动，不再参与 camToRobot 计算。
    Eigen::Matrix4d camToHead = Eigen::Matrix4d::Identity();

    void calcMapLines();
    void calcMapMarkings();

    void handle();
    
    void print(ostream &os);

private:
    Brain* brain;
};