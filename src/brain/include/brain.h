#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_interface/msg/detections.hpp>
#include <vision_interface/msg/line_segments.hpp>
#include <vision_interface/msg/cal_param.hpp>
#include <vision_interface/msg/segmentation_result.hpp>
#include <game_controller_interface/msg/game_control_data.hpp>
#include "brain/msg/kick.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// 必须先于 RoboCupGameControlData.h 引入：该头文件定义了 STATE_READY 等宏，
// 会与 robot_interfaces/msg/RobotState.msg 的 STATE_* 常量同名而展开成非法代码。
// robot_client.h -> robot_io/zl_io.h 会引入 zl 消息头，故提前到这里。
#include "robot_client.h"

#include "RoboCupGameControlData.h"
#include "team_communication_msg.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>

#include "brain_config.h"
#include "ball_estimator.h"
#include "brain_data.h"
#include "brain_log.h"
#include "brain_tree.h"
#include "brain_communication.h"
#include "locator.h"
#include "visualization_publisher.h"

using namespace std;

/**
 * Core `Brain` class. As shared pointers are passed to `BrainTree`, this class
 * must support shared_from_this semantics.
 * Data is encapsulated in sub-objects rather than stored directly in `Brain`.
 * Static configuration values live in `config`; dynamic runtime data lives in `data`.
 * TODO: Some data is also stored in the BehaviorTree blackboard; consider removing duplicates.
 * Current design passes the `brain` pointer into BehaviorTree nodes which directly access `brain->config` and `brain->data`.
 */
class Brain : public rclcpp::Node
{
public:
    // 球位“新鲜”阈值（ms）：本帧无检测、或距最后一次检测超过该时长，位置即视为过期。
    // publishBallPosition / CamTrackBall / Kick 等消费端统一用它判断，避免各节点
    // 各自依赖 ball_location_known（2 s 记忆）或干脆不判，导致过期球位继续驱动动作与上报。
    static constexpr double kBallFreshMs = 300.0;

    // BrainConfig object, mainly contains configuration values needed at runtime (static)
    std::shared_ptr<BrainConfig> config;
    // BrainLog object, encapsulates log-related operations
    std::shared_ptr<BrainLog> log;
    // BrainData object, all runtime values of Brain are stored here
    std::shared_ptr<BrainData> data;
    // RobotClient object, contains all operations related to the robot
    std::shared_ptr<RobotClient> client;
    // Locator object, contains robot localization related operations
    std::shared_ptr<Locator> locator;
    // VisualizationPublisher object, used to publish visualization markers
    std::shared_ptr<VisualizationPublisher> visualizer;
    // BrainTree object, contains BehaviorTree related operations
    std::shared_ptr<BrainTree> tree;
    // Communication object, contains communication related operations, mainly for dual-machine communication and referee communication
    std::shared_ptr<BrainCommunication> communication;

    // Constructor: create ROS2 node (nodeName handled in implementation)
    Brain();

    ~Brain();

    // Initialize operations, only need to be called once in the main function. If the initialization process does not meet expectations, an exception can be thrown to terminate the program.
    void init();

    // Called in the ROS2 loop
    void tick();

    // Handle special states, such as kickoff, free kick, etc.
    void handleSpecialStates();

    void registerLocatorNodes(BT::BehaviorTreeFactory &factory)
    {
        RegisterLocatorNodes(factory, this);
    }

    /**
     * @brief Calculate the vector angles from the current ball position to the opponent's goalposts, in the field coordinate system
     *
     * @param  margin double, when calculating the angles, the returned values are moved inward from the actual goal positions by this distance, because the ball will be blocked by the goalposts at this angle. The larger this value, the higher the shooting accuracy, but the longer it takes to adjust the angle.
     *
     * @return vector<double> The returned vector contains vec[0] as the left goalpost and vec[1] as the right goalpost. Note that left and right are based on the opponent's direction as the front.
     */
    vector<double> getGoalPostAngles(const double margin = 0.3);

    double calcKickDir(double goalPostMargin = 0.3);

    // type: kick: dribble, shoot: rl kick
    bool isAngleGood(double goalPostMargin = 0.3, string type = "kick");

   
    bool isBallOut(double locCompareDist = 3.0, double lineCompareDist = 1.0);

    
    void updateBallOut();

    
    double distToBorder();

    bool isDefensing();

    bool isBoundingBoxInCenter(BoundingBox boundingBox, double xRatio = 0.8, double yRatio = 0.8);

    void calibrateOdom(double x, double y, double theta);

    double msecsSince(rclcpp::Time time);

    // 丢球扫场（2026-11-17）：CamFindBall 与 CamFastScan 共用同一份点位/节拍/相位，
    // 由 findSweepTarget 按「距本轮开始的时间」给出当前目标；再经单一出口去重。
    static constexpr int kFindSweepCount = 6;

    /** 扫场每点驻留时长（ms，参数 robot.head_sweep_interval_ms，内部夹到 ≥500）。 */
    double findSweepIntervalMs();

    /** 当前扫场目标（相机光轴语义，弧度）。每拍可重复调用，同拍内结果一致。 */
    void findSweepTarget(double &pitch, double &yaw);

    bool isFreekickStartPlacing();

    rclcpp::Time timePointFromHeader(std_msgs::msg::Header header);


    void updateCostToKick();

    /**
     * @brief Publish visualization markers (robot position, ball position, field lines, mark points)
     */
    void publishVisualizationMarkers();

    /**
     * @brief Publish odom to map TF transform
     */
    void publishOdomToMapTF();

    /**
     * @brief Publish field dimensions as static messages
     */
    void publishFieldDimensions();

    /**
     * @brief Publish robot position (field coordinate system)
     */
    void publishRobotPose();

    /**
     * @brief Publish ball position (field coordinate system)
     */
    void publishBallPosition();

    /**
     * @brief Publish teammates' positions (field coordinate system)
     */
    void publishTeammatesPoses();

    void pubKickMsg();

    /**
     * @brief Publish the behavior-tree strategy chain executed this tick, as a JSON string.
     *        Used by sim_console to show which strategy branch is active (UI telemetry only).
     */
    void publishBrainStrategy();

    // ------------------------------------------------------ SUB CALLBACKS ------------------------------------------------------

    void gameControlCallback(const game_controller_interface::msg::GameControlData &msg);

    void detectionsCallback(const vision_interface::msg::Detections &msg);

    void fieldLineCallback(const vision_interface::msg::LineSegments &msg);

    void imageCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    void depthCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    void depthImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    void compressedDepthImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

    void processDepthImage(const cv::Mat &depthFloat, int width, int height, const std_msgs::msg::Header &header);

    // /robot/head_to_base 自 2026-11-11 起为 geometry_msgs/PoseStamped（zl 侧新增 header.stamp，frame_id 留空）。
    void headPoseCallback(const geometry_msgs::msg::PoseStamped &msg);

    /**
     * Ground truth bypass callbacks (mocap/simulation localization injection)
     */
    void gtPoseCallback(const geometry_msgs::msg::Pose2D &msg);

    void gtBallCallback(const geometry_msgs::msg::Point &msg);

    void updateRelativePos(GameObject &obj);

    void updateFieldPos(GameObject &obj);

    /**
     * @brief Calculate the collision distance
     * 
     * @param angle double, target angle
     * 
     * @return double, collision distance
     */
    double distToObstacle(double angle);

    /**
     * @brief 距离-速度刹车模型：在剩余距离 d 内能刹停的速度上限（含感知/执行滞后）。
     *
     *   v = −a·τ + √((a·τ)² + 2a·d)
     * 推导：制动距离 v²/(2a) 加上滞后期间走过的 v·τ 不超过 d。
     * a 默认取 robot.brake_decel，τ 默认取 robot.brake_latency；显式传入可用于包络的
     * 独立参数（obstacle_avoidance.speed_envelope_decel/latency）。
     *
     * @param distance 到目标的净距离（m，<=0 返回 0）
     * @param decel 减速度（m/s²，<=0 时取 robot.brake_decel；两者都<=0 则不约束，返回 1e9）
     * @param latency 总滞后（s，<0 时取 robot.brake_latency）
     * @return double 允许速度上限（m/s），调用方自行与节点限速取 min
     */
    double brakeSpeed(double distance, double decel = -1.0, double latency = -1.0);

    /**
     * @brief 速度包络：按“最紧清距”给出本拍允许的平面速度上限（只降不升）。
     *
     * v_allow = gain × (clearance − margin)，再夹到 [floor, ceiling]；clearance 取三者最小：
     *   1) 沿 headingRad（机体系）的障碍走廊清距（distToObstacle，栅格外恒 1e9 不约束）；
     *   2) 最近存活队友的直线距离；
     *   3) 沿行进方向（场系，由 robotPoseToField.theta + headingRad 换算）到场地边线的距离。
     *
     * 用途：“只在空档大时才快”：远处空档可到 ceiling，接近障碍/队友/边线自动落回 floor。
     * 必须与调用方自身速度取 min 使用；参数来自 obstacle_avoidance.speed_envelope_*，
     * enable=false 或 ceiling<=0 时直接返回 robot.vx_limit（等价关闭）。
     *
     * @param headingRad 行进方向（机体系，rad）
     * @param tag 日志标签（"chase"/"assist"），仅用于 1 Hz 观测
     * @return double 允许的速度上限（m/s）
     */
    double speedEnvelopeCap(double headingRad, const char *tag = nullptr);

    vector<double> findSafeDirections(double startAngle, double safeDist, double step=deg2rad(10));

    double calcAvoidDir(double startAngle, double safeDist);

    /**
     * @brief calcAvoidDir 的“盲区安全”版本：把偏转夹到 ±maxOffsetRad 内。
     *
     * calcAvoidDir 依赖 distToObstacle 判“安全方向”，而栅格只覆盖相机 FOV
     * （水平半角约 40°）以内；FOV 外的方向一律返回 1e9，会被误判为安全。
     * 直接采用会得到指向盲区的绕行角。
     *
     * @param startAngle double, 期望朝向（robot 系）
     * @param safeDist double, 安全距离（必须小于栅格量程 max_x）
     * @param maxOffsetRad double, 允许的最大偏转（弧度）
     * @param outDir double&, 限幅后的绕行角
     * @return bool, false = 限幅后该方向仍近距受阻，调用方应减速直行而非转向
     */
    bool calcSafeAvoidDir(double startAngle, double safeDist, double maxOffsetRad, double &outDir);

    /**
     * @brief 队友贴身让位速度（侧向切出版，2026-11-12 改）。
     *
     * 最近的存活队友距离小于 rSep 时，输出沿「队友连线」垂直线切出的速度（robot 坐标系），
     * 不再沿背离方向直退：实测两机对撞时双方都按背离方向退，退让方向正好被对方顶住，
     * 机身重叠 1.2 s（间距 0.03 m）也撤不开。切出的一侧选「切出后离球更远」的那条，
     * 既离开队友又不把球让出去；贴边时改选指向场内的一侧，两侧都贴边才退回背离方向。
     * 调用方应直接下发该速度并跳过本周期到位判定。
     *
     * @param vx vy vtheta double&, 退让速度输出（robot 坐标系；vtheta 保持面向球）
     * @param rSep double, 触发距离（默认 0.7；追球/绕球链路传 0.9~1.2 做提前量）
     * @param speed double, 切出线速度（默认 0.3；避碰用 0.6~0.8，贴身解纠缠用 0.3）
     * @param headingRad double, 我可选：我本拍想走的方向（场地系）。传了就只对「挡在我行进
     *        方向 ±70° 内」的队友让位；不传（1e9）退化为只看距离。带方向判据可以避免
     *        并排走位/站位时无谓地反复让开又回去（大于 0.5 m 才生效，贴身时一律让）。
     *
     * @return bool, true=输出有效退让速度
     */
    bool calcTeammateSeparationVelocity(double &vx, double &vy, double &vtheta, double rSep = 0.7, double speed = 0.3,
                                        double headingRad = 1e9);

    /**
     * @brief 追球/绕球阶段（Chase、Adjust）的单侧队友让位。
     *
     * 与 calcTeammateSeparationVelocity 的区别：本函数先仲裁决定「该不该由我退」，只有
     * 明显更差的（或同档且 id 更大的）一台才输出切出速度，因此同一时刻只有一台在退。
     * 仲裁分两层（2026-11-12）：
     *   1) 距离 < HARD_YIELD_DIST(0.5 m)：不看 cost，直接按 player_id 大者退。实测两台
     *      前锋会各自判「对方更差 → 我不退」（cost 一个是本机实时值、一个是对方 100 ms
     *      前广播 + 中值滤波），结论不互补，于是双双不让、直接顶在一起；硬门限保证
     *      贴近时必然有一方在退。
     *   2) 距离 ≥ 0.5 m：仍用 (cost, player_id) 仲裁，避免正常追球时无谓退让。
     * 球已进入踢球距离（minBallRange）时只在硬门限内干预，避免把正在起脚的一台拽离球。
     * 最近队友是门将时无条件让位（门将拥有优先权）。
     *
     * @param vx vy vtheta double&, 让位速度输出（robot 坐标系）
     * @param minBallRange double, 球距小于该值时不触发让位（Chase 用 0.8，Adjust 用 0.55）
     * @param rSep double, 触发距离（提前量，Chase 用 1.2，Adjust 用 0.9）
     * @param speed double, 切出线速度（Chase 用 0.8，Adjust 用 0.6）
     * @param headingRad double, 我本拍想走的方向（场地系），透传给
     *        calcTeammateSeparationVelocity 做方向判据；不传则只看距离
     * @return bool, true=输出有效退让速度
     */
    bool calcChaseYieldVelocity(double &vx, double &vy, double &vtheta, double minBallRange = 0.8,
                                double rSep = 1.2, double speed = 0.8, double headingRad = 1e9);

    /**
     * @brief 贴身脱离兜底（2026-11-16）。行为树 tick 之后调用，压过本拍任何节点的速度。
     * 与队友距离 < 0.40 m 持续 0.6 s 时，由 player_id 大者沿「背离队友 ∪ 远离球」的角平分线后退 1 s。
     * 只在 control_state==3、非 go_manual、gc_game_state==PLAY 时接管。
     * @return bool, true=已下发脱离速度
     */
    bool applySeparationBackoff();

    /**
     * @brief 起脚前队友门：最近存活队友是否挡在出球方向（kickDir）的扇形内。
     *
     * 实测（2026-11-12，39/45 双前锋）：一台 adjust 超时兜底强行起脚时，另一台距其 0.41 m，
     * Kick / RLVisionKick 全程没有任何队友判据，球直接从队友身上过。判据取「我→队友」方位
     * 与 kickDir 的夹角：夹角小说明队友站在我出球的那条线上。
     * 调用方应据此把 decision 从 kick 降级为 adjust（由 Adjust 先侧移让开）。
     *
     * @param clearDist double, 队友距离小于该值算挡路（默认 0.8）
     * @param coneRad double, 出球方向两侧的判定半角（默认 1.05 rad ≈ 60°）
     * @param outIdx int*, 可选输出：最近队友下标
     * @param outDist double*, 可选输出：最近队友距离
     * @return bool, true = 有队友挡在出球方向上
     */
    bool isTeammateBlockingKick(double clearDist = 0.8, double coneRad = 1.05, int *outIdx = nullptr, double *outDist = nullptr);


    /**
     * agent related
     */
    void agentCommandCallback(const std_msgs::msg::String::SharedPtr msg);

private:
    void loadConfig();

    void updateBallMemory();

    // 球速估计 + 黑盒量输出（ball_vel_x/ball_vel_y/ball_speed/ball_age_ms/ball_est_valid）
    void updateBallEstimate();

    void updateRobotMemory();

    void updateObstacleMemory();

    void updateKickoffMemory();

    void updateMemory();

    bool gtBallFresh();

    void handleCooperation();

    // ------------------------------------------------------ Vision Processing ------------------------------------------------------

    int markCntOnFieldLine(const string MarkType, const FieldLine line, const double margin = 0.2);

    int goalpostCntOnFieldLine(const FieldLine line, const double margin = 0.2);

    bool isBallOnFieldLine(const FieldLine line, const double margin = 0.3); 

    void identifyFieldLine(FieldLine &line);

    void identifyMarking(GameObject& marking);

    void identifyGoalpost(GameObject& goalpost);

    void updateLinePosToField(FieldLine &line);

    vector<FieldLine> processFieldLines(vector<FieldLine> &fieldLines);

    vector<GameObject> getGameObjects(const vision_interface::msg::Detections &msg);
    void detectProcessBalls(const vector<GameObject> &ballObjs);

    void detectProcessMarkings(const vector<GameObject> &markingObjs);

    void detectProcessRobots(const vector<GameObject> &robotObjs);

    void detectProcessGoalposts(const vector<GameObject> &goalpostObjs);

    void detectProcessVisionBox(const vision_interface::msg::Detections &msg);
    
    void logDepth(int grid_x_count, int grid_y_count, vector<vector<int>> &grid, vector<std::array<float, 3>> &points);

    void logDebugInfo();


    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joySubscription;
    rclcpp::Subscription<game_controller_interface::msg::GameControlData>::SharedPtr gameControlSubscription;
    rclcpp::Subscription<vision_interface::msg::Detections>::SharedPtr detectionsSubscription;
    rclcpp::Subscription<vision_interface::msg::LineSegments>::SharedPtr subFieldLine;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr imageCameraInfoSubscription;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depthCameraInfoSubscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthImageSubscription;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressedDepthImageSubscription;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr headPoseSubscription;
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr gtPoseSubscription;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr gtBallSubscription;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubVisualizationMarkers;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pubFieldDimensions;
    rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pubRobotPose;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pubBallPosition;
    rclcpp::Publisher<brain::msg::Kick>::SharedPtr pubKickBall;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pubTeammatesPoses;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pubBrainStrategy;

    // brain_strategy telemetry: last published body (change detection) and throttle/heartbeat clock
    string lastStrategyBody;
    rclcpp::Time lastStrategyPubTime;

    // ------------------------------------------------------ debug logs related ------------------------------------------------------
    void logLags();
    // 1 Hz 障碍/深度链观测：obstacles 数量、最近距离、前向距离、深度断流时长
    void logObstacleStatus();
    void logStatusToConsole();
    string getComLogString(); 
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ------------------------------------------------------ ground truth localization bypass ------------------------------------------------------
    // timestamp of the last received ground truth ball message (zero = never received)
    rclcpp::Time lastGtBallTime;
    // when the ground truth ball was declared lost after a timeout (zero = not lost)
    rclcpp::Time gtBallLostTime;
    // whether the current ball state originates from ground truth
    bool gtBallActive = false;

    // ------------------------------------------------------ cost smoothing & lead hysteresis ------------------------------------------------------
    // cost 中值滤波：抑制丢球/摔倒等台阶项在单拍造成的突变，避免 lead 反复易手
    static constexpr int COST_HISTORY_LEN = 5;
    double costHistory[COST_HISTORY_LEN] = {0.};
    int costHistoryIdx = 0;
    int costHistoryFilled = 0;
    // 摔倒惩罚斜坡计时（0→15 在 1.5 s 内爬升，替代原 0/15 台阶）
    bool fallTimerActive = false;
    rclcpp::Time fallStartTime;
    // lead 切换迟滞：候选结论须连续保持 LEAD_SWITCH_HOLD_MS 才真正换手
    bool leadSwitchPending = false;
    bool leadSwitchTarget = true;
    rclcpp::Time leadSwitchStartTime;

    // ------------------------------------------------------ teammate ball sharing ------------------------------------------------------
    // 球位置来源（本地检测 / 队友共享）与交接保护窗：来源切换后短时间内禁止触发踢球，
    // 避免两套场系间 0.7~1.4 m 的位置跳变直接变成一脚乱踢。
    bool ballPosFromTeammate = false;
    rclcpp::Time ballPosHandoffTime;
    // 贴身 hint（队友球位离我 <2 m 或队友自己也在用记忆）：300 ms 内有效，仅用于转头
    rclcpp::Time lastTmBallHintTime;

    // ------------------------------------------------------ agent related ------------------------------------------------------

    // ----- 丢球扫场相位（见 findSweepTarget）-----
    bool findSweepActive = false;
    double findSweepStartMs = 0.0;   // 本轮扫描起点（ms，墙钟）
    double findSweepLastReqMs = 0.0; // 最近一次取目标时刻（判“中间被别人占头”用）

    // In agent mode, subscribe to parameter changes and handle callbacks
    std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> team_id_handle_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> player_role_handle_;
};
