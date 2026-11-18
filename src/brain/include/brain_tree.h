#pragma once

#include <tuple>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <behaviortree_cpp/behavior_tree.h>
#include <behaviortree_cpp/bt_factory.h>
#include <algorithm>

#include "types.h"
#include "split_localization_schedule.h"

class Brain;

using namespace std;
using namespace BT;

class BrainTree
{
public:
    BrainTree(Brain *argBrain) : brain(argBrain) {}

    void init();

    void tick();

    // 所有定位行为树节点共享一个顶层 tick 的调度状态。
    SplitLocalizationSchedule localizationSchedule;
    int64_t localizationTickNs = 0;
    // 计数只在行为树 tick 线程更新，按方法记录跳过、尝试和成功。
    std::array<uint64_t, 6> locGeomSkipped{}, locGeomAttempts{}, locGeomSuccess{};
    uint64_t locTrustSkipped = 0, locTrustAttempts = 0, locTrustSuccess = 0;
    int64_t lastLocalizationDiagNs = -1;
    int64_t lastLocalizationTraceNs = -1;
    std::string localizationLastMode = "NONE";
    bool localizationSplitWasEnabled = false;

    /// 本次 tick 实际执行的行为树主链（从根到最深执行节点的 XML name 序列；UI 策略回读用）
    const vector<string> &getStrategyPath() const { return _strategyPath; }

    /// 本次 tick 实际执行过的全部具名/动作节点（树序、去重；UI「本轮执行」用）
    const vector<string> &getStrategyActive() const { return _strategyActive; }

    // get entry on blackboard
    template <typename T>
    inline T getEntry(const string &key)
    {
        T value{};
        [[maybe_unused]] auto res = tree.rootBlackboard()->get<T>(key, value);
        return value;
    }

    // set entry on blackboard
    template <typename T>
    inline void setEntry(const string &key, const T &value)
    {
        tree.rootBlackboard()->set<T>(key, value);
    }

private:
    Tree tree;
    Brain *brain;

    // ---- 策略链快照（brain_strategy 遥测）----
    // tick 监视器记录每个节点本次 tick 的返回状态；被 _while 等 precondition
    // 替换的 tick 不触发回调（BT.CPP 语义），因此该表恰好等于「真正执行过」的集合。
    unordered_map<const TreeNode *, NodeStatus> _lastStatus;
    vector<string> _strategyPath;
    vector<string> _strategyActive;

    void attachTickMonitors();

    void buildStrategyPath();

    /**
     * Initialize entries in the blackboard. When adding new fields, set a default value here.
     */
    void initEntry();
};


class CalcKickDir : public SyncActionNode 
{
public:
    CalcKickDir(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("cross_threshold", 0.2, "If the angle range for scoring is less than this value, then pass")
        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};


class StrikerDecide : public SyncActionNode
{
public:
    StrikerDecide(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("chase_threshold", 1.0, "If the distance exceeds this value, execute the chase action"),
            // 2026-11-06：以下四项原先硬编码在 StrikerDecide::tick 内，现场无法调整。
            // 参数化后默认值保持旧口径（0.22 / 0.26 / 0 / 3000），不传则行为不变。
            InputPort<double>("kick_dir_align", 0.22, "|kickDir - robot->ball bearing| threshold for handoff (rad)"),
            InputPort<double>("kick_body_align", 0.26, "|kickDir - body heading| threshold for handoff (rad)"),
            InputPort<double>("kick_range_min", 0.0, "Walking kick is blocked while ball range is above this value, leaving the ball to Adjust for fine alignment; 0 disables the guard"),
            InputPort<double>("kick_force_msec", 3000.0, "Fallback: allow walking kick after staying in adjust for this long"),
            // 2026-11-06：交棒要求球已基本静止。t4_kicking 是“走向静止球再起脚”的定点动作，
            // 追着滚动的球交棒会被 zl 侧 KICK_ESCAPE_M 立刻收尾（实测 <1 s 秒退）。
            // 设很大值（如 99）可关闭该门限；超时用于球被持续推动时放行。
            InputPort<double>("handoff_ball_speed_max", 0.35, "Max ball speed (m/s) allowed at visual-kick handoff; ball must be nearly still"),
            // 2026-11-09：超时放行也要有速度上限（否则“持续滚动的球”会被无条件交棒，模型追不上）。
            InputPort<double>("handoff_ball_speed_relax_max", 0.80, "Relaxed ball-speed cap applied when the still-timeout fires; above this the handoff is still refused"),
            InputPort<double>("handoff_ball_still_timeout_ms", 1000.0, "Fallback: allow handoff after waiting this long for the ball to stop"),
            // 模型自动瞄门（正向 180°）后方向对齐门限失效，改判“球/门都在机前锥内 + 瞄准目标有效”。
            InputPort<double>("visual_kick_ball_front", 1.30, "Max |ball yaw| (rad) allowed at visual-kick handoff"),
            InputPort<double>("visual_kick_goal_front", 1.40, "Max |kickDir - body heading| (rad) allowed at visual-kick handoff"),
            InputPort<bool>("visual_kick_require_goal_target", true, "Require kick_targets (goal pose synthesis) to be publishing"),
            InputPort<bool>("visual_kick_shoot_only", true, "Only hand off to visual kick when kickType == 'shoot'"),
            InputPort<string>("decision_in", "", "Used to read the previous decision"),
            InputPort<string>("position", "offense", "offense | defense, determines the direction to kick the ball"),
            OutputPort<string>("decision_out")};
    }

    NodeStatus tick() override;

private:
    Brain *brain;
    double lastDeltaDir; 
    rclcpp::Time timeLastTick; 
};


class GoalieDecide : public SyncActionNode
{
public:
    GoalieDecide(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("chase_threshold", 1.0, "If the distance exceeds this value, execute the chase action"),
            InputPort<double>("close_threshold", 2.0, "Always leave the blocking position and chase when the ball is closer than this distance, regardless of the defense zone"),
            InputPort<double>("adjust_angle_tolerance", 0.1, "If the angle is less than this value, consider adjust successful"), //
            InputPort<double>("adjust_y_tolerance", 0.1, "If the y-direction offset is less than this value, consider y-direction adjust successful"),  //
            InputPort<string>("decision_in", "", "Used to read the previous decision"),
            InputPort<double>("auto_visual_kick_enable_dist_min", 2.0, "Minimum distance for enabling auto visual kick"),
            InputPort<double>("auto_visual_kick_enable_dist_max", 3.0, "Maximum distance for enabling auto visual kick"),
            InputPort<double>("auto_visual_kick_enable_angle", 0.785, "Angle range for enabling auto visual kick"),
            InputPort<double>("auto_visual_kick_obstacle_dist_threshold", 3.0, "Obstacle distance threshold for auto visual kick, if an obstacle is within this distance, auto visual kick is not executed"),
            InputPort<double>("auto_visual_kick_obstacle_angle_threshold", 1.744, "Obstacle angle threshold for auto visual kick, if an obstacle is within this angle, auto visual kick is not executed"),
            OutputPort<string>("decision_out"),
        };
    }

    BT::NodeStatus tick() override;

private:
    Brain *brain;
};


class CamTrackBall : public SyncActionNode
{
public:
    CamTrackBall(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {};
    }
    NodeStatus tick() override;

private:
    Brain *brain;
};


class CamFindBall : public SyncActionNode
{
public:
    CamFindBall(const string &name, const NodeConfig &config, Brain *_brain);

    NodeStatus tick() override;

private:
    Brain *brain;

};


class RobotFindBall : public StatefulActionNode
{
public:
    RobotFindBall(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("vyaw_limit", 1.0, "Maximum turning speed when searching for the ball"),
        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    void onHalted() override;

private:
    double _turnDir; 
    Brain *brain;
};


class CamFastScan : public StatefulActionNode
{
public:
    CamFastScan(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            // 2026-11-17: 端口已移除——点位/节拍统一到 Brain::findSweepTarget() 与
            // robot.head_sweep_interval_ms。XML 均为 <CamFastScan />，不传端口，删之不碍加载。
        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    void onHalted() override {};

private:
    // 2026-11-17: 本节点只保留“扫一轮（6 点）后交回 FindBall 序列（转身/回位）”的时序；
    // 每个点的目标姿态由 Brain::findSweepTarget() 统一给出，与 CamFindBall 同源同相位。
    rclcpp::Time _startTime;
    Brain *brain;
};

class TurnOnSpot : public StatefulActionNode
{
public:
    TurnOnSpot(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("rad", 0, "How many radians to turn, positive for left"),
            InputPort<bool>("towards_ball", false, "If true, ignore the sign of rad and turn towards the last seen ball direction")
        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    void onHalted() override {};

private:
    double _lastAngle; 
    double _angle;
    double _cumAngle; 
    double _msecLimit = 5000;  
    rclcpp::Time _timeStart;
    Brain *brain;
};


class Chase : public SyncActionNode
{
public:
    Chase(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("vx_limit", 0.6, "Maximum x speed when chasing the ball"),
            InputPort<double>("vy_limit", 0.4, "Maximum y speed when chasing the ball"),
            InputPort<double>("vtheta_limit", 1.0, "Maximum angular speed when chasing the ball"),
            InputPort<double>("dist", 0.1, "Target distance behind the ball when chasing"),
            InputPort<double>("safe_dist", 4.0, "Safe distance to maintain when circling back"),
        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
    string _state;     
    double _dir = 1.0; 
};


class Adjust : public SyncActionNode
{
public:
    Adjust(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("turn_threshold", 3.25, "If the ball angle is greater than this value, the robot will turn to face the ball and pause linear movement"),
            InputPort<double>("vx_limit", 0.05, "Limit for vx during adjustment [-limit, limit]"),
            InputPort<double>("vy_limit", 0.05, "Limit for vy during adjustment [-limit, limit]"),
            InputPort<double>("vtheta_limit", 0.1, "Limit for vtheta during adjustment [-limit, limit]"),
            InputPort<double>("range", 2.25, "Maintain this ball range"),
            // 2026-11-09：绕球防蹭。未对准时把目标半径推远 backoff_margin，让机器人在球外绕行
            // （range=0.38 时腿尖距球仅 ~3 cm，绕一圈必然把球蹭走，实测 adjust 兜底 3→6 次、
            // 行走踢 3→7 次）；并且允许负向径向项（后退）才能真正退出去。
            InputPort<double>("backoff_margin", 0.12, "When not aligned, orbit at range+margin instead of range (anti-brush)"),
            InputPort<double>("backoff_speed", 0.30, "Max radial back-off speed (m/s) when the ball is closer than target; 0 disables"),
            InputPort<double>("vtheta_factor", 3.0, "Multiplier for vtheta when adjusting angle, higher values result in faster turning"),
            InputPort<double>("tangential_speed_far", 0.2, "Tangential speed when far from the target"),
            InputPort<double>("tangential_speed_near", 0.15, "Tangential speed when near the target"),
            InputPort<double>("near_threshold", 0.8, "Use near speed when distance to target is less than this value"),
            InputPort<double>("no_turn_threshold", 0.1, "Do not turn if angle difference is less than this value"),
            InputPort<double>("turn_first_threshold", 0.5, "Turn first without moving if angle difference is greater than this value"),

        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};

// Execute kick action
class Kick : public StatefulActionNode
{
public:
    Kick(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("min_msec_kick", 500, "Minimum duration for the kick action (milliseconds)"),
            InputPort<double>("msecs_stablize", 1000, "Duration to stabilize before kicking (milliseconds)"),
            InputPort<double>("speed_limit", 0.8, "Maximum speed for the kick action"),
        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    // callback to execute if the action was aborted by another node
    void onHalted() override;

private:
    Brain *brain;
    rclcpp::Time _startTime; 
    string _state = "kick"; // stablize | kick
    int _msecKick = 1000;    
    double _speed; 
    double _minRange; 
    tuple<double, double, double> _calcSpeed();
    // 行走击球期间保持机身对准出球方向的 vtheta（球不在正前时先转球）。
    double _alignVtheta(double ballYaw);
};

class RLVisionKick : public StatefulActionNode
{
public:
    RLVisionKick(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("max_msec_kick", 10000, "Maximum duration for the kick action (milliseconds)"),
            InputPort<double>("min_msec_kick", 3000, "Minimum duration for the kick action (milliseconds)"),
            InputPort<double>("range", 1.5, "Range within which the strategy is considered effective"),
            InputPort<double>("auto_visual_kick_enable_dist_min", 0.0, "Minimum distance for enabling auto visual kick"),
            InputPort<double>("auto_visual_kick_enable_dist_max", 4.0, "Maximum distance for enabling auto visual kick"),
            InputPort<double>("auto_visual_kick_enable_angle", 0.785, "Angle range for enabling auto visual kick"),
            InputPort<double>("auto_visual_kick_enable_goal_angle", 0.35, "Angle range for enabling auto visual kick towards the goal"),
            InputPort<double>("auto_visual_kick_obstacle_dist_threshold", 1.0, "Distance threshold for obstacles during auto visual kick, if an obstacle is within this distance, auto visual kick will not be executed"),
            InputPort<double>("auto_visual_kick_obstacle_angle_threshold", 1.744, "Angle threshold for obstacles in front during auto visual kick, if an obstacle is within this angle, auto visual kick will not be executed"),
        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    void onHalted() override;
    
    static rclcpp::Time getLastExitTime() { return _lastExitTime; }
    
    static bool isMinIntervalSatisfied(double minIntervalMsec);

private:
    Brain *brain;
    rclcpp::Time _startTime;              // Start time of the kick action
    rclcpp::Time _headScanStartTime;      // Start time for head scanning
    
    // Non-blocking deceleration state
    bool _isDecelerating = false;         // Whether deceleration is in progress
    bool _visionKickStarted = false;     // Whether visual kick mode has started
    bool _pendingRobocupWalk = false;    // Whether waiting to switch to robocupWalk after deceleration
    rclcpp::Time _decelStartTime;        // Deceleration start time
    double _decelDurationMs = 500.0;     // Deceleration duration (milliseconds)
    
    // Static variable: records the last exit time of RLVisionKick to limit frequent switching
    static rclcpp::Time _lastExitTime;
    
    // Helper methods
    void recordExitTime();
    void startDecelerate(double durationMs = 500.0);
    bool stepDecelerate();
};

class StandStill : public StatefulActionNode
{
public:
    StandStill(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<int>("msecs", 1000, "Duration to stand still (milliseconds)"),
        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    // callback to execute if the action was aborted by another node
    void onHalted() override;

private:
    Brain *brain;
    rclcpp::Time _startTime; 
};


class CamScanField : public SyncActionNode
{
public:
    CamScanField(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("low_pitch", 0.5, "Maximum pitch when looking down"),
            InputPort<double>("high_pitch", 0.22, "Minimum pitch when looking up (new head: >= 0.20, physical limit)"),
            InputPort<double>("left_yaw", 0.8, "Maximum yaw when looking left"),
            InputPort<double>("right_yaw", -0.8, "Minimum yaw when looking right"),
            InputPort<int>("msec_cycle", 4000, "Duration for a full scan cycle (milliseconds)"),
            InputPort<int>("send_interval_msec", 500, "Minimum interval between head commands; lets the robot-body head smoothing settle (0 = send every tick)"),
        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
    double _lastScanSendSec = -1e9;  // 上次下发头部指令的时刻（秒），用于降频节流
};


class MoveToPoseOnField : public SyncActionNode
{
public:
    MoveToPoseOnField(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("x", 0, "Target x coordinate in the Field frame"),
            InputPort<double>("y", 0, "Target y coordinate in the Field frame"),
            InputPort<double>("theta", 0, "Target final orientation in the Field frame"),
            InputPort<double>("long_range_threshold", 1.5, "If the distance to the target point exceeds this value, prioritize moving towards it rather than fine-tuning position and orientation"),
            InputPort<double>("turn_threshold", 0.4, "When the distance is long, if the direction to the target point exceeds this value, turn towards the target point first"),
            InputPort<double>("vx_limit", 0.8, "x speed limit"),
            InputPort<double>("vy_limit", 0.5, "y speed limit"),
            InputPort<double>("vtheta_limit", 0.2, "theta speed limit"),
            InputPort<double>("x_tolerance", 0.5, "x tolerance"),
            InputPort<double>("y_tolerance", 0.5, "y tolerance"),
            InputPort<double>("theta_tolerance", 0.5, "theta tolerance"),
            InputPort<bool>("avoid_obstacle", false, "Whether to avoid obstacles")
        };
    }

    BT::NodeStatus tick() override;

private:
    Brain *brain;
};

class GoToReadyPosition : public SyncActionNode
{
public:
    GoToReadyPosition(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("dist_tolerance", 0.8, "x tolerance"),
            InputPort<double>("theta_tolerance", 0.5, "theta tolerance"),
            InputPort<double>("vx_limit", 0.8, "vx limit"),
            InputPort<double>("vy_limit", 0.5, "vy limit"),
        };
    }

    BT::NodeStatus tick() override;

private:
    Brain *brain;
};


class GoToGoalBlockingPosition : public SyncActionNode
{
public:
    GoToGoalBlockingPosition(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("dist_tolerance", 0.8, "dist tolerance, within which considered arrived."),
            InputPort<double>("theta_tolerance", 0.8, "theta tolerance, winin which considered arrived."),
            InputPort<double>("vx_limit", 0.1, "x speed limit"),
            InputPort<double>("vy_limit", 0.1, "y speed limit"),
            InputPort<double>("dist_to_goalline", 2.5, "Distance from the robot to the goal line"),
        };
    }

    BT::NodeStatus tick() override;

private:
    Brain *brain;
    // 门将站位目标缓存（2026-11-16）：球位超过 strategy.gk_ball_stale_ms 没有新检测时
    // 不再用旧球位重算目标（1.2 s 记忆期会把门将一路拽向旧位置），冻结最后一次目标原地守门。
    // 每个 XML 实例各自一份，避免三个 _while 分支共享状态。
    bool hasLastTarget_ = false;
    double lastTargetX_ = 0.0;
    double lastTargetY_ = 0.0;
    // 上一拍目标时间：目标点速率限幅（抗定位跳变）的 dt 基准
    rclcpp::Time lastTargetTime_{0, 0, RCL_ROS_TIME};
    // 定位跳变门（2026-11-17）：odom→field 跳变会让“机体系目标误差”单拍突变，
    // 直接变成反向速度。这里缓存上一拍的误差/指令，突变时沿用旧指令。
    bool hasLastCmd_ = false;
    double lastErrX_ = 0.0;
    double lastErrY_ = 0.0;
    double lastCmdVx_ = 0.0;
    double lastCmdVy_ = 0.0;
    double lastCmdVtheta_ = 0.0;
    bool jumping_ = false;
    rclcpp::Time jumpStart_{0, 0, RCL_ROS_TIME};
    // 上拍指令时间：间隔过大（节点刚被重新激活/长时间未 tick）时不做跳变判断，
    // 避免拿上一个回合的误差去比对。
    rclcpp::Time lastCmdTime_{0, 0, RCL_ROS_TIME};
};


class Assist : public SyncActionNode
{
public:
    Assist(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("dist_tolerance", 0.8, "dist tolerance, within which considered arrived."),
            InputPort<double>("theta_tolerance", 0.8, "theta tolerance, winin which considered arrived."),
            InputPort<double>("vx_limit", 0.1, "x speed limit"),
            InputPort<double>("vy_limit", 0.1, "y speed limit"),
            InputPort<double>("dist_to_goalline", 2.5, "Distance from the robot to the goal line"),
        };
    }

    BT::NodeStatus tick() override;

private:
    Brain *brain;
};


/**
 * @brief Set the robot's velocity
 *
 * @param x,y,theta double, Robot's velocity in the x, y directions (m/s) and counterclockwise angular velocity (rad/s), default is 0. When all are 0, it is equivalent to giving a standstill command.
 *
 */
class SetVelocity : public SyncActionNode
{
public:
    SetVelocity(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    NodeStatus tick() override;
    static PortsList providedPorts()
    {
        return {
            InputPort<double>("x", 0, "Default x is 0"),
            InputPort<double>("y", 0, "Default y is 0"),
            InputPort<double>("theta", 0, "Default  theta is 0"),
        };
    }

private:
    Brain *brain;
};

class StepOnSpot : public SyncActionNode
{
public:
    StepOnSpot(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    NodeStatus tick() override;
    static PortsList providedPorts()
    {
        return {};
    }

private:
    Brain *brain;
};

class WaveHand : public SyncActionNode
{
public:
    WaveHand(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain)
    {
    }

    NodeStatus tick() override;

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<string>("action", "start", "start | stop"),
        };
    }

private:
    Brain *brain;
};

class MoveHead : public SyncActionNode
{
public:
    MoveHead(const std::string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain)
    {
    }

    NodeStatus tick() override;

    static BT::PortsList providedPorts()
    {
        return {
            InputPort<double>("pitch", 0, "target head pitch"),
            InputPort<double>("yaw", 0, "target head yaw"),
        };
    }

private:
    Brain *brain;
};


class CheckAndStandUp : public SyncActionNode
{
public:
CheckAndStandUp(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts() {
        return {};
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};


class GoToFreekickPosition : public StatefulActionNode
{
public:
    GoToFreekickPosition(const string &name, const NodeConfig &config, Brain *_brain) : StatefulActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<string>("side", "attack", "attack | defense"),
            InputPort<double>("attack_dist", 0.7, "attack side target dist to ball"),
            InputPort<double>("defense_dist", 1.9, "defense side target dist to ball"),
            InputPort<double>("vx_limit", 1.2, "vx limit"),
            InputPort<double>("vy_limit", 0.5, "vy limit"),

        };
    }

    NodeStatus onStart() override;

    NodeStatus onRunning() override;

    void onHalted() override;

private:
    Brain *brain;
    bool _isInFinalAdjust = false; 
};

class GoBackInField : public SyncActionNode
{
public:
    GoBackInField(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("valve", 0.5, "Distance from the boundary to stop when returning to the field"),
        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};


class SimpleChase : public SyncActionNode
{
public:
    SimpleChase(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("stop_dist", 1.0, "Distance from the ball to stop moving towards it"),
            InputPort<double>("stop_angle", 0.1, "Angle of the ball to stop turning towards it"),
            InputPort<double>("vy_limit", 0.2, "Limit Y direction speed to prevent walking instability. Must be less than the robot's maximum speed 0.4 to take effect"),
            InputPort<double>("vx_limit", 0.6, "Limit X direction speed to prevent walking instability. Must be less than the robot's maximum speed 1.2 to take effect"),
        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};


class CalibrateOdom : public SyncActionNode
{
public:
    CalibrateOdom(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("x", 0, "x"),
            InputPort<double>("y", 0, "y"),
            InputPort<double>("theta", 0, "theta"),
        };
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};


class PrintMsg : public SyncActionNode
{
public:
    PrintMsg(const std::string &name, const NodeConfig &config, Brain *_brain)
        : SyncActionNode(name, config)
    {
    }

    NodeStatus tick() override;

    static PortsList providedPorts()
    {
        return {InputPort<std::string>("msg")};
    }

private:
    Brain *brain;
};
