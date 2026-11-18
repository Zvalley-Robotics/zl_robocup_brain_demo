#include <cmath>
#include <cstdlib>
#include "brain_tree.h"
#include "locator.h"
#include "brain.h"
#include "utils/math.h"
#include "utils/print.h"
#include "utils/misc.h"
#include "locator.h"
#include "std_msgs/msg/string.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <fstream>
#include <ios>

/**
 * Here we use a macro definition to reduce the code for RegisterBuilder. The effect of REGISTER_BUILDER(Test) after expansion is
 * factory.registerBuilder<Test>(  \
 *      "Test",                    \
 *     [this](const string& name, const NodeConfig& config) { return make_unique<Test>(name, config, brain); });
 */
#define REGISTER_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [this](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });

// 2026-11-05：球方位 → 角速度的跟踪增益。原为 SimpleChase / GoToGoalBlockingPosition /
// Assist 三处硬编码 *4.0；3588 取消速度映射缩减后本体按下发值 1:1 执行，等效增益翻倍，
// |yaw|>0.3 rad（17°）就顶到 vtheta_limit 满幅反向转。改为可配增益（默认 2.0 ≈ 提速前等效值）
// 并加死区：死区内直接归零，避免 min_vtheta 抬升把近零方位误差放大成持续原地转。
static double yawTrackVtheta(Brain *brain, double yawToBall)
{
    if (!std::isfinite(yawToBall)) return 0.0;
    if (fabs(yawToBall) < brain->config->get_yaw_track_deadband()) return 0.0;
    return yawToBall * brain->config->get_yaw_track_gain();
}

// 2026-11-06：纯 P 转向（vtheta = Kp*err）在执行滞后（步态 + 板端 0.05 平滑，约 100~200 ms）
// 叠加 3588 取消速度折半后增益翻倍，工作点贴着极限环临界点：球静止时机器人绕球来回摆
// （实测球位连续 157 帧不变，vx=vy=0 而 vtheta 在 ±1.20 反复换向）。
// 补 D 项（阻尼）后 vtheta = Kp*err - Kd*omega_measured：omega 涨到 Kp*err/Kd 时输出归零，
// 系统自动进入滑行段，不再一路打满转过冲。
// 关键取舍：D 作用在"实测角速度"上而非误差差分——目标量（ballYaw / kickDir）跳变时
// 不会产生微分尖峰，只抑制"已经转得太快"。
// 角速度来源是 odom 的 atan2(yaw)（zl_io.cpp onOdom）：连续、不受定位重置影响，可安全差分。
// yaw_track_damping=0 时整条通道退化为原纯 P 行为（回退开关）。
struct YawCtrlState
{
    double lastTheta = 0.0;
    double lastTimeSec = 0.0;
    double rateFiltered = 0.0;
    bool init = false;
};

static double yawRateFromOdom(Brain *brain, YawCtrlState &st)
{
    const double theta = brain->data->robotPoseToOdom.theta;
    const double nowSec = brain->get_clock()->now().seconds();
    if (!st.init || nowSec <= st.lastTimeSec)
    {
        st.lastTheta = theta;
        st.lastTimeSec = nowSec;
        st.init = true;
        return st.rateFiltered;
    }
    const double dt = nowSec - st.lastTimeSec;
    const double dTheta = toPInPI(theta - st.lastTheta);
    st.lastTheta = theta;
    st.lastTimeSec = nowSec;
    // odom 断流/卡顿时不更新差分（dt 极小会把量化噪声放大成尖峰），沿用上次滤波值
    if (dt < 1e-3 || dt > 0.2) return st.rateFiltered;
    double raw = cap(dTheta / dt, 3.0, -3.0); // 打滑/被撞时实测角速度失真，先限幅
    const double lpfHz = brain->config->get_yaw_rate_lpf_hz();
    const double lpfSafe = lpfHz > 1e-3 ? lpfHz : 1e-3;
    const double alpha = 1.0 - std::exp(-2.0 * M_PI * lpfSafe * dt);
    st.rateFiltered += alpha * (raw - st.rateFiltered);
    return st.rateFiltered;
}

// kp 由调用方给出（Adjust 用 XML 的 vtheta_factor，其余节点用 yaw_track_gain），
// 避免把不同节点的增益口径混在一起。deadband 同样由调用方决定。
static double yawControl(Brain *brain, double err, YawCtrlState &st,
                         double limit, double kp, double deadband)
{
    if (!std::isfinite(err)) return 0.0;
    if (fabs(err) < deadband) return 0.0;
    const double kd = brain->config->get_yaw_track_damping();
    if (kd <= 1e-6) return cap(kp * err, limit, -limit); // 回退：纯 P
    return cap(kp * err - kd * yawRateFromOdom(brain, st), limit, -limit);
}

void BrainTree::init()
{
    BehaviorTreeFactory factory;

    // Action Nodes
    REGISTER_BUILDER(RobotFindBall)
    REGISTER_BUILDER(Chase)
    REGISTER_BUILDER(SimpleChase)
    REGISTER_BUILDER(Adjust)
    REGISTER_BUILDER(Kick)
    REGISTER_BUILDER(StandStill)
    REGISTER_BUILDER(CalcKickDir)
    REGISTER_BUILDER(StrikerDecide)
    REGISTER_BUILDER(CamTrackBall)
    REGISTER_BUILDER(CamFindBall)
    REGISTER_BUILDER(CamFastScan)
    REGISTER_BUILDER(CamScanField)
    REGISTER_BUILDER(SetVelocity)
    REGISTER_BUILDER(StepOnSpot)
    REGISTER_BUILDER(GoToFreekickPosition)
    REGISTER_BUILDER(GoToReadyPosition)
    REGISTER_BUILDER(GoToGoalBlockingPosition)
    REGISTER_BUILDER(TurnOnSpot)
    REGISTER_BUILDER(MoveToPoseOnField)
    REGISTER_BUILDER(GoBackInField)
    REGISTER_BUILDER(GoalieDecide)
    REGISTER_BUILDER(WaveHand)
    REGISTER_BUILDER(MoveHead)
    REGISTER_BUILDER(CheckAndStandUp)
    REGISTER_BUILDER(RLVisionKick)
    REGISTER_BUILDER(Assist)

    // Register Locator related nodes
    brain->registerLocatorNodes(factory);

    // Action Nodes for debug
    REGISTER_BUILDER(CalibrateOdom)
    REGISTER_BUILDER(PrintMsg)

    factory.registerBehaviorTreeFromFile(brain->config->get_tree_file_path());
    tree = factory.createTree("MainTree");

    // 挂 tick 监视器（必须在首次 tick 之前；节点集合此后不再变化）
    attachTickMonitors();

    // After construction, initialize blackboard entries
    initEntry();
}

void BrainTree::initEntry()
{
    setEntry<string>("player_role", brain->config->get_player_role());
    setEntry<bool>("ball_location_known", false);
    setEntry<bool>("tm_ball_pos_reliable", false);
    // 球位置共享（2026-11-10）：hint 只给转头用；guard 在 LOCAL/TM 来源切换后的短窗内置真
    setEntry<bool>("tm_ball_pos_hint", false);
    setEntry<bool>("ball_src_guard", false);
    setEntry<bool>("ball_out", false);
    setEntry<bool>("track_ball", true);
    // 当 ground_truth 启用时直接标定为已校准，SelfLocate 粒子滤波变为无害空操作
    setEntry<bool>("odom_calibrated", brain->config->get_gt_pose_enabled());
    setEntry<string>("decision", "");
    setEntry<string>("defend_decision", "chase");
    setEntry<double>("ball_range", 0);
    // 球速估计（Brain::updateBallEstimate 每拍覆写）：门将用 vel 预判过线点、用 age 判陈旧
    setEntry<double>("ball_vel_x", 0.0);
    setEntry<double>("ball_vel_y", 0.0);
    setEntry<double>("ball_speed", 0.0);
    setEntry<double>("ball_age_ms", 1e9);
    setEntry<bool>("ball_est_valid", false);

    setEntry<bool>("gamecontroller_isKickOff", true);
    setEntry<string>("gc_game_state", "");
    setEntry<string>("gc_game_sub_state_type", "NONE");
    setEntry<string>("gc_game_sub_state", "");
    setEntry<bool>("gc_is_kickoff_side", false);
    setEntry<bool>("gc_is_sub_state_kickoff_side", false);
    setEntry<bool>("gc_is_under_penalty", false);

    setEntry<bool>("need_check_behind", false);

    setEntry<bool>("is_lead", true); 
    setEntry<string>("goalie_mode", "attack"); 

    setEntry<int>("test_choice", 0);
    setEntry<int>("control_state", 0);
    setEntry<bool>("assist_chase", false);
    setEntry<bool>("assist_kick", false);
    setEntry<bool>("go_manual", false);

    setEntry<bool>("we_just_scored", false);
    setEntry<bool>("wait_for_opponent_kickoff", false);

    // Automatic vision calibration related
    setEntry<string>("calibrate_state", "pitch");
    setEntry<double>("calibrate_pitch_center", 0.0);
    setEntry<double>("calibrate_pitch_step", 1.0);
    setEntry<double>("calibrate_yaw_center", 0.0);
    setEntry<double>("calibrate_yaw_step", 1.0);
    setEntry<double>("calibrate_z_center", 0.0);
    setEntry<double>("calibrate_z_step", 0.01);
}

void BrainTree::tick()
{
    _lastStatus.clear();
    const bool splitEnabled = brain->get_parameter("locator.split_schedule").as_bool();
    if (splitEnabled != localizationSplitWasEnabled) localizationSchedule.clear();
    localizationSplitWasEnabled = splitEnabled;
    if (splitEnabled) {
        localizationTickNs = brain->get_clock()->now().nanoseconds();
        if (!getEntry<bool>("odom_calibrated") || brain->config->get_gt_pose_enabled()) {
            localizationSchedule.clear();
        } else if (localizationSchedule.beginTick(localizationTickNs)) {
            RCLCPP_WARN(brain->get_logger(), "[LOC_SPLIT] ROS时间回退，重新设置调度参考");
        }
    }
    tree.tickOnce();
    if (brain->get_parameter("locator.diagnostics").as_bool()) {
        const auto now = brain->get_clock()->now().nanoseconds();
        if (lastLocalizationDiagNs < 0 || now < lastLocalizationDiagNs || now - lastLocalizationDiagNs >= 5000000000LL) {
            lastLocalizationDiagNs = now;
            RCLCPP_INFO(brain->get_logger(), "[LOC_SCHED] method=TRUSTDIR skipped=%llu attempts=%llu success=%llu last=%s elapsed_ms=%.1f split=%d",
                (unsigned long long)locTrustSkipped, (unsigned long long)locTrustAttempts, (unsigned long long)locTrustSuccess,
                localizationLastMode.c_str(), brain->msecsSince(brain->data->lastSuccessfulLocalizeTime),
                int(brain->get_parameter("locator.split_schedule").as_bool()));
            static const char *names[] = {"1M", "2T", "PT", "LT", "2X", "Border"};
            for (size_t i = 0; i < locGeomAttempts.size(); ++i)
                RCLCPP_INFO(brain->get_logger(), "[LOC_SCHED] method=%s skipped=%llu attempts=%llu success=%llu",
                    names[i], (unsigned long long)locGeomSkipped[i], (unsigned long long)locGeomAttempts[i], (unsigned long long)locGeomSuccess[i]);
        }
    }
    // 在同一次行为树 tick 后读取定位状态；仅诊断，不改变定位结果。
    if (brain->get_parameter("locator.trace_enabled").as_bool()) {
        const auto now = brain->get_clock()->now().nanoseconds();
        const double interval = brain->get_parameter("locator.trace_interval_sec").as_double();
        if (std::isfinite(interval) && interval > 0.0 && interval <= 3600.0 &&
            (lastLocalizationTraceNs < 0 || now < lastLocalizationTraceNs ||
             (now - lastLocalizationTraceNs) * 1e-9 >= interval)) {
            lastLocalizationTraceNs = now;
            const auto &odom = brain->data->robotPoseToOdom;
            const auto &field = brain->data->robotPoseToField;
            const auto &otf = brain->data->odomToField;
            RCLCPP_INFO(brain->get_logger(),
                "[BRAIN_TRACE] t_ns=%lld odom=(%.6f,%.6f,%.6f) field=(%.6f,%.6f,%.6f) otf=(%.6f,%.6f,%.6f) calib=%d",
                (long long)now, odom.x, odom.y, odom.theta, field.x, field.y, field.theta,
                otf.x, otf.y, otf.theta, int(getEntry<bool>("odom_calibrated")));
        }
    }
    buildStrategyPath();
}

void BrainTree::attachTickMonitors()
{
    // tree.subtrees 覆盖含 MainTree 在内的全部子树；回调在第 n 节点 tick 结束后触发，
    // 记录其返回状态。以 this 捕获：监视器生命周期与 BrainTree 一致。
    for (auto &subtree : tree.subtrees)
        for (auto &node : subtree->nodes)
            node->setTickMonitorCallback([this](TreeNode &n, NodeStatus status, std::chrono::microseconds)
                                         { _lastStatus.emplace(&n, status); });
}

void BrainTree::buildStrategyPath()
{
    // 展示过滤：XML 里显式命名（name != registrationName）或动作/条件叶子，
    // 跳过未命名的组合/装饰节点（它们只是结构，不表达策略）。
    auto keep = [](const TreeNode *n) {
        const string &nm = n->name();
        return !nm.empty() && (nm != n->registrationName() ||
                               n->type() == NodeType::ACTION ||
                               n->type() == NodeType::CONDITION);
    };

    // 主链：从根出发，每层跟随本次实际执行的分支；多个候选时优先 RUNNING
    // （正在推进的分支），否则取最后一个被执行者（序列语义下即最深推进的分支）。
    _strategyPath.clear();
    const TreeNode *node = tree.rootNode();
    while (node != nullptr && _lastStatus.count(node) > 0)
    {
        if (node != tree.rootNode() && keep(node))
            _strategyPath.push_back(node->name());

        if (node->type() == NodeType::CONTROL)
        {
            const TreeNode *follow = nullptr;
            for (auto *child : dynamic_cast<const ControlNode *>(node)->children())
            {
                auto it = _lastStatus.find(child);
                if (it == _lastStatus.end()) continue;
                follow = child;
                if (it->second == NodeStatus::RUNNING) break;
            }
            node = follow;
        }
        else if (node->type() == NodeType::DECORATOR || node->type() == NodeType::SUBTREE)
        {
            auto *dec = dynamic_cast<const DecoratorNode *>(node);
            node = dec ? dec->child() : nullptr;
        }
        else
        {
            node = nullptr;
        }
    }

    // 本轮执行过的全部具名/动作节点（按子树/节点声明序，按名字去重）
    _strategyActive.clear();
    unordered_set<string> seen;
    for (auto &subtree : tree.subtrees)
        for (auto &n : subtree->nodes)
            if (keep(n.get()) && _lastStatus.count(n.get()) > 0 && seen.insert(n->name()).second)
                _strategyActive.push_back(n->name());
}

NodeStatus SetVelocity::tick()
{
    double x, y, theta;
    vector<double> targetVec;
    getInput("x", x);
    getInput("y", y);
    getInput("theta", theta);

    auto res = brain->client->setVelocity(x, y, theta);
    return NodeStatus::SUCCESS;
}

NodeStatus StepOnSpot::tick()
{
    std::srand(std::time(0));
    double vx = (std::rand() / (RAND_MAX / 0.02)) - 0.01;

    auto res = brain->client->setVelocity(vx, 0, 0);
    return NodeStatus::SUCCESS;
}

NodeStatus CamTrackBall::tick()
{
    double pitch, yaw, ballX, ballY, deltaX, deltaY;
    // 2026-10-30: 死区 30% → 15%。原值下球在画面中心 60%×60% 区域内时完全
    // 不下发指令，实机观感是“球快出画面了头才动”。15% 让球稍偏即开始跟。
    const double pixToleranceX = brain->config->cameraImageWidth * 15 / 100.;
    const double pixToleranceY = brain->config->cameraImageHeight * 15 / 100.;
    const double xCenter = brain->config->cameraImageWidth / 2;
    const double yCenter = brain->config->cameraImageHeight / 2; 


    // “看见球”= 本帧有检测且位置未过期。只看 ballDetected 会把“刚丢失一帧”
    // 也当成可见；只看 ball_location_known 则会在记忆期内一直信任旧位置。
    const double ballAgeMs = brain->msecsSince(brain->data->ball.timePoint);
    // GT 球只有场地/机器人系坐标，没有视觉 boundingBox；若仍走下面的图像框分支，
    // 会用未初始化的框（通常全 0）把 yaw/pitch 持续推到限位。GT 球改走相对角度分支
    // （updateBallMemory 每 tick 已维护 pitchToRobot/yawToRobot）。
    // 实机（gt ball 关闭）行为不变。
    const bool gtBall = brain->config->get_gt_ball_enabled();
    bool iSeeBall = !gtBall && brain->data->ballDetected && ballAgeMs <= Brain::kBallFreshMs;
    bool iKnowBallPos = brain->tree->getEntry<bool>("ball_location_known");
    bool tmBallPosReliable = brain->tree->getEntry<bool>("tm_ball_pos_reliable");
    // 贴身 hint（2026-11-10）：队友报的球位离我 <2 m 或队友自己也在用记忆时只能算“提示”，
    // 不置 ball_location_known、不驱动追球/踢球；唯一用途是把头转向该位置，
    // 让机器人尽快自己看到球，恢复真实检测。
    bool tmBallPosHint = brain->tree->getEntry<bool>("tm_ball_pos_hint");
    if (!(iKnowBallPos || tmBallPosReliable || tmBallPosHint))
        return NodeStatus::SUCCESS;

    // 2026-10-30: 丢球宽限期按距离分段。
    //   近距（<0.6 m，踢球/贴脚）：300 ms 就放弃，避免把头顶到脚边（range 小时
    //   pitchToRobot 会被 asin 限幅到最低头）；
    //   远距：放宽到 1500 ms，球短暂出框/被腿挡住时继续朝最后方向转头找，
    //   不要立刻切 CamFindBall 的 6 点扫描。
    const double rangeForGrace = iKnowBallPos ? brain->data->ball.range : brain->data->tmBall.range;
    const double lostGraceMs = (rangeForGrace < 0.6) ? 300.0 : 1500.0;
    if (!iSeeBall && ballAgeMs > lostGraceMs)
    {
        // 超过宽限期：保持当前头位，交给 CamFindBall 扫场
        return NodeStatus::SUCCESS;
    }

    // 2026-10-30: 指令节流。本体头部平滑节点每条指令 ~0.5 s 收敛，原来每 tick
    // （~25 ms）就发新目标，平滑被反复重置、越跟越慢；这里限制最小下发间隔。
    static constexpr double kSendIntervalMs = 100.0;
    static rclcpp::Time lastSendTime(0, 0, RCL_ROS_TIME);
    if (lastSendTime.nanoseconds() > 0 && brain->msecsSince(lastSendTime) < kSendIntervalMs)
        return NodeStatus::SUCCESS;

    if (!iSeeBall)
    { 
        // 宽限期内看不见球：朝最后已知位置收敛。节流后每 100 ms 才发一次，
        // 收敛系数 0.01 → 0.15（时间常数 ~0.6 s），否则跟不上。
        const GameObject &trackedBall = iKnowBallPos ? brain->data->ball : brain->data->tmBall;
        if (!std::isfinite(trackedBall.pitchToRobot) || !std::isfinite(trackedBall.yawToRobot))
            return NodeStatus::SUCCESS;
        pitch = brain->data->headPitch + (trackedBall.pitchToRobot - brain->data->headPitch) * 0.15;
        yaw = brain->data->headYaw + (trackedBall.yawToRobot - brain->data->headYaw) * 0.15;
    }
    else {      
        ballX = mean(brain->data->ball.boundingBox.xmax, brain->data->ball.boundingBox.xmin);
        ballY = mean(brain->data->ball.boundingBox.ymax, brain->data->ball.boundingBox.ymin);
        deltaX = ballX - xCenter;
        deltaY = ballY - yCenter; 
        
        if (std::fabs(deltaX) < pixToleranceX && std::fabs(deltaY) < pixToleranceY)
        {
            return NodeStatus::SUCCESS;
        }

        // 2026-10-30: 增益 3.5 → 2.0，跟得更快一些。
        double smoother = 2.0;
        double deltaYaw = deltaX / brain->config->cameraImageWidth * brain->config->depthCameraFovX / smoother;
        double deltaPitch = deltaY / brain->config->cameraImageHeight * brain->config->depthCameraFovY / smoother;

        pitch = brain->data->headPitch + deltaPitch;
        yaw = brain->data->headYaw - deltaYaw;
    }

    brain->client->moveHead(pitch, yaw);
    lastSendTime = brain->get_clock()->now();
    return NodeStatus::SUCCESS;
}

CamFindBall::CamFindBall(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain)
{
    double lowPitch = 1.0;
    double midPitch = 0.6;
    double highPitch = 0.22; // 高位取光轴最抬头附近（新头 11°~61° 下俯，下限 0.20）
    double leftYaw = 1.1;
    double rightYaw = -1.1;

    // 先抬头看远处（球被踢走后大概率在 2~6 m），最后才低头确认脚下。
    // 旧顺序是“低 3 格 → 高 3 格”，丢球后前 3 s 相机一直对着地面
    // （pitch=1.0 时地面可视只有 ~0.5 m），正好错过刚被踢出去的球。
    _cmdSequence[0][0] = highPitch;
    _cmdSequence[0][1] = 0;
    _cmdSequence[1][0] = highPitch;
    _cmdSequence[1][1] = leftYaw;
    _cmdSequence[2][0] = highPitch;
    _cmdSequence[2][1] = rightYaw;
    _cmdSequence[3][0] = midPitch;
    _cmdSequence[3][1] = rightYaw;
    _cmdSequence[4][0] = lowPitch;
    _cmdSequence[4][1] = 0;
    _cmdSequence[5][0] = lowPitch;
    _cmdSequence[5][1] = leftYaw;

    _cmdIndex = 0;
    // 2026-10-30: 1000 → 700 ms。本体头部 ~0.5 s 收敛，1 s 中有近一半是空等；
    // 700 ms 可覆盖一轮 6 点 ≈ 4.2 s（原 6.0 s）。
    _cmdIntervalMSec = 700;
    // 重新进入节点超过 2 个节拍即视为新一轮扫描，必须从 [0] 开始（原来 60 s 才重置，
    // 中途切走再回来会接着扫后半段，导致“朝最后球方向看第一眼”形同虚设）。
    _cmdRestartIntervalMSec = _cmdIntervalMSec * 2;
    _timeLastCmd = brain->get_clock()->now();
}

NodeStatus CamFindBall::tick()
{
    if (brain->data->ballDetected)
    {
        return NodeStatus::SUCCESS;
    } // Currently, all nodes return Success. Returning Failure would affect the execution of subsequent nodes.

    auto curTime = brain->get_clock()->now();
    auto timeSinceLastCmd = (curTime - _timeLastCmd).nanoseconds() / 1e6;
    if (timeSinceLastCmd < _cmdIntervalMSec)
    {
        return NodeStatus::SUCCESS;
    } // Not yet time for the next command
    else if (timeSinceLastCmd > _cmdRestartIntervalMSec)
    {                  // Exceeded a certain time, consider this as restarting from the beginning
        _cmdIndex = 0; // Note that we don't return here
    }
    else
    { // Reached the time, execute the next command, also do not return
        _cmdIndex = (_cmdIndex + 1) % (sizeof(_cmdSequence) / sizeof(_cmdSequence[0]));
    }

    // 新一轮扫描优先朝“最后一次看到球的方向”抬头（球的 yawToRobot 在失球后
    // 仍会随位姿更新），而不是固定从正前方开始；5 s 内没看到过球则退回正前。
    if (_cmdIndex == 0)
    {
        double yaw0 = 0.0;
        const double ballAgeSec = brain->msecsSince(brain->data->ball.timePoint) / 1000.0;
        if (ballAgeSec <= 5.0 && std::isfinite(brain->data->ball.yawToRobot))
        {
            yaw0 = cap(brain->data->ball.yawToRobot, 1.1, -1.1);
        }
        _cmdSequence[0][1] = yaw0;
    }

    brain->client->moveHead(_cmdSequence[_cmdIndex][0], _cmdSequence[_cmdIndex][1]);
    _timeLastCmd = brain->get_clock()->now();
    return NodeStatus::SUCCESS;
}

NodeStatus CamScanField::tick()
{
    auto sec = brain->get_clock()->now().seconds();
    auto msec = static_cast<unsigned long long>(sec * 1000);
    double lowPitch, highPitch, leftYaw, rightYaw;
    getInput("low_pitch", lowPitch);
    getInput("high_pitch", highPitch);
    getInput("left_yaw", leftYaw);
    getInput("right_yaw", rightYaw);
    int msecCycle;
    getInput("msec_cycle", msecCycle);
    int sendIntervalMsec = 0;
    getInput("send_interval_msec", sendIntervalMsec);

    int cycleTime = msec % msecCycle;
    double pitch = cycleTime > (msecCycle / 2.0) ? lowPitch : highPitch;
    double yaw = cycleTime < (msecCycle / 2.0) ? (leftYaw - rightYaw) * (2.0 * cycleTime / msecCycle) + rightYaw : (leftYaw - rightYaw) * (2.0 * (msecCycle - cycleTime) / msecCycle) + rightYaw;

    // 降频节流：本体 head 平滑节点每条指令需 ~0.5s 收敛，90Hz 连发会被反复重置导致
    // 幅度衰减。目标值仍按绝对时间连续计算，只按 send_interval_msec 节拍下发采样点，
    // 使本体侧每个航点都能收敛到位（0 表示不节流，每 tick 下发）。
    if (sendIntervalMsec > 0 && sec - _lastScanSendSec < sendIntervalMsec / 1000.0)
    {
        return NodeStatus::SUCCESS;
    }
    _lastScanSendSec = sec;

    brain->client->moveHead(pitch, yaw);
    return NodeStatus::SUCCESS;
}

NodeStatus Chase::tick()
{
    auto log = [=](string msg) {
        brain->log->debug("Chase4", msg);
    };
    log("ticked");
    
    double vxLimit, vyLimit, vthetaLimit, dist, safeDist;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    getInput("vtheta_limit", vthetaLimit);
    getInput("dist", dist);
    getInput("safe_dist", safeDist);

    bool avoidObstacle = brain->config->get_avoid_during_chase();
    double oaSafeDist = brain->config->get_chase_ao_safe_dist();

    if (
        brain->config->get_limit_near_ball_speed()
        && brain->data->ball.range < brain->config->get_near_ball_range()
    ) {
        vxLimit = min(brain->config->get_near_ball_speed_limit(), vxLimit);
    }

    double ballRange = brain->data->ball.range;
    double ballYaw = brain->data->ball.yawToRobot;
    double kickDir = brain->data->kickDir;
    double theta_br = atan2(
        brain->data->robotPoseToField.y - brain->data->ball.posToField.y,
        brain->data->robotPoseToField.x - brain->data->ball.posToField.x
    );
    double theta_rb = brain->data->robotBallAngleToField;
    auto ballPos = brain->data->ball.posToField;


    double vx, vy, vtheta;
    Pose2D target_f, target_r; 
    static string targetType = "direct"; 
    static double circleBackDir = 1.0; 
    double dirThreshold = M_PI / 2;
    if (targetType == "direct") dirThreshold *= 1.2;


    // Calculate target point
    if (fabs(toPInPI(kickDir - theta_rb)) < dirThreshold) {
        log("targetType = direct");
        targetType = "direct";
        target_f.x = ballPos.x - dist * cos(kickDir);
        target_f.y = ballPos.y - dist * sin(kickDir);
    } else {
        targetType = "circle_back";
        double cbDirThreshold = 0.0; 
        cbDirThreshold -= 0.2 * circleBackDir; 
        circleBackDir = toPInPI(theta_br - kickDir) > cbDirThreshold ? 1.0 : -1.0;
        log(format("targetType = circle_back, circleBackDir = %.1f", circleBackDir));
        double tanTheta = theta_br + circleBackDir * acos(min(1.0, safeDist/max(ballRange, 1e-5))); 
        target_f.x = ballPos.x + safeDist * cos(tanTheta);
        target_f.y = ballPos.y + safeDist * sin(tanTheta);
    }
    target_r = brain->data->field2robot(target_f);
            
    double targetDir = atan2(target_r.y, target_r.x);
    double distToObstacle = brain->distToObstacle(targetDir);
    // 避障决策的 INFO 观测（1 Hz 限频）。Chase 原有的 "avoid obstacle" 是 debug 级，
    // 默认不进 /tmp/brain.log，实机上无法区分"未触发"与"未打印"。
    // 判据字段：trig=1 已进绕行分支；ok=1 表示限幅后方向可用；ok=0 为减速/停车。
    static rclcpp::Time lastAvoidLog = brain->get_clock()->now();
    const bool avoidLogNow = brain->msecsSince(lastAvoidLog) > 1000;
    if (avoidObstacle && distToObstacle < oaSafeDist) {
        log("avoid obstacle");
        // 只允许在相机 FOV 内偏转：FOV 外栅格恒返回“安全”(1e9)，直接采用
        // calcAvoidDir 会得到指向盲区的绕行角；限幅后仍受阻则减速直行。
        // 绕行速度已参数化（obstacle_avoidance.avoid_speed）：原为硬编码 0.5，
        // 它不随 vx_limit/XML 限速变化，提速时须单独调这一项。
        const double speed = brain->config->get_avoid_speed();
        const double maxOffset = deg2rad(brain->config->get_avoid_max_offset_deg());
        double avoidDir = 0;
        const bool dirOk = brain->calcSafeAvoidDir(targetDir, oaSafeDist, maxOffset, avoidDir);
        bool quantized = false;
        if (dirOk) {
            vx = speed * cos(avoidDir);
            vy = speed * sin(avoidDir);
            // setVelocity 会把 <min_vy 的分量抬到 min_vy，实际转向角远大于决策角
            // （实测 9.2°→31.5°、-2.9°→-31°），会造成方向振荡。
            // 表达不出来的微偏转直接归零，用略低的 vx 表示“减速直行”。
            if (fabs(vy) < brain->config->get_min_vy()) {
                vy = 0.0;
                vx = speed * 0.7;
                quantized = true;
            }
            vtheta = ballYaw;
        } else {
            // 贴身区或 FOV 内绕不开：贴近时停车，稍远时半速减速。
            // 不能只半速直行——以 0.25 m/s 逼近 1.3 m 的目标只需 5 s 就会撞上。
            vx = (distToObstacle < brain->config->get_avoid_min_clearance_m()) ? 0.0 : speed * 0.5;
            vy = 0.0;
            vtheta = ballYaw;
        }
        if (avoidLogNow) {
            lastAvoidLog = brain->get_clock()->now();
            brain->log->log("chase/avoid", format("trig=1 targetDir=%.2f dist=%.2f safe=%.2f dir=%.2f ok=%d quant=%d vx=%.2f vy=%.2f",
                targetDir, distToObstacle, oaSafeDist, avoidDir, dirOk ? 1 : 0, quantized ? 1 : 0, vx, vy));
        }
    } else {
        // 接球段速度：用刹车模型按“到目标点的剩余距离”限速。旧写法 min(vxLimit, ball.range)
        // 相当于 1 m/s per meter，在 2 m/s 档位最后 1 m 只有 1.0 的上限、制动距离不够，
        // 实测会冲过球/冲过头。圆弧绕后另受“可跟踪角速度”约束，否则 ω=v/r 超限绕不回来。
        const double toTarget = max(0.0, ballRange - dist);
        double approachCap = brain->brakeSpeed(toTarget);
        if (targetType == "circle_back") {
            approachCap = min(approachCap, brain->config->get_vtheta_limit() * max(0.3, safeDist));
        }
        vx = min(vxLimit, approachCap);
        vy = 0;
        vtheta = targetDir;
        if (fabs(targetDir) < 0.1 && ballRange > 2.0) vtheta = 0.0;
        vx *= sigmoid((fabs(vtheta)), 1, 3); 
    }

    vx = cap(vx, vxLimit, -vxLimit);
    vy = cap(vy, vyLimit, -vyLimit);
    vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

    // 队友贴身单侧让位：Chase 自身只看深度障碍栅格（不含队友坐标），双 lead 瞬态或两台
    // 同时收敛到同一球后点时没有任何去冲突，实测会直接顶在一起。由 (cost, id) 仲裁后
    // 只让较差的一台退开；球已进入 0.8 m 踢球窗时不干预（交给 Adjust/kick）。
    {
        double yieldVx, yieldVy, yieldVtheta;
        if (brain->calcChaseYieldVelocity(yieldVx, yieldVy, yieldVtheta, 0.8)) {
            brain->client->setVelocity(yieldVx, yieldVy, yieldVtheta);
            return NodeStatus::SUCCESS;
        }
    }

    static double smoothVx = 0.0;
    static double smoothVy = 0.0;
    static double smoothVtheta = 0.0;
    smoothVx = smoothVx * 0.7 + vx * 0.3;
    smoothVy = smoothVy * 0.7 + vy * 0.3;
    smoothVtheta = smoothVtheta * 0.7 + vtheta * 0.3;

    brain->client->setVelocity(vx, vy, vtheta);
    return NodeStatus::SUCCESS;
}

NodeStatus SimpleChase::tick()
{
    double stopDist, stopAngle, vyLimit, vxLimit;
    getInput("stop_dist", stopDist);
    getInput("stop_angle", stopAngle);
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);

    if (!brain->tree->getEntry<bool>("ball_location_known"))
    {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    double vx = brain->data->ball.posToRobot.x;
    double vy = brain->data->ball.posToRobot.y;
    double vtheta = yawTrackVtheta(brain, brain->data->ball.yawToRobot);

    double linearFactor = 1 / (1 + exp(3 * (brain->data->ball.range * fabs(brain->data->ball.yawToRobot)) - 3)); 
    vx *= linearFactor;
    vy *= linearFactor;

    vx = cap(vx, vxLimit, -1.0);    
    vy = cap(vy, vyLimit, -vyLimit); 

    if (brain->data->ball.range < stopDist)
    {
        vx = 0;
        vy = 0;
    }

    brain->client->setVelocity(vx, vy, vtheta);
    return NodeStatus::SUCCESS;
}


NodeStatus GoToFreekickPosition::onStart() {
    _isInFinalAdjust = false;
    return NodeStatus::RUNNING;
}

NodeStatus GoToFreekickPosition::onRunning() {

    string side;
    getInput("side", side);
    if (side !="attack" && side != "defense") return NodeStatus::SUCCESS;
    
    Pose2D targetPose;
    auto fd = brain->config->fieldDimensions;
    auto ballPos = brain->data->ball.posToField;
    auto robotPose = brain->data->robotPoseToField;

    double kickDir = brain->data->kickDir;
    double defenseDir = atan2(ballPos.y, ballPos.x + fd.length / 2);
    bool stanceAligned = false;   // 主罚者站位是否已按“沿 kickDir 收缩”重解（见下）
    if (side == "attack") {
       double dist;
       getInput("attack_dist", dist);
        
       if (brain->data->myStrikerIDRank == 0) {
        // 理想站位点在球后 dist·(-kickDir)。判罚点贴底线/边线（角球）时该点会落到
        // 场外；若按 x/y 硬夹进场内，站位会滑到球的斜前方——对齐角丢失后 PLAY 里
        // reachedKickDir 永远不满足，只能落 Adjust 顶着贴线球绕，容易把球蹭出界。
        // 这里沿“球→站位”射线收缩站位距离：机器人→球方位仍严格等于 kickDir
        // （起脚对齐不丢），只把距离收进“场内留 0.05 m 边距”的可达范围。
        double ux = cos(kickDir), uy = sin(kickDir);
        double dMax = dist;
        if (ux > 1e-9)       dMax = min(dMax, (ballPos.x + fd.length / 2.0 - 0.05) / ux);
        else if (ux < -1e-9) dMax = min(dMax, (fd.length / 2.0 - 0.05 - ballPos.x) / (-ux));
        if (uy > 1e-9)       dMax = min(dMax, (ballPos.y + fd.width / 2.0 - 0.05) / uy);
        else if (uy < -1e-9) dMax = min(dMax, (fd.width / 2.0 - 0.05 - ballPos.y) / (-uy));
        if (dMax > 0.15) {   // 收缩后仍留得出起码站位距离才用重解，否则退回统一 clamp 兜底
            dist = dMax;
            stanceAligned = true;
        }
        targetPose.x = ballPos.x - dist * cos(kickDir);
        targetPose.y = ballPos.y - dist * sin(kickDir);
        targetPose.theta = kickDir;
       } else if (brain->data->myStrikerIDRank == 1) {
        targetPose.x = ballPos.x - 2.0 * cos(defenseDir);
        targetPose.y = ballPos.y - 2.0 * sin(defenseDir);
        targetPose.theta = defenseDir;
        } else if (brain->data->myStrikerIDRank == 2) {
            targetPose.x = - fd.length / 2.0 + fd.penaltyDist;
            targetPose.y = fd.goalAreaWidth / 2.0;
        } else if (brain->data->myStrikerIDRank == 3) {
            targetPose.x = - fd.length / 2.0 + fd.penaltyDist;
            targetPose.y = - fd.goalAreaWidth / 2.0;
        }
    } else if (side == "defense") {
        if (brain->data->myStrikerIDRank == 0) {
            targetPose.x = ballPos.x - 3.0 * cos(defenseDir);  // Changed from 2 to 3, defensive player is farther from the ball
            targetPose.y = ballPos.y - 2.5 * sin(defenseDir);
            targetPose.theta = defenseDir;
           } else if (brain->data->myStrikerIDRank == 1) {
            targetPose.x = ballPos.x - 3.5 * cos(defenseDir);
            targetPose.y = ballPos.y - 4.0 * sin(defenseDir);
            targetPose.theta = defenseDir;
            } else if (brain->data->myStrikerIDRank == 2) {
                targetPose.x = - fd.length / 2.0 + fd.penaltyDist;
                targetPose.y = fd.goalAreaWidth / 2.0;
            } else if (brain->data->myStrikerIDRank == 3) {
                targetPose.x = - fd.length / 2.0 + fd.penaltyDist;
                targetPose.y = - fd.goalAreaWidth / 2.0;
            }
    }

    // 判罚点靠近底线/边线时，“站到球后/挡在球与球门之间”的几何解会落到场外
    // （最多可在球门线后数米），统一收进场内并留 0.3 m 边距，避免径直走出场；
    // 主罚者站位已在上方沿 kickDir 收缩重解并保持在 0.05 m 边距内，跳过 clamp
    // 以免再次破坏起脚对齐。
    if (!stanceAligned) {
        targetPose.x = cap(targetPose.x, fd.length / 2.0 - 0.3, - fd.length / 2.0 + 0.3);
        targetPose.y = cap(targetPose.y, fd.width / 2.0 - 0.3, - fd.width / 2.0 + 0.3);
    }

    double dist = norm(targetPose.x - robotPose.x, targetPose.y - robotPose.y);
    double deltaDir = toPInPI(targetPose.theta - robotPose.theta);


    if ( // Considered to have reached the target position
        dist < 0.2 
        && fabs(deltaDir) < 0.1
    ) {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }
    
    if (!brain->config->get_enable_obstacle_avoidance() || dist < 1.0 || _isInFinalAdjust) {
        _isInFinalAdjust = true; // Entering the final adjustment phase
        auto targetPose_r = brain->data->field2robot(targetPose);

        double vx = targetPose_r.x;
        double vy = targetPose_r.y;
        double vtheta = brain->data->ball.yawToRobot * 2.0; // The larger the multiplier, the faster the rotation

        double linearFactor = 1 / (1 + exp(3 * (brain->data->ball.range * fabs(brain->data->ball.yawToRobot)) - 3)); // When the distance is far, prioritize turning
        vx *= linearFactor;
        vy *= linearFactor;

        // Prevent collision with the ball
        Line path = {robotPose.x, robotPose.y, targetPose.x, targetPose.y};
        if (
            pointMinDistToLine(Point2D({ballPos.x, ballPos.y}), path) < 0.5
            && brain->data->ball.range < 1.0
        ) {
            vx = min(0.0, vx);
            vy = vy >= 0 ? vy + 0.1: vy - 0.1;
        }

        double vxLimit, vyLimit;
        getInput("vx_limit", vxLimit);
        getInput("vy_limit", vyLimit);
        vx = cap(vx, vxLimit, -0.4);     // Further limit speed
        vy = cap(vy, vyLimit, -vyLimit);     // Further limit speed
        

        brain->client->setVelocity(vx, vy, vtheta);
        return NodeStatus::RUNNING;
    }

    double longRangeThreshold = 1.4;
    double turnThreshold = 0.4;
    double vxLimit = 0.6;
    double vyLimit = 0.5;
    double vthetaLimit = 1.5;
    bool avoidObstacle = true;
    brain->client->moveToPoseOnField3(targetPose.x, targetPose.y, targetPose.theta, longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit, 0.2, 0.2, 0.1, avoidObstacle);

    return NodeStatus::RUNNING;
}
void GoToFreekickPosition::onHalted() {
}

NodeStatus GoToGoalBlockingPosition::tick() {
    
    double distTolerance = getInput<double>("dist_tolerance").value();
    double thetaTolerance = getInput<double>("theta_tolerance").value();
    double distToGoalline = getInput<double>("dist_to_goalline").value();

    auto fd = brain->config->fieldDimensions;
    auto ballPos = brain->data->ball.posToField;
    auto robotPose = brain->data->robotPoseToField;

    string curRole = brain->tree->getEntry<string>("player_role");
    const bool isGk = curRole != "striker";

    // ---- 球速预测（2026-11-16，门将拦截第 1 档）----
    // data->ball 是最近一次检测的位置，本身没有速度：先把球位按估计速度外推到“现在”，
    // 再据此求“过拦截线点”，替掉纯几何投影（静止球两者等价，滚动的球投影永远慢半拍）。
    // 陈旧保护：超过 gk_ball_stale_ms 没有新检测就不再追旧球位（球位记忆里保留的是 1.2 s
    // 前的点），冻结上次目标原地守门。
    const double ballAgeMs = brain->msecsSince(brain->data->ball.timePoint);
    const double staleMs = brain->get_parameter_or("strategy.gk_ball_stale_ms", 500.0);
    const double predMaxAgeMs = brain->get_parameter_or("strategy.gk_ball_pred_max_age_ms", 400.0);
    // 预判门限（2026-11-17）：原硬编码 0.35 m/s / TTI<2 s 会漏掉慢速地滚球。
    // 50 号机日志：t=276 s 球以 1.9 m/s 滚向门，2.44 s 后过线 y=+1.41，门将原地不动 4 s。
    const double predMinSpeed = brain->get_parameter_or("strategy.gk_pred_min_speed", 0.25);
    const double predMaxTTI = brain->get_parameter_or("strategy.gk_pred_max_tti_s", 3.5);
    const bool ballStale = ballAgeMs > staleMs;
    const bool estValid = brain->tree->getEntry<bool>("ball_est_valid") && ballAgeMs <= predMaxAgeMs;
    double bvx = 0.0, bvy = 0.0;
    if (isGk && estValid) {
        bvx = brain->tree->getEntry<double>("ball_vel_x");
        bvy = brain->tree->getEntry<double>("ball_vel_y");
    }

    // 目标时间基准：冻结分支与正常分支都要“消费”本拍，否则解冻后 dt 过大、限幅直接失效。
    const auto nowTarget = brain->get_clock()->now();
    double dtTarget = hasLastTarget_ ? (nowTarget - lastTargetTime_).seconds() : 0.0;
    if (!(dtTarget > 0.0 && dtTarget < 0.3)) dtTarget = 0.0;
    lastTargetTime_ = nowTarget;

    Pose2D targetPose;
    targetPose.x = isGk ? (- fd.length / 2.0 + distToGoalline)
            : (std::max(- fd.length / 2.0 + distToGoalline, ballPos.x - 1.5));

    if (isGk && ballStale && hasLastTarget_) {
        // 球位已过期：拿旧位置重算会把门将一路拽向 2~3 s 前的球，改为冻结上次目标。
        targetPose.x = lastTargetX_;
        targetPose.y = lastTargetY_;
    } else {
        // 球位外推到当前时刻（age 已在上面用 pred_max_age_ms 卡住）；前锋分支保持原样。
        double bx = ballPos.x, by = ballPos.y;
        if (isGk) {
            bx += bvx * (ballAgeMs / 1000.0);
            by += bvy * (ballAgeMs / 1000.0);
        }

        const double xLine = - fd.length / 2.0 + distToGoalline;
        const double ballSpeed = std::hypot(bvx, bvy);
        bool crossValid = false;
        double yCross = 0.0;
        if (isGk && ballSpeed > predMinSpeed && bvx < -0.2 && bx > xLine) {
            const double tCross = (xLine - bx) / bvx;
            if (tCross > 0.0 && tCross < predMaxTTI) {
                yCross = by + bvy * tCross;
                crossValid = true;
            }
        }

        if (isGk) {
            // 站位点 = 球到两门柱射线的角平分线与拦截线 x=xLine 的交点（2026-11-17）。
            // 原“球→门心”射线在侧翼会让近柱一侧空出更多（50 号机实测最坏偏差 1.55 m）；
            // 平分线对两柱是对称的最小最大偏差站位。
            const double pX = -fd.length / 2.0 - bx;   // 球→门线的 x 分量（本方门在 -x）
            const double pLy = fd.goalWidth / 2.0 - by;  // 球→左门柱(+y)
            const double pRy = -fd.goalWidth / 2.0 - by; // 球→右门柱(-y)
            const double nL = std::hypot(pX, pLy);
            const double nR = std::hypot(pX, pRy);
            bool bisectorOk = false;
            double yBisector = 0.0;
            if (bx > xLine && nL > 1e-6 && nR > 1e-6) {
                const double ux = pX / nL + pX / nR;
                const double uy = pLy / nL + pRy / nR;
                const double un = std::hypot(ux, uy);
                if (un > 1e-6 && ux < 0.0) {  // 平分线必须指向本方门
                    const double s = (xLine - bx) / (ux / un);
                    yBisector = by + (uy / un) * s;
                    bisectorOk = std::isfinite(yBisector);
                }
            }
            targetPose.y = bisectorOk ? yBisector
                : (by > 0 ? fd.goalWidth / 4.0 : -fd.goalWidth / 4.0);
            targetPose.y = cap(targetPose.y, fd.penaltyAreaWidth / 2.0, -fd.penaltyAreaWidth / 2.0);
        } else {
            if (bx + fd.length / 2.0 < distToGoalline) {
                targetPose.y = by > 0 ? fd.goalWidth / 2.0 : -fd.goalWidth / 2.0;
            } else {
                targetPose.y = by * distToGoalline / (bx + fd.length / 2.0);
                targetPose.y = cap(targetPose.y, fd.goalWidth / 2.0, -fd.goalWidth / 2.0);
            }
        }
        if (crossValid) {
            // 交点只夹到“门宽 + 0.4 m”：射向门柱外侧的球不追出场；x 锁在拦截线
            targetPose.y = cap(yCross, fd.goalWidth / 2.0 + 0.4, -(fd.goalWidth / 2.0 + 0.4));
            targetPose.x = xLine;
        }
        if (isGk) {
            // 目标点速率限幅（2026-11-17）：定位（odom→field）实测每 10 s 跳一次 >0.3 m
            // （最大 5.38 m），会把目标点整体平移，指令随之反向/空转（实测 cmd vy=-0.40
            // 只走 -0.02，下一拍又 vx=+0.60 回位）。把目标点变化摊成 ≤ gk_target_track_mps
            // 的平滑跟随；它同时限制了“预判过线点”的跳变幅度，与门将横移能力同量级。
            const double trackMps = brain->get_parameter_or("strategy.gk_target_track_mps", 0.6);
            if (hasLastTarget_ && dtTarget > 1e-3 && trackMps > 0.0) {
                const double dMax = trackMps * dtTarget;
                targetPose.y = lastTargetY_ + cap(targetPose.y - lastTargetY_, dMax, -dMax);
            }
            lastTargetX_ = targetPose.x;
            lastTargetY_ = targetPose.y;
            hasLastTarget_ = true;
        }
    }

    double dist = norm(targetPose.x - robotPose.x, targetPose.y - robotPose.y);
    bool atTarget = dist < distTolerance;
    if ( // Considered to have reached the target position
        atTarget
        && fabs(brain->data->ball.yawToRobot) < thetaTolerance
    ) {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    auto targetPose_r = brain->data->field2robot(targetPose);
    double vx = targetPose_r.x;
    double vy = targetPose_r.y;
    double vtheta = yawTrackVtheta(brain, brain->data->ball.yawToRobot);

    if (atTarget) {
        // 位置已到位但尚未面向球：原地转向对准（平移清零，
        // 避免 min_vx/vy 下限把小偏差放大成横向乱飘）
        vx = 0.0;
        vy = 0.0;
    }


    double vxLimit, vyLimit;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    vx = cap(vx, vxLimit, -vxLimit);    
    vy = cap(vy, vyLimit, -vyLimit);    

    // 定位跳变门（2026-11-17）：odom→field 每 10 s 跳一次 >0.3 m（最大 5.38 m），跳变会让
    // “机体系目标误差”单拍突变 0.3~0.9 m，而走路每拍只走 <0.02 m。判为跳变时本拍沿用
    // 上拍指令，不把跳变变成一次反向/空转（50 号机实测 cmd vy=-0.40 实测只走 -0.02，
    // 下一拍又 vx=+0.60 回位）。持续跳变超 0.35 s 视为真校正，正常采纳。
    if (isGk) {
        const double errJump = std::hypot(targetPose_r.x - lastErrX_, targetPose_r.y - lastErrY_);
        // 只在“上一拍刚跑过”时判跳变：节点刚被重新激活（decision 切换）时误差本来就大，
        // 那不是定位跳变，不能沿用旧指令。
        const bool recentCmd = hasLastCmd_ && brain->msecsSince(lastCmdTime_) < 150.0;
        const bool jumpDetected = recentCmd && errJump > 0.25;
        if (jumpDetected) {
            if (!jumping_) {
                jumping_ = true;
                jumpStart_ = brain->get_clock()->now();
            }
            const bool holdExpired = brain->msecsSince(jumpStart_) > 350.0;
            if (!holdExpired) {
                vx = lastCmdVx_;
                vy = lastCmdVy_;
                vtheta = lastCmdVtheta_;
            } else {
                lastErrX_ = targetPose_r.x;
                lastErrY_ = targetPose_r.y;
                lastCmdVx_ = vx;
                lastCmdVy_ = vy;
                lastCmdVtheta_ = vtheta;
                lastCmdTime_ = brain->get_clock()->now();
                jumping_ = false;
            }
        } else {
            jumping_ = false;
            lastErrX_ = targetPose_r.x;
            lastErrY_ = targetPose_r.y;
            lastCmdVx_ = vx;
            lastCmdVy_ = vy;
            lastCmdVtheta_ = vtheta;
            lastCmdTime_ = brain->get_clock()->now();
            hasLastCmd_ = true;
        }
    }
    
    // 门将站位是唯一需要更宽横移的节点：把 XML 的 vy_limit 透传给 setVelocity 覆盖全局
    // robot.vy_limit（0.4），前锋分支仍走全局上限，不影响其他角色。
    // accel 同理：全局 1.2 m/s² 从 0 到 0.6 要 0.5 s，门将改走 gk_accel_vy（默认 2.5）。
    const double gkAccelVy = brain->get_parameter_or("strategy.gk_accel_vy", 2.5);
    brain->client->setVelocity(vx, vy, vtheta, isGk ? vyLimit : -1.0, isGk ? gkAccelVy : -1.0);
    return NodeStatus::SUCCESS;
}

NodeStatus Assist::tick() {
    auto log = [=](string msg) {
        brain->log->debug("Assist", msg);
    };
    log("ticked");

    double distTolerance = getInput<double>("dist_tolerance").value();
    double thetaTolerance = getInput<double>("theta_tolerance").value();
    double distToGoalline = getInput<double>("dist_to_goalline").value();

    auto fd = brain->config->fieldDimensions;
    auto ballPos = brain->data->ball.posToField;
    auto robotPose = brain->data->robotPoseToField;
    string curRole = brain->tree->getEntry<string>("player_role");

    bool isSecondary = false; 
    bool has2Assists = false;
    int selfIdx = brain->config->get_player_id() - 1;
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == selfIdx) continue; 

        auto tmStatus = brain->data->tmStatus[i];
        if (!tmStatus.isAlive) continue; 
        if (tmStatus.isLead) continue; 
        if (tmStatus.role != "striker") continue; 

        has2Assists = true;
        log("2 assists found");
        if (tmStatus.robotPoseToField.x > robotPose.x) {
            log("i am secondary");
            isSecondary = true; 
        }
    }
    log(format("has2Assists: %d, isSecondary: %d", has2Assists, isSecondary));


    Pose2D targetPose;
    // GK 管辖区（球 x < -L/2+penaltyAreaLength+1.0，与 GoalieDecide/StrikerDecide 同一条线）：
    // 球进自家后场后由 GK 负责清球，两个接应前锋就近分工：离本方门更近的一台退守门线前
    // 补门（二次门将），另一台退中场接应位等 GK 清球后上前拦截反击；
    // 保证球被压到禁区时门前始终有人协防，避免全队只留 GK 一人守空门。
    // 原"球前方 2m/4m"公式经 floor 后会在球进后场时塌到门线前与 GK 站位点重合，
    // 接应前锋也挤在门前——这是"都去抢球"画面的另一半来源。
    bool gkZone = ballPos.x < -fd.length / 2.0 + fd.penaltyAreaLength + 1.0;
    if (gkZone) {
        // 门前协防分配（去中心化，两台独立计算须互斥且一致）：
        // 到本方门中心距离最近的前锋守门前；队友明显更近（>0.5m）或距离接近（<0.5m 死区）
        // 但 player_id 更小时让位——死区按 id 裁决，保证两机算出同一结果（恰有一台去门前）。
        bool goFront = true;
        double myGoalDist = norm(robotPose.x + fd.length / 2.0, robotPose.y);
        for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
            if (i == selfIdx) continue;
            auto &tm = brain->data->tmStatus[i];
            if (tm.role != "striker" || !tm.isAlive) continue;
            if (brain->msecsSince(tm.timeLastCom) > 1000) continue; // 失联队友不作为分配依据
            double tmGoalDist = norm(tm.robotPoseToField.x + fd.length / 2.0, tm.robotPoseToField.y);
            if (tmGoalDist < myGoalDist - 0.5) {
                goFront = false; // 队友明显更近，门前让给他
                break;
            }
            if (tmGoalDist <= myGoalDist + 0.5 && i + 1 < brain->config->get_player_id()) {
                goFront = false; // 距离接近时按 player_id 小者优先（两机视角一致）
                break;
            }
        }
        if (goFront) {
            // 门前补位（二次门将）：站门线前 0.6m 的"球—门中心"射线上封门中路，
            // 与 GK 站位点（门线前 1.5m，x=-5.5）前后错开 ~0.9m，不挤占 GK 出击空间。
            targetPose.x = -fd.length / 2.0 + 0.6;
            double frontDenom = max(ballPos.x + fd.length / 2.0, 0.5);
            targetPose.y = ballPos.y * (targetPose.x + fd.length / 2.0) / frontDenom;
            targetPose.y = cap(targetPose.y, fd.goalWidth / 2.0 - 0.2, -fd.goalWidth / 2.0 + 0.2);
        } else {
            targetPose.x = -1.2;
            targetPose.y = (brain->config->get_player_id() % 2 == 0) ? 1.5 : -1.5;
        }
    } else {
        targetPose.x = isSecondary ? ballPos.x - 4.0 : ballPos.x - 2.0;
    targetPose.x = max(targetPose.x, - fd.length / 2.0 + distToGoalline); 
    // 球贴在我方底线附近时（球心未过出界判定线，会长时间停在门线上），
    // 球门中心→球的投影线近乎竖直、分母趋零，接应点 y 会被甩到无穷把机器人带出场边，
    // 这里给分母设下限，并统一把接应点限制在场内（留 0.3 m 边距）
    double assistDenom = max(ballPos.x + fd.length / 2.0, 0.5);
    targetPose.y = ballPos.y * (targetPose.x + fd.length / 2.0) / assistDenom;
    if (has2Assists) { 
        targetPose.y += isSecondary ? - 0.5 : 0.5;
    }
    }
    targetPose.y = cap(targetPose.y, fd.width / 2.0 - 0.3, - fd.width / 2.0 + 0.3);


    double dist = norm(targetPose.x - robotPose.x, targetPose.y - robotPose.y);

    // 队友贴身让位：贴近最近队友时沿背离方向低速退开（门线防守等队友拥有优先权），
    // 让位期间不判定到位，避免双方互相顶住形成僵持（实现在 Brain::calcTeammateSeparationVelocity）
    double sepVx, sepVy, sepVtheta;
    if (brain->calcTeammateSeparationVelocity(sepVx, sepVy, sepVtheta)) {
        brain->client->setVelocity(sepVx, sepVy, sepVtheta);
        return NodeStatus::SUCCESS;
    }
    if ( 
        dist < distTolerance
        && fabs(brain->data->ball.yawToRobot) < thetaTolerance
    ) {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    double vx, vy, vtheta;
    auto targetPose_r = brain->data->field2robot(targetPose);
    double targetDir = atan2(targetPose_r.y, targetPose_r.x);
    double distToObstacle = brain->distToObstacle(targetDir);

    bool avoidObstacle = brain->config->get_avoid_during_chase();
    double oaSafeDist = brain->config->get_chase_ao_safe_dist();

    static rclcpp::Time lastAvoidLogAssist = brain->get_clock()->now();
    const bool avoidLogNowAssist = brain->msecsSince(lastAvoidLogAssist) > 1000;
    if (avoidObstacle && distToObstacle < oaSafeDist) {
        log("avoid obstacle");
        // 同 Chase：绕行角夹在相机 FOV 内，否则会被栅格外的 1e9 “安全”值引入盲区
        const double speed = brain->config->get_avoid_speed();
        const double maxOffset = deg2rad(brain->config->get_avoid_max_offset_deg());
        double avoidDir = 0;
        const bool dirOk = brain->calcSafeAvoidDir(targetDir, oaSafeDist, maxOffset, avoidDir);
        bool quantized = false;
        if (dirOk) {
            vx = speed * cos(avoidDir);
            vy = speed * sin(avoidDir);
            // 同 Chase：微偏转被 min_vy 放大后会变成急转，表达不出来就归零
            if (fabs(vy) < brain->config->get_min_vy()) {
                vy = 0.0;
                vx = speed * 0.7;
                quantized = true;
            }
            vtheta = brain->data->ball.yawToRobot;
        } else {
            // 同 Chase：贴身区停车，稍远半速
            vx = (distToObstacle < brain->config->get_avoid_min_clearance_m()) ? 0.0 : speed * 0.5;
            vy = 0.0;
            vtheta = brain->data->ball.yawToRobot;
        }
        if (avoidLogNowAssist) {
            lastAvoidLogAssist = brain->get_clock()->now();
            brain->log->log("assist/avoid", format("trig=1 targetDir=%.2f dist=%.2f safe=%.2f dir=%.2f ok=%d quant=%d vx=%.2f vy=%.2f",
                targetDir, distToObstacle, oaSafeDist, avoidDir, dirOk ? 1 : 0, quantized ? 1 : 0, vx, vy));
        }
    } else {
        vx = targetPose_r.x;
        vy = targetPose_r.y;
        vtheta = yawTrackVtheta(brain, brain->data->ball.yawToRobot);
    }


    double vxLimit, vyLimit;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    vx = cap(vx, vxLimit, -vxLimit);     
    vy = cap(vy, vyLimit, -vyLimit);     
    
    // 速度包络（2026-11-17）：Assist 是矢量平移（vx/vy 同时非零），按合速度等比例收缩，
    // 方向不变；行进方向用本拍实际指令方向，而不是 targetDir。
    {
        const double mag = norm(vx, vy);
        double capMag = brain->speedEnvelopeCap(atan2(vy, vx), "assist");
        // 到位缓冲：按到站位点的剩余距离再夹一道刹车上限（否则 1.0 m/s 会冲过站位点）
        capMag = min(capMag, brain->brakeSpeed(norm(targetPose_r.x, targetPose_r.y)));
        if (mag > capMag && mag > 1e-6) {
            const double s = capMag / mag;
            vx *= s;
            vy *= s;
        }
    }
    
    brain->client->setVelocity(vx, vy, vtheta);
    return NodeStatus::SUCCESS;
}

NodeStatus Adjust::tick()
{
    auto log = [=](string msg) { 
        brain->log->debug("adjust5", msg); 
    };
    log("enter");
    if (!brain->tree->getEntry<bool>("ball_location_known"))
    {
        return NodeStatus::SUCCESS;
    }

    double turnThreshold, vxLimit, vyLimit, vthetaLimit, range, st_far, st_near, vtheta_factor, NEAR_THRESHOLD;
    getInput("near_threshold", NEAR_THRESHOLD);
    getInput("tangential_speed_far", st_far);
    getInput("tangential_speed_near", st_near);
    getInput("vtheta_factor", vtheta_factor);
    getInput("turn_threshold", turnThreshold);
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    getInput("vtheta_limit", vthetaLimit);
    getInput("range", range);
    double backoffMargin, backoffSpeed;
    getInput("backoff_margin", backoffMargin);
    getInput("backoff_speed", backoffSpeed);
    log(format("ballX: %.1f ballY: %.1f ballYaw: %.1f", brain->data->ball.posToRobot.x, brain->data->ball.posToRobot.y, brain->data->ball.yawToRobot));
    double NO_TURN_THRESHOLD, TURN_FIRST_THRESHOLD;
    getInput("no_turn_threshold", NO_TURN_THRESHOLD);
    getInput("turn_first_threshold", TURN_FIRST_THRESHOLD);


    double vx = 0, vy = 0, vtheta = 0;
    double kickDir = brain->data->kickDir;
    double dir_rb_f = brain->data->robotBallAngleToField; 
    double deltaDir = toPInPI(kickDir - dir_rb_f);
    double ballRange = brain->data->ball.range;
    double ballYaw = brain->data->ball.yawToRobot;
    double st = st_far; 
    double R = ballRange; 
    double r = range;
    // 球位单拍会跳（实测 0.35→0.60 m/0.3 s，远超本体能力），R 与 δ 取 3 拍中值：
    // 对真实运动无感，但能剔掉单拍离群，切断「球位跳变 → sr/st 跳变」的链路。
    {
        static double R1 = -1.0, R2 = -1.0, D1 = 1e9, D2 = 1e9;
        auto med3 = [](double a, double b, double c) {
            return std::max(std::min(a, b), std::min(std::max(a, b), c));
        };
        if (R1 < 0.0) { R1 = R2 = R; D1 = D2 = deltaDir; }
        const double Rf = med3(R, R1, R2);
        const double Df = med3(deltaDir, D1, D2);
        R2 = R1; R1 = R;
        D2 = D1; D1 = deltaDir;
        R = Rf;
        deltaDir = Df;
    }
    // 绕球防蹭：未对准时目标半径抬到 range + backoff_margin，让机器人绕在球外一圈
    // （range=0.38 时腿尖距球仅约 3 cm，沿内圈切向绕行会把球蹭走）；对准后前压到起脚距离。
    // 取值与实测依据：docs/03_design/adjust_align_tuning.md#adjust-align
    constexpr double ALIGN_CONE = 0.24;   // ≈13.7°，须 > StrikerDecide 的 kick_dir_align(0.22)
    // 半径随 |δ| 连续抬升（替掉硬切换）再一阶低通：δ 穿门限时半径阶跃会把 sr 打反，
    // 表现为“球没动、机器人在球前后来回走”。
    constexpr double RTARGET_TAU = 0.10;  // s，仅作噪声滤器；过大会把半径平均在 0.44 卡住前压
    const double alignFrac = cap(fabs(deltaDir) / ALIGN_CONE, 1.0, 0.0);
    const double rTargetRaw = r + backoffMargin * alignFrac;
    static double rTargetSmoothed = -1.0;
    static rclcpp::Time lastRTargetTick{0, 0, RCL_ROS_TIME};
    const auto rTargetNow = brain->get_clock()->now();
    const double rTargetDt = (lastRTargetTick.nanoseconds() > 0)
                                 ? (rTargetNow - lastRTargetTick).seconds() : 0.0;
    lastRTargetTick = rTargetNow;
    if (rTargetSmoothed < 0.0 || rTargetDt < 1e-3 || rTargetDt > 0.5) {
        rTargetSmoothed = rTargetRaw; // 首拍 / 断流 / 长间隔：直接对齐，不做平滑
    } else {
        const double alpha = 1.0 - std::exp(-rTargetDt / RTARGET_TAU);
        rTargetSmoothed += alpha * (rTargetRaw - rTargetSmoothed);
    }
    double srCap = max(0.5, min(1.2, brain->brakeSpeed(max(0.0, R - rTargetSmoothed))));
    double sr = cap(R - rTargetSmoothed, srCap, -backoffSpeed); 
    // 径向死区 5 cm：球位跳变 0.25 m 会直接变成一次满幅后退，观感是“在球前后蹭”。
    // 进入对准锥后保证 0.12 m/s 前压：连续化半径在此处只剩 0.02 m/s，要爬 5 s 才吃掉 10 cm
    // （贴线球分支优先级更高，会把 sr 重新压到 0）。
    if (fabs(R - rTargetSmoothed) < 0.05) sr = 0.0;
    if (fabs(deltaDir) < ALIGN_CONE) sr = max(sr, 0.12);
    log(format("R: %.2f, r: %.2f, rT: %.2f/%.2f, sr: %.2f", R, r, rTargetRaw, rTargetSmoothed, sr));

    // 贴线球保护：沿“机器人→球”推球 1.5 m 内会出线时取消径向靠近，只保留切向绕行，
    // 绕到 kickDir 对齐后交给决策层起脚，避免把摆好的贴线球顶出界。
    {
        auto fd = brain->config->fieldDimensions;
        auto bPos = brain->data->ball.posToField;
        double ux = cos(dir_rb_f), uy = sin(dir_rb_f);
        double tExit = 1e9;
        if (ux > 1e-9)       tExit = min(tExit, (fd.length / 2.0 - bPos.x) / ux);
        else if (ux < -1e-9) tExit = min(tExit, (-fd.length / 2.0 - bPos.x) / ux);
        if (uy > 1e-9)       tExit = min(tExit, (fd.width / 2.0 - bPos.y) / uy);
        else if (uy < -1e-9) tExit = min(tExit, (-fd.width / 2.0 - bPos.y) / uy);
        if (tExit < 1.5) {
            sr = R > 0.6 ? min(sr, 0.3) : 0.;
            log(format("boundary ball: tExit %.2f -> sr %.2f", tExit, sr));
        }
    }

    log(format("deltaDir = %.1f", deltaDir));
    // 切向速度按「弧长 / 期望绕行时间」给：大偏差自然快，小偏差降到 st_near，
    // 不再只看球距（旧法对上 140° 起手要 4.7 s，吃穿 kick_force_msec）。
    // near_threshold 已不参与计算，保留声明仅为 XML 兼容。
    {
        constexpr double ST_ARC_TIME = 1.2;    // s，绕完当前弧长的期望时间
        // 近球上限 0.45：0.50 侧向过多且会蹭球，0.30 太慢（140° 起手要 4.7 s）。
        constexpr double ST_MAX_NEAR = 0.45;   // m/s
        const double arcLen = fabs(deltaDir) * R;
        const double nearFar01 = cap((R - 0.60) / 0.60, 1.0, 0.0);
        const double stCap = ST_MAX_NEAR + (st_far - ST_MAX_NEAR) * nearFar01;
        st = cap(arcLen / ST_ARC_TIME, stCap, st_near);
        // 切向死区：δ 穿零时方向翻转，配合下限会形成 ±0.3 rad 极限环；进死区就停止绕行，
        // 交给径向项把 R 压到 0.38（XML 的 tangential_speed_near 已同步降到 0.08）。
        if (fabs(deltaDir) < 0.08) st = 0.0;
        log(format("st: %.2f (arcLen=%.2f R=%.2f cap=%.2f)", st, arcLen, R, stCap));
    }

    double theta_robot_f = brain->data->robotPoseToField.theta; 
    double thetat_r = dir_rb_f + M_PI / 2 * (deltaDir > 0 ? -1.0 : 1.0) - theta_robot_f; 
    double thetar_r = dir_rb_f - theta_robot_f; 

    vx = st * cos(thetat_r) + sr * cos(thetar_r); 
    vy = st * sin(thetat_r) + sr * sin(thetar_r); 
    {
        // 球方位跟踪：纯 P -> PD（D 作用在 odom 实测角速度上），抑制球静止时的来回摆。
        // kp 仍用 XML 的 vtheta_factor，保持与原纯 P 增益口径一致，只加阻尼不改增益。
        static YawCtrlState stAdjustBall;
        vtheta = yawControl(brain, ballYaw, stAdjustBall, vthetaLimit, vtheta_factor, NO_TURN_THRESHOLD);
    }

    // 球已在正前但机身未对准出球方向时继续转向 kickDir：否则 StrikerDecide 的
    // 机身朝向门限永远不满足（Adjust 只跟球，球正前时 vtheta=0 会卡死在 adjust）。
    // 与跟球分支目标量不同（球方位 vs kickDir），必须用独立状态，否则微分项互相污染。
    {
        static YawCtrlState stAdjustKickDir;
        const double bodyErrAdjust = toPInPI(kickDir - theta_robot_f);
        if (fabs(ballYaw) < 0.35 && fabs(bodyErrAdjust) > 0.15) {
            vtheta = yawControl(brain, bodyErrAdjust, stAdjustKickDir, vthetaLimit, vtheta_factor, 0.0);
        }
    } 
    // 「先转身再平移」只在球明显在侧方且 δ 未对准时驻停：机身转向不改变「机器人→球」
    // 方位，δ 只能靠平移收敛；已对准时冻结平移只是干等（实测白掉 1~2 s）。
    if (
        fabs(ballYaw) > TURN_FIRST_THRESHOLD 
        && fabs(deltaDir) > M_PI / 4
    ) { 
        vx = 0;
        vy = 0;
    }

    // 缓冲（2026-11-17）：按“径向剩余 + 剩余弧长”的总路径限速，防止 1 m/s 档位冲过
    // 0.38 m 的收口点后又要倒车（旧行为只在径向用固定 0.5 上限，提速后不够用）。
    {
        const double pathLen = max(0.0, R - rTargetSmoothed) + fabs(deltaDir) * max(0.0, R);
        const double capMag = brain->brakeSpeed(pathLen);
        const double mag = norm(vx, vy);
        if (mag > capMag && mag > 1e-6) {
            const double s = capMag / mag;
            vx *= s;
            vy *= s;
        }
    }

    vx = cap(vx, vxLimit, -backoffSpeed);   // 下限放开：允许负向径向后退（防蹭不可少）
    vy = cap(vy, vyLimit, -vyLimit);
    vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

    // 队友贴身单侧让位（同 Chase）：只让较差的一台侧向切出。提前量 0.9 m（小于 Chase 的
    // 1.2）：绕球本就要贴到球后 0.4~0.6 m，太早退会让两台都对不准 kickDir。
    {
        double yieldVx, yieldVy, yieldVtheta;
        // heading 用本拍绕球速度方向（机体系 vx/vy + 机身朝角）换算到场系：
        // 绕球是切向运动，只有队友挡在切向弧线上才让，避免“队友在球另一侧”的无谓让位
        const double adjustHeading = brain->data->robotPoseToField.theta + atan2(vy, vx);
        if (brain->calcChaseYieldVelocity(yieldVx, yieldVy, yieldVtheta, 0.8, 1.4, 1.0, adjustHeading)) {
            log(format("yield to teammate, vx: %.2f vy: %.2f", yieldVx, yieldVy));
            brain->client->setVelocity(yieldVx, yieldVy, yieldVtheta);
            return NodeStatus::SUCCESS;
        }
    }
    
    log(format("vx: %.1f vy: %.1f vtheta: %.1f", vx, vy, vtheta));
    brain->client->setVelocity(vx, vy, vtheta);
    return NodeStatus::SUCCESS;
}

NodeStatus CalcKickDir::tick()
{
    double crossThreshold;
    getInput("cross_threshold", crossThreshold);

    string lastKickType = brain->data->kickType;
    if (lastKickType == "cross") crossThreshold += 0.1;

    auto gpAngles = brain->getGoalPostAngles(0.0);
    auto thetal = gpAngles[0]; auto thetar = gpAngles[1];
    auto bPos = brain->data->ball.posToField;
    auto fd = brain->config->fieldDimensions;

    if (thetal - thetar < crossThreshold && brain->data->ball.posToField.x > fd.circleRadius) {
        brain->data->kickType = "cross";
        brain->data->kickDir = atan2(
            - bPos.y,
            fd.length/2 - fd.penaltyDist/2 - bPos.x
        );
    }
    else if (brain->isDefensing()) {
        brain->data->kickType = "block";
        brain->data->kickDir = atan2(
            bPos.y,
            bPos.x + fd.length/2
        );

    } else { 
        brain->data->kickType = "shoot";
        brain->data->kickDir = atan2(
            - bPos.y,
            fd.length/2 - bPos.x
        );
        if (brain->data->ball.posToField.x > brain->config->fieldDimensions.length / 2) brain->data->kickDir = 0; 
    }

    brain->log->log(
        "field/kick_dir",
        format("Kick direction: %.2f rad at ball position (%.2f, %.2f)", 
               brain->data->kickDir, brain->data->ball.posToField.x, brain->data->ball.posToField.y)
    );

    return NodeStatus::SUCCESS;
}

NodeStatus StrikerDecide::tick() {
    auto log = [=](string msg) {
        brain->log->debug("striker_decide", msg);
    };

    double chaseRangeThreshold;
    getInput("chase_threshold", chaseRangeThreshold);
    // 2026-11-06 参数化：原先硬编码在下方 KICK_DIR_ALIGN/KICK_BODY_ALIGN 与本分支条件里，
    // 现场只能重编 brain 才能调；现全部由 XML 传入（默认值保持旧口径，不传则行为不变）。
    double kickDirAlign, kickBodyAlign, kickRangeMin, kickForceMsec;
    getInput("kick_dir_align", kickDirAlign);
    getInput("kick_body_align", kickBodyAlign);
    getInput("kick_range_min", kickRangeMin);
    getInput("kick_force_msec", kickForceMsec);
    double handoffBallSpeedMax, handoffBallStillTimeoutMs, handoffBallSpeedRelaxMax;
    getInput("handoff_ball_speed_max", handoffBallSpeedMax);
    getInput("handoff_ball_still_timeout_ms", handoffBallStillTimeoutMs);
    getInput("handoff_ball_speed_relax_max", handoffBallSpeedRelaxMax);
    string lastDecision, position;
    getInput("decision_in", lastDecision);
    getInput("position", position);

    // --- 视觉踢退出冷却（R2，2026-10-31）--------------------------------
    // 提前退出（ballTooFar / max_msec_kick）后若立刻允许再触发，会出现“两次踢球只隔
    // 2~3 s”的连踢（实测：好回合 1.4~1.6 s 结束后 1.4~1.6 s 又进一次，第二发必坏）。
    // 2026-10-31 二次调：2500 ms 仍拦不住“P3 秒退循环”（brain 侧球 0.5 m、adapter 侧 5.8 m
    // 的球源不一致导致每次交接被秒退，2 s 一圈），抬到 5000 ms；
    // 真正的根治在 vision 的 Ball 选球策略（见交接单），本条仅止血。
    constexpr double VISUAL_KICK_COOLDOWN_MS = 5000.0;
    static bool wasInVisualKick = false;
    static rclcpp::Time lastVisualKickEnd;
    if (brain->data->tmImInVisualKick) {
        wasInVisualKick = true;
    } else if (wasInVisualKick) {
        wasInVisualKick = false;
        lastVisualKickEnd = brain->get_clock()->now();
    }
    const bool visualKickCooling =
        lastVisualKickEnd.nanoseconds() > 0 &&
        brain->msecsSince(lastVisualKickEnd) < VISUAL_KICK_COOLDOWN_MS;

    // --- 交棒前置：球必须是“刚被检测到”，不能是跟踪预测的保持值（R1）---------
    // 实机 14:29 那轮 4 个回合里有 3 个在交棒后 adapter 侧几乎看不见球（可见拍 3%~26%），
    // 模型只能拿零阶保持的旧球位原地迈步 3~5 s（角速度冲到 4.4 rad/s，表现为踢完要摔）。
    // 这里要求最近 200 ms 内每一拍都拿到新鲜检测（ballDetected 且 ballAge<=200 ms），
    // 否则不交棒，让 chase/adjust 继续跟到稳定可见为止。
    constexpr double BALL_FRESH_MS = 200.0;
    static bool ballStaleInit = false;
    static rclcpp::Time lastBallStale;
    if (!ballStaleInit) {
        lastBallStale = brain->get_clock()->now();
        ballStaleInit = true;
    }
    const double ballAgeMs = brain->msecsSince(brain->data->ball.timePoint);
    if (!brain->data->ballDetected || ballAgeMs > BALL_FRESH_MS) {
        lastBallStale = brain->get_clock()->now();
    }
    const bool ballFreshForHandoff =
        brain->msecsSince(lastBallStale) >= BALL_FRESH_MS;

    // --- 球速估计：交棒要求球已基本静止（2026-11-06）-------------------------
    // 现象（实机）：Chase 冲进 <0.45 → 行走踢推球（球获 0.5~0.8 m/s 初速）→ 球刚滚到
    // 0.6~0.9 且方位恰好对着 kickDir → 立刻交棒；切模型要 0.5 s，期间球再滚出 zl 侧
    // t4_kicking 的 KICK_ESCAPE_M=1.2 → 回合 0.97 s 秒退；随后 5 s 冷却只能再推一次，
    // 形成“推一下→踢一下→秒退”的怪异循环。实测：交棒后 1 s 球位移 >1 m 的 5 次里 3 次
    // 秒退，而 <0.5 m 的 26 次里 0 次。t4_kicking 本质是“走向静止球、在球心外 0.35 m
    // 起脚”的定点动作，追着滚动的球交棒必然失败，所以交棒前要求球速低于阈值。
    // 估计方法（修正：2026-11-06 实机回测后）：用 200 ms 窗口的“端点位移 / 窗口时长”，
    // 不用相邻帧差分。相邻帧差分（dt≈20 ms）会把 ±3 cm 的球位检测噪声放大成 1.5 m/s
    // 的假速度——实测日志 ballSpeed 均值 0.78、100% 走 1 s 超时兜底，而同一时刻用
    // 200 ms 端点差分只有中位 0.23 m/s（13/17 次 <0.35），且“交棒后 1 s 球位移”
    // 中位仅 0.11 m，证明球确实静止。零阶保持帧不会入选（窗口取第一个/最后一个
    // 不同位置）。超 5 m/s 的窗口跳过（检测跳变），丢球 >0.5 s 重置。
    static bool ballPosInit = false;
    static double ballRefStampSec = 0.0;
    static double ballRefX = 0.0, ballRefY = 0.0;
    static double ballSpeedEst = 0.0;
    {
        const double stampSec = brain->data->ball.timePoint.seconds();
        // 球丢失超过 0.5 s 时重新初始化，避免拿上回合的陈旧球速去卡本轮交棒。
        if (!brain->data->ballDetected || ballAgeMs > 500.0) {
            ballPosInit = false;
            ballSpeedEst = 0.0;
        }
        if (brain->data->ballDetected) {
            if (!ballPosInit) {
                ballPosInit = true;
                ballRefStampSec = stampSec;
                ballRefX = brain->data->ball.posToField.x;
                ballRefY = brain->data->ball.posToField.y;
                ballSpeedEst = 0.0;
            } else {
                const double dts = stampSec - ballRefStampSec;
                if (dts >= 0.2 && dts < 1.0) {
                    const double disp = hypot(brain->data->ball.posToField.x - ballRefX,
                                              brain->data->ball.posToField.y - ballRefY);
                    const double rawSpeed = disp / dts;
                    if (rawSpeed < 5.0) {   // 剔除检测跳变
                        ballSpeedEst = rawSpeed;
                    }
                    ballRefStampSec = stampSec;
                    ballRefX = brain->data->ball.posToField.x;
                    ballRefY = brain->data->ball.posToField.y;
                }
            }
        }
    }

    // adjust 兜底计时的静态状态（2026-11-06）：声明点提到视觉踢早退之前，
    // 以便在早退路径里清零——否则“上一拍是 adjust、随后交棒”时，视觉踢结束后
    // 首拍会拿着陈旧的 adjustStart 误判超时，直接把行走踢放行。
    static bool wasInAdjust = false;
    static rclcpp::Time adjustStart;

    // 视觉踢进行中：锁存 decision，不再每拍重算。
    // zl 侧 t4_kicking（2026-10-30 版）接管后会自己转身、走向球，球距/球方位角/机身朝向
    // 都会越出下面的触发窗；若继续每拍重算，RLVisionKick 的 _while 会判假把节点中途 halt，
    // 表现为“刚切模型就停”/“踢一半回行走”。退出由 RLVisionKick 自己负责：shouldExit 与
    // onHalted 两条路径都会清 tmImInVisualKick，清掉后这里立刻恢复逐拍决策。
    if (brain->data->tmImInVisualKick) {
        wasInAdjust = false;
        setOutput("decision_out", "auto_visual_kick");
        return NodeStatus::SUCCESS;
    }

    double kickDir = brain->data->kickDir;
    double dir_rb_f = brain->data->robotBallAngleToField; 
    auto ball = brain->data->ball;
    double ballRange = ball.range;
    double ballYaw = ball.yawToRobot;
    double ballX = ball.posToRobot.x;
    double ballY = ball.posToRobot.y;
    
    const double goalpostMargin = 0.3; 
    bool angleGoodForKick = brain->isAngleGood(goalpostMargin, "kick");

    bool avoidPushing = brain->config->get_avoid_during_kick();
    double kickAoSafeDist = brain->config->get_kick_ao_safe_dist();
    bool avoidKick = avoidPushing 
        && brain->data->robotPoseToField.x < brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist;

    double deltaDir = toPInPI(kickDir - dir_rb_f);
    auto now = brain->get_clock()->now();
    // 机身朝向门限（2026-10-29）：行走击球出球方向=机身朝向，Kick 期间只能小幅修正，
    // 机身偏离 kickDir 过大时必须先留在 adjust 转正，否则会把球从身体侧前方推出去
    // （表现为“侧身踢”）。0.26 rad ≈ 15°。2026-11-06 起改由 XML kick_body_align 传入。
    const double KICK_BODY_ALIGN = kickBodyAlign;
    const bool bodyAlignedForKick =
        fabs(toPInPI(kickDir - brain->data->robotPoseToField.theta)) < KICK_BODY_ALIGN;
    // 出球方向门限（2026-10-31）：视觉踢交出去之后，zl 侧 t4_kicking 已删掉“绕到球后 →
    // 到位转向门”的内部对准，只按“走向球、停在球心外 0.35 m、始终面向球”走位并起脚，
    // 出球方向 ≈ 交接瞬间“机器人→球”的方位。所以必须在 brain 侧先把自己摆到球后 kickDir
    // 线上，否则斜着进场就会把球踢偏。判据用 kickDir 而非门柱角，cross/block 两种踢法同样适用。
    // 0.22 rad ≈ 12.6°：比行走击球的 reachedKickDir（0.1）松，便于由 adjust 收敛到位后放行。
    // 注意（2026-10-31 实测回退）：曾试过“≤0.5 m 时放宽到 0.40”，结果放进了一批“球已在
    // 0.32~0.37 m、一交棒就进相机盲区”的回合（adapter 侧可见拍只有 3%~26%，原地迈步 3~5 s）。
    // 近距要的是“站到球后”而不是放松，所以恢复统一 0.22。
    // 2026-11-06：改为由 XML kick_dir_align 传入（默认仍 0.22），现场放宽到 0.28~0.30
    // 不必再重编；但注意放宽直接换成射门偏角：0.30 rad 在 3 m 处横向偏约 0.93 m。
    const double KICK_DIR_ALIGN = kickDirAlign;
    const bool dirAlignedForKick = fabs(deltaDir) < KICK_DIR_ALIGN;

    // 球静止判定（2026-11-06，2026-11-09 补放宽上限）：阈值与超时由 XML 的
    // handoff_ball_speed_max / handoff_ball_still_timeout_ms 控制。
    // 超时存在的目的是“球被对手推着走时不要永远不交棒”；但**无条件**超时放行会交出
    // 正在快速滚动的球（实测 11-09 场 6 次轻踢里有 3 次交棒球速 1.02 / 1.29 / 3.29 m/s
    // 全部是超时放行），模型启动后球已跑远，只能蹭到 0.6~0.9 m 或被 adapter 的
    // KICK_ESCAPE_M=1.2 直接收尾。所以超时后仍需 ballSpeed < handoff_ball_speed_relax_max
    // （0.8 m/s）：超时的作用是“放宽”而不是“取消”门限。
    static bool ballStillWaiting = false;
    static rclcpp::Time ballStillStart;
    const bool ballStill = ballSpeedEst < handoffBallSpeedMax;
    const bool ballSlowEnoughRelaxed = ballSpeedEst < handoffBallSpeedRelaxMax;
    if (!ballStill) {
        if (!ballStillWaiting) {
            ballStillWaiting = true;
            ballStillStart = now;
        }
    } else {
        ballStillWaiting = false;
    }
    const bool ballStillTimedOut =
        ballStillWaiting && brain->msecsSince(ballStillStart) > handoffBallStillTimeoutMs;
    const bool ballStillReady = ballStill || (ballStillTimedOut && ballSlowEnoughRelaxed);

    // --- adjust 兜底计时（2026-11-06）------------------------------------
    // 下面的 kick 分支新增了“球距 ≥ kick_range_min 时不允许行走踢抢跑”的门控：
    // 窗内（0.4~1.0）的拍子必须留给 Adjust 绕位，否则 Kick 一旦 RUNNING 就会冻结
    // 本节点 1~4.8 s，窗口机会全部流失（实测 11-05 第二段 60 个合格拍就是这么被吞的）。
    // 但贴线球/球位抖动可能让 dirAlign 永远收敛不了，所以必须留兜底：连续 adjust
    // 超过 kick_force_msec 就放行行走踢（此时仍要求 bodyAlignedForKick，不会侧身推球）。
    // 状态变量 wasInAdjust / adjustStart 已在视觉踢早退之前声明。
    const bool adjustTimedOut =
        wasInAdjust && brain->msecsSince(adjustStart) > kickForceMsec;

    log(format("ballRange: %.2f, ballYaw: %.2f, ballX:%.2f, ballY: %.2f kickDir: %.2f, dir_rb_f: %.2f, angleGoodForKick: %d, dirAlign: %d, bodyAlign: %d, fresh: %d, cooling: %d",
        ballRange, ballYaw, ballX, ballY, kickDir, dir_rb_f, angleGoodForKick,
        dirAlignedForKick, bodyAlignedForKick, ballFreshForHandoff, visualKickCooling));
    auto dt = brain->msecsSince(timeLastTick);
    bool reachedKickDir = 
        deltaDir * lastDeltaDir <= 0 
        && fabs(deltaDir) < M_PI / 6
        && dt < 100;
    reachedKickDir = reachedKickDir || fabs(deltaDir) < 0.1;
    timeLastTick = now;
    lastDeltaDir = deltaDir;

    string newDecision;
    bool iKnowBallPos = brain->tree->getEntry<bool>("ball_location_known");
    bool tmBallPosReliable = brain->tree->getEntry<bool>("tm_ball_pos_reliable");
    // 球位来源刚在 LOCAL/TM 之间切换（两套场系差 0.7~1.4 m）：短暂禁踢，让位置先稳定。
    // 被拒的回合会落到 adjust 继续绕球，保护窗（默认 150 ms）过后自动恢复。
    const bool ballSrcGuard = brain->tree->getEntry<bool>("ball_src_guard");
    // 贴身可踢预判（复用原 kick 分支条件，仅用于 GK 区特判；isFreekickKickingOff 状态清理仍只在真正进入 kick 分支时执行）
    bool canKickNow =
        (
            (angleGoodForKick && !brain->data->isFreekickKickingOff) 
            || reachedKickDir
        )
        && brain->data->ballDetected
        && fabs(brain->data->ball.yawToRobot) < M_PI / 2.
        && !avoidKick
        && ball.range < 1.5
        && !ballSrcGuard;
    // GK 管辖区（与 GoalieDecide 后场线一致：球 x < -L/2+penaltyAreaLength+1.0，成年场 x<-3.0）：
    // 球进入自家后场后由 GK 负责清球，前锋（含 lead）不再上抢——只在已贴身（range<1.0）可踢时
    // 顺路把球踢出（清球），否则转 assist 退到中场接应，避免"门将和前锋都去追球"把全队吸回门前。
    auto fd = brain->config->fieldDimensions;
    bool ballInGkZone = brain->data->ball.posToField.x < -fd.length / 2.0 + fd.penaltyAreaLength + 1.0;
    if (!(iKnowBallPos || tmBallPosReliable))
    {
        newDecision = "find";
    } else if (
                brain->config->get_enable_auto_visual_kick() &&
                !ballSrcGuard &&
                brain->data->tmImLead && 
                brain->data->tmMyCostRank == 0 && 
                !brain->tree->getEntry<bool>("ball_out") && 
                brain->data->lose_ball == false &&
                brain->data->tmMyCost < 7.0 &&
                // 摆位与朝向必须先对准 kickDir：模型内部不再做绕球/转向门修正，
                // 交接时斜着进来就会斜着出球（未对准时落到 adjust 分支继续绕球摆位）。
                // 另外：球必须刚被检测到（fresh）且不在踢后冷却期（cooling），
                // 否则会交出“盲踢”回合 / 连续两次踢球。
                dirAlignedForKick &&
                bodyAlignedForKick &&
                ballFreshForHandoff &&
                // 球已基本静止才交棒：t4_kicking 追滚动的球必被 zl 侧 ESCAPE 收尾（<1 s 秒退）。
                ballStillReady &&
                !visualKickCooling &&
                brain->data->ball.range < brain->config->get_auto_visual_kick_enable_dist_max() &&
                brain->data->ball.range > brain->config->get_auto_visual_kick_enable_dist_min() &&
                fabs(brain->data->ball.yawToRobot) < brain->config->get_auto_visual_kick_enable_angle() * 1.3 &&
                brain->data->ball.posToField.x > brain->config->fieldDimensions.length / 2 - 14.3 &&
                fabs(brain->data->ball.posToField.y) < 5 &&
                brain->data->robotPoseToField.x > brain->config->fieldDimensions.length / 2 - 14.3 &&
                fabs(brain->data->robotPoseToField.y) < 5 
            ) {
        // GK 管辖区内禁用视觉踢：球在自家后场时前锋让位给 GK，视觉踢（1.5~7s 霸球动作）会把前锋
        // 钉在门前与 GK 抢球；让位后退中场，若已贴身则走下面的 GK 区踢球分支顺路清球
        if (ballInGkZone) {
            newDecision = "assist";
        } else {
            newDecision = "auto_visual_kick";
            brain->data->tmImInVisualKick = true;
        }
    } else if (!brain->data->tmImLead) {
        newDecision = "assist";
    } else if (ballInGkZone) {
        // GK 管辖区：先与 GK 做球权仲裁——读其通信球距，若 GK 明显更近（近 0.3m 以上）则让 GK 清球；
        // 否则已贴身可踢则顺路清球，不然让位退中场接应（不进 chase、不与 GK 抢球）
        bool gkOnBall = false;
        for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
            if (i == brain->config->get_player_id() - 1) continue;
            auto &tm = brain->data->tmStatus[i];
            if (tm.role != "goal_keeper" || !tm.isAlive || !tm.ballDetected) continue;
            if (brain->msecsSince(tm.timeLastCom) > 1000) continue; // 消息过期视为失联，不为其让位
            if (tm.ballRange < ball.range - 0.3) {
                gkOnBall = true;
                break;
            }
        }
        if (canKickNow && ball.range < 1.0 && !gkOnBall) {
            if (brain->data->kickType == "cross") newDecision = "cross";
            else newDecision = "kick";
            brain->data->isFreekickKickingOff = false;
        } else {
            newDecision = "assist";
        }
    } else if (ballRange > chaseRangeThreshold * (lastDecision == "chase" ? 0.9 : 1.0))
    {
        newDecision = "chase";
    } else if (
        (
            (angleGoodForKick && !brain->data->isFreekickKickingOff) 
            || reachedKickDir
        )
        && brain->data->ballDetected
        && fabs(brain->data->ball.yawToRobot) < M_PI / 2.
        && !avoidKick
        && bodyAlignedForKick
        && !ballSrcGuard
        && ball.range < 1.5
        // 2026-11-06：行走踢不得抢跑。Kick 分支原本只要求 bodyAlign + range<1.5，
        // 没有 dirAlign 也没有距离下限，于是“球距还在视觉踢窗口内、只是方位没对上”
        // 的拍子会立刻转 kick；Kick 一 RUNNING 就冻结 StrikerDecide 1~4.8 s，窗口
        // 机会全部流失。现在球距 ≥ kick_range_min 时强制留在 adjust 绕位，直到对准
        // 或兜底超时（kick_force_msec）。kick_range_min=0 时退回旧行为。
        && (ball.range < kickRangeMin || adjustTimedOut)
    ) {
        if (brain->data->kickType == "cross") newDecision = "cross";
        else newDecision = "kick";      
        brain->data->isFreekickKickingOff = false; 
    }
    else
    {
        newDecision = "adjust";
    }

    setOutput("decision_out", newDecision);

    // 兜底放行必须留痕（2026-11-06）：否则现场无法区分“正常贴脸起脚”与“绕位超时
    // 被强制放行”——两者需要完全相反的调参方向（前者正常，后者要考虑抬 kick_force_msec
    // 或改 dirAlign）。必须用 log->log 而非 log->debug，后者不落盘。
    if ((newDecision == "kick" || newDecision == "cross") &&
        adjustTimedOut && ball.range >= kickRangeMin) {
        brain->log->log(
            "striker_decide",
            format("walking kick forced by adjust timeout: ballRange=%.2f adjustMs=%.0f dirAlign=%d bodyAlign=%d thr_dir=%.2f",
                   ball.range, brain->msecsSince(adjustStart),
                   dirAlignedForKick, bodyAlignedForKick, kickDirAlign));
    }

    // adjust 兜底计时：只在决策连续为 adjust 期间累计，进入其它分支立即清零。
    // 必须在 newDecision 定案之后更新，本拍用到的 adjustTimedOut 取自上一拍的状态。
    if (newDecision == "adjust") {
        if (!wasInAdjust) {
            wasInAdjust = true;
            adjustStart = now;
        }
    } else {
        wasInAdjust = false;
    }

    // P4：交棒瞬间打一行 INFO（每次交棒一行）。debug 全关时 striker_decide 看不到，
    // 这行专门用来定位“brain 与 adapter 球源不一致”（实测出现过 brain 侧≤1 m、
    // adapter 侧首帧 3.7~4.7 m，导致模型走很远去追假目标）。
    if (newDecision == "auto_visual_kick" && lastDecision != "auto_visual_kick") {
        brain->log->log(
            "striker_decide",
            format("visual kick handoff: ballRange=%.2f ballAge=%.0fms dirAlign=%d bodyAlign=%d fresh=%d ballSpeed=%.2f still=%d stillReady=%d relaxOK=%d cooling=%d cost=%.1f window=[%.2f,%.2f]",
                   ballRange, ballAgeMs, dirAlignedForKick, bodyAlignedForKick,
                   ballFreshForHandoff, ballSpeedEst, ballStill, ballStillReady,
                   ballSlowEnoughRelaxed, visualKickCooling,
                   brain->data->tmMyCost,
                   brain->config->get_auto_visual_kick_enable_dist_min(),
                   brain->config->get_auto_visual_kick_enable_dist_max()));
    }

    // Publish player_decide message
    brain->visualizer->publishPlayerDecision(format("striker-%s", newDecision.c_str()));
    
    // Publish decision information through visualization_publisher
    auto decision_marker = brain->visualizer->createDecisionInfoMarker(
        "striker",
        newDecision,
        ballRange,
        ballYaw,
        kickDir,
        dir_rb_f,
        angleGoodForKick,
        brain->data->tmImLead,
        "map"
    );
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.push_back(decision_marker);
    brain->visualizer->publishMarkers(marker_array);
    
    return NodeStatus::SUCCESS;
}

NodeStatus GoalieDecide::tick()
{

    double chaseRangeThreshold;
    getInput("chase_threshold", chaseRangeThreshold);
    double closeThreshold;
    getInput("close_threshold", closeThreshold);
    string lastDecision, position;
    getInput("decision_in", lastDecision);

    double kickDir = atan2(brain->data->ball.posToField.y, brain->data->ball.posToField.x + brain->config->fieldDimensions.length / 2);
    double dir_rb_f = brain->data->robotBallAngleToField;
    auto goalPostAngles = brain->getGoalPostAngles(0.3);
    double theta_l = goalPostAngles[0]; 
    double theta_r = goalPostAngles[1]; 
    bool angleIsGood = (dir_rb_f > -M_PI / 2 && dir_rb_f < M_PI / 2);
    double ballRange = brain->data->ball.range;
    double ballYaw = brain->data->ball.yawToRobot;
    // 机体系朝向门限：球相对机头方位角过大时不允许进入 kick（先由 Adjust 转正）。
    // 原 angleIsGood 只看“球在场地图守门员哪一侧”，与机身朝向无关且几乎恒真；
    // 取 1.0 rad（约 57°）与 sim 行走击球 75° 弧留出收敛余量。
    bool facingBallForKick = fabs(ballYaw) < 1.0;

    string newDecision;
    // 出击范围：只有球进入己方罚球区纵深（x < -L/2 + penaltyAreaLength）时才允许离开站位点出击，
    // 其余时段一律 hold 回位。原判据只看“球离我多近”，球停在中场/边路也能把距离压进 3m
    // 把守门员勾出去，形成守门员远离球门追球、球门空门、前锋被压成看戏的问题。
    auto fd = brain->config->fieldDimensions;
    bool ballInDefenseZone = brain->data->ball.posToField.x < -fd.length / 2.0 + fd.penaltyAreaLength;
    // 近距触发门限 close_threshold：球逼到身边时无论防区判据如何都要动（hold 态 0.9 倍滞回）。
    // 取 2.6 m：站位容差 0.4 + 球在禁区线外沿时最坏球距 2.05 m，取 2.0 的 hold 门限会卡十几秒。
    // 还必须要求球在己方后场，否则 GK 被球领出去后距离恒 <2.6 m，会一路追到中线。
    double closeThr = closeThreshold * (lastDecision == "hold" ? 0.9 : 1.0);
    bool ballInCloseZone = brain->data->ball.posToField.x < -fd.length / 2.0 + fd.penaltyAreaLength + 1.0;
    bool ballClose = ballRange <= closeThr && ballInCloseZone;
    bool iKnowBallPos = brain->tree->getEntry<bool>("ball_location_known");
    bool tmBallPosReliable = brain->tree->getEntry<bool>("tm_ball_pos_reliable");
    // 与前锋的球权仲裁（解决“门将和前锋都去抢球”）：取本队最靠近球且消息新鲜的前锋通信球距。
    // 前锋“贴身清球”阈值是自身 ball.range<1.0（StrikerDecide GK 区分支），这里留 0.2m 通信
    // 滞后余量；前锋在处理球（<1.2m）时 GK 让位守门，只有明显更近（0.3m 以上）才接手清球。
    double strikerBallRange = 1e9;
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        if (i == brain->config->get_player_id() - 1) continue;
        auto &tm = brain->data->tmStatus[i];
        if (tm.role != "striker" || !tm.isAlive || !tm.ballDetected) continue;
        if (brain->msecsSince(tm.timeLastCom) > 1000) continue; // 消息过期视为失联，不为其让位
        strikerBallRange = min(strikerBallRange, tm.ballRange);
    }
    bool strikerOnBall = strikerBallRange < 1.2;
    bool gkCloserThanStriker = ballRange < strikerBallRange - 0.3;
    // 球位新鲜度门（2026-11-16，拦截第 1 档）：视觉 30 Hz，球被踢走/滚出视野后 data->ball
    // 仍保留最后一次检测位置（记忆期 1.2 s），拿它进 kick 就是对着旧位置起脚。
    // 超过 gk_ball_stale_ms 没有新检测时一律回 hold 守门，不做 kick / adjust。
    const double ballAgeMs = brain->msecsSince(brain->data->ball.timePoint);
    const double staleMs = brain->get_parameter_or("strategy.gk_ball_stale_ms", 500.0);
    // 只用球龄判新鲜，不看 data->ballDetected：30 Hz 视觉的单帧掉帧不应让门将掉回 hold。
    const bool ballFreshEnough = ballAgeMs <= staleMs;
    const double ballVelX = brain->tree->getEntry<double>("ball_vel_x");
    const double ballVelY = brain->tree->getEntry<double>("ball_vel_y");
    const bool ballVelValid = brain->tree->getEntry<bool>("ball_est_valid");
    if (!(iKnowBallPos || tmBallPosReliable))
    {
        newDecision = "find";
    }
    else if (!ballFreshEnough)
    {
        newDecision = "hold";
    }
    else if (brain->data->ball.posToField.x > 0 - static_cast<double>(lastDecision == "retreat"))
    {
        newDecision = "retreat";
    }
    // 两级出击触发：① 球进罚球区纵深且 ballRange ≤ chase_threshold；② ballRange ≤
    // close_threshold（防区外也出击清球）。滞回与 chase 一致：进 hold 后用 0.9 倍门限退出。
    else if (
        (
            !ballInDefenseZone
            || ballRange > chaseRangeThreshold * (lastDecision == "hold" ? 0.9 : 1.0)
        )
        && !ballClose
    )
    {
        newDecision = "hold";
    }
    else if (strikerOnBall && !gkCloserThanStriker)
    {
        // 前锋正在贴身处理球（其通信球距 < 1.2m）：GK 不出去抢，回站位点守门防打空门，
        // 等前锋把球清出后由中场接应位接手；GK 明显更近（近 0.3m 以上）时才接手清球
        newDecision = "hold";
    }
    else if (angleIsGood && facingBallForKick)
    {
        newDecision = "kick";
    }
    else
    {
        newDecision = "adjust";
    }

    setOutput("decision_out", newDecision);

    // 1 Hz INFO：门将决策不落盘就无法区分“该 hold 却出击”与“正常清球”（2026-11-16）；
    // 2026-11-16 补 age/vel：现场要能分开“没看到球（age 大）”与“看到了但速度上限不够”。
    static rclcpp::Time lastDecideLog = brain->get_clock()->now();
    if (brain->msecsSince(lastDecideLog) > 1000) {
        lastDecideLog = brain->get_clock()->now();
        RCLCPP_INFO(brain->get_logger(),
            "[gk_decide] decision=%s last=%s ball=(%.2f,%.2f) range=%.2f age=%.0fms vel=(%.2f,%.2f) velOK=%d fresh=%d close=%d defZone=%d strikerRange=%.2f iKnow=%d tmReliable=%d",
            newDecision.c_str(), lastDecision.c_str(), brain->data->ball.posToField.x, brain->data->ball.posToField.y,
            ballRange, ballAgeMs, ballVelX, ballVelY, ballVelValid ? 1 : 0, ballFreshEnough ? 1 : 0,
            ballClose ? 1 : 0, ballInDefenseZone ? 1 : 0, strikerBallRange > 1e8 ? -1.0 : strikerBallRange,
            iKnowBallPos ? 1 : 0, tmBallPosReliable ? 1 : 0);
    }

    // Publish player_decide message
    brain->visualizer->publishPlayerDecision(format("goalie-%s", newDecision.c_str()));
    
    // Publish decision information through visualization_publisher
    auto decision_marker = brain->visualizer->createDecisionInfoMarker(
        "goalie",
        newDecision,
        ballRange,
        ballYaw,
        kickDir,
        dir_rb_f,
        angleIsGood,
        false,  // goalie does not need is_lead information
        "map"
    );
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.push_back(decision_marker);
    brain->visualizer->publishMarkers(marker_array);
    
    return NodeStatus::SUCCESS;
}

tuple<double, double, double> Kick::_calcSpeed() {
    double vx, vy, msecKick;


    double vxLimit, vyLimit;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    int minMSecKick;
    getInput("min_msec_kick", minMSecKick);
    double vxFactor = brain->config->get_vx_factor();   
    double yawOffset = brain->config->get_yaw_offset(); 


    double adjustedYaw = brain->data->ball.yawToRobot + yawOffset;
    double tx = cos(adjustedYaw) * brain->data->ball.range; 
    double ty = sin(adjustedYaw) * brain->data->ball.range;

    if (fabs(ty) < 0.01 && fabs(adjustedYaw) < 0.01)
    { 
        vx = vxLimit;
        vy = 0.0;
    }
    else
    { 
        vy = ty > 0 ? vyLimit : -vyLimit;
        vx = vy / ty * tx * vxFactor;
        if (fabs(vx) > vxLimit)
        {
            vy *= vxLimit / vx;
            vx = vxLimit;
        }
    }


    double speed = norm(vx, vy);
    msecKick = speed > 1e-5 ? minMSecKick + static_cast<int>(brain->data->ball.range / speed * 1000) : minMSecKick;
    
    return make_tuple(vx, vy, msecKick);
}

double Kick::_alignVtheta(double ballYaw)
{
    // 行走击球的出球方向 = 机身朝向，而 Kick 期间原实现冻结 vtheta（纯平移），
    // 会把球从身体侧前方推出去，表现为“侧身踢”。这里补两层：
    //   1) 球不在正前锥内：先转球（与守门员分支同一控制律）；
    //   2) 球在正前锥内：让机身对准本次出球方向 kickDir（场地系），边推边转正。
    const double BALL_FRONT_CONE = 0.35;  // rad ≈ 20°
    const double VT_LIMIT = 0.8;          // rad/s
    if (fabs(ballYaw) > BALL_FRONT_CONE) {
        return cap(ballYaw * 1.5, VT_LIMIT, -VT_LIMIT);
    }
    const double bodyErr = toPInPI(brain->data->kickDir - brain->data->robotPoseToField.theta);
    return cap(bodyErr * 1.5, VT_LIMIT, -VT_LIMIT);
}

NodeStatus Kick::onStart()
{
    _minRange = brain->data->ball.range;
    _speed = 0.5;
    _startTime = brain->get_clock()->now();


    bool avoidPushing = brain->config->get_avoid_during_kick();
    double kickAoSafeDist = brain->config->get_kick_ao_safe_dist();
    string role = brain->tree->getEntry<string>("player_role");
    if (
        avoidPushing
        && (role != "goal_keeper")
        && brain->data->robotPoseToField.x < brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist
    ) {
        brain->client->setVelocity(-0.1, 0, 0);
        return NodeStatus::SUCCESS;
    }

    // Publish movement command
    double angle = brain->data->ball.yawToRobot;
    if (role == "goal_keeper") {
        // 守门员：推进中保持机身转向球（行走击球出球方向=机身朝向，
        // 只平移不转向会造成侧身横移蹭球、出球方向失控）
        double vtheta = cap(angle * 1.5, 0.8, -0.8);
        brain->client->crabWalk(angle, _speed, vtheta);
    } else {
        brain->client->crabWalk(angle, _speed, _alignVtheta(angle));
    }
    return NodeStatus::RUNNING;
}

NodeStatus Kick::onRunning()
{
    bool enableAbort = brain->config->get_abort_kick_when_ball_moved();
    auto ballRange = brain->data->ball.range;
    const double ballAgeMs = brain->msecsSince(brain->data->ball.timePoint);
    // 行走踢本身就是“推着球走”，球被推开 0.3 m 属于正常推进，原阈值会把
    // 一次连续推球切成多段“点刹”（实测 58 次 abort 中 42 次由它触发，
    // 11 次间隔 <1 s）。放宽到 0.6 m；真丢球仍由下面的球龄判据兜底。
    const double MOVE_RANGE_THRESHOLD = 0.6;
    // 近距推球阶段（球在脚下 <0.6 m）：相机低头常贴限位、球被自身遮挡或滑出画面，
    // 视觉短暂断流是常态。该区间把“新鲜”窗口从 300 ms 放宽到 500 ms，
    // 避免把短暂看不见误判成丢球（实测这里是 Kick abort 的主要来源）。
    // 远端球仍用 kBallFreshMs：那里的断流通常意味着球真的被踢走。
    const double NEAR_BALL_RANGE_M = 0.6;
    const double NEAR_BALL_FRESH_MS = 500.0;
    const double ballFreshMs = (ballRange < NEAR_BALL_RANGE_M) ? NEAR_BALL_FRESH_MS : Brain::kBallFreshMs;
    const bool ballLost = ballAgeMs > ballFreshMs;
    const bool ballMoved = brain->data->ballDetected && ballRange - _minRange > MOVE_RANGE_THRESHOLD;
    if (
        enableAbort 
        && (
            ballMoved
            || ballLost 
        )
    ) {
        // 两类中止分开记：kicked away 是正常结束（球被推/踢出脚下），
        // lost 才是需要关注的“看不见球”。日志分流便于统计。
        if (ballMoved) {
            brain->log->warn("Kick", format("ball kicked away (range %.2f -> %.2f, moved %.2fm), end kick",
                                            _minRange, ballRange, ballRange - _minRange));
        } else {
            brain->log->warn("Kick", format("ball lost (age=%.0fms > %.0fms, range=%.2f), abort kick",
                                            ballAgeMs, ballFreshMs, ballRange));
        }
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }


    if (ballRange < _minRange) _minRange = ballRange;    

    
    bool avoidPushing = brain->config->get_avoid_during_kick();
    double kickAoSafeDist = brain->config->get_kick_ao_safe_dist();
    if (
        avoidPushing
        && brain->data->robotPoseToField.x < brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist
    ) {
        brain->client->setVelocity(-0.1, 0, 0);
        return NodeStatus::SUCCESS;
    }


    double msecs = getInput<double>("min_msec_kick").value();
    double speed = getInput<double>("speed_limit").value();
    msecs = msecs + brain->data->ball.range / speed * 1000;
    if (brain->msecsSince(_startTime) > msecs) { 
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }


    if (brain->data->ballDetected && ballAgeMs <= ballFreshMs) {
        double angle = brain->data->ball.yawToRobot;
        double speed = getInput<double>("speed_limit").value();
        _speed += 0.1; 
        speed = min(speed, _speed);
        if (brain->tree->getEntry<string>("player_role") == "goal_keeper") {
            // 守门员：推进中逐 tick 跟踪球方位保持转向（同上）
            double vtheta = cap(angle * 1.5, 0.8, -0.8);
            brain->client->crabWalk(angle, speed, vtheta);
        } else {
            double vtheta = _alignVtheta(angle);
            static uint64_t align_log_cnt = 0;
            if (++align_log_cnt % 10 == 0) {
                const double bodyErr =
                    toPInPI(brain->data->kickDir - brain->data->robotPoseToField.theta);
                brain->log->log("Kick", format("align bodyErr=%.2f ballYaw=%.2f vtheta=%.2f range=%.2f",
                                               bodyErr, angle, vtheta, ballRange));
            }
            brain->client->crabWalk(angle, speed, vtheta);
        }
    } else if (ballAgeMs > ballFreshMs) {
        // 球已看不见（被踢走 / 被腿遮挡超过当前窗口：近距 500 ms、远端 300 ms）：原实现没有 else，
        // 会沿用上一条 crabWalk 速度继续推空，表现为“以为球还在脚下继续顶”。
        brain->log->warn("Kick", format("ball lost (age=%.0fms), stop push kick", ballAgeMs));
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    return NodeStatus::RUNNING;
}

void Kick::onHalted()
{
    _startTime -= rclcpp::Duration(100, 0);
}


// Static variable definition
rclcpp::Time RLVisionKick::_lastExitTime = rclcpp::Time(0, 0, RCL_ROS_TIME);

NodeStatus RLVisionKick::onStart()
{
    _startTime = brain->get_clock()->now();
    _isDecelerating = false;
    _visionKickStarted = false;
    _pendingRobocupWalk = false;
    
    // Start deceleration
    startDecelerate(500.0);
    stepDecelerate();
    
    return NodeStatus::RUNNING;
}

NodeStatus RLVisionKick::onRunning()
{
    // Check exit flag
    if (brain->data->shouldExitRLVisionKick) {
        brain->data->shouldExitRLVisionKick = false;
        brain->data->tmImInVisualKick = false;
        recordExitTime();
        return NodeStatus::SUCCESS;
    }
    
    // Handle deceleration phase
    if (_isDecelerating) {
        stepDecelerate();
        
        if (!_isDecelerating) {
            // Deceleration completed
            if (_pendingRobocupWalk) {
                brain->client->robocupWalk();
                _pendingRobocupWalk = false;
                // 防御：任何返回 SUCCESS 的出口都不得留下“视觉踢进行中”标志。
                brain->data->tmImInVisualKick = false;
                return NodeStatus::SUCCESS;
            } else if (!_visionKickStarted) {
                // Start vision kick after deceleration
                if (brain->client->RLVisionKick() != 0) {
                    // 位姿未就绪时必须结束本次启动尝试，不能标记成已经在踢球。
                    brain->data->tmImInVisualKick = false;
                    recordExitTime();
                    return NodeStatus::FAILURE;
                }
                _headScanStartTime = brain->get_clock()->now();
                _visionKickStarted = true;
            }
        }
        return NodeStatus::RUNNING;
    }
    
    if (_visionKickStarted) {
        double headMsec = brain->msecsSince(_headScanStartTime);
        if (headMsec < 300.0) {
            brain->client->moveHead(0.4, 0.0);
        } else if (headMsec < 550.0) {
            brain->client->moveHead(0.7, 0.0);
        }
    }

    // Check exit conditions
    double elapsed = brain->msecsSince(_startTime);
    double minMsecKick = getInput<double>("min_msec_kick").value();
    double maxMsecKick = getInput<double>("max_msec_kick").value();
    
    // Check if ball is too far or cost is too high
    // 退出条件里的“球已飞远”阀值：视觉踢把球踢到 2 m 以外就立即交还决策层。
    // 原值 5.0 m 在实机几乎不可达（典型踢出 3~4 m），会一直挂到 max_msec_kick 超时，
    // 表现为“踢完之后站着发呆再突然走”。
    bool ballTooFar = brain->data->ballDetected && brain->data->ball.range > 2.0;
    // 超时兜底（XML max_msec_kick）：踢球动作播完（约 5.7s）后 rl 侧已自动切回行走模型，
    // 若球没被踢远、视野也没丢，前面的退出条件会一直不满足——前锋会站着不动。
    // 到点强制退出走 robocupWalk()，由决策树重新评估下一拍。
    bool kickTimeout = (maxMsecKick > 0.0) && (elapsed > maxMsecKick);
    bool shouldExit = kickTimeout || (((ballTooFar || brain->data->tmMyCost > 8.0) && (elapsed > minMsecKick)) || brain->data->lose_ball || brain->tree->getEntry<bool>("ball_out"));
    
    if (shouldExit) {
        recordExitTime();
        // 必须在这里清 tmImInVisualKick：本分支是“本地判定回合结束”的主路径，
        // 若只靠 onHalted 清，就会跟 StrikerDecide 的 decision 锁存互锁——
        // 锁存让 decision 一直等于 auto_visual_kick → 节点返回 SUCCESS 后又被重新 tick，
        // 每次都重发 kVisualKick（实机复现：每 3.5 s 一次、模型/模式反复切换）。
        brain->data->tmImInVisualKick = false;
        startDecelerate(500.0);
        _pendingRobocupWalk = true;
        stepDecelerate();
        return NodeStatus::RUNNING;
    }

    return NodeStatus::RUNNING;
}

void RLVisionKick::onHalted()
{
    brain->data->tmImInVisualKick = false;
    brain->client->setVelocity(0.0, 0.0, 0.0);
    brain->client->robocupWalk();
    recordExitTime();
    
    _isDecelerating = false;
    _visionKickStarted = false;
    _pendingRobocupWalk = false;
}

bool RLVisionKick::isMinIntervalSatisfied(double minIntervalMsec)
{
    // This is a static method, so we can't access brain instance
    // For now, always return true. If needed, pass brain pointer as parameter
    return true;
}

void RLVisionKick::recordExitTime()
{
    _lastExitTime = brain->get_clock()->now();
}

void RLVisionKick::startDecelerate(double durationMs)
{
    if (_isDecelerating) {
        return;
    }
    
    _isDecelerating = true;
    _decelStartTime = brain->get_clock()->now();
    _decelDurationMs = durationMs;
}

bool RLVisionKick::stepDecelerate()
{
    if (!_isDecelerating) {
        return true;
    }
    
    double elapsed = brain->msecsSince(_decelStartTime);
    brain->client->setVelocity(0.0, 0.0, 0.0);
    
    if (elapsed >= _decelDurationMs) {
        _isDecelerating = false;
        return true;
    }
    
    return false;
}

NodeStatus StandStill::onStart()
{

    _startTime = brain->get_clock()->now();


    brain->client->setVelocity(0, 0, 0);
    return NodeStatus::RUNNING;
}

NodeStatus StandStill::onRunning()
{
    double msecs;
    getInput("msecs", msecs);
    if (brain->msecsSince(_startTime) < msecs) {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::RUNNING;
    }


    return NodeStatus::SUCCESS;
}

void StandStill::onHalted()
{
    double msecs;
    getInput("msecs", msecs);
    _startTime -= rclcpp::Duration(- 2 * msecs, 0);
}


NodeStatus RobotFindBall::onStart()
{
    if (brain->data->ballDetected)
    {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }
    _turnDir = brain->data->ball.yawToRobot > 0 ? 1.0 : -1.0;

    return NodeStatus::RUNNING;
}

NodeStatus RobotFindBall::onRunning()
{
    if (brain->data->ballDetected)
    {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    double vyawLimit;
    getInput("vyaw_limit", vyawLimit);

    brain->client->setVelocity(0, 0, vyawLimit * _turnDir);
    return NodeStatus::RUNNING;
}

void RobotFindBall::onHalted()
{
    _turnDir = 1.0;
}

NodeStatus CamFastScan::onStart()
{
    _cmdIndex = 0;
    _timeLastCmd = brain->get_clock()->now();
    brain->client->moveHead(_cmdSequence[_cmdIndex][0], _cmdSequence[_cmdIndex][1]);
    return NodeStatus::RUNNING;
}

NodeStatus CamFastScan::onRunning()
{
    double interval = getInput<double>("msecs_interval").value();
    if (brain->msecsSince(_timeLastCmd) < interval) return NodeStatus::RUNNING;

    // else 
    // 序列点数与 _cmdSequence 自动同步（点数为 N 时最后一个索引是 N-1）
    constexpr int kCmdCount = sizeof(_cmdSequence) / sizeof(_cmdSequence[0]);
    if (_cmdIndex >= kCmdCount - 1) return NodeStatus::SUCCESS;

    // else
    _cmdIndex++;
    _timeLastCmd = brain->get_clock()->now();
    brain->client->moveHead(_cmdSequence[_cmdIndex][0], _cmdSequence[_cmdIndex][1]);
    return NodeStatus::RUNNING;
}

NodeStatus TurnOnSpot::onStart()
{
    _timeStart = brain->get_clock()->now();
    _lastAngle = brain->data->robotPoseToOdom.theta;
    _cumAngle = 0.0;

    bool towardsBall = false;
    _angle = getInput<double>("rad").value();
    getInput("towards_ball", towardsBall);
    if (towardsBall) {
        // 原来用 bbox 中心判方向，但失球时 bbox 已被清零 → ballPixX=0 →
        // 永远朝同一侧转，表现为原地反复打转。改为：
        //   1) 5 s 内看到过球 → 朝最后球方位转；
        //   2) 本帧有检测但无有效方位 → 用 bbox 中心；
        //   3) 完全未知 → 左右交替，避免总是同向。
        double turnSign = 1.0;
        const double ballAgeSec = brain->msecsSince(brain->data->ball.timePoint) / 1000.0;
        if (ballAgeSec <= 5.0 && std::isfinite(brain->data->ball.yawToRobot)
            && fabs(brain->data->ball.yawToRobot) > 0.15) {
            turnSign = brain->data->ball.yawToRobot > 0 ? 1.0 : -1.0;
        } else if (brain->data->ballDetected) {
            double ballPixX = (brain->data->ball.boundingBox.xmin + brain->data->ball.boundingBox.xmax) / 2;
            turnSign = ballPixX < brain->config->cameraImageWidth / 2 ? 1.0 : -1.0;
        } else {
            static int flip = 0;
            turnSign = (flip++ % 2 == 0) ? 1.0 : -1.0;
        }
        _angle = fabs(_angle) * turnSign;
    }

    brain->client->setVelocity(0, 0, _angle);
    return NodeStatus::RUNNING;
}

NodeStatus TurnOnSpot::onRunning()
{
    double curAngle = brain->data->robotPoseToOdom.theta;
    double deltaAngle = toPInPI(curAngle - _lastAngle);
    _lastAngle = curAngle;
    _cumAngle += deltaAngle;
    double turnTime = brain->msecsSince(_timeStart);
    if (
        fabs(_cumAngle) - fabs(_angle) > -0.1
        || turnTime > _msecLimit
    ) {
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    // else 
    brain->client->setVelocity(0, 0, (_angle - _cumAngle)*2);
    return NodeStatus::RUNNING;
}

NodeStatus MoveToPoseOnField::tick()
{

    double tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit, xTolerance, yTolerance, thetaTolerance;
    getInput("x", tx);
    getInput("y", ty);
    getInput("theta", ttheta);
    getInput("long_range_threshold", longRangeThreshold);
    getInput("turn_threshold", turnThreshold);
    getInput("vx_limit", vxLimit);
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    getInput("vtheta_limit", vthetaLimit);
    getInput("x_tolerance", xTolerance);
    getInput("y_tolerance", yTolerance);
    getInput("theta_tolerance", thetaTolerance);
    bool avoidObstacle;
    getInput("avoid_obstacle", avoidObstacle);

    brain->client->moveToPoseOnField2(tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit, xTolerance, yTolerance, thetaTolerance, avoidObstacle);
    return NodeStatus::SUCCESS;
}

NodeStatus GoToReadyPosition::tick()
{
    double distTolerance, thetaTolerance;
    getInput("dist_tolerance", distTolerance);
    getInput("theta_tolerance", thetaTolerance);
    string role = brain->tree->getEntry<string>("player_role");
    bool isKickoff = brain->tree->getEntry<bool>("gc_is_kickoff_side");
    auto fd = brain->config->fieldDimensions;

    // default values, override with different conditions
    double tx = 0, ty = 0, ttheta = 0; 
    double longRangeThreshold = 1.0;
    double turnThreshold = 0.4;
    double vxLimit, vyLimit;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    if (brain->distToBorder() > - 1.0) { // near border
        vxLimit = 0.6;
        vyLimit = 0.4;
    }
    double vthetaLimit = 1.5;
    bool avoidObstacle = true;

    if (role == "striker") {
        if (brain->data->myStrikerIDRank == 0) {
            tx = isKickoff ? - fd.circleRadius - 0.5 : - fd.circleRadius * 2;
            ty = 0.0;
        } else if (brain->data->myStrikerIDRank == 1) {
            tx = isKickoff ? - fd.circleRadius - 0.5 : - fd.circleRadius * 2;
            ty = -1.5;
        } else if (brain->data->myStrikerIDRank == 2) {
            tx = - fd.length / 2.0 + fd.penaltyAreaLength;
            ty = fd.circleRadius / 2.0;
        } else if (brain->data->myStrikerIDRank == 3) {
            tx = - fd.length / 2.0 + fd.penaltyDist;
            ty = - fd.circleRadius / 2.0;
        }
    } else if (role == "goal_keeper") {
        tx = -fd.length / 2.0 + fd.goalAreaLength;
        ty = 0;
        ttheta = 0;
    }

    brain->client->moveToPoseOnField2(tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit, distTolerance / 1.5, distTolerance / 1.5, thetaTolerance, avoidObstacle);
    return NodeStatus::SUCCESS;
}

NodeStatus GoBackInField::tick()
{
    auto log = [=](string msg) {
        brain->log->debug("GoBackInField", msg);
    };
    log("GoBackInField ticked");

    double valve;
    getInput("valve", valve);
    double vx = 0; 
    double vy = 0; 
    double dir = 0;
    auto fd = brain->config->fieldDimensions;
    if (brain->data->robotPoseToField.x > fd.length / 2.0 - valve) dir = - M_PI;
    else if (brain->data->robotPoseToField.x < - fd.length / 2.0 + valve) dir = 0;
    else if (brain->data->robotPoseToField.y > fd.width / 2.0 + valve) dir = - M_PI / 2.0;
    else if (brain->data->robotPoseToField.y < - fd.width / 2.0 - valve) dir = M_PI / 2.0;
    else { 
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    
    double dir_r = toPInPI(dir - brain->data->robotPoseToField.theta);
    vx = 0.4 * cos(dir_r);
    vy = 0.4 * sin(dir_r);
    brain->client->setVelocity(vx, vy, 0);
    return NodeStatus::SUCCESS;
}

NodeStatus WaveHand::tick()
{
    string action;
    getInput("action", action);
    if (action == "start")
        brain->client->waveHand(true);
    else
        brain->client->waveHand(false);
    return NodeStatus::SUCCESS;
}

NodeStatus MoveHead::tick()
{
    double pitch, yaw;
    getInput("pitch", pitch);
    getInput("yaw", yaw);
    brain->client->moveHead(pitch, yaw);
    return NodeStatus::SUCCESS;
}

NodeStatus CheckAndStandUp::tick()
{
    if (brain->tree->getEntry<bool>("gc_is_under_penalty") || brain->data->currentRobotModeIndex == 2) {
        brain->data->recoveryPerformedRetryCount = 0;
        brain->data->recoveryPerformed = false;
        brain->log->debug("recovery", "reset recovery");
        return NodeStatus::SUCCESS;
    }
    brain->log->debug("recovery", format("Recovery retry count: %d, recoveryPerformed: %d recoveryState: %d currentRobotModeIndex: %d", brain->data->recoveryPerformedRetryCount, brain->data->recoveryPerformed, brain->data->recoveryState, brain->data->currentRobotModeIndex));

    if (!brain->data->recoveryPerformed &&
        brain->data->recoveryState == RobotRecoveryState::HAS_FALLEN &&
        brain->data->currentRobotModeIndex == 1 && 
        brain->data->recoveryPerformedRetryCount < brain->config->get_retry_max_count()) {
        brain->data->shouldExitRLVisionKick = true;
        brain->client->standUp();
        brain->data->recoveryPerformed = true;
        brain->log->debug("recovery", format("Recovery retry count: %d", brain->data->recoveryPerformedRetryCount));
        return NodeStatus::SUCCESS;
    }

    if (brain->data->recoveryPerformed && brain->data->currentRobotModeIndex == 10) {
        brain->data->recoveryPerformedRetryCount +=1;
        brain->data->recoveryPerformed = false;
        brain->log->debug("recovery", format("Add retry count: %d", brain->data->recoveryPerformedRetryCount));
    }


    if (brain->data->recoveryState == RobotRecoveryState::IS_READY &&
        (brain->data->currentRobotModeIndex == 8 || brain->data->currentRobotModeIndex == 20)) { 
        brain->data->recoveryPerformedRetryCount = 0;
        brain->data->recoveryPerformed = false;
        brain->data->shouldExitRLVisionKick = false;
        brain->log->debug("recovery", "Reset recovery, recoveryState: " + to_string(static_cast<int>(brain->data->recoveryState)));
    }

    return NodeStatus::SUCCESS;
}


NodeStatus CalibrateOdom::tick()
{
    double x, y, theta;
    getInput("x", x);
    getInput("y", y);
    getInput("theta", theta);

    brain->calibrateOdom(x, y, theta);
    return NodeStatus::SUCCESS;
}

NodeStatus PrintMsg::tick()
{
    Expected<std::string> msg = getInput<std::string>("msg");
    if (!msg)
    {
        throw RuntimeError("missing required input [msg]: ", msg.error());
    }
    std::cout << "[MSG] " << msg.value() << std::endl;
    return NodeStatus::SUCCESS;
}

