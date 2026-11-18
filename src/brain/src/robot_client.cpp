#include <cmath>

#include "brain.h"
#include "robot_client.h"
#include "utils/math.h"
#include "utils/print.h"
#include "utils/misc.h"

void RobotClient::init(const std::string & robot_name)
{
    // zl 侧话题使用契约原名（实机一机一图，不带前后缀）；robot_name 只影响
    // brain 自身的 /soccer/* 话题，这里仅做一次提示。
    if (!robot_name.empty())
    {
        brain->log->warn("robot_client",
                         format("robot_name='%s' 仅作用于 brain 自身话题；zl 侧 I/O 使用契约原名",
                                robot_name.c_str()));
    }
    zl = std::make_unique<robot_io::ZlIo>(brain);
    zl->init();

    // 头部仲裁参数（brain.cpp 中 declare_parameter，config.yaml 可覆盖）。
    // stale：节点被 halt 后不再 tick，多久算“不再请求”（200 ms ≈ 20 拍，覆盖 tick 抖动）；
    // dedup：与上一条已下发目标的死区，小于它不重发（板端每条消息都会重设 0.5 s 插值斜坡）。
    _headStaleMs = brain->get_parameter_or<double>("robot.head_arb_stale_ms", 200.0);
    _headDedupRad = brain->get_parameter_or<double>("robot.head_arb_dedup_rad", 0.01);

    // 原先由 bridge 工作区的两个独立节点承担，现收进 brain 进程。
    kick_targets = std::make_unique<robot_io::KickTargets>(brain);
    kick_targets->init();
    ball_fwd = std::make_unique<robot_io::BallFwd>(brain);
    ball_fwd->init();
}

void RobotClient::tick()
{
    zl->tick();

    // 真实球门位姿合成（原 goal_pose_synth）：用 Brain 场地系位姿 + IMU 姿态，50Hz 输出。
    kick_targets->update(
        brain->data->robotPoseToField.x, brain->data->robotPoseToField.y,
        brain->data->robotPoseToField.theta, zl->lastOdomSec(), zl->navSnapshot(),
        brain->now().seconds());
}

bool RobotClient::kickTargetsActive() const
{
    return kick_targets && kick_targets->active(brain->now().seconds());
}

int RobotClient::moveHead(double pitch, double yaw)
{
    // 保留旧签名给 MoveHead 节点：行为树显式指定的姿态优先级最高，且允许立即接管。
    return requestHead(HeadSource::Explicit, pitch, yaw, 0.0);
}

int RobotClient::requestHead(HeadSource src, double pitch, double yaw, double minHoldMs)
{
    // 上游（头部位姿/图像偏差）偶发 NaN/Inf：直接丢弃该帧，保持上一帧头部姿态。
    if (!std::isfinite(pitch) || !std::isfinite(yaw))
    {
        brain->log->warn("move_head", format("非有限值已丢弃: pitch=%.2f yaw=%.2f", pitch, yaw));
        return -1;
    }

    const int idx = static_cast<int>(src);
    if (idx <= 0 || idx >= kHeadSourceCount)
    {
        brain->log->warn("move_head", format("非法头部来源 %d，已丢弃", idx));
        return -1;
    }

    HeadRequest &req = _headReq[idx];
    req.valid = true;
    req.pitch = pitch;
    req.yaw = yaw;
    req.stamp = brain->get_clock()->now();
    req.minHoldMs = minHoldMs;
    return 0;
}

void RobotClient::flushHead()
{
    const auto now = brain->get_clock()->now();

    // 1) 作废过期请求：节点被 _while 判假/halt 后不再 tick，不能继续占着出口。
    //    一次遍历同时挑出最高优先级（枚举值最大）的存活来源。
    int best = static_cast<int>(HeadSource::None);
    for (int i = 1; i < kHeadSourceCount; ++i)
    {
        if (!_headReq[i].valid)
            continue;
        if (brain->msecsSince(_headReq[i].stamp) > _headStaleMs)
        {
            _headReq[i].valid = false;
            continue;
        }
        if (i > best)
            best = i;
    }
    if (best == static_cast<int>(HeadSource::None))
    {
        _headOwner = 0;  // 无人请求：保持上一姿态（与旧行为一致，不主动回中）
        return;
    }

    // 2) 抢占规则：高优先级立即抢；同级/更低必须等当前持有者驻留期满且它已不再请求。
    //    这条是本轮修复的核心——两套 6 点扫描同拍请求时也只有一个在动，不会 550/700 ms 交替。
    if (_headOwner != 0 && best != _headOwner)
    {
        const bool ownerAlive = _headReq[_headOwner].valid;
        const bool byPriority = best > _headOwner;
        const bool afterHold =
            !ownerAlive && brain->msecsSince(_headOwnerSince) >= _headOwnerMinHoldMs;
        if (!byPriority && !afterHold)
        {
            if (!ownerAlive)
                return;  // 老持有者已不请求但驻留期未满：本拍不下发
            best = _headOwner;
        }
    }

    const HeadRequest &r = _headReq[best];
    if (best != _headOwner)
    {
        // 切换留痕（debug 级，仅在换手时打一条）：现场用它区分“谁在动头”。
        brain->log->debug("move_head", format("来源切换 %d -> %d (minHold %.0fms)",
                                             _headOwner, best, r.minHoldMs));
        _headOwner = best;
        _headOwnerSince = now;
    }
    _headOwnerMinHoldMs = r.minHoldMs;

    // 3) 限幅（与旧 moveHead 完全一致，保证实机行为不变）
    const double rawPitch = r.pitch, rawYaw = r.yaw;  // 限幅前的原始指令
    double yaw = cap(r.yaw, brain->config->get_head_yaw_limit_left(),
                     brain->config->get_head_yaw_limit_right());
    // pitch 是「相机光轴下俯角」（zl 侧按 head_pitch_offset_deg 换算成关节角）。
    // 双向限幅到新头硬件的物理可达区间（约 0.20~1.06 rad = 11°~61° 下俯）：
    // 更小会被机械限位挡住并让 head_pose 与实际不符，更大同样会顶到限位。
    double pitch = cap(r.pitch, brain->config->get_head_pitch_limit_down(),
                       brain->config->get_head_pitch_limit_up());

    // 4) 去重：与上一条已下发目标几乎相同就不重发。板端 external_node 对每条消息都
    //    重设 0.5 s 线性插值斜坡（起点=当前实测位置），重复发同一目标会让斜坡永远走不完。
    if (_headPubValid && std::fabs(pitch - _headPubPitch) < _headDedupRad &&
        std::fabs(yaw - _headPubYaw) < _headDedupRad)
    {
        return;
    }

    brain->log->debug("move_head", format("src=%d pitch: %.2f -> %.2f, yaw: %.2f -> %.2f",
                                          best, rawPitch, pitch, rawYaw, yaw));

    zl->sendHead(pitch, yaw);
    _headPubPitch = pitch;
    _headPubYaw = yaw;
    _headPubValid = true;
}

int RobotClient::standUp()
{
    // zl 侧无独立 "getup"：起身/恢复统一走行走序列（damp -> reinit -> ready -> walk）。
    zl->requestWalk();
    return 0;
}

int RobotClient::RLVisionKick(bool start)
{
    if (!start)
    {
        // 回合结束由 rl_manager 的 active_model 回切自动处理，无需显式退出。
        return 0;
    }
    if (!zl->odometryAllowsMotion())
    {
        RCLCPP_WARN_THROTTLE(brain->get_logger(), *brain->get_clock(), 1000,
                             "consistent 里程计未就绪或超时，暂不启动踢球");
        return -1;
    }
    zl->startVisualKick();
    return 0;
}

int RobotClient::robocupWalk()
{
    zl->requestWalk();
    return 0;
}

int RobotClient::enterDamping()
{
    zl->requestDamp();
    return 0;
}

int RobotClient::waveHand(bool doWaveHand)
{
    static rclcpp::Time lastWarn = brain->now();
    (void)doWaveHand;
    if ((brain->now() - lastWarn).seconds() > 5.0)
    {
        lastWarn = brain->now();
        brain->log->warn("wave_hand", "zl 人形无手部自由度，忽略挥手指令");
    }
    return 0;
}

int RobotClient::setVelocity(double x, double y, double theta, double vyLimitOverride,
                           double accelVyOverride)
{
    // 非有限速度按 0 处理后继续下发：既不把 null 送进链路，也避免沿用上一帧的危险速度。
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(theta))
    {
        brain->log->warn("RobotClient/setVelocity_in",
                         format("非有限速度已置零: vx=%.2f vy=%.2f vtheta=%.2f", x, y, theta));
        if (!std::isfinite(x)) x = 0;
        if (!std::isfinite(y)) y = 0;
        if (!std::isfinite(theta)) theta = 0;
    }

    const double rawX = x, rawY = y, rawTheta = theta; // 限幅/最小速度补偿前的原始指令
    if (!zl->odometryAllowsMotion())
    {
        // 在最小速度补偿和命令记忆前归零；仍允许零速请求完成行走模式初始化，避免预热死锁。
        x = y = theta = 0.0;
        RCLCPP_WARN_THROTTLE(brain->get_logger(), *brain->get_clock(), 1000,
                             "consistent 里程计未就绪或超时，本拍速度归零");
    }
    brain->log->debug("RobotClient/setVelocity_in",
                      format("vx: %.2f  vy: %.2f  vtheta: %.2f", rawX, rawY, rawTheta));

    // speed_scale：3588 侧取消速度映射缩减后本体按下发值 1:1 执行（旧版实际约折半）。
    // 缩放必须放在 min_v 抬升与限幅之前，使各门限仍按“本体侧单位”解释；<1.0 仅作回退开关。
    double speedScale = brain->config->get_speed_scale();
    if (!std::isfinite(speedScale) || speedScale <= 0.0) speedScale = 1.0;
    if (speedScale != 1.0)
    {
        x *= speedScale;
        y *= speedScale;
        theta *= speedScale;
    }

    // 低速指令抬升：本体判据是“合速度模 < 0.1 m/s 不响应”，不是逐轴 <0.1。按合模等比例
    // 放大、方向不变；后退（vx<0）不放大——逐轴抬升会把可忽略的后退分量放大成持续倒退。
    // vtheta 单位不同仍单独抬升；min_vx/min_vy 保留供行为树判断“某分量是否可表达”。
    const double minLin = brain->config->get_min_lin_speed();
    const double mintheta = brain->config->get_min_vtheta();
    const double linMag = std::hypot(x, y);
    if (minLin > 0.0 && linMag > 1e-5 && linMag < minLin && x >= 0.0)
    {
        const double scale = minLin / linMag; // 保持方向，只放大模长
        x *= scale;
        y *= scale;
    }
    if (fabs(theta) < mintheta && fabs(theta) > 1e-5)
        theta = theta > 0 ? mintheta : -mintheta;
    x = cap(x, brain->config->get_vx_limit(), -brain->config->get_vx_limit());
    // y 轴上限：默认走 config 的 robot.vy_limit（前锋/Assist 共享）。门将站位是唯一需要
    // 更宽横移的场景（射门时需 0.6~1.0 m/s 才能封住两柱），由调用方显式传 override，
    // 且仍夹一道硬上限，避免 XML 写错让本体失稳；override<=全局值时等于不生效。
    constexpr double kCmdVyHardMax = 0.80;
    double vyCap = brain->config->get_vy_limit();
    if (vyLimitOverride > 0.0 && vyLimitOverride > vyCap)
        vyCap = vyLimitOverride < kCmdVyHardMax ? vyLimitOverride : kCmdVyHardMax;
    y = cap(y, vyCap, -vyCap);
    theta = cap(theta, brain->config->get_vtheta_limit(), -brain->config->get_vtheta_limit());

    // 后退单独上限：vx_limit 正负对称，各节点都能要求 1.0 m/s 退避，而双足后退重心不稳。
    // 在出口统一夹一道（夹在对称 cap 之后、加速度限幅之前）；<=0 或 >= vx_limit 等于关闭。
    {
        const double vxBackward = brain->config->get_vx_limit_backward();
        if (vxBackward > 0.0 && x < -vxBackward) x = -vxBackward;
    }

    // 加速度（斜率）限幅：各阶段各自打补丁太分散，改在下发出口统一限幅，所有节点自动受益，
    // 阶跃指令被摊成 0.4~0.8 s 斜坡。它不消除“方向反复”本身；板端 rl_manager 另有 ≈2.5 m/s²
    // 平滑，本层须更小才有增量意义。0 或负值关闭，三轴可独立回退。
    const double accelVx = brain->config->get_accel_limit_vx();
    // 门将专项：全局 1.2 m/s² 从 0 到 0.6 要 0.5 s，在 1 s 级扑救窗口里占一半时间，
    // 因此允许调用方把 y 轴斜率提到本体自身平滑值（≈2.5 m/s²），不改全局值。
    const double accelVy = (accelVyOverride > 0.0) ? accelVyOverride : brain->config->get_accel_limit_vy();
    const double accelVtheta = brain->config->get_accel_limit_vtheta();
    // 刹车（|指令| 变小或换向）走 decel_limit_*：加速受步态稳定限制，刹车只受摩擦/重心
    // 限制，二者共用一个值会让高速档刹不住（2.2 m/s 按 3.5 减速需 0.63 s / 0.7 m）。
    // <=0 回退到对应加速度值；且刹车永不低于加速度（否则等于没有刹车增益）。
    const double decelVx = max(brain->config->get_decel_limit_vx(), accelVx);
    const double decelVy = max(brain->config->get_decel_limit_vy(), accelVy);
    const double decelVtheta = max(brain->config->get_decel_limit_vtheta(), accelVtheta);
    const auto nowCmd = brain->get_clock()->now();
    const double dtCmd = (_lastCmdTime.nanoseconds() > 0)
                             ? (nowCmd - _lastCmdTime).seconds()
                             : -1.0;
    // 单向斜率限幅：制动（绝对值变小或换向）用 down，其余用 up。
    auto slew = [](double target, double current, double up, double down, double dt) {
        const bool braking = fabs(target) < fabs(current) || target * current < 0.0;
        const double rate = braking ? down : up;
        if (rate <= 0.0) return target; // 未配置 = 不限速
        return cap(target, current + rate * dt, current - rate * dt);
    };
    // dt 异常（首拍 / tick 抖动 / 断流）时不限速，直接对齐，避免卡住
    if (dtCmd > 1e-3 && dtCmd < 0.2)
    {
        x = slew(x, _vx, accelVx, decelVx, dtCmd);
        y = slew(y, _vy, accelVy, decelVy, dtCmd);
        theta = slew(theta, _vtheta, accelVtheta, decelVtheta, dtCmd);
    }
    _vx = x; _vy = y; _vtheta = theta; // remember last command. Can be approximately used as the current robot's velocity.
    _lastCmdTime = nowCmd;
    if (fabs(_vx) > 1e-3 || fabs(_vy) > 1e-3 || fabs(_vtheta) > 1e-3) _lastNonZeroCmdTime = brain->get_clock()->now();

    // 行为树每拍都会下发一次速度，逐拍打 INFO 会刷屏并拖快日志轮转。
    // robot.cmd_log_interval_ms：>0 按该间隔节流（默认 1000ms），0 逐拍全量输出（专项排查），
    // <0 关闭；静止<->运动切换时总是立即输出一行，便于事后定位起步/骤停时刻。
    const int logIntervalMs = brain->config->get_cmd_log_interval_ms();
    if (logIntervalMs >= 0)
    {
        const bool moving = fabs(x) > 1e-3 || fabs(y) > 1e-3 || fabs(theta) > 1e-3;
        const bool lastMoving = fabs(_lastLoggedVx) > 1e-3 || fabs(_lastLoggedVy) > 1e-3 || fabs(_lastLoggedVtheta) > 1e-3;
        const bool stateChanged = !_hasLoggedCmd || (moving != lastMoving);
        const bool intervalElapsed = _hasLoggedCmd
            && (brain->get_clock()->now() - _lastCmdLogTime).nanoseconds() >=
               static_cast<int64_t>(logIntervalMs) * 1000000;
        if (logIntervalMs == 0 || stateChanged || intervalElapsed)
        {
            string msg = format("vx: %.2f  vy: %.2f  vtheta: %.2f", x, y, theta);
            // 只有限幅/最小速度补偿真的改了下发值时才附带原始值，避免多打一行
            if (fabs(x - rawX) > 1e-3 || fabs(y - rawY) > 1e-3 || fabs(theta - rawTheta) > 1e-3)
            {
                msg += format("  (cmd vx: %.2f  vy: %.2f  vtheta: %.2f)", rawX, rawY, rawTheta);
            }
            brain->log->log("RobotClient/setVelocity_out", msg);
            _lastLoggedVx = x;
            _lastLoggedVy = y;
            _lastLoggedVtheta = theta;
            _lastCmdLogTime = brain->get_clock()->now();
            _hasLoggedCmd = true;
        }
    }

    zl->sendTwist(x, y, theta);
    return 0;
}

int RobotClient::crabWalk(double angle, double speed) {
    return crabWalk(angle, speed, 0);
}

int RobotClient::crabWalk(double angle, double speed, double vtheta) {
    double vxFactor = brain->config->get_vx_factor();   // Used to adjust vx, vx *= vxFactor, to compensate for the deviation between the speed parameters in the x and y directions and the actual speed ratio, making the movement direction accurate
    double yawOffset = brain->config->get_yaw_offset(); // Used to compensate for the deviation in the localization angle
    double vxLimit = brain->config->get_vx_limit();
    double vyLimit = brain->config->get_vy_limit();

    // Calculate velocity command
    double cmdAngle = angle + yawOffset;
    double vx = cos(cmdAngle) * speed * vxFactor;
    double vy = sin(cmdAngle) * speed;

    if (fabs(vy) > vyLimit) {
        vx *= vyLimit / fabs(vy);
        vy = vyLimit * fabs(vy) / vy;
    }

    return setVelocity(vx, vy, vtheta);
}

bool RobotClient::isStandingStill(double timeBuffer) {
    return fabs(_vx) < 1e-3 && fabs(_vy) < 1e-3 && fabs(_vtheta) < 1e-3 && brain->msecsSince(_lastNonZeroCmdTime) > timeBuffer;
}

void RobotClient::checkStuck() {
    constexpr double CMD_MIN = 0.30;  // 指令合速度门限（避开 min_lin_speed 抬升的 0.12）
    constexpr double MOVE_MIN = 0.05; // 窗口内应有位移速度，低于此值算“没走”
    constexpr double WIN_SEC = 1.0;

    static rclcpp::Time winStart = brain->get_clock()->now();
    static Pose2D winPose = brain->data->robotPoseToField;
    static double stuckMs = 0.0;
    static rclcpp::Time lastLog = brain->get_clock()->now();

    const auto now = brain->get_clock()->now();
    const double dt = (now - winStart).seconds();
    // 窗口未满或时间跳变过长时重开窗口
    if (dt < WIN_SEC || dt > 5.0) {
        winStart = now;
        winPose = brain->data->robotPoseToField;
        return;
    }

    const auto p = brain->data->robotPoseToField;
    const double rate = norm(p.x - winPose.x, p.y - winPose.y) / dt;
    const double cmd = std::hypot(_vx, _vy);
    winStart = now;
    winPose = p;

    if (cmd >= CMD_MIN && rate < MOVE_MIN) stuckMs += dt * 1000.0;
    else stuckMs = 0.0;

    if (stuckMs >= 1000.0 && brain->msecsSince(lastLog) > 1000) {
        lastLog = now;
        RCLCPP_WARN(brain->get_logger(), "[stuck] cmd=%.2f m/s 实测 %.3f m/s 持续 %.1f s，机体未跟上指令（接触/被顶住/打滑）",
                    cmd, rate, stuckMs / 1000.0);
    }
}

int RobotClient::moveToPoseOnField2(double tx, double ty, double ttheta, double longRangeThreshold, double turnThreshold, double vxLimit, double vyLimit, double vthetaLimit, double xTolerance, double yTolerance, double thetaTolerance, bool avoidObstacle) {
    static string mode = "longRange";
    static bool isBacking = false;
    static rclcpp::Time timeEndAvoid = brain->get_clock()->now();
    static double avoidDir = 1.0; // 1.0 for turn left, -1.0 for turn right
    const double SAFE_DIST = brain->config->get_safe_distance();
    const double AVOID_SECS = brain->config->get_avoid_secs();

    double range = norm(tx - brain->data->robotPoseToField.x, ty - brain->data->robotPoseToField.y);
    if (range > longRangeThreshold * (mode == "longRange" ? 0.9 : 1.0)) {
        mode = "longRange";
    } else {
        mode = "shortRange";
    }

    double vx, vy, vtheta;
    double tarDir = atan2(ty - brain->data->robotPoseToField.y, tx - brain->data->robotPoseToField.x);
    double faceDir = brain->data->robotPoseToField.theta;
    double tarDir_r = toPInPI(tarDir - faceDir);
    auto now = brain->get_clock()->now();
    if (mode == "longRange") {

        if (fabs(tarDir_r) > turnThreshold) {
            vx = 0.0;
            vy = 0.0;
            vtheta = tarDir_r;
        } else {
            // 长距段：用刹车模型按剩余距离限速（旧式 cap(range) 是 1 m/s per meter，
            // 1.4 m/s 档位会在目标点前冲过去）
            vx = min(vxLimit, brain->brakeSpeed(max(0.0, range - xTolerance)));
            vy = 0.0;
            vtheta = tarDir_r;
        }

        if (avoidObstacle) {
            if (now < timeEndAvoid) {
                double distToObstacle = brain->distToObstacle(0);
                if (distToObstacle < SAFE_DIST / 2.0) {
                    isBacking = true;
                    timeEndAvoid = now + rclcpp::Duration(AVOID_SECS, 0.0);
                    avoidDir = brain->calcAvoidDir(tarDir_r, SAFE_DIST) > 0 ? 1.0 : -1.0;
                    vx = -0.2;
                    vy = avoidDir * 0.2;
                    vtheta = 0.0;
                }else if (brain->distToObstacle(0) < SAFE_DIST + (isBacking ? 0.5 : 0.0)) {
                    isBacking = false;
                    timeEndAvoid = now + rclcpp::Duration(AVOID_SECS, 0.0);
                    avoidDir = brain->calcAvoidDir(tarDir_r, SAFE_DIST) > 0 ? 1.0 : -1.0;
                    vx = 0.0;
                    vy = 0.0;
                    vtheta = avoidDir * brain->config->get_vtheta_limit();
                } else {
                    vx = vxLimit;
                    if (brain->distToObstacle(tarDir_r) < SAFE_DIST * 2) vxLimit *= 0.5;
                    vy = 0.0;
                    vtheta = 0.0;
                }
            } else {
                double distToObstacle = brain->distToObstacle(tarDir_r);
                // 旧式“障碍进入 2×SAFE_DIST 就一律降到 0.5”在 safe_distance 抬到 3.0 后
                // 等于 6 m 内全部半速；改为按清距的刹车模型降速（仍保留 0.5 的硬下限）。
                if (distToObstacle < SAFE_DIST * 2) {
                    vxLimit = min(vxLimit, max(0.5, brain->brakeSpeed(distToObstacle - 0.35)));
                }
                if (distToObstacle < SAFE_DIST) {
                    timeEndAvoid = now + rclcpp::Duration(AVOID_SECS, 0.0);
                    avoidDir = brain->calcAvoidDir(tarDir_r, SAFE_DIST) > 0 ? 1.0 : -1.0;

                    vx = 0.0;
                    vy = 0.0;
                    vtheta = 0.0;
                }
            }

        }
    } else if (mode == "shortRange") {

        vx = range * cos(tarDir_r);
        vy = range * sin(tarDir_r);
        // 短距段同样是 1 m/s per meter，用刹车模型封顶，避免近距冲过站位点
        {
            const double speedCap = min(vxLimit, brain->brakeSpeed(range));
            const double mag = norm(vx, vy);
            if (mag > speedCap && mag > 1e-6) {
                const double s = speedCap / mag;
                vx *= s;
                vy *= s;
            }
        }
        vtheta = toPInPI(ttheta - faceDir);
        if (fabs(vx) < xTolerance && fabs(vy) < yTolerance && fabs(vtheta) < thetaTolerance) {
            vx = 0.0;
            vy = 0.0;
            vtheta = 0.0;
        }


        if (avoidObstacle) {
            double distToObstacle = brain->distToObstacle(tarDir_r);

            if (distToObstacle < SAFE_DIST) {
                vx = 0.0;
                vy = 0.0;
                vtheta = tarDir_r;
            }
        }
    }

    vx = cap(vx, vxLimit, -vxLimit);
    vy = cap(vy, vyLimit, -vyLimit);
    vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);
    return setVelocity(vx, vy, vtheta);
}

int RobotClient::moveToPoseOnField3(double tx, double ty, double ttheta, double longRangeThreshold, double turnThreshold, double vxLimit, double vyLimit, double vthetaLimit, double xTolerance, double yTolerance, double thetaTolerance, bool avoidObstacle) {
    static string mode = "longRange";
    static bool should_avoid_in_move = false;
    bool isFreekickStartPlacing = brain->isFreekickStartPlacing();

    const double SAFE_DIST = isFreekickStartPlacing ? brain->config->get_freekick_start_placing_safe_distance() : brain->config->get_safe_distance();
    const double AVOID_SECS = isFreekickStartPlacing ? brain->config->get_freekick_start_placing_avoid_secs() : brain->config->get_avoid_secs();

    double range = norm(tx - brain->data->robotPoseToField.x, ty - brain->data->robotPoseToField.y);
    if (range > longRangeThreshold * (mode == "longRange" ? 0.9 : 1.0)) {
        mode = "longRange";
    } else {
        mode = "shortRange";
    }

    double vx, vy, vtheta;
    double tarDir = atan2(ty - brain->data->robotPoseToField.y, tx - brain->data->robotPoseToField.x);
    double faceDir = brain->data->robotPoseToField.theta;
    double tarDir_r = toPInPI(tarDir - faceDir);
    if (mode == "longRange") {
        if (fabs(tarDir_r) > turnThreshold) {
            vx = 0.0;
            vy = 0.0;
            vtheta = tarDir_r;
        } else {
            // 长距段：用刹车模型按剩余距离限速（旧式 cap(range) 是 1 m/s per meter，
            // 1.4 m/s 档位会在目标点前冲过去）
            vx = min(vxLimit, brain->brakeSpeed(max(0.0, range - xTolerance)));
            vy = 0.0;
            vtheta = tarDir_r;
        }

        if (avoidObstacle) {
            double mostViableDir = 0;
            double distToObstacleMostViable = brain->distToObstacle(0);
            for (size_t i = 1; i <= 10; i++)
            {
                double tarDir_r_temp = toPInPI(0 + M_PI / 10 * i);
                double distToObstacle_temp = brain->distToObstacle(tarDir_r_temp);
                if (distToObstacle_temp > SAFE_DIST) {
                    mostViableDir = tarDir_r_temp;
                    break;
                }

                tarDir_r_temp = toPInPI(0 - M_PI / 10 * i);
                distToObstacle_temp = brain->distToObstacle(tarDir_r_temp);
                if (distToObstacle_temp > SAFE_DIST) {
                    mostViableDir = tarDir_r_temp;
                    break;
                }
            }
            if (should_avoid_in_move) {
                double distToObstacle = brain->distToObstacle(0);
                vx = 0;
                vy = (mostViableDir - tarDir_r > 0 ? 0.3 : -0.3);
                vtheta = 0.0;
                if (distToObstacle < SAFE_DIST) {
                    should_avoid_in_move = true;
                } else {
                    vx = vxLimit;
                    if (brain->distToObstacle(tarDir_r) < SAFE_DIST * 2) vxLimit *= 0.5;
                    vy = 0.0;
                    vtheta = 0.0;
                    should_avoid_in_move = false;
                }
            } else {
                double distToObstacle = brain->distToObstacle(tarDir_r);
                if (distToObstacle < SAFE_DIST * 3) vxLimit = 0.5;
                if (distToObstacle < SAFE_DIST) {
                    should_avoid_in_move = true;

                    vx = 0.0;
                    vy = 0.0;
                    vtheta = 0.0;
                }
            }

        }
    } else if (mode == "shortRange") {

        vx = range * cos(tarDir_r);
        vy = range * sin(tarDir_r);
        // 短距段同样是 1 m/s per meter，用刹车模型封顶，避免近距冲过站位点
        {
            const double speedCap = min(vxLimit, brain->brakeSpeed(range));
            const double mag = norm(vx, vy);
            if (mag > speedCap && mag > 1e-6) {
                const double s = speedCap / mag;
                vx *= s;
                vy *= s;
            }
        }
        vtheta = toPInPI(ttheta - faceDir);
        if (fabs(vx) < xTolerance && fabs(vy) < yTolerance && fabs(vtheta) < thetaTolerance) {
            vx = 0.0;
            vy = 0.0;
            vtheta = 0.0;
        }


        if (avoidObstacle) {
            double distToObstacle = brain->distToObstacle(tarDir_r);

            double mostViableDir = tarDir_r;
            double distToObstacleMostViable = distToObstacle;
            for (size_t i = 1; i <= 10; i++)
            {
                double tarDir_r_temp = toPInPI(tarDir_r + M_PI / 10 * i);
                double distToObstacle_temp = brain->distToObstacle(tarDir_r_temp);
                if (distToObstacle_temp > SAFE_DIST) {
                    mostViableDir = tarDir_r_temp;
                    break;
                }

                tarDir_r_temp = toPInPI(tarDir_r - M_PI / 10 * i);
                distToObstacle_temp = brain->distToObstacle(tarDir_r_temp);
                if (distToObstacle_temp > SAFE_DIST) {
                    mostViableDir = tarDir_r_temp;
                    break;
                }
            }

            if (distToObstacle < SAFE_DIST) {
                vx = 0.0;
                vy = mostViableDir == tarDir_r ? 0.0 : (mostViableDir - tarDir_r > 0 ? 0.2 : -0.2);
                vtheta = tarDir_r;
            }
        }
    }

    vx = cap(vx, vxLimit, -vxLimit);
    vy = cap(vy, vyLimit, -vyLimit);
    vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);
    return setVelocity(vx, vy, vtheta);
}

double RobotClient::msecsToCollide(double vx, double vy, double vtheta, double maxTime) {
    auto robots = brain->data->getRobots();

    if (robots.size() == 0) return maxTime;
    if (vx < 1e-3 && vy < 1e-3) return maxTime;

    // simplify the calculation by not considering vtheta for now. This is a
    // conservative estimation.
    double x0 = brain->data->robotPoseToField.x;
    double y0 = brain->data->robotPoseToField.y;
    double robotTheta = brain->data->robotPoseToField.theta;

    Line path = {
        x0, y0,
        x0 + vx * cos(robotTheta) * maxTime / 1000, y0 + vy * sin(robotTheta) * maxTime / 1000
    };

    double minDist = 1e6;
    const double SAFE_DIST = 0.6;
    for (int i = 0; i < robots.size(); i++) {
        auto pose = robots[i].posToField;
        Point2D p = {pose.x, pose.y};
        double distToPath = pointMinDistToLine(p, path);
        brain->log->debug("msecsToCollide", format(
            "i = %d, rbt at (%.2f, %.2f), obstacle at (%.2f, %.2f), distToPath = %.2f",
            i, x0, y0, p.x, p.y, distToPath
        ));
        if (distToPath < SAFE_DIST) {
            Line line = {x0, y0, pose.x, pose.y};
            double angle = angleBetweenLines(line, path);
            double l = norm(pose.x - x0, pose.y - y0);
            double d = l*cos(angle) - sqrt(SAFE_DIST*SAFE_DIST - l*l*sin(angle)*sin(angle));
            brain->log->debug("msecsToCollide", format(
                "angle: %.2f, l: %.2f, d: %.2f",
                angle, l, d
            ));
            if (d < minDist) minDist = d;
        }
    }

    return min(maxTime, minDist / norm(vx, vy) * 1000);
}
