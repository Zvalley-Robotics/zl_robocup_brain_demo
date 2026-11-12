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

int RobotClient::moveHead(double pitch, double yaw)
{
    // 上游（头部位姿/图像偏差）偶发 NaN/Inf：直接丢弃该帧，保持上一帧头部姿态。
    if (!std::isfinite(pitch) || !std::isfinite(yaw))
    {
        brain->log->warn("move_head", format("非有限值已丢弃: pitch=%.2f yaw=%.2f", pitch, yaw));
        return -1;
    }

    const double rawPitch = pitch, rawYaw = yaw; // 限幅前的原始指令
    yaw = cap(yaw, brain->config->get_head_yaw_limit_left(), brain->config->get_head_yaw_limit_right());
    // pitch 是「相机光轴下俯角」（zl 侧按 head_pitch_offset_deg 换算成关节角）。
    // 双向限幅到新头硬件的物理可达区间（约 0.20~1.06 rad = 11°~61° 下俯）：
    // 更小会被机械限位挡住并让 head_pose 与实际不符，更大同样会顶到限位。
    pitch = cap(pitch, brain->config->get_head_pitch_limit_down(), brain->config->get_head_pitch_limit_up());

    brain->log->debug("move_head", format("pitch: %.2f -> %.2f, yaw: %.2f -> %.2f", rawPitch, pitch, rawYaw, yaw));

    zl->sendHead(pitch, yaw);
    return 0;
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

int RobotClient::setVelocity(double x, double y, double theta)
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

    // 2026-11-05：3588 侧取消速度映射缩减后，本体按下发值 1:1 执行（旧版本实际约折半）。
    // 缩放必须放在 min_v 抬升与限幅之前，使 min_vx/vy/vtheta 与 vx/vy/vtheta_limit 仍按
    // “本体侧单位”解释：speed_scale=0.5 时 0.2 的意图值仍抬到 0.3（本体 0.15），与旧手感一致。
    // 默认 1.0 = 保持新真实速度；小于 1.0 仅作快速回退开关。
    double speedScale = brain->config->get_speed_scale();
    if (!std::isfinite(speedScale) || speedScale <= 0.0) speedScale = 1.0;
    if (speedScale != 1.0)
    {
        x *= speedScale;
        y *= speedScale;
        theta *= speedScale;
    }

    double minx = brain->config->get_min_vx();
    double miny = brain->config->get_min_vy();
    double mintheta = brain->config->get_min_vtheta();
    if (fabs(x) < minx && fabs(x) > 1e-5)
        x = x > 0 ? minx : -minx;
    if (fabs(y) < miny && fabs(y) > 1e-5)
        y = y > 0 ? miny : -miny;
    if (fabs(theta) < mintheta && fabs(theta) > 1e-5)
        theta = theta > 0 ? mintheta : -mintheta;
    x = cap(x, brain->config->get_vx_limit(), -brain->config->get_vx_limit());
    y = cap(y, brain->config->get_vy_limit(), -brain->config->get_vy_limit());
    theta = cap(theta, brain->config->get_vtheta_limit(), -brain->config->get_vtheta_limit());

    _vx = x; _vy = y; _vtheta = theta; // remember last command. Can be approximately used as the current robot's velocity.
    _lastCmdTime = brain->get_clock()->now();
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
            vx = cap(range, vxLimit, -vxLimit);
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
                if (distToObstacle < SAFE_DIST * 2) vxLimit = 0.5;
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
            vx = cap(range, vxLimit, -vxLimit);
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

    // simplify the calculation by not considering vtheta for now. This is a conservative estimation.
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
