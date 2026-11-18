#pragma once

#include <memory>
#include <string>

#include "robot_io/ball_fwd.h"
#include "robot_io/kick_targets.h"
#include "robot_io/zl_io.h"

class Brain;  // Mutually dependent classes; forward declaration

/**
 * `RobotClient` contains all robot-body operations used by the Brain.
 *
 * 2026-11-03 起本类直接对接 zl_robot 原生接口（取代 brain_zl_bridge 进程）：
 * 下行 = /cmd_vel、/robot/input/intent、/robot/external/head_control；
 * 上行由 ZlIo 直接写入 BrainData。速度限幅、避障、日志等决策侧处理保留在本类。
 */
class RobotClient
{
public:
    explicit RobotClient(Brain* argBrain) : brain(argBrain) {}

    /** 建立 zl 侧收发接口（参数见 config.yaml 的 zl_io 段）。 */
    void init(const std::string& robot_name);

    /** 100Hz 驱动 zl 状态推进序列器与踢球交接状态机（由 Brain::tick 调用）。 */
    void tick();

    /**
     * @brief 头部位姿（相机光轴语义，弧度；内部转关节角并限幅）
     * @return int , 0 indicates success
     */
    int moveHead(double pitch, double yaw);

    /**
     * @brief 速度指令（x/y/theta 已在内部限幅并做最小速度补偿）
     * @param vyLimitOverride >0 时用该值替代 config 的 robot.vy_limit 做 y 轴限幅，
     *        仍会被内部硬上限（kCmdVyHardMax）夹住；<=0 等于不生效。
     *        目前只有守门员站位用（射门横移需要越过全局 0.4 的横移上限，
     *        而不把全局上限抬高以免影响前锋/Assist）。
     * @param accelVyOverride >0 时用该值替代 robot.accel_limit_vy 做 y 轴斜率限幅。
     *        门将被射门时需要从 0 加速到 0.6（全局 1.2 m/s² 要 0.5 s，吃掉一半扑救时间），
     *        故给门将单独放宽到本体自身平滑值；<=0 等于不生效。
     * @return int , 0 indicates success
     */
    int setVelocity(double x, double y, double theta, double vyLimitOverride = -1.0,
                    double accelVyOverride = -1.0);

    int crabWalk(double angle, double speed);

    /**
     * @brief crab walk with heading tracking: same as crabWalk(angle, speed) but
     *        also outputs vtheta (e.g. goal keeper turning to face the ball while advancing)
     */
    int crabWalk(double angle, double speed, double vtheta);

    /**
     * @brief 场地系目标点运动（避障版 2/3）
     * @return int Return value of motion-control command; 0 indicates success
     */
    int moveToPoseOnField2(double tx, double ty, double ttheta, double longRangeThreshold, double turnThreshold, double vxLimit, double vyLimit, double vthetaLimit, double xTolerance, double yTolerance, double thetaTolerance, bool avoidObstacle = false);

    /**
     * @brief 场地系目标点运动（避障版 3，含自由球就位专用安全距离）
     * @return int Return value of motion-control command; 0 indicates success
     */
    int moveToPoseOnField3(double tx, double ty, double ttheta, double longRangeThreshold, double turnThreshold, double vxLimit, double vyLimit, double vthetaLimit, double xTolerance, double yTolerance, double thetaTolerance, bool avoidObstacle = false);

    /**
     * @brief Wave hand (zl 人形无手部自由度：忽略并节流提示)
     */
    int waveHand(bool doWaveHand);

    /**
     * @brief 起身/恢复：请求 zl 行走序列（ready -> walk）
     */
    int standUp();

    /**
     * @brief switch to RL-based vision kick mode
     * @param start true 进入视觉踢球交接（切 RL + t4_kicking），false 忽略
     */
    int RLVisionKick(bool start = true);

    /**
     * @brief request robocup gait（等价于请求行走序列）
     */
    int robocupWalk();

    /**
     * @brief Enter damping mode
     */
    int enterDamping();

    double msecsToCollide(double vx, double vy, double vtheta, double maxTime = 10000);

    bool isStandingStill(double timeBuffer = 1000);

    /** 指令-位移一致性观测：下发在跑但机体不动（接触/被顶住/打滑）时 1 Hz 告警，只报警不改速度。 */
    void checkStuck();

private:
    Brain* brain;
    std::unique_ptr<robot_io::ZlIo> zl;
    // zl 侧适配（2026-11-03 由独立节点并入）：真实球门位姿合成、球类型转换
    std::unique_ptr<robot_io::KickTargets> kick_targets;
    std::unique_ptr<robot_io::BallFwd> ball_fwd;
    // 加速度限制需要读“上一拍下发值”，必须确定初值（原先未初始化）
    double _vx = 0.0, _vy = 0.0, _vtheta = 0.0;
    rclcpp::Time _lastCmdTime;
    rclcpp::Time _lastNonZeroCmdTime;
    // setVelocity 日志节流状态：上次输出的指令、时间与是否已输出过
    double _lastLoggedVx = 0.0, _lastLoggedVy = 0.0, _lastLoggedVtheta = 0.0;
    rclcpp::Time _lastCmdLogTime;
    bool _hasLoggedCmd = false;
};
