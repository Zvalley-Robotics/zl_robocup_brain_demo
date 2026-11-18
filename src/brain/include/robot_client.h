#pragma once

#include <memory>
#include <string>

#include "robot_io/ball_fwd.h"
#include "robot_io/kick_targets.h"
#include "robot_io/zl_io.h"

class Brain;  // Mutually dependent classes; forward declaration

/**
 * 头部指令来源（2026-11-17）。枚举值即优先级，数值越大越优先。
 *
 * 同一拍内只有一个来源能成为“当前持有者”，其余请求直接丢弃——头部从「各节点直发」
 * 改为单一出口。背景：丢球时 CamFindBall（700 ms 6 点）与 FindBall 子树里的
 * CamFastScan（550 ms 6 点）挂在同一个 ReactiveSequence 下，每拍都会先后下发两套
 * 完全不同的目标姿态；板端 external_node 每收到一条消息都把 0.5 s 线性插值斜坡从头
 * 重来，于是头部在两个满行程端点（yaw ±1.1 rad）之间反复反向，实机表现为剧烈晃头。
 * 速度链早有 setVelocity 单一出口 + 全局加速度限幅，头部此前完全没有。
 */
enum class HeadSource : int
{
    None = 0,        // 无请求（内部占位，不得作为请求来源）
    ScanField = 1,   // CamScanField：未定位时巡视场地线，可被任何找球动作打断
    FindBall = 2,    // CamFindBall / CamFastScan：丢球后的 6 点扫场（两者点位已同源）
    TrackBall = 3,   // CamTrackBall：看见球或宽限期内朝最后已知位置收敛
    VisionKick = 4,  // RLVisionKick：踢球前低头两段，时间敏感
    Explicit = 5,    // MoveHead：行为树显式指定姿态（READY 抬头看场地线等）
};

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

    /** 门位姿合成（kick_targets）是否在稳定发布；视觉踢交棒门用它确认模型瞄门输入有效。 */
    bool kickTargetsActive() const;

    /**
     * @brief 头部位姿（相机光轴语义，弧度；内部转关节角并限幅）
     * @return int , 0 indicates success
     */
    int moveHead(double pitch, double yaw);

    /**
     * @brief 头部指令唯一入口（2026-11-17）：只登记请求，真正的下发由 tick() 里的
     *        flushHead() 每拍仲裁后执行一次，因此同一拍最多只有一条指令出门。
     *
     * @param src       来源，枚举值即优先级（见 HeadSource）
     * @param pitch     相机光轴下俯角（rad），语义与 moveHead 一致
     * @param yaw       头部偏航角（rad，正=左），语义与 moveHead 一致
     * @param minHoldMs 本来源成为持有者后，同级/更低优先级不得抢占的最短驻留时间。
     *                  离散扫描点必须传「一个点位的驻留时长」（板端插值需 0.5 s 收敛，
     *                  小于它等于让头部每次都到不了位）；连续跟踪传 0。
     * @return int , 0 indicates success
     */
    int requestHead(HeadSource src, double pitch, double yaw, double minHoldMs = 0.0);

    /**
     * @brief 头部单一出口收口（每拍一次）：在行为树 tick 之后由 Brain::tick 调用。
     *        先按优先级/驻留期选出本拍来源，再限幅、去重后下发。
     */
    void flushHead();

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
    /** 头部请求槽位：每拍由 requestHead 写入，flushHead 读取仲裁。 */
    struct HeadRequest
    {
        bool valid = false;
        double pitch = 0.0;
        double yaw = 0.0;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        double minHoldMs = 0.0;
    };
    static constexpr int kHeadSourceCount = 6;  // = HeadSource 枚举取值个数（0..5）

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

    // ---- 头部单一出口仲裁状态（见 requestHead / flushHead）----
    HeadRequest _headReq[kHeadSourceCount];
    int _headOwner = 0;                              // 当前持有者，0 = 无
    rclcpp::Time _headOwnerSince{0, 0, RCL_ROS_TIME}; // 持有者上任时刻（驻留期起算）
    double _headOwnerMinHoldMs = 0.0;                // 持有者声明的驻留时长
    double _headPubPitch = 0.0, _headPubYaw = 0.0;   // 上一条已下发指令（去重用）
    bool _headPubValid = false;
    double _headStaleMs = 200.0;   // 请求过期时长（参数 robot.head_arb_stale_ms）
    double _headDedupRad = 0.01;   // 去重死区（参数 robot.head_arb_dedup_rad）
};
