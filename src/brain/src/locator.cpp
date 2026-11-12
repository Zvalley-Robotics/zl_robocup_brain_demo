#include "locator.h"
#include "utils/math.h"
#include "utils/misc.h"
#include "brain.h"
#include "utils/print.h"
#include "brain_tree.h"
#include "one_m_candidate_selection.h"
#include <chrono>
#include <limits>

// 仅拆分调度模式使用新计时，真值模式继续遵循现有路径。
static bool splitEnabled(Brain *b) {
    return b->get_parameter("locator.split_schedule").as_bool() && !b->config->get_gt_pose_enabled();
}
static bool geomDue(Brain *b, double interval, double legacyAge, size_t method) {
    const bool due = splitEnabled(b) ? b->tree->localizationSchedule.geomDue(b->tree->localizationTickNs, interval)
                                     : legacyAge >= interval;
    if (b->get_parameter("locator.diagnostics").as_bool()) {
        if (due) ++b->tree->locGeomAttempts[method]; else ++b->tree->locGeomSkipped[method];
    }
    return due;
}
// 记录校正后位姿，事件点使用本次计算结果，避免按低频轨迹近邻推测。
static void traceSuccess(Brain *b, const char *method) {
    if (!b->get_parameter("locator.trace_enabled").as_bool()) return;
    const auto &p = b->data->robotPoseToField;
    RCLCPP_INFO(b->get_logger(), "[BRAIN_LOC_OK] t_ns=%lld method=%s pose=(%.6f,%.6f,%.6f)",
        (long long)b->get_clock()->now().nanoseconds(), method, p.x, p.y, p.theta);
}
static void geomSucceeded(Brain *b, size_t method) {
    static const char *eventNames[] = {"1M", "2T", "PT", "LT", "2X", "Border"};
    traceSuccess(b, eventNames[method]);
    if (splitEnabled(b)) b->tree->localizationSchedule.onGeomSuccess(b->tree->localizationTickNs);
    if (b->get_parameter("locator.diagnostics").as_bool()) {
        static const char *names[] = {"1M", "2T", "PT", "LT", "2X", "Border"};
        ++b->tree->locGeomSuccess[method];
        b->tree->localizationLastMode = names[method];
    }
}
static int particleCount(Brain *b) {
    const auto n = b->get_parameter("locator.num_particles").as_int();
    if (n < 2 || n > std::numeric_limits<int>::max()) throw std::invalid_argument("locator.num_particles invalid");
    return static_cast<int>(n);
}

#define REGISTER_LOCATOR_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [brain](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });

void RegisterLocatorNodes(BT::BehaviorTreeFactory &factory, Brain* brain)
{
    REGISTER_LOCATOR_BUILDER(SelfLocate);
    REGISTER_LOCATOR_BUILDER(SelfLocateEnterField);
    REGISTER_LOCATOR_BUILDER(SelfLocate1M);
    REGISTER_LOCATOR_BUILDER(SelfLocateBorder);
    REGISTER_LOCATOR_BUILDER(SelfLocate2T);
    REGISTER_LOCATOR_BUILDER(SelfLocateLT);
    REGISTER_LOCATOR_BUILDER(SelfLocatePT);
    REGISTER_LOCATOR_BUILDER(SelfLocate2X);
}

void Locator::init(FieldDimensions fd, int minMarkerCntParam, double residualToleranceParam, double muOffestParam)
{
    fieldDimensions = fd;
    calcFieldMarkers(fd);
    minMarkerCnt = minMarkerCntParam;
    residualTolerance = residualToleranceParam;
    muOffset = muOffestParam;
}

void Locator::calcFieldMarkers(FieldDimensions fd)
{
    
    fieldMarkers.push_back(FieldMarker{'X', 0.0, -fd.circleRadius, 0.0});
    fieldMarkers.push_back(FieldMarker{'X', 0.0, fd.circleRadius, 0.0});

    fieldMarkers.push_back(FieldMarker{'P', fd.length / 2 - fd.penaltyDist, 0.0, 0.0});
    fieldMarkers.push_back(FieldMarker{'P', -fd.length / 2 + fd.penaltyDist, 0.0, 0.0});

    fieldMarkers.push_back(FieldMarker{'T', 0.0, fd.width / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', 0.0, -fd.width / 2, 0.0});

    fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.penaltyAreaLength), fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.penaltyAreaLength), -fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.penaltyAreaLength), fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.penaltyAreaLength), -fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, -fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, fd.penaltyAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, -fd.penaltyAreaWidth / 2, 0.0});

    fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.goalAreaLength), fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', (fd.length / 2 - fd.goalAreaLength), -fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.goalAreaLength), fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', -(fd.length / 2 - fd.goalAreaLength), -fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', fd.length / 2, -fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, fd.goalAreaWidth / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'T', -fd.length / 2, -fd.goalAreaWidth / 2, 0.0});

    fieldMarkers.push_back(FieldMarker{'L', fd.length / 2, fd.width / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', fd.length / 2, -fd.width / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', -fd.length / 2, fd.width / 2, 0.0});
    fieldMarkers.push_back(FieldMarker{'L', -fd.length / 2, -fd.width / 2, 0.0});
}

FieldMarker Locator::markerToFieldFrame(FieldMarker marker_r, Pose2D pose_r2f)
{
    auto [x, y, theta] = pose_r2f;

    Eigen::Matrix3d transform;
    transform << cos(theta), -sin(theta), x,
        sin(theta), cos(theta), y,
        0, 0, 1;

    Eigen::Vector3d point_r;
    point_r << marker_r.x, marker_r.y, 1.0;

    auto point_f = transform * point_r;

    return FieldMarker{marker_r.type, point_f.x(), point_f.y(), marker_r.confidence};
}

int Locator::genInitialParticles(int num)
{
    unsigned long long seed = chr::duration_cast<std::chrono::milliseconds>(chr::system_clock::now().time_since_epoch()).count();
    srand(seed);
    hypos.resize(num, 6);
    hypos.leftCols(3) = Eigen::ArrayXXd::Random(num, 3);

    auto [xmin, xmax, ymin, ymax, thetamin, thetamax] = constraints;
    hypos.col(0) = hypos.col(0) * (xmax - xmin) / 2 + (xmin + xmax) / 2;
    hypos.col(1) = hypos.col(1) * (ymax - ymin) / 2 + (ymin + ymax) / 2;
    hypos.col(2) = hypos.col(2) * (thetamax - thetamin) / 2 + (thetamin + thetamax) / 2;

    return 0;
}

int Locator::genParticles()
{
    auto old_hypos = hypos;

    int num = static_cast<int>(hypos.rows() * numShrinkRatio);
    if (num <= 0)
        return 1;
    hypos.resize(num, 6);
    hypos.setZero();
    Eigen::ArrayXd rands = (Eigen::ArrayXd::Random(num) + 1) / 2;

    for (int i = 0; i < rands.size(); i++)
    {
        double rand = rands(i);
        int j;
        for (j = 0; j < old_hypos.rows(); j++)
        {
            if (old_hypos(j, 5) >= rand)
                break;
        }
        hypos.row(i).head(3) = old_hypos.row(j).head(3);
    }

    offsetX *= offsetShrinkRatio;
    offsetY *= offsetShrinkRatio;
    offsetTheta *= offsetShrinkRatio;
    Eigen::ArrayXXd offsets = Eigen::ArrayXXd::Random(num, 3);
    offsets.col(0) *= offsetX;
    offsets.col(1) *= offsetY;
    offsets.col(2) *= offsetTheta;
    hypos.leftCols(3) += offsets;

    hypos.col(0) = hypos.col(0).cwiseMax(constraints.xmin).cwiseMin(constraints.xmax);
    hypos.col(1) = hypos.col(1).cwiseMax(constraints.ymin).cwiseMin(constraints.ymax);
    hypos.col(2) = hypos.col(2).cwiseMax(constraints.thetamin).cwiseMin(constraints.thetamax);

    return 0;
}

double Locator::minDist(FieldMarker marker)
{
    double minDist = std::numeric_limits<double>::infinity();
    double dist;
    for (int i = 0; i < fieldMarkers.size(); i++)
    {
        auto target = fieldMarkers[i];
        if (target.type != marker.type)
        {
            continue;
        }
        dist = sqrt(pow((target.x - marker.x), 2.0) + pow((target.y - marker.y), 2.0));
        if (dist < minDist)
            minDist = dist;
    }
    return minDist;
}

vector<double> Locator::getOffset(FieldMarker marker)
{
    double minDist = std::numeric_limits<double>::infinity();
    FieldMarker nearestTarget{' ', 0.0, 0.0, 0.0};
    double dist;
    for (int i = 0; i < fieldMarkers.size(); i++)
    {
        auto target = fieldMarkers[i];
        if (target.type != marker.type)
        {
            continue;
        }
        dist = sqrt(pow((target.x - marker.x), 2.0) + pow((target.y - marker.y), 2.0));
        if (dist < minDist)
        {
            minDist = dist;
            nearestTarget = target;
        }
    }

    return vector<double>{nearestTarget.x - marker.x, nearestTarget.y - marker.y};
}

double Locator::residual(vector<FieldMarker> markers_r, Pose2D pose)
{
    double res = 0;

    for (int i = 0; i < markers_r.size(); i++)
    {
        auto marker_r = markers_r[i];
        double dist = max(norm(marker_r.x, marker_r.y), 0.1);
        auto marker_f = markerToFieldFrame(marker_r, pose);
        double conf = max(marker_r.confidence, 0.1);
        res += minDist(marker_f) / dist * 3; // weighted
    }

    return res;
}

Pose2D Locator::finalAdjust(vector<FieldMarker> markers_r, Pose2D pose)
{
    if (markers_r.size() == 0)
        return Pose2D{0, 0, 0};

    double dx = 0;
    double dy = 0;
    for (int i = 0; i < markers_r.size(); i++)
    {
        auto marker_r = markers_r[i];
        auto marker_f = markerToFieldFrame(marker_r, pose);
        auto offset = getOffset(marker_f);
        dx += offset[0];
        dy += offset[1];
    }
    dx /= markers_r.size();
    dy /= markers_r.size();

    return Pose2D{pose.x + dx, pose.y + dy, pose.theta};
}

int Locator::calcProbs(vector<FieldMarker> markers_r)
{
    int rows = hypos.rows();
    if (rows <= 1)
        return 1; 

    for (int i = 0; i < rows; i++)
    {
        Pose2D pose{hypos(i, 0), hypos(i, 1), hypos(i, 2)};
        double res = residual(markers_r, pose);
        hypos(i, 3) = res;

        if (res < bestResidual)
        {
            bestResidual = res;
            bestPose = pose;
        }
    }

    double mean = hypos.col(3).mean();
    double sqSum = ((hypos.col(3) - mean).square().sum());
    double sigma = std::sqrt(sqSum / (rows - 1));
    double mu = hypos.col(3).minCoeff() - muOffset * sigma;

    for (int i = 0; i < rows; i++)
    {
        hypos(i, 4) = probDesity(hypos(i, 3), mu, sigma);
    }

    double probSum = hypos.col(4).sum();
    if (fabs(probSum) < 1e-5)
        return 1;

    hypos.col(4) = hypos.col(4) / probSum;

    double acc = 0;
    for (int i = 0; i < rows; i++)
    {
        acc += hypos(i, 4);
        hypos(i, 5) = acc;
    }

    return 0;
}

bool Locator::isConverged()
{
    return (
        (hypos.col(0).maxCoeff() - hypos.col(0).minCoeff() < convergeTolerance)
        && (hypos.col(1).maxCoeff() - hypos.col(1).minCoeff() < convergeTolerance)
        && (hypos.col(2).maxCoeff() - hypos.col(2).minCoeff() < convergeTolerance)
    );
}

LocateResult Locator::locateRobot(vector<FieldMarker> markers_r, PoseBox2D constraintsParam, int numParticles, double offsetXParam, double offsetYParam, double offsetThetaParam)
{
    auto start_time = chr::high_resolution_clock::now();
    LocateResult res;
    double bigEnoughNum = 1e6;
    res.residual = bigEnoughNum;
    bestResidual = bigEnoughNum;
    bestPose = Pose2D{0.0, 0.0, 0.0};

    if (markers_r.size() < minMarkerCnt)
    {
        res.success = false;
        res.code = 4;
        res.msecs = msecsSince(start_time);
        return res;
    }

    constraints = constraintsParam;
    offsetX = offsetXParam;
    offsetY = offsetYParam;
    offsetTheta = offsetThetaParam;

    genInitialParticles(numParticles);
    if (calcProbs(markers_r))
    {
        res.success = false;
        res.code = 5;
        res.msecs = msecsSince(start_time);
        return res;
    }

    for (int i = 0; i < maxIteration; i++)
    {
        res.residual = bestResidual / markers_r.size();
        if (isConverged())
        {
            if (res.residual > residualTolerance)
            {
                res.success = false;
                res.code = 2;
                res.msecs = msecsSince(start_time);
                return res;
            }
            res.success = true;
            res.code = 0;
            res.pose = bestPose;
            res.pose.theta = toPInPI(res.pose.theta);
            res.msecs = msecsSince(start_time);
            return res;
        }

        if (genParticles())
        {
            res.success = false;
            res.code = 1;
            res.msecs = msecsSince(start_time);
            return res;
        }

        if (calcProbs(markers_r))
        {
            res.success = false;
            res.code = 5;
            res.msecs = msecsSince(start_time);
            return res;
        }
    }

    res.success = false;
    res.code = 3;
    res.msecs = msecsSince(start_time);
    return res;
}


NodeStatus SelfLocate::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    string mode = getInput<string>("mode").value();
    const bool splitTrust = mode == "trust_direction" && splitEnabled(brain);
    if (splitTrust) {
        const double retry = brain->get_parameter("locator.split_retry_ms").as_double();
        const double lo = brain->get_parameter("locator.split_min_drift_m").as_double();
        const double hi = brain->get_parameter("locator.split_max_drift_m").as_double();
        const double speed = brain->get_parameter("locator.max_drift_speed").as_double();
        const double yaw = brain->get_parameter("locator.trust_yaw_half_width_deg").as_double();
        if (!std::isfinite(retry) || retry <= 0 || !std::isfinite(lo) || lo < 0 ||
            !std::isfinite(hi) || hi < lo || !std::isfinite(speed) || speed < 0 ||
            !std::isfinite(yaw) || yaw <= 0 || yaw > 180) {
            RCLCPP_ERROR_THROTTLE(brain->get_logger(), *brain->get_clock(), 3000, "[LOC_SPLIT] 参数非法，跳过搜索");
            return NodeStatus::SUCCESS;
        }
        if (!brain->tree->localizationSchedule.trustDue(brain->tree->localizationTickNs, interval, retry)) {
            if (brain->get_parameter("locator.diagnostics").as_bool()) ++brain->tree->locTrustSkipped;
            return NodeStatus::SUCCESS;
        }
        brain->tree->localizationSchedule.onTrustAttempt(brain->tree->localizationTickNs);
    } else if (brain->msecsSince(brain->data->lastSuccessfulLocalizeTime) < interval) {
        if (mode == "trust_direction" && brain->get_parameter("locator.diagnostics").as_bool()) ++brain->tree->locTrustSkipped;
        return NodeStatus::SUCCESS;
    }
    double xMin = 0.0, xMax = 0.0, yMin = 0.0, yMax = 0.0, thetaMin = 0.0, thetaMax = 0.0; 
    auto markers = brain->data->getMarkersForLocator();

    if (mode == "face_forward")
    {
        xMin = -brain->config->fieldDimensions.length / 2;
        xMax = brain->config->fieldDimensions.length / 2;
        yMin = -brain->config->fieldDimensions.width / 2;
        yMax = brain->config->fieldDimensions.width / 2;
        thetaMin = -M_PI / 4;
        thetaMax = M_PI / 4;
    }
    else if (mode == "trust_direction")
    {
        int msec = static_cast<int>(brain->msecsSince(brain->data->lastSuccessfulLocalizeTime));
        double maxDriftSpeed = brain->get_parameter("locator.max_drift_speed").as_double();
        double maxDrift = splitTrust
            ? brain->tree->localizationSchedule.searchRadius(brain->tree->localizationTickNs, maxDriftSpeed,
                brain->get_parameter("locator.split_min_drift_m").as_double(),
                brain->get_parameter("locator.split_max_drift_m").as_double(),
                brain->get_parameter("locator.split_unbounded").as_bool())
            : msec / 1000.0 * maxDriftSpeed;

        xMin = max(-brain->config->fieldDimensions.length / 2 - 2, brain->data->robotPoseToField.x - maxDrift);
        xMax = min(brain->config->fieldDimensions.length / 2 + 2, brain->data->robotPoseToField.x + maxDrift);
        yMin = max(-brain->config->fieldDimensions.width / 2 - 2, brain->data->robotPoseToField.y - maxDrift);
        yMax = min(brain->config->fieldDimensions.width / 2 + 2, brain->data->robotPoseToField.y + maxDrift);
        const double yawDeg = brain->get_parameter("locator.trust_yaw_half_width_deg").as_double();
        if (!std::isfinite(yawDeg) || yawDeg <= 0 || yawDeg > 180 || !std::isfinite(maxDrift) || maxDrift < 0) {
            RCLCPP_ERROR_THROTTLE(brain->get_logger(), *brain->get_clock(), 3000, "[LOC_SEARCH] 航向半宽或搜索范围非法");
            return NodeStatus::SUCCESS;
        }
        const double halfYaw = yawDeg * M_PI / 180.0;
        thetaMin = brain->data->robotPoseToField.theta - halfYaw;
        thetaMax = brain->data->robotPoseToField.theta + halfYaw;
    }
    else if (mode == "fall_recovery")
    {
        int msec = static_cast<int>(brain->msecsSince(brain->data->lastSuccessfulLocalizeTime));
        double maxDriftSpeed = 0.1;                      
        double maxDrift = msec / 1000.0 * maxDriftSpeed; 

        xMin = -brain->config->fieldDimensions.length / 2 - 2;
        xMax = brain->config->fieldDimensions.length / 2 + 2;
        yMin = -brain->config->fieldDimensions.width / 2 - 2;
        yMax = brain->config->fieldDimensions.width / 2 + 2;
        thetaMin = brain->data->robotPoseToField.theta - M_PI / 180;
        thetaMax = brain->data->robotPoseToField.theta + M_PI / 180;
    }

    
    
    PoseBox2D constraints{xMin, xMax, yMin, yMax, thetaMin, thetaMax};
    double residual;
    if (splitTrust && (xMin > xMax || yMin > yMax || thetaMin > thetaMax)) {
        RCLCPP_WARN_THROTTLE(brain->get_logger(), *brain->get_clock(), 3000, "[LOC_SEARCH] 无效搜索范围，跳过定位");
        return NodeStatus::SUCCESS;
    }
    if (mode == "trust_direction" && brain->get_parameter("locator.diagnostics").as_bool()) ++brain->tree->locTrustAttempts;
    const auto start = std::chrono::steady_clock::now();
    auto res = brain->locator->locateRobot(markers, constraints, particleCount(brain));
    if (brain->get_parameter("locator.diagnostics").as_bool()) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        RCLCPP_INFO(brain->get_logger(), "[LOC_SEARCH] mode=%s code=%d success=%d markers=%zu residual=%.4f ms=%.3f box=(%.3f,%.3f,%.3f,%.3f,%.2f,%.2f)",
            mode.c_str(), res.code, int(res.success), markers.size(), res.residual, ms,
            xMin, xMax, yMin, yMax, thetaMin*180/M_PI, thetaMax*180/M_PI);
    }

    string mstring = "";
    for (int i = 0; i < markers.size(); i++) {
        auto m = markers[i];
        mstring += format("type: %c  x: %.1f y: %.1f", m.type, m.x, m.y).c_str();
    }
    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr(
            "%s success: %d  residual: %.2f  marker.size: %d  minMarkerCnt: %d  resTolerance: %.2f marker: %s",
            "self_locate",
            res.success,
            res.residual,
            markers.size(),
            brain->locator->minMarkerCnt,
            brain->locator->residualTolerance,
            mstring.c_str()
        )
    );
    
    if (!res.success)
        return NodeStatus::SUCCESS; 

    brain->calibrateOdom(res.pose.x, res.pose.y, res.pose.theta);
    brain->tree->setEntry<bool>("odom_calibrated", true);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    if (splitTrust) brain->tree->localizationSchedule.onTrustSuccess(brain->tree->localizationTickNs);
    else if (splitEnabled(brain)) brain->tree->localizationSchedule.initialize(brain->data->lastSuccessfulLocalizeTime.nanoseconds());
    traceSuccess(brain, mode == "trust_direction" ? "TRUSTDIR" : mode.c_str());
    if (mode == "trust_direction" && brain->get_parameter("locator.diagnostics").as_bool()) ++brain->tree->locTrustSuccess;
    if (brain->get_parameter("locator.diagnostics").as_bool()) brain->tree->localizationLastMode = mode;
    prtDebug("Locate Success: " + to_string(res.pose.x) + " " + to_string(res.pose.y) + " " + to_string(rad2deg(res.pose.theta)) + " Dur: " + to_string(res.msecs));

    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocateEnterField::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    if (brain->msecsSince(brain->data->lastSuccessfulLocalizeTime) < interval) return NodeStatus::SUCCESS;

    auto markers = brain->data->getMarkersForLocator();
    auto fd = brain->config->fieldDimensions;
    PoseBox2D cEnterLeft = {-fd.length / 2, -fd.circleRadius, fd.width / 2, fd.width / 2 + 1, -M_PI / 2 - M_PI / 6, -M_PI / 2 + M_PI / 6};
    PoseBox2D cEnterRight = {-fd.length / 2, -fd.circleRadius, -fd.width / 2 - 1, -fd.width / 2, M_PI / 2 - M_PI / 6, M_PI / 2 + M_PI / 6};


    auto resLeft = brain->locator->locateRobot(markers, cEnterLeft, particleCount(brain));
    auto resRight = brain->locator->locateRobot(markers, cEnterRight, particleCount(brain));
    LocateResult res;

    static string lastReport = "";
    string report = lastReport;
    if (resLeft.success && !resRight.success) {
        res = resLeft;
        report = "Entering Left";
    }
    else if (!resLeft.success && resRight.success) {
        res = resRight;
        report = "Entering Right";
    }
    else if (resLeft.success && resRight.success) {
        // 2026-10-14 双解歧义消解（镜像翻转修复）：入场左右解关于场地镜像对称，
        // 观测组合恰逢歧义时错误侧的单帧 residual 可能偶尔更小（实测右场真值下
        // left 0.16 < right 0.21，calibrateOdom 把估计瞬移 9.8m 到镜像位，~10s
        // 后才被拉回，期间 brain 以错误坐标决策）。与 SelfLocate1M 围绕当前估计
        // 限位的思路一致：某侧解与当前估计足够近时认定同侧（连续性优先，两解
        // 都远离当前估计——首次定位/位置被外部重置——时退回 residual 比较）。
        const double kSameSideDist = 2.0; // m：解与当前估计小于该距离视为同侧
        const Pose2D& cur = brain->data->robotPoseToField;
        const double dl = std::sqrt((resLeft.pose.x - cur.x) * (resLeft.pose.x - cur.x)
                                  + (resLeft.pose.y - cur.y) * (resLeft.pose.y - cur.y));
        const double dr = std::sqrt((resRight.pose.x - cur.x) * (resRight.pose.x - cur.x)
                                  + (resRight.pose.y - cur.y) * (resRight.pose.y - cur.y));
        if (dl < kSameSideDist || dr < kSameSideDist) {
            if (dl < dr) {
                res = resLeft;
                report = "Entering Left";
            } else {
                res = resRight;
                report = "Entering Right";
            }
        } else if (resLeft.residual < resRight.residual) {
            res = resLeft;
            report = "Entering Left";
        } else {
            res = resRight;
            report = "Entering Right";
        }
    } else res = resLeft;

    if (report != lastReport) {
        lastReport = report;
    }

    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr(
            "%s %s left success: %d  left residual: %.2f  right success %d  right residual %.2f resTolerance: %.2f markers: %d minMarkerCnt: %d ",
            "SelfLocateEnterField",
            report.c_str(),
            resLeft.success, 
                resLeft.residual,
                resRight.success,
                resRight.residual,
                brain->locator->residualTolerance,
                markers.size(),
                brain->locator->minMarkerCnt
            ),
            res.success
        );

    if (!res.success) return NodeStatus::SUCCESS; 


    // else, success.
    brain->calibrateOdom(res.pose.x, res.pose.y, res.pose.theta);
    brain->tree->setEntry<bool>("odom_calibrated", true);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    if (splitEnabled(brain)) brain->tree->localizationSchedule.initialize(brain->data->lastSuccessfulLocalizeTime.nanoseconds());
    traceSuccess(brain, "ENTERFIELD");
    if (brain->get_parameter("locator.diagnostics").as_bool()) brain->tree->localizationLastMode = "ENTERFIELD";
    prtDebug("Locate Success: " + to_string(res.pose.x) + " " + to_string(res.pose.y) + " " + to_string(rad2deg(res.pose.theta)) + " Dur: " + to_string(res.msecs));

    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocate1M::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    double maxDist = getInput<double>("max_dist").value();
    if (brain->client->isStandingStill(2000)) maxDist *= 1.5; 
    double maxDrift = getInput<double>("max_drift").value();
    bool validate = getInput<bool>("validate").value();
    
    string logPathS = "/locate/1m/success";
    string logPathF = "/locate/1m/fail";

    
    auto msecs = brain->msecsSince(brain->data->lastSuccessfulLocalizeTime);
    if (!geomDue(brain, interval, msecs, 0)){
        RCLCPP_DEBUG(
            brain->get_logger(),
            format2cstr("%s, Failed, msecs(%.1f).c_str() < interval(%.1f)", logPathF.c_str(), msecs, interval)
        );
        return NodeStatus::SUCCESS;
    }

    int markerIndex = -1;
    GameObject marker;
    MapMarking mapMarker; 
    double minDist = 100;
    auto markings = brain->data->getMarkings();
    if (brain->get_parameter("locator.one_m_multi_candidate").as_bool()) {
        const double margin = brain->get_parameter("locator.one_m_score_margin").as_double();
        const double separation = brain->get_parameter("locator.one_m_ambiguity_separation_m").as_double();
        if (!std::isfinite(margin) || margin < 0 || !std::isfinite(separation) || separation < 0) {
            RCLCPP_ERROR_THROTTLE(brain->get_logger(), *brain->get_clock(), 3000, "[LOC_1M_MULTI] 歧义参数非法");
            return NodeStatus::SUCCESS;
        }
        // 一次读取标志，同时用于候选生成和残差验证，避免跨视觉帧混用。
        std::vector<FieldMarker> observations;
        for (const auto &m : markings) {
            char type = 0;
            if (m.label == "LCross") type = 'L';
            else if (m.label == "TCross") type = 'T';
            else if (m.label == "XCross") type = 'X';
            else if (m.label == "PenaltyPoint") type = 'P';
            if (type) observations.push_back(FieldMarker{type, m.posToRobot.x, m.posToRobot.y, m.confidence});
        }
        std::vector<OneMCandidateChoice> candidates;
        const Pose2D prior = brain->data->robotPoseToField;
        for (size_t i = 0; i < markings.size(); ++i) {
            const auto &m = markings[i];
            if (m.name == "LOLG" || m.name == "LORG" || m.name == "LSLG" || m.name == "LSRG") continue;
            if (m.id < 0 || static_cast<size_t>(m.id) >= brain->config->mapMarkings.size()) continue;
            if (!std::isfinite(m.range) || !std::isfinite(m.posToField.x) || !std::isfinite(m.posToField.y) ||
                !std::isfinite(m.boundingBox.xmin) || !std::isfinite(m.boundingBox.xmax) ||
                !std::isfinite(m.boundingBox.ymin) || !std::isfinite(m.boundingBox.ymax)) continue;
            if (m.range > maxDist || !brain->isBoundingBoxInCenter(m.boundingBox)) continue;
            const auto &target = brain->config->mapMarkings[m.id];
            const double dx = target.x - m.posToField.x, dy = target.y - m.posToField.y;
            if (!std::isfinite(dx) || !std::isfinite(dy) || norm(dx, dy) > maxDrift || observations.empty()) continue;
            Pose2D candidate = prior; candidate.x += dx; candidate.y += dy;
            const double score = brain->locator->residual(observations, candidate) / observations.size();
            if (!std::isfinite(score) || score > brain->locator->residualTolerance) continue;
            candidates.push_back({i, candidate.x, candidate.y, score});
        }
        const auto selected = selectOneMCandidate(candidates, margin, separation);
        if (brain->get_parameter("locator.diagnostics").as_bool())
            RCLCPP_INFO_THROTTLE(brain->get_logger(), *brain->get_clock(), 5000,
                "[LOC_1M_MULTI] total=%zu candidates=%zu found=%d ambiguous=%d score=%.4f",
                markings.size(), candidates.size(), int(selected.found), int(selected.ambiguous), selected.score);
        if (!selected.found || selected.ambiguous) return NodeStatus::SUCCESS;
        const auto &m = markings[selected.index];
        const auto &target = brain->config->mapMarkings[m.id];
        Pose2D pose = prior;
        pose.x += target.x - m.posToField.x; pose.y += target.y - m.posToField.y;
        brain->calibrateOdom(pose.x, pose.y, pose.theta);
        brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
        geomSucceeded(brain, 0);
        return NodeStatus::SUCCESS;
    }
    for (int i = 0; i < markings.size(); i++) {
        auto m = markings[i];

        
        if (m.name == "LOLG" || m.name == "LORG" || m.name == "LSLG" || m.name == "LSRG") continue; 

        if (m.range < minDist) {
            minDist = m.range;
            markerIndex = i;
            marker = m;
        }
    }
    
    if (
        markerIndex < 0 || markerIndex >= markings.size()
        || marker.id < 0 || marker.id >= brain->config->mapMarkings.size()
    ) {
        if (brain->get_parameter("locator.one_m_candidate_diagnostics").as_bool()) {
            size_t other = 0, validId = 0, staticPass = 0;
            for (size_t i = 0; i < markings.size(); ++i) {
                if (static_cast<int>(i) == markerIndex) continue;
                const auto &m = markings[i];
                if (m.name == "LOLG" || m.name == "LORG" || m.name == "LSLG" || m.name == "LSRG") continue;
                ++other;
                if (m.id < 0 || static_cast<size_t>(m.id) >= brain->config->mapMarkings.size()) continue;
                ++validId;
                if (std::isfinite(m.range) && m.range <= maxDist && brain->isBoundingBoxInCenter(m.boundingBox)) ++staticPass;
            }
            RCLCPP_INFO_THROTTLE(brain->get_logger(), *brain->get_clock(), 5000,
                "[LOC_1M_CAND] nearest=%d other=%zu valid_id=%zu static_pass=%zu", markerIndex, other, validId, staticPass);
        }
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, No markings Found. Or marker id invalid.", logPathF.c_str())
        );
        return NodeStatus::SUCCESS;
    }
    mapMarker = brain->config->mapMarkings[marker.id];

    if (marker.range > maxDist) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, min marker Dist(%.2f).c_str() > maxDist(%.2f)", logPathF.c_str(), marker.range, maxDist)
        );
        return NodeStatus::SUCCESS;
    }

    if (!brain->isBoundingBoxInCenter(marker.boundingBox)) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, boundingbox is not in the center area", logPathF.c_str())
        );
        return NodeStatus::SUCCESS;
    }

    double dx, dy; 
    dx = mapMarker.x - marker.posToField.x;
    dy = mapMarker.y - marker.posToField.y;

    double drift = norm(dx, dy);
    if (drift > maxDrift) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, drift(%.2f).c_str() > maxDrift(%.2f)", logPathF.c_str(), drift, maxDrift)
        );
        return NodeStatus::SUCCESS;
    }
    
    
    Pose2D hypoPose = brain->data->robotPoseToField;
    hypoPose.x += dx;
    hypoPose.y += dy;

    auto allMarkers = brain->data->getMarkersForLocator();
    if (allMarkers.size() > 0) {
        double residual = brain->locator->residual(allMarkers, hypoPose) / allMarkers.size();
        if (residual > brain->locator->residualTolerance) 
        { 
            RCLCPP_INFO(
                brain->get_logger(),
                format2cstr("%s, Failed, validation residual(%.2f).c_str() > tolerance(%.2f)", logPathF.c_str(), residual, brain->locator->residualTolerance)
            );
            return NodeStatus::SUCCESS;
        }
    }

    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr("%s, Success. Drift = %.2f", logPathS.c_str(), drift)
    );
    brain->calibrateOdom(hypoPose.x, hypoPose.y, hypoPose.theta);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    geomSucceeded(brain, 0);
    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocate2X::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    double maxDist = getInput<double>("max_dist").value();
    if (brain->client->isStandingStill(2000)) maxDist *= 1.5; 
    double maxDrift = getInput<double>("max_drift").value();
    bool validate = getInput<bool>("validate").value();
    
    string logPathS = "/locate/2x/success";
    string logPathF = "/locate/2x/fail";

    auto msecs = brain->msecsSince(brain->data->lastSuccessfulLocalizeTime);
    if (!geomDue(brain, interval, msecs, 4)){
        RCLCPP_DEBUG(
            brain->get_logger(),
            format2cstr("%s, Failed, msecs(%.1f).c_str() < interval(%.1f)", logPathF.c_str(), msecs, interval)
        );
        return NodeStatus::SUCCESS;
    }

    auto points = brain->data->getMarkingsByType({"XCross"});
    if (points.size() != 2) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, point cnt(%d).c_str() != 2", logPathF.c_str(), points.size())
        );
        return NodeStatus::SUCCESS;
    }

    auto p0 = points[0]; auto p1 = points[1];
    
    if (p0.range > maxDist || p1.range > maxDist) 
    { 
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, p0 range (%.2f).c_str() or p1 range (%.2f) > maxDist(%.2f)", logPathF.c_str(), p0.range, p1.range, maxDist)
        );
        return NodeStatus::SUCCESS;
    }

    double xDist = fabs(p0.posToField.x - p1.posToField.x);
    if (xDist > 0.5) 
    { 
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, xDist(%.2f).c_str() > maxDist(%.2f)", logPathF.c_str(), xDist, 0.5)
        );
        return NodeStatus::SUCCESS;
    }

    double yDist = fabs(p0.posToField.y - p1.posToField.y);
    double mapYDist = brain->config->fieldDimensions.circleRadius * 2.0;
    if (fabs(yDist - mapYDist) > 0.5) { 
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, yDist(%.2f).c_str() too far (%.2f) from mapYDist(%.2f)", logPathF.c_str(), yDist, 0.5, mapYDist)
        );
        return NodeStatus::SUCCESS;
    }

    
    double dx = - (p0.posToField.x + p1.posToField.x) / 2.0;
    double dy = - (p1.posToField.y + p1.posToField.y) / 2.0;
    double drift = norm(dx, dy);

    if (drift > maxDrift) { 
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, dirft(%.2f).c_str() > maxDrift(%.2f)", logPathF.c_str(), drift, maxDrift)
        );
        return NodeStatus::SUCCESS;
    }

    
    Pose2D hypoPose = brain->data->robotPoseToField;
    hypoPose.x += dx;
    hypoPose.y += dy;

    
    auto allMarkers = brain->data->getMarkersForLocator();
    if (allMarkers.size() > 0) {
        double residual = brain->locator->residual(allMarkers, hypoPose) / allMarkers.size();
        if (residual > brain->locator->residualTolerance) 
        { 
            RCLCPP_INFO(
                brain->get_logger(),
                format2cstr("%s, Failed, validation residual(%.2f).c_str() > tolerance(%.2f)", logPathF.c_str(), residual, brain->locator->residualTolerance)
            );
            return NodeStatus::SUCCESS;
        }
    }

    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr("%s, Success. Dist = %.2f", logPathS.c_str(), drift)
    );
    brain->calibrateOdom(hypoPose.x, hypoPose.y, hypoPose.theta);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    geomSucceeded(brain, 4);
    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocate2T::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    double maxDist = getInput<double>("max_dist").value();
    if (brain->client->isStandingStill(2000)) maxDist *= 1.5; 
    double maxDrift = getInput<double>("max_drift").value();
    bool validate = getInput<bool>("validate").value();
    
    string logPathS = "/locate/2t/success";
    string logPathF = "/locate/2t/fail";

    auto msecs = brain->msecsSince(brain->data->lastSuccessfulLocalizeTime);
    if (!geomDue(brain, interval, msecs, 1)){
        RCLCPP_DEBUG(
            brain->get_logger(),
            format2cstr("%s, Failed, msecs(%.1f).c_str() < interval(%.1f)", logPathF.c_str(), msecs, interval)
        );
        return NodeStatus::SUCCESS;
    }

    auto markers = brain->data->getMarkingsByType({"TCross"});
    GameObject m1, m2;
    bool found = false;
    auto fd = brain->config->fieldDimensions;
    for (int i = 0; i < markers.size(); i++) {
        m1 = markers[i];
        
        if (m1.range > maxDist) continue; 

        for (int j = i + 1; j < markers.size(); j++) {
            m2 = markers[j];

            if (m2.range > maxDist) continue; 

            if (
                fabs(m1.posToField.x - m2.posToField.x) < 0.3
                && fabs(fabs(m1.posToField.y - m2.posToField.y) - fabs(fd.goalAreaWidth - fd.penaltyAreaWidth)/2.0)< 0.3
            ) {
                found = true;
                break;
            }
        }
        if (found) break;
    }


    if (!found) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, No pattern within maxDist(%.2f).c_str() Found", logPathF.c_str(), maxDist)
        );
        return NodeStatus::SUCCESS;
    }

    Point2D pos_o = { 
        (m1.posToField.x + m2.posToField.x)/2,
        (m1.posToField.y + m2.posToField.y)/2
    };
    Point2D pos_m; 

    vector<double> halfs = {-1, 1};
    vector<double> sides = {-1, 1};
    bool matched = false;
    for (auto half: halfs) {
        for (auto side: sides) {
            pos_m = {
                half * (fd.length / 2.0), 
                side * (fd.penaltyAreaWidth + fd.goalAreaWidth) / 4.0
            };
            double dist = norm(pos_o.x - pos_m.x, pos_o.y - pos_m.y);
            if (dist < maxDrift) {
                matched = true;
                break;
            }
        }
        if (matched) break;
    }

    if (!matched) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, can not match to any map positions within maxDrift(%.2f).c_str()", logPathF.c_str(), maxDrift)
        );
        return NodeStatus::SUCCESS;
    }

    double dx = pos_m.x - pos_o.x;
    double dy = pos_m.y - pos_o.y;
    double drift = norm(dx, dy);

    Pose2D hypoPose = brain->data->robotPoseToField;
    hypoPose.x += dx;
    hypoPose.y += dy;

    auto allMarkers = brain->data->getMarkersForLocator();
    if (allMarkers.size() > 0) {
        double residual = brain->locator->residual(allMarkers, hypoPose) / allMarkers.size();
        if (residual > brain->locator->residualTolerance) 
        { 
            RCLCPP_INFO(
                brain->get_logger(),
                format2cstr("%s, Failed, validation residual(%.2f).c_str() > tolerance(%.2f)", logPathF.c_str(), residual, brain->locator->residualTolerance)
            );
            return NodeStatus::SUCCESS;
        }
    }

    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr("%s, Success. Dist = %.2f", logPathS.c_str(), drift)
    );
    brain->calibrateOdom(hypoPose.x, hypoPose.y, hypoPose.theta);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    geomSucceeded(brain, 1);
    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocateLT::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    double maxDist = getInput<double>("max_dist").value();
    if (brain->client->isStandingStill(2000)) maxDist *= 1.5; 
    double maxDrift = getInput<double>("max_drift").value();
    bool validate = getInput<bool>("validate").value();
    
    string logPathS = "/locate/lt/success";
    string logPathF = "/locate/lt/fail";

    auto msecs = brain->msecsSince(brain->data->lastSuccessfulLocalizeTime);
    if (!geomDue(brain, interval, msecs, 3)){
        RCLCPP_DEBUG(
            brain->get_logger(),
            format2cstr("%s, Failed, msecs(%.1f).c_str() < interval(%.1f)", logPathF.c_str(), msecs, interval)
        );
        return NodeStatus::SUCCESS;
    }

    auto tMarkers = brain->data->getMarkingsByType({"TCross"});
    auto lMarkers = brain->data->getMarkingsByType({"LCross"});

    GameObject t, l;
    bool found = false;
    auto fd = brain->config->fieldDimensions;
    for (int i = 0; i < tMarkers.size(); i++) {
        t = tMarkers[i];
        
        if (t.range > maxDist) continue; 

        for (int j = i + 1; j < lMarkers.size(); j++) {
            l = lMarkers[j];

            if (l.range > maxDist) continue;

            if (
                fabs(t.posToField.y - l.posToField.y) < 0.3
                && fabs(fabs(t.posToField.x - l.posToField.x) - fd.goalAreaLength)< 0.3
            ) {
                found = true;
                break;
            }
        }

        if (found) break;
    }


    if (!found) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, No pattern within maxDist(%.2f).c_str() Found", logPathF.c_str(), maxDist)
        );
        return NodeStatus::SUCCESS;
    }

    Point2D pos_o = { 
        (t.posToField.x + l.posToField.x)/2,
        (t.posToField.y + l.posToField.y)/2
    };
    Point2D pos_m; 

    vector<double> halfs = {-1, 1};
    vector<double> sides = {-1, 1};
    bool matched = false;
    for (auto half: halfs) {
        for (auto side: sides) {
            pos_m = {
                half * (fd.length / 2.0 - fd.goalAreaLength / 2.0), 
                side * (fd.goalAreaWidth / 2.0)
            };
            double dist = norm(pos_o.x - pos_m.x, pos_o.y - pos_m.y);
            if (dist < maxDrift) {
                matched = true;
                break;
            }
        }
        if (matched) break;
    }
    if (!matched) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, can not match to any map positions within maxDrift(%.2f).c_str()", logPathF.c_str(), maxDrift)
        );
        return NodeStatus::SUCCESS;
    }

    double dx = pos_m.x - pos_o.x;
    double dy = pos_m.y - pos_o.y;
    double drift = norm(dx, dy);

    Pose2D hypoPose = brain->data->robotPoseToField;
    hypoPose.x += dx;
    hypoPose.y += dy;

    auto allMarkers = brain->data->getMarkersForLocator();
    if (allMarkers.size() > 0) {
        double residual = brain->locator->residual(allMarkers, hypoPose) / allMarkers.size();
        if (residual > brain->locator->residualTolerance) { 
            RCLCPP_INFO(
                brain->get_logger(),
                format2cstr("%s, Failed, validation residual(%.2f).c_str() > tolerance(%.2f)", logPathF.c_str(), residual, brain->locator->residualTolerance)
            );
            return NodeStatus::SUCCESS;
        }
    }

    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr("%s, Success. Dist = %.2f", logPathS.c_str(), drift)
    );
    brain->calibrateOdom(hypoPose.x, hypoPose.y, hypoPose.theta);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    geomSucceeded(brain, 3);
    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocatePT::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    double maxDist = getInput<double>("max_dist").value();
    if (brain->client->isStandingStill(2000)) maxDist *= 1.5; 
    double maxDrift = getInput<double>("max_drift").value();
    bool validate = getInput<bool>("validate").value();
    
    string logPathS = "/locate/pt/success";
    string logPathF = "/locate/pt/fail";

    auto msecs = brain->msecsSince(brain->data->lastSuccessfulLocalizeTime);
    if (!geomDue(brain, interval, msecs, 2)){
        RCLCPP_DEBUG(
            brain->get_logger(),
            format2cstr("%s, Failed, msecs(%.1f).c_str() < interval(%.1f)", logPathF.c_str(), msecs, interval)
        );
        return NodeStatus::SUCCESS;
    }

    auto posts = brain->data->getGoalposts();
    auto tMarkers = brain->data->getMarkingsByType({"TCross"});
    
    GameObject p, t;
    bool found = false;
    auto fd = brain->config->fieldDimensions;
    for (int i = 0; i < posts.size(); i++) {
        p = posts[i];
        if (p.range > maxDist) continue;

        for (int j = i + 1; j < tMarkers.size(); j++) {
            t = tMarkers[j];
            if (t.range > maxDist) continue;
            if (
                fabs(t.posToField.x - p.posToField.x) < 0.5
                && fabs(fabs(t.posToField.x - p.posToField.x) - fabs(fd.goalAreaWidth - fd.goalWidth) / 2.0) < 0.3
            ) {
                found = true;
                break;
            }
        }
        
        if (found) break;
    }


    if (!found) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, No pattern within maxDist(%.2f).c_str() Found", logPathF.c_str(), maxDist)
        );
        return NodeStatus::SUCCESS;
    }

    Point2D pos_o = { 
        t.posToField.x,
        t.posToField.y
    };
    Point2D pos_m; 

    vector<double> halfs = {-1, 1};
    vector<double> sides = {-1, 1};
    bool matched = false;
    for (auto half: halfs) {
        for (auto side: sides) {
            pos_m = {
                half * (fd.length), 
                side * (fd.goalAreaWidth / 2.0)
            };
            double dist = norm(pos_o.x - pos_m.x, pos_o.y - pos_m.y);
            if (dist < maxDrift) {
                matched = true;
                break;
            }
        }
        if (matched) break;
    }
    if (!matched) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, can not match to any map positions within maxDrift(%.2f).c_str()", logPathF.c_str(), maxDrift)
        );
        return NodeStatus::SUCCESS;
    }

    double dx = pos_m.x - pos_o.x;
    double dy = pos_m.y - pos_o.y;
    double drift = norm(dx, dy);

    Pose2D hypoPose = brain->data->robotPoseToField;
    hypoPose.x += dx;
    hypoPose.y += dy;

    auto allMarkers = brain->data->getMarkersForLocator();
    if (allMarkers.size() > 0) {
        double residual = brain->locator->residual(allMarkers, hypoPose) / allMarkers.size();
        if (residual > brain->locator->residualTolerance) { 

            RCLCPP_INFO(
                brain->get_logger(),
                format2cstr("%s, Failed, validation residual(%.2f).c_str() > tolerance(%.2f)", logPathF.c_str(), residual, brain->locator->residualTolerance)
            );
            return NodeStatus::SUCCESS;
        }
    }

    
    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr("%s, Success. Dist = %.2f", logPathS.c_str(), drift)
    );
    brain->calibrateOdom(hypoPose.x, hypoPose.y, hypoPose.theta);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    geomSucceeded(brain, 2);
    return NodeStatus::SUCCESS;
}

NodeStatus SelfLocateBorder::tick()
{
    double interval = getInput<double>("msecs_interval").value();
    double maxDist = getInput<double>("max_dist").value();
    if (brain->client->isStandingStill(2000)) maxDist *= 1.5; 
    double maxDrift = getInput<double>("max_drift").value();
    bool validate = getInput<bool>("validate").value();
    
    string logPathS = "/locate/border/success";
    string logPathF = "/locate/border/fail";

    auto msecs = brain->msecsSince(brain->data->lastSuccessfulLocalizeTime);
    if (!geomDue(brain, interval, msecs, 5)){
        RCLCPP_DEBUG(
            brain->get_logger(),
            format2cstr("%s, Failed, msecs(%.1f).c_str() < interval(%.1f)", logPathF.c_str(), msecs, interval)
        );
        return NodeStatus::SUCCESS;
    }

    
    bool touchLineFound = false;
    FieldLine touchLine;
    double bestConfidenceTouchline = 0.0;
    bool goalLineFound = false;
    FieldLine goalLine;
    double bestConfidenceGoalline = 0.0;
    bool middleLineFound = false;
    FieldLine middleLine;
    double bestConfidenceMiddleLine = 0.0;

    auto fieldLines = brain->data->getFieldLines();
    for (int i = 0; i < fieldLines.size(); i++) {
        auto line = fieldLines[i];
        if (line.type != LineType::TouchLine && line.type != LineType::GoalLine && line.type != LineType::MiddleLine) continue;
        if (line.confidence < 0.8) continue;
        
        double dist = pointMinDistToLine(
            Point2D({brain->data->robotPoseToField.x, brain->data->robotPoseToField.y}), 
            line.posToField
        );
        if (dist > maxDist) continue;

        if (line.type == LineType::TouchLine) {
           if (line.confidence > bestConfidenceTouchline) {
               bestConfidenceTouchline = line.confidence;
               touchLine = line;
               touchLineFound = true;
           }
        } else if (line.type == LineType::GoalLine) {
            if (line.confidence > bestConfidenceGoalline) {
                bestConfidenceGoalline = line.confidence;
                goalLine = line;
                goalLineFound = true;
            }
        } else if (line.type == LineType::MiddleLine) {
            if (line.confidence > bestConfidenceMiddleLine) {
                bestConfidenceMiddleLine = line.confidence;
                middleLine = line;
                middleLineFound = true;
            }
        }
    }

    double dx = 0; 
    double dy = 0; 
    auto fd = brain->config->fieldDimensions;
    if (touchLineFound) {
       double y_m = touchLine.side == LineSide::Left ? fd.width / 2.0 : - fd.width / 2.0;
       double perpDist = pointPerpDistToLine(
           Point2D({brain->data->robotPoseToField.x, brain->data->robotPoseToField.y}),
           touchLine.posToField
       );
       double y_o = touchLine.side == LineSide::Left ? 
           brain->data->robotPoseToField.y - perpDist :
           brain->data->robotPoseToField.y + perpDist;
       dy = y_m - y_o;
    }
    if (goalLineFound) {
        double x_m = goalLine.half == LineHalf::Opponent ? fd.length / 2.0: - fd.length / 2.0;
        double perpDist = pointPerpDistToLine(
            Point2D({brain->data->robotPoseToField.x, brain->data->robotPoseToField.y}),
            goalLine.posToField
        );
        double x_o = goalLine.half == LineHalf::Opponent?
            brain->data->robotPoseToField.x - perpDist :
            brain->data->robotPoseToField.x + perpDist;
        dx = x_m - x_o;
    } else if (middleLineFound) {
        double x_m = 0;
        auto linePos = middleLine.posToField;
        auto robotPose = brain->data->robotPoseToField;
        vector<double> pointA(2);
        vector<double> pointB(2);
        vector<double> pointR = {robotPose.x, robotPose.y};

        if (linePos.y0 > linePos.y1) {
            pointA = {linePos.x0, linePos.y0};
            pointB = {linePos.x1, linePos.y1};
        } else {
            pointA = {linePos.x1, linePos.y1};
            pointB = {linePos.x0, linePos.y0};
        }

        vector<double> vl = {pointB[0] - pointA[0], pointB[1] - pointA[1]};
        vector<double> vr = {pointR[0] - pointA[0], pointR[1] - pointA[1]};

        double normvl = norm(vl);
        double normvr = norm(vr);
        if (normvl < 1e-3 || normvr < 1e-3) {
            dx = 10000; 
        } else {
            double dist = crossProduct(vr, vl) / normvl;
            double x_o = robotPose.x + dist;
            dx = x_m - x_o;
        }
    }

    
    if ((!touchLineFound && !goalLineFound && !middleLineFound)) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, No touchline or goalline or middleLine found.", logPathF.c_str())
        );
        return NodeStatus::SUCCESS;
    }

    double drift = norm(dx, dy);
    if (drift > maxDrift) {
        RCLCPP_INFO(
            brain->get_logger(),
            format2cstr("%s, Failed, drift(%.2f).c_str() > maxDrift(%.2f)", logPathF.c_str(), drift, maxDrift)
        );
        return NodeStatus::SUCCESS;
    }
    
    Pose2D hypoPose = brain->data->robotPoseToField;
    hypoPose.x += dx;
    hypoPose.y += dy;

    auto allMarkers = brain->data->getMarkersForLocator();
    if (allMarkers.size() > 0) {
        double residual = brain->locator->residual(allMarkers, hypoPose) / allMarkers.size();
        if (residual > brain->locator->residualTolerance) { 
            RCLCPP_INFO(
                brain->get_logger(),
                format2cstr("%s, Failed, validation residual(%.2f).c_str() > tolerance(%.2f)", logPathF.c_str(), residual, brain->locator->residualTolerance)
            );
            return NodeStatus::SUCCESS;
        }
    }

    RCLCPP_INFO(
        brain->get_logger(),
        format2cstr("%s, Success. Drift = %.2f", logPathS.c_str(), drift)
    );
    string label = "";
    if (touchLineFound) label += "TouchLine";
    if (touchLineFound && (goalLineFound || middleLineFound)) label += " ";
    if (goalLineFound) label += "GoalLine";
    if (middleLineFound) label += "MiddleLine";

    brain->calibrateOdom(hypoPose.x, hypoPose.y, hypoPose.theta);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
    geomSucceeded(brain, 5);
    return NodeStatus::SUCCESS;
}
