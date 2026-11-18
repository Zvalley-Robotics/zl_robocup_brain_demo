#pragma once

#include <cmath>
#include <deque>

#include <rclcpp/rclcpp.hpp>

#include "types.h"

/**
 * 球速估计器（2026-11-16，门将拦截第 1 档）。
 *
 * 视觉 /ball 是 30 Hz 的单帧离散球位，brain 侧 data->ball 只有“最近一次检测的位置 + 时间戳”，
 * 没有任何速度量，门将因此只能做“球 → 球门中心”的瞬时投影，对滚动的球永远慢半拍。
 *
 * 本类把连续检测帧拟合成 (t, x) / (t, y) 两条直线，输出场系速度：
 *   - 只有时间戳前进的新检测帧入样：视觉 30 Hz 而 brain tick 86 Hz，同一帧会被读到多次；
 *   - 拟合窗口默认 0.35 s，样本数 ≥ minSamples 且跨度 ≥ kMinSpanS 才输出，避免
 *     “相邻帧差分把 ±3 cm 检测噪声放大成 1.5 m/s 假速度”；
 *   - 单帧跳变（相对上一拍外推残差 > kResetResidualM，或拟合速度 > maxSpeed）整段重置：
 *     视觉误检、球源 LOCAL↔TM 切换（两套场系差 0.7~1.4 m）都不会被当成球在飞；
 *   - 无检测超过 resetMs 即清空，旧速度不得用于外推。
 *
 * 消费端契约：只有 state().valid 为真时速度可用，且必须同时看 state().ageMs——
 * 最近一次检测越旧外推误差越大（门将默认只用 ≤400 ms 的球位做预测）。
 */
class BallEstimator
{
public:
    struct State
    {
        double vx = 0.0;     // 场系速度（m/s）
        double vy = 0.0;
        double speed = 0.0;  // 合速度模（m/s）
        double ageMs = 1e9;  // 距最近一次检测的时长
        int samples = 0;     // 当前窗口内样本数
        bool valid = false;  // 速度可信（样本够、跨度够、未超时）
    };

    void reset()
    {
        buf_.clear();
        hasStamp_ = false;
        st_ = State{};
    }

    /**
     * 每拍调用一次（允许同一检测帧被重复调用，内部按时戳去重）。
     *
     * @param ball       当前球：只取 timePoint（判新帧）与 posToField（位置）
     * @param detected   本帧是否有检测（data->ballDetected）
     * @param ageMs      调用方算好的球龄 msecsSince(ball.timePoint)
     * @param windowS    拟合窗口（s）
     * @param maxSpeed   超过该速度视为检测跳变（m/s）
     * @param minSamples 输出速度所需的最少样本数
     * @param resetMs    无检测超过该时长即清空（ms）
     */
    void update(const GameObject &ball, bool detected, double ageMs,
                double windowS, double maxSpeed, int minSamples, double resetMs)
    {
        st_.ageMs = ageMs;
        if (!std::isfinite(ball.posToField.x) || !std::isfinite(ball.posToField.y) ||
            !(windowS > 1e-3) || ageMs > resetMs)
        {
            reset();
            st_.ageMs = ageMs; // reset() 会把 ageMs 归零，这里恢复真实球龄
            return;
        }
        if (!detected)
        {
            // 单帧掉帧（30 Hz 视觉常见）：保留窗口不输出速度，下一帧接着拟合；
            // 真正的长时间失球由上方的 ageMs > resetMs 清空。
            st_.valid = false;
            return;
        }

        const double stampSec = ball.timePoint.seconds();
        if (!(stampSec > 0.0) || !std::isfinite(stampSec))
        {
            st_.valid = false;
            return;
        }

        // 同一检测帧被多个 tick 读到：只刷新有效性，不重复入样
        if (hasStamp_ && stampSec <= lastStampSec_)
        {
            st_.valid = fitReady(minSamples);
            return;
        }

        // 时间戳大幅前进（球丢了很久又出现）：旧样本已无参考价值
        if (hasStamp_ && stampSec - lastStampSec_ > 4.0 * windowS)
        {
            buf_.clear();
            st_ = State{};
        }
        lastStampSec_ = stampSec;
        hasStamp_ = true;

        // 单帧跳变：按上一拍速度外推到本帧，残差过大说明不是同一个球（误检 / 球源切换）
        if (!buf_.empty() && st_.valid)
        {
            const Sample &prev = buf_.back();
            const double dt = stampSec - prev.t;
            if (dt > 1e-4)
            {
                const double predX = prev.x + st_.vx * dt;
                const double predY = prev.y + st_.vy * dt;
                if (std::hypot(ball.posToField.x - predX, ball.posToField.y - predY) > kResetResidualM)
                {
                    buf_.clear();
                }
            }
        }

        buf_.push_back(Sample{stampSec, ball.posToField.x, ball.posToField.y});
        while (!buf_.empty() &&
               (stampSec - buf_.front().t > windowS || buf_.size() > kMaxSamples))
        {
            buf_.pop_front();
        }

        fit(maxSpeed, minSamples);
    }

    const State &state() const { return st_; }

private:
    struct Sample
    {
        double t;
        double x;
        double y;
    };

    static constexpr double kMinSpanS = 0.08;      // 拟合所需最小时间跨度
    static constexpr double kResetResidualM = 1.0; // 单帧外推残差超过即判跳变
    static constexpr size_t kMaxSamples = 24;      // 窗口内样本上限（防高频抖动堆积）

    bool fitReady(int minSamples) const
    {
        if (static_cast<int>(buf_.size()) < minSamples || buf_.size() < 2) return false;
        return (buf_.back().t - buf_.front().t) >= kMinSpanS;
    }

    void fit(double maxSpeed, int minSamples)
    {
        st_.vx = 0.0;
        st_.vy = 0.0;
        st_.speed = 0.0;
        st_.samples = static_cast<int>(buf_.size());
        st_.valid = false;
        if (!fitReady(minSamples)) return;

        // 最小二乘 y = a + b·t：t 取相对首样本的秒数，量级小、数值更稳
        const double t0 = buf_.front().t;
        double n = 0.0, sumT = 0.0, sumX = 0.0, sumY = 0.0, sumTT = 0.0, sumTX = 0.0, sumTY = 0.0;
        for (const auto &s : buf_)
        {
            const double dt = s.t - t0;
            n += 1.0;
            sumT += dt;
            sumX += s.x;
            sumY += s.y;
            sumTT += dt * dt;
            sumTX += dt * s.x;
            sumTY += dt * s.y;
        }
        const double den = n * sumTT - sumT * sumT;
        if (std::fabs(den) < 1e-9) return;

        const double vx = (n * sumTX - sumT * sumX) / den;
        const double vy = (n * sumTY - sumT * sumY) / den;
        const double speed = std::hypot(vx, vy);
        if (!std::isfinite(speed) || speed > maxSpeed)
        {
            // 拟合结果超出本体能力：判为跳变，只保留最新一帧重新起算
            const Sample last = buf_.back();
            buf_.clear();
            buf_.push_back(last);
            st_.samples = 1;
            return;
        }

        st_.vx = vx;
        st_.vy = vy;
        st_.speed = speed;
        st_.valid = true;
    }

    std::deque<Sample> buf_;
    double lastStampSec_ = 0.0;
    bool hasStamp_ = false;
    State st_;
};
