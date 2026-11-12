#pragma once

#include <algorithm>
#include <cstdint>

// 所有时间均为节点 ROS 时钟的纳秒；-1 代表尚无记录，零时刻合法。
struct SplitLocalizationSchedule
{
    int64_t trust_success = -1, geom_success = -1, trust_attempt = -1;
    int64_t search_anchor = -1, observed = -1;
    bool updated_this_tick = false;

    void clear() { *this = SplitLocalizationSchedule{}; }
    void initialize(int64_t now)
    {
        trust_success = geom_success = search_anchor = observed = now;
        trust_attempt = -1;
        updated_this_tick = false;
    }
    // 时间回退后将间隔及搜索参考重置到当前时刻；这不是新的定位成功。
    bool beginTick(int64_t now)
    {
        if (observed < 0) { initialize(now); return false; }
        if (now < observed) { initialize(now); return true; }
        observed = now;
        updated_this_tick = false;
        return false;
    }
    static double ageMs(int64_t now, int64_t reference)
    {
        return reference < 0 || now < reference ? 0.0 : static_cast<double>(now - reference) / 1e6;
    }
    bool trustDue(int64_t now, double success_interval_ms, double retry_interval_ms) const
    {
        return !updated_this_tick && trust_success >= 0 &&
            ageMs(now, trust_success) >= success_interval_ms &&
            (trust_attempt < 0 || ageMs(now, trust_attempt) >= retry_interval_ms);
    }
    bool geomDue(int64_t now, double interval_ms) const
    {
        return !updated_this_tick && geom_success >= 0 && ageMs(now, geom_success) >= interval_ms;
    }
    void onTrustAttempt(int64_t now) { trust_attempt = now; }
    void onTrustSuccess(int64_t now) { trust_success = search_anchor = now; updated_this_tick = true; }
    void onGeomSuccess(int64_t now) { geom_success = search_anchor = now; updated_this_tick = true; }
    double searchRadius(int64_t now, double speed_mps, double minimum_m, double maximum_m, bool unbounded = false) const
    {
        const double elapsed_sec = ageMs(now, search_anchor) / 1000.0;
        const double radius = elapsed_sec * speed_mps;
        // 实验模式仅取消半宽上下限；场地边界仍由调用方裁剪。
        return unbounded ? radius : std::clamp(radius, minimum_m, maximum_m);
    }
};
