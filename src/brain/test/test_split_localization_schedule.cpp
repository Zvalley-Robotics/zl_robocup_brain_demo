#include "split_localization_schedule.h"
#include <cassert>
#include <cmath>

#ifdef NDEBUG
#error "测试必须启用 assert"
#endif

int main()
{
    constexpr int64_t ms = 1000000;
    SplitLocalizationSchedule s;
    s.initialize(0); // 模拟时钟 0 是有效的时间点。
    assert(!s.trustDue(9999 * ms, 10000, 500));
    s.beginTick(9999 * ms);
    s.onGeomSuccess(9999 * ms);
    assert(!s.geomDue(9999 * ms, 0)); // 同一个 tick 最多一次更新。
    s.beginTick(10000 * ms);
    assert(s.trustDue(10000 * ms, 10000, 500)); // 几何成功不推迟 TRUSTDIR。
    s.onTrustAttempt(10000 * ms); // 失败后只刷新尝试时刻。
    s.beginTick(10499 * ms);
    assert(!s.trustDue(10499 * ms, 10000, 500));
    s.beginTick(10500 * ms);
    assert(s.trustDue(10500 * ms, 10000, 500));
    s.onTrustSuccess(10500 * ms);
    assert(!s.geomDue(10500 * ms, 300));
    s.beginTick(10501 * ms);
    assert(s.geomDue(10501 * ms, 300)); // TRUSTDIR 成功不重置几何间隔。
    assert(std::fabs(s.searchRadius(10501 * ms, 0.1, 0.2, 2.0) - 0.2) < 1e-9);
    assert(std::fabs(s.searchRadius(50500 * ms, 0.1, 0.2, 2.0) - 2.0) < 1e-9);
    assert(std::fabs(s.searchRadius(10501 * ms, 0.1, 0.2, 2.0, true) - 0.0001) < 1e-9);
    assert(std::fabs(s.searchRadius(50500 * ms, 0.1, 0.2, 2.0, true) - 4.0) < 1e-9);
    assert(s.beginTick(9000 * ms)); // 时间回退后重新设定时间参考。
    assert(!s.trustDue(9000 * ms, 10000, 500));
    assert(!s.geomDue(9000 * ms, 300));
    assert(s.trust_attempt == -1);
    s.clear();
    assert(s.trust_success == -1 && s.geom_success == -1);
}
