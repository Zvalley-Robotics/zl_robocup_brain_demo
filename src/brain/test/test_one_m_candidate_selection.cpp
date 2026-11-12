#include "one_m_candidate_selection.h"
#include <cassert>
#include <limits>

#ifdef NDEBUG
#error "测试必须启用 assert"
#endif

int main()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(!selectOneMCandidate({}, 0.02, 0.2).found);
    assert(!selectOneMCandidate({{0, nan, 0, 0.1}}, 0.02, 0.2).found);
    auto one = selectOneMCandidate({{4, 1, 2, 0.1}}, 0.02, 0.2);
    assert(one.found && !one.ambiguous && one.index == 4);
    auto ranked = selectOneMCandidate({{2, 0, 0, 0.2}, {5, 1, 0, 0.1}}, 0.02, 0.2);
    assert(ranked.found && !ranked.ambiguous && ranked.index == 5);
    auto tie = selectOneMCandidate({{7, 0, 0, 0.1}, {8, 1, 0, 0.1}}, 0.02, 0.2);
    assert(tie.found && tie.ambiguous && tie.index == 7);
    auto same = selectOneMCandidate({{7, 0, 0, 0.1}, {8, 0.1, 0, 0.1}}, 0.02, 0.2);
    assert(same.found && !same.ambiguous && same.index == 7);
    auto ambiguous_second = selectOneMCandidate({{1, 0, 0, 0.1}, {2, 0.1, 0, 0.11}, {3, 1, 0, 0.119}}, 0.02, 0.2);
    assert(ambiguous_second.found && ambiguous_second.ambiguous);
}
