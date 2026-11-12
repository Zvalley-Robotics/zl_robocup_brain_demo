#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// 候选评分沿用 Locator 的无量纲归一化评分，不是米制误差。
struct OneMCandidateChoice
{
    size_t index;
    double x, y, score;
};

struct OneMSelection
{
    bool found = false, ambiguous = false;
    size_t index = 0;
    double score = 0.0;
};

// 同分时按原输入顺序取首个；近似同分但位置明显不同则拒绝更新。
inline OneMSelection selectOneMCandidate(const std::vector<OneMCandidateChoice> &candidates,
                                         double score_margin, double separation_m)
{
    OneMSelection out;
    size_t best = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const auto &c = candidates[i];
        if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.score)) continue;
        if (c.score < best_score) { best_score = c.score; best = i; out.found = true; }
    }
    if (!out.found) return out;
    out.index = candidates[best].index;
    out.score = best_score;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (i == best) continue;
        const auto &c = candidates[i];
        if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.score)) continue;
        if (c.score - best_score <= score_margin &&
            std::hypot(c.x - candidates[best].x, c.y - candidates[best].y) > separation_m)
            out.ambiguous = true;
    }
    return out;
}
