#pragma once
#include <string>

#include "model/panel.h"

namespace clean {

enum class Action { kNone, kSetZero, kReject, kFlag };

struct PointResult {
  Action action = Action::kNone;
  double value = 0.0;   // transformed value (NaN when rejected)
  bool quarantine = false;
  std::string rule_id;
  std::string action_name;
  std::string reason;
};

// NR1–NR5：单个光伏格的开关规则（NR3 负值先拦）。
PointResult apply_pv_point(double value, bool in_core, bool in_shoulder, const model::RuleParams& p);

// 拒绝（置 NaN 并进隔离清单）的公共封装，NR7 预报侧也用。
PointResult make_reject(const std::string& rule_id, const std::string& reason, double value);

// 规则说明表，自检和质量报告要用。
const char* rule_description(const std::string& rule_id);

}  // namespace clean
