#include "clean/rules.h"

#include <cmath>
#include <limits>

namespace clean {

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
}

PointResult make_reject(const std::string& rule_id, const std::string& reason, double value) {
  PointResult r;
  r.action = Action::kReject;
  r.value = kNaN;
  r.quarantine = true;
  r.rule_id = rule_id;
  r.action_name = "reject";
  r.reason = reason + " (old=" + std::to_string(value) + ")";
  return r;
}

PointResult apply_pv_point(double value, bool in_core, bool in_shoulder, const model::RuleParams& p) {
  PointResult r;
  r.value = value;
  // NR3：负功率不可能出现，任何时段都拦
  if (std::isnan(value)) {
    r.rule_id = "NR3";
    r.action = Action::kFlag;
    r.action_name = "flag";
    r.reason = "missing value inside daylight window (NR6 candidate)";
    return r;
  }
  if (value < 0.0) return make_reject("NR3", "negative PV power", value);

  if (in_core) {
    if (value > p.noise_kw) return make_reject("NR2", "core-night value above noise band", value);
    if (value > 0.0) {  // 0 < v <= 5 kW : sensor zero drift
      r.action = Action::kSetZero;
      r.value = 0.0;
      r.rule_id = "NR1";
      r.action_name = "set_zero";
      r.reason = "core-night zero-drift within noise band";
      return r;
    }
    return r;  // exact zero: keep
  }

  if (in_shoulder) {
    if (value >= p.shoulder_kw && value <= p.noise_kw) {  // [1, 5] kW: weak but kept
      r.action = Action::kFlag;
      r.value = value;
      r.rule_id = "NR5";
      r.action_name = "flag";
      r.reason = "shoulder weak signal, excluded from sigma estimation";
      return r;
    }
    if (value > 0.0 && value < p.shoulder_kw) {  // (0, 1) kW: sub-kW noise
      r.action = Action::kSetZero;
      r.value = 0.0;
      r.rule_id = "NR4";
      r.action_name = "set_zero";
      r.reason = "shoulder sub-kW noise";
      return r;
    }
  }
  return r;
}

const char* rule_description(const std::string& rule_id) {
  if (rule_id == "NR1") return "core-night zero drift (0,5] kW -> 0";
  if (rule_id == "NR2") return "core-night hard outlier (>5 kW) -> NaN + quarantine";
  if (rule_id == "NR3") return "negative PV (any segment) -> NaN + quarantine";
  if (rule_id == "NR4") return "shoulder sub-kW noise (0,1) kW -> 0";
  if (rule_id == "NR5") return "shoulder weak signal [1,5] kW -> flag (keep)";
  if (rule_id == "NR6") return "daylight-window gap interpolation (<= max_gap slots)";
  if (rule_id == "NR7") return "forecast-side cleaning with identical night semantics";
  if (rule_id == "NR8") return "conformal scale isolation with floor";
  if (rule_id == "NR9") return "daylight jump > 20% -> flag only";
  if (rule_id == "NR10") return "time convention assertion (right endpoint, slot<->column)";
  if (rule_id == "NR11") return "price < 0.1 CNY/kWh -> near-zero flag";
  if (rule_id == "NR12") return "night load outside [1500,6000] kW -> flag";
  if (rule_id == "WINDOW_JUMP") return "sunrise/sunset window jump > 3 slots";
  return "unknown rule";
}

}  // namespace clean
