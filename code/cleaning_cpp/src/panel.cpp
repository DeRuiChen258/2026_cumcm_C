#include "model/panel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

#include "clean/rules.h"
#include "xlsx/xlsx.h"

namespace model {
namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

std::string col_letter(int col0) {
  std::string s;
  int c = col0 + 1;
  while (c > 0) {
    const int rem = (c - 1) % 26;
    s.insert(s.begin(), static_cast<char>('A' + rem));
    c = (c - 1) / 26;
  }
  return s;
}

std::string num(double v, int precision = 10) {
  char buf[64];
  if (std::isnan(v)) return "nan";
  std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
  return std::string(buf);
}

double cell_number(const xlsx::Grid& g, int row0, int col0) {
  if (row0 < 0 || row0 >= static_cast<int>(g.size())) return kNaN;
  if (col0 < 0 || col0 >= static_cast<int>(g[row0].size())) return kNaN;
  const xlsx::CellValue& v = g[row0][col0];
  if (v.is_number()) return v.number;
  if (v.is_text()) {
    const char* t = v.text.c_str();
    char* end = nullptr;
    const double d = std::strtod(t, &end);
    if (end != nullptr && end != t && *end == '\0') return d;
  }
  return kNaN;
}

std::string cell_text(const xlsx::Grid& g, int row0, int col0) {
  if (row0 < 0 || row0 >= static_cast<int>(g.size())) return std::string();
  if (col0 < 0 || col0 >= static_cast<int>(g[row0].size())) return std::string();
  const xlsx::CellValue& v = g[row0][col0];
  if (v.is_text()) return v.text;
  if (v.is_number()) {
    if (v.number > 30000.0) return xlsx::excel_date_to_iso(v.number);
    return std::to_string(static_cast<int>(v.number));
  }
  return std::string();
}

int parse_iso_day_index(const std::string& iso) {
  // "2025-01-01" 换成第 0 天
  if (iso.size() < 8) return -1;
  int y = 0, m = 0, d = 0;
  if (std::sscanf(iso.c_str(), "%d-%d-%d", &y, &m, &d) != 3) return -1;
  const int month_days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int doy = d - 1;
  for (int i = 0; i < m - 1; ++i) doy += month_days[i];
  if (y == 2025) return doy;
  if (y == 2026 && m == 1) return 365 + doy;
  return -1;
}

int vintage_index(const std::string& label, int fallback) {
  int h = -1, mi = -1;
  if (std::sscanf(label.c_str(), "%d:%d", &h, &mi) == 2) {
    if (h == 0) return 0;
    if (h == 6) return 1;
    if (h == 12) return 2;
    if (h == 18) return 3;
  }
  return fallback;
}

}  // namespace

Panel load_panel(const InputPaths& paths, const RuleParams& params) {
  Panel panel;

  // --- 时段标签：数据侧口径，右端点 ---
  panel.slots.reserve(kSlots);
  for (int k = 1; k <= kSlots; ++k) {
    // 右端点标签：0:10 … 23:50，最后一个收尾是 0:00+1
    panel.slots.push_back(k == kSlots ? "0:00+1" : xlsx::minutes_to_label(k * 10));
  }
  if (panel.slots[143] != "0:00+1") throw std::runtime_error("slot label convention broken (NR10)");

  // --- 附件2：负荷 + 实际光伏 ---
  const xlsx::Workbook wb2 = xlsx::read_workbook(paths.att2);
  const xlsx::Sheet* load_sheet = wb2.find("小区负载");
  const xlsx::Sheet* pv_sheet = wb2.find("光伏发电实际功率");
  if (load_sheet == nullptr || pv_sheet == nullptr) throw std::runtime_error("附件2 sheets not found");

  panel.load = Matrix(kDays, kSlots);
  panel.pv_actual = Matrix(kDays, kSlots);
  panel.dates.reserve(kDays);
  for (int d = 0; d < kDays; ++d) {
    const int row0 = d + 1;  // skip header row
    const std::string date_iso = xlsx::excel_date_to_iso(cell_number(load_sheet->grid, row0, 0));
    panel.dates.push_back(date_iso);
    for (int k = 0; k < kSlots; ++k) {
      panel.load.at(d, k) = cell_number(load_sheet->grid, row0, k + 1);
      panel.pv_actual.at(d, k) = cell_number(pv_sheet->grid, row0, k + 1);
    }
  }
  if (panel.dates.front() != "2025-01-01" || panel.dates.back() != "2025-12-31") {
    throw std::runtime_error("附件2 date range unexpected");
  }

  // --- 附件4：电价 ---
  const xlsx::Workbook wb4 = xlsx::read_workbook(paths.att4);
  if (wb4.size() == 0) throw std::runtime_error("附件4 empty");
  const xlsx::Sheet& price_sheet = wb4.at(0);
  panel.price = Matrix(kDays, kSlots);
  for (int d = 0; d < kDays; ++d) {
    for (int k = 0; k < kSlots; ++k) panel.price.at(d, k) = cell_number(price_sheet.grid, d + 1, k + 1);
  }

  // --- 附件1：代表性日（电价、负荷、光伏预测） ---
  const xlsx::Workbook wb1 = xlsx::read_workbook(paths.att1);
  if (wb1.size() == 0) throw std::runtime_error("附件1 empty");
  const xlsx::Sheet& day_sheet = wb1.at(0);
  panel.day1.assign(kSlots * 3, 0.0);
  for (int k = 0; k < kSlots; ++k) {
    const int row0 = k + 1;
    // 按 C 序排成 (144, 3)：第 k 行就是 [电价, 负荷, 光伏预测]
    panel.day1[3 * k + 0] = cell_number(day_sheet.grid, row0, 1);
    panel.day1[3 * k + 1] = cell_number(day_sheet.grid, row0, 2);
    panel.day1[3 * k + 2] = cell_number(day_sheet.grid, row0, 3);
  }

  // --- 附件3：四个发布时次的 24 小时整点光伏预报 ---
  const xlsx::Workbook wb3 = xlsx::read_workbook(paths.att3);
  if (wb3.size() == 0) throw std::runtime_error("附件3 empty");
  const xlsx::Sheet& fc_sheet = wb3.at(0);
  panel.pv_forecast.assign(static_cast<size_t>(kDays) * kVintages * kForecastHours, kNaN);
  std::map<int, int> vintage_seen;
  std::string last_date;
  for (int r = 1; r < static_cast<int>(fc_sheet.grid.size()); ++r) {
    const std::string date_txt = cell_text(fc_sheet.grid, r, 0);
    if (!date_txt.empty()) last_date = date_txt;
    int day = parse_iso_day_index(last_date);
    if (day < 0 || day >= kDays) continue;
    const int j = vintage_index(cell_text(fc_sheet.grid, r, 1), vintage_seen[day] % kVintages);
    vintage_seen[day] += 1;
    for (int h = 1; h <= kForecastHours; ++h) {
      const double v = cell_number(fc_sheet.grid, r, h + 1);
      panel.pv_forecast[(static_cast<size_t>(day) * kVintages + j) * kForecastHours + (h - 1)] =
          std::isnan(v) ? 0.0 : v;
    }
  }

  // --- 日照窗口：按日自适应，不写死时钟 ---
  panel.windows.assign(kDays, DayWindow{});
  panel.pv_global_max = 0.0;
  for (int d = 0; d < kDays; ++d) {
    double peak = 0.0;
    for (int k = 0; k < kSlots; ++k) peak = std::max(peak, panel.pv_actual.at(d, k));
    panel.pv_global_max = std::max(panel.pv_global_max, peak);
    const double theta = std::max(params.theta_rel * peak, params.theta_abs);
    int k_rise = -1, k_set = -1;
    for (int k = 0; k < kSlots; ++k) {
      if (panel.pv_actual.at(d, k) > theta) {
        if (k_rise < 0) k_rise = k;
        k_set = k;
      }
    }
    DayWindow& w = panel.windows[d];
    w.theta = theta;
    w.peak = peak;
    w.valid = (k_rise >= 0 && k_set > k_rise);
    w.k_rise = w.valid ? k_rise : 0;
    w.k_set = w.valid ? k_set : 0;
    if (!w.valid) throw std::runtime_error("window inference failed on day " + std::to_string(d));
  }

  const std::string run_id = "clean-2025-v1";
  auto push_audit = [&](const std::string& file, const std::string& sheet, int excel_row, int excel_col,
                        const std::string& date, int slot1, const std::string& rule_id,
                        const std::string& action, const std::string& old_v, const std::string& new_v,
                        const std::string& reason) {
    AuditRecord a;
    a.run_id = run_id;
    a.file = file;
    a.sheet = sheet;
    a.excel_row = excel_row;
    a.excel_col = col_letter(excel_col);
    a.date = date;
    a.slot_index = slot1;
    a.slot_label = panel.slots[slot1 - 1];
    char hb[16];
    std::snprintf(hb, sizeof(hb), "%d:%02d", (slot1 * 10) / 60, (slot1 * 10) % 60);
    a.hour_end = std::string(hb);
    a.rule_id = rule_id;
    a.action = action;
    a.old_value = old_v;
    a.new_value = new_v;
    a.reason = reason;
    panel.audit.push_back(a);
  };
  auto push_quarantine = [&](const std::string& file, const std::string& sheet, int excel_row, int excel_col,
                             const std::string& date, int slot1, const std::string& value,
                             const std::string& rule_id, const std::string& reason) {
    QuarantineRecord q;
    q.run_id = run_id;
    q.file = file;
    q.sheet = sheet;
    q.excel_row = excel_row;
    q.excel_col = col_letter(excel_col);
    q.date = date;
    q.slot_index = slot1;
    q.slot_label = panel.slots[slot1 - 1];
    q.value = value;
    q.rule_id = rule_id;
    q.reason = reason;
    panel.quarantine.push_back(q);
  };

  // --- 实际光伏走 NR1–NR6 ---
  panel.pv_clean = panel.pv_actual;
  std::vector<std::string> last_rule(kDays * kSlots);
  for (int d = 0; d < kDays; ++d) {
    const DayWindow& w = panel.windows[d];
    for (int k = 0; k < kSlots; ++k) {
      const double raw = panel.pv_actual.at(d, k);
      const bool in_shoulder = (k == w.k_rise - 1 || k == w.k_rise || k == w.k_set || k == w.k_set + 1);
      const bool in_core = (k < w.k_rise - 1) || (k > w.k_set + 1);
      const clean::PointResult pr = clean::apply_pv_point(raw, in_core, in_shoulder, params);
      if (pr.action == clean::Action::kNone) continue;
      const int excel_row = d + 2;
      const int excel_col = k + 1;  // column B (index 1) holds slot 1
      const std::string old_s = num(raw);
      if (pr.action == clean::Action::kSetZero) {
        panel.pv_clean.at(d, k) = 0.0;
        push_audit("附件2", "光伏发电实际功率", excel_row, excel_col, panel.dates[d], k + 1, pr.rule_id,
                   "set_zero", old_s, "0", pr.reason);
        last_rule[static_cast<size_t>(d) * kSlots + k] = pr.rule_id;
      } else if (pr.action == clean::Action::kReject) {
        panel.pv_clean.at(d, k) = kNaN;
        push_audit("附件2", "光伏发电实际功率", excel_row, excel_col, panel.dates[d], k + 1, pr.rule_id,
                   "reject", old_s, "nan", pr.reason);
        push_quarantine("附件2", "光伏发电实际功率", excel_row, excel_col, panel.dates[d], k + 1, old_s,
                        pr.rule_id, pr.reason);
        last_rule[static_cast<size_t>(d) * kSlots + k] = pr.rule_id;
      } else if (pr.action == clean::Action::kFlag) {
        push_audit("附件2", "光伏发电实际功率", excel_row, excel_col, panel.dates[d], k + 1, pr.rule_id,
                   "flag", old_s, old_s, pr.reason);
        last_rule[static_cast<size_t>(d) * kSlots + k] = pr.rule_id;
      }
    }
  }

  // --- NR6：日照窗口内缺测插值，只在当天内插，不跨夜 ---
  for (int d = 0; d < kDays; ++d) {
    const DayWindow& w = panel.windows[d];
    int k = w.k_rise;
    while (k <= w.k_set) {
      if (!std::isnan(panel.pv_clean.at(d, k))) {
        ++k;
        continue;
      }
      int gap_end = k;
      while (gap_end <= w.k_set && std::isnan(panel.pv_clean.at(d, gap_end))) ++gap_end;
      const int gap_len = gap_end - k;
      const int left = k - 1, right = gap_end;
      if (gap_len <= params.interp_max_gap && left >= w.k_rise && right <= w.k_set &&
          !std::isnan(panel.pv_clean.at(d, left)) && !std::isnan(panel.pv_clean.at(d, right))) {
        for (int g = k; g < gap_end; ++g) {
          const double t = static_cast<double>(g - left) / static_cast<double>(right - left);
          const double v = panel.pv_clean.at(d, left) * (1.0 - t) + panel.pv_clean.at(d, right) * t;
          panel.pv_clean.at(d, g) = v;
          push_audit("附件2", "光伏发电实际功率", d + 2, g + 1, panel.dates[d], g + 1, "NR6", "interpolate",
                     "nan", num(v), "daylight-window gap interpolation");
        }
      } else {
        for (int g = k; g < gap_end; ++g) {
          push_quarantine("附件2", "光伏发电实际功率", d + 2, g + 1, panel.dates[d], g + 1, "nan", "NR6",
                          "gap longer than interp_max_gap or touches window edge");
        }
        for (int g = k; g < gap_end; ++g) panel.pv_clean.at(d, g) = kNaN;
      }
      k = gap_end;
    }
  }

  // --- NR9：日间相邻时段跳变超过 20%，只标记不改值 ---
  for (int d = 0; d < kDays; ++d) {
    const DayWindow& w = panel.windows[d];
    for (int k = w.k_rise; k < w.k_set; ++k) {
      const double a = panel.pv_clean.at(d, k), b = panel.pv_clean.at(d, k + 1);
      if (std::isnan(a) || std::isnan(b)) continue;
      const double denom = std::max(std::fabs(a), std::fabs(b));
      if (denom <= 1e-9) continue;
      if (std::fabs(b - a) / denom > 0.20) {
        push_audit("附件2", "光伏发电实际功率", d + 2, k + 2, panel.dates[d], k + 2, "NR9", "flag",
                   "nan", "nan", "daylight jump > 20%");
      }
    }
  }

  // --- NR7：预报侧按同样的夜间口径处理 ---
  panel.pv_forecast_clean = panel.pv_forecast;
  const int vintage_hour[4] = {0, 6, 12, 18};
  for (int d = 0; d < kDays; ++d) {
    for (int j = 0; j < kVintages; ++j) {
      for (int h = 1; h <= kForecastHours; ++h) {
        const int abs_hour = vintage_hour[j] + h;
        int target_day = d;
        int hour_of_day = abs_hour;
        if (abs_hour > 24) {
          target_day = std::min(d + 1, kDays - 1);
          hour_of_day = abs_hour - 24;
        }
        const int slot = hour_of_day * 6 - 1;  // last slot of that clock hour
        if (slot < 0 || slot >= kSlots) continue;
        const DayWindow& w = panel.windows[target_day];
        const bool in_day = (slot >= w.k_rise && slot <= w.k_set);
        const bool in_shoulder =
            (slot == w.k_rise - 1 || slot == w.k_rise || slot == w.k_set || slot == w.k_set + 1);
        const bool in_core = (slot < w.k_rise - 1) || (slot > w.k_set + 1);
        if (in_day && !in_shoulder) continue;
        const double raw = panel.pv_forecast[(static_cast<size_t>(d) * kVintages + j) * kForecastHours + (h - 1)];
        const clean::PointResult pr = clean::apply_pv_point(raw, in_core, in_shoulder, params);
        if (pr.action == clean::Action::kNone) continue;
        const int excel_row = d * kVintages + j + 2;
        const int excel_col = h + 1;
        const std::string old_s = std::to_string(raw);
        if (pr.action == clean::Action::kSetZero) {
          panel.pv_forecast_clean[(static_cast<size_t>(d) * kVintages + j) * kForecastHours + (h - 1)] = 0.0;
          push_audit("附件3", "Sheet1", excel_row, excel_col, panel.dates[d], slot + 1, "NR7>NR1", "set_zero",
                     old_s, "0", "forecast-side core-night zero drift");
        } else if (pr.action == clean::Action::kReject) {
          panel.pv_forecast_clean[(static_cast<size_t>(d) * kVintages + j) * kForecastHours + (h - 1)] = kNaN;
          push_audit("附件3", "Sheet1", excel_row, excel_col, panel.dates[d], slot + 1, "NR7>NR2", "reject",
                     old_s, "nan", "forecast-side core-night outlier");
          push_quarantine("附件3", "Sheet1", excel_row, excel_col, panel.dates[d], slot + 1, old_s, "NR7>NR2",
                          "forecast-side core-night outlier");
        } else if (pr.action == clean::Action::kFlag) {
          push_audit("附件3", "Sheet1", excel_row, excel_col, panel.dates[d], slot + 1,
                     "NR7>" + pr.rule_id, "flag", old_s, old_s, "forecast-side shoulder flag");
        }
      }
    }
  }

  // --- NR11 / NR12：只标记不改值 ---
  for (int d = 0; d < kDays; ++d) {
    for (int k = 0; k < kSlots; ++k) {
      const double price = panel.price.at(d, k);
      if (price < 0.1) {
        push_audit("附件4", "Sheet1", d + 2, k + 1, panel.dates[d], k + 1, "NR11", "flag",
                   num(price), num(price), "near-zero price (<0.1 CNY/kWh), kept for Q4 scenarios");
      }
      const DayWindow& w = panel.windows[d];
      const bool in_core = (k < w.k_rise - 1) || (k > w.k_set + 1);
      const double load = panel.load.at(d, k);
      if (in_core && (load < 1500.0 || load > 6000.0)) {
        push_audit("附件2", "小区负载", d + 2, k + 1, panel.dates[d], k + 1, "NR12", "flag", num(load),
                   num(load), "night load outside [1500,6000] kW");
      }
    }
  }

  // --- 窗口跳变检查：日出日落一天挪太多就记账 ---
  for (int d = 1; d < kDays; ++d) {
    const int dr = std::abs(panel.windows[d].k_rise - panel.windows[d - 1].k_rise);
    const int ds = std::abs(panel.windows[d].k_set - panel.windows[d - 1].k_set);
    if (dr > params.window_jump_slots || ds > params.window_jump_slots) {
      push_audit("附件2", "光伏发电实际功率", d + 2, 1, panel.dates[d], 1, "WINDOW_JUMP", "flag", "nan",
                 "nan", "sunrise/sunset shift exceeded window_jump_slots");
    }
  }

  // --- NR8：共形尺度隔离，夜间用 floor 兜底 ---
  panel.sigma_floor = std::max(1.0, 0.005 * panel.pv_global_max);
  panel.sigma_raw.assign(kSlots, 0.0);
  panel.sigma_used.assign(kSlots, 0.0);
  panel.sigma_n.assign(kSlots, 0);
  for (int k = 0; k < kSlots; ++k) {
    double sum = 0.0, sum2 = 0.0;
    int n = 0;
    for (int d = 0; d < kDays; ++d) {
      const DayWindow& w = panel.windows[d];
      if (!(k >= w.k_rise && k <= w.k_set)) continue;
      const double act = panel.pv_clean.at(d, k);
      if (std::isnan(act) || act <= w.theta) continue;
      // 用同一小时、0:00 时次的预报算日前残差
      const int hour = std::min(k / 6 + 1, kForecastHours);
      const double fc = panel.pv_forecast_clean[(static_cast<size_t>(d) * kVintages + 0) * kForecastHours + (hour - 1)];
      if (std::isnan(fc)) continue;
      const double res = act - fc;
      sum += res;
      sum2 += res * res;
      ++n;
    }
    panel.sigma_n[k] = n;
    const double mean = n > 0 ? sum / n : 0.0;
    const double var = n > 1 ? std::max(0.0, sum2 / n - mean * mean) : 0.0;
    panel.sigma_raw[k] = std::sqrt(var);
    panel.sigma_used[k] = std::max(panel.sigma_raw[k], panel.sigma_floor);
  }

  return panel;
}

}  // namespace model
