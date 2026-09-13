// cleaning_cpp —— 数据清洗（C++17）：附件解析、口径归一、NR1–NR12 规则、审计与导出。
// 凡是改数值的事都在这里做，Python 那边只读 clean/ 的结果。
// 子命令：inspect | extract | clean | export | verify | selftest
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "clean/rules.h"
#include "io/writers.h"
#include "model/panel.h"
#include "util/sha256.h"
#include "xlsx/xlsx.h"

namespace fs = std::filesystem;
using model::kDays;
using model::kSlots;

namespace {

struct Options {
  std::string in_dir = "Data";
  std::string out_dir = "clean";
  std::string templates_dir = "templates/附件5";
  bool strict = false;
  model::RuleParams params;
};

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--in") o.in_dir = next();
    else if (a == "--out") o.out_dir = next();
    else if (a == "--templates") o.templates_dir = next();
    else if (a == "--strict") o.strict = true;
    else if (a == "--noise-kw") o.params.noise_kw = std::atof(next().c_str());
    else if (a == "--shoulder-kw") o.params.shoulder_kw = std::atof(next().c_str());
    else if (a == "--theta-rel") o.params.theta_rel = std::atof(next().c_str());
    else if (a == "--theta-abs") o.params.theta_abs = std::atof(next().c_str());
    else if (a == "--interp-max-gap") o.params.interp_max_gap = std::atoi(next().c_str());
    else throw std::runtime_error("unknown option: " + a);
  }
  return o;
}

model::InputPaths make_paths(const Options& o) {
  model::InputPaths p;
  p.att1 = (fs::path(o.in_dir) / "附件1.xlsx").string();
  p.att2 = (fs::path(o.in_dir) / "附件2.xlsx").string();
  p.att3 = (fs::path(o.in_dir) / "附件3.xlsx").string();
  p.att4 = (fs::path(o.in_dir) / "附件4.xlsx").string();
  p.templates_dir = o.templates_dir;
  return p;
}

long peak_rss_kb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;
}

std::vector<std::string> template_files(const std::string& dir) {
  std::vector<std::string> out;
  if (!fs::exists(dir)) return out;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".xlsx") out.push_back(e.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string template_schema_json(const std::string& dir) {
  std::string out = "{\n  \"templates\": [\n";
  const std::vector<std::string> files = template_files(dir);
  for (size_t fi = 0; fi < files.size(); ++fi) {
    const xlsx::Workbook wb = xlsx::read_workbook(files[fi]);
    out += "    {\n      \"file\": \"" + io::json_escape(fs::path(files[fi]).filename().string()) + "\",\n";
    out += "      \"sheets\": [\n";
    for (size_t si = 0; si < wb.size(); ++si) {
      const xlsx::Sheet& s = wb.at(si);
      const int rows = static_cast<int>(s.grid.size());
      int cols = 0;
      for (const auto& r : s.grid) cols = std::max(cols, static_cast<int>(r.size()));
      out += "        {\"name\": \"" + io::json_escape(s.name) + "\", \"rows\": " + std::to_string(rows) +
             ", \"cols\": " + std::to_string(cols) + ", \"header\": [";
      if (rows > 0) {
        for (int c = 0; c < static_cast<int>(s.grid[0].size()); ++c) {
          if (c) out += ", ";
          out += "\"" + io::json_escape(s.grid[0][c].text) + "\"";
        }
      }
      out += "], \"row_labels\": [";
      for (int r = 1; r < rows; ++r) {
        if (r > 1) out += ", ";
        out += "\"" + io::json_escape(s.grid[r].empty() ? std::string() : s.grid[r][0].text) + "\"";
      }
      out += "]}" + std::string(si + 1 < wb.size() ? "," : "") + "\n";
    }
    out += "      ]\n    }" + std::string(fi + 1 < files.size() ? "," : "") + "\n";
  }
  out += "  ]\n}\n";
  return out;
}

// ---------- inspect：只读地打印结构，不改数据 ----------
int cmd_inspect(const Options& o) {
  const std::vector<std::string> files = {(fs::path(o.in_dir) / "附件1.xlsx").string(),
                                          (fs::path(o.in_dir) / "附件2.xlsx").string(),
                                          (fs::path(o.in_dir) / "附件3.xlsx").string(),
                                          (fs::path(o.in_dir) / "附件4.xlsx").string()};
  for (const std::string& f : files) {
    std::cout << "== " << f << " sha256=" << util::Sha256::file(f).substr(0, 16) << "\n";
    const xlsx::Workbook wb = xlsx::read_workbook(f);
    for (const xlsx::Sheet& s : wb.sheets) {
      int cols = 0;
      int num_cells = 0, text_cells = 0, empty_cells = 0;
      for (const auto& row : s.grid) {
        cols = std::max(cols, static_cast<int>(row.size()));
        for (const auto& c : row) {
          if (c.empty()) ++empty_cells;
          else if (c.is_text()) ++text_cells;
          else ++num_cells;
        }
      }
      std::cout << "   sheet '" << s.name << "' rows=" << s.grid.size() << " cols=" << cols
                << " numeric=" << num_cells << " text=" << text_cells << " empty=" << empty_cells << "\n";
      if (s.grid.size() > 1) {
        std::cout << "      row2 sample:";
        for (int c = 0; c < std::min(6, static_cast<int>(s.grid[1].size())); ++c) {
          const xlsx::CellValue& v = s.grid[1][c];
          std::cout << " [" << (v.is_text() ? v.text : io::fmt(v.number, 6)) << "]";
        }
        std::cout << "\n";
      }
    }
  }
  for (const std::string& t : template_files(o.templates_dir)) {
    const xlsx::Workbook wb = xlsx::read_workbook(t);
    std::cout << "== template " << fs::path(t).filename().string() << " sheets=";
    for (size_t i = 0; i < wb.size(); ++i) {
      int cols = 0;
      for (const auto& r : wb.at(i).grid) cols = std::max(cols, static_cast<int>(r.size()));
      std::cout << "[" << wb.at(i).name << " " << wb.at(i).grid.size() << "x" << cols << "]";
    }
    std::cout << "\n";
  }
  return 0;
}

// ---------- 各类 CSV 输出：窗口表、审计表、质量报告、sigma ----------
void write_support_csv(const model::Panel& p, const std::string& out_dir) {
  // 日照窗口表
std::vector<std::vector<std::string>> w_rows;
  w_rows.push_back({"date", "k_rise", "k_set", "sunrise", "sunset", "day_hours",
                    "night_core_hours", "theta_d", "peak_d"});
  for (int d = 0; d < kDays; ++d) {
    const model::DayWindow& w = p.windows[d];
    // 口径：日出取当天第一个有效时段的起点，日落取最后一个有效时段的起点，
    // 白天时长 = (k_set-k_rise)/6 小时（和提示词 N5 的数字对得上）
    char rb[16], sb[16];
    std::snprintf(rb, sizeof(rb), "%d:%02d", (w.k_rise * 10) / 60, (w.k_rise * 10) % 60);
    std::snprintf(sb, sizeof(sb), "%d:%02d", (w.k_set * 10) / 60, (w.k_set * 10) % 60);
    const std::string rise = rb, set = sb;
    const double day_hours = (w.k_set - w.k_rise) / 6.0;
    const double night_hours = 24.0 - day_hours;
    w_rows.push_back({p.dates[d], std::to_string(w.k_rise + 1), std::to_string(w.k_set + 1), rise, set,
                      io::fmt(day_hours, 6), io::fmt(night_hours, 6), io::fmt(w.theta, 6),
                      io::fmt(w.peak, 8)});
  }
  io::write_csv((fs::path(out_dir) / "day_windows.csv").string(), w_rows);

  // 审计表（列顺序按提示词固定）
  std::vector<std::vector<std::string>> a_rows;
  a_rows.push_back({"run_id", "file", "sheet", "excel_row", "excel_col", "date", "slot_index",
                    "slot_label", "hour_end", "rule_id", "action", "old_value", "new_value", "reason"});
  for (const model::AuditRecord& a : p.audit) {
    a_rows.push_back({a.run_id, a.file, a.sheet, std::to_string(a.excel_row), a.excel_col, a.date,
                      std::to_string(a.slot_index), a.slot_label, a.hour_end, a.rule_id, a.action,
                      a.old_value, a.new_value, a.reason});
  }
  io::write_csv((fs::path(out_dir) / "cleaning_audit.csv").string(), a_rows);

  // 隔离清单
  std::vector<std::vector<std::string>> q_rows;
  q_rows.push_back({"run_id", "file", "sheet", "excel_row", "excel_col", "date", "slot_index",
                    "slot_label", "value", "rule_id", "reason"});
  for (const model::QuarantineRecord& q : p.quarantine) {
    q_rows.push_back({q.run_id, q.file, q.sheet, std::to_string(q.excel_row), q.excel_col, q.date,
                      std::to_string(q.slot_index), q.slot_label, q.value, q.rule_id, q.reason});
  }
  io::write_csv((fs::path(out_dir) / "quarantine.csv").string(), q_rows);

  // 质量报告：按整点小时统计（清洗后）
  std::vector<std::vector<std::string>> h_rows;
  h_rows.push_back({"hour", "cells", "zero_rate", "noise_rate", "reject_count", "mean_pv", "max_pv"});
  for (int hour = 0; hour < 24; ++hour) {
    int cells = 0, zeros = 0, noise = 0, rejects = 0;
    double sum = 0.0, max_v = 0.0;
    for (int k = hour * 6; k < hour * 6 + 6; ++k) {
      for (int d = 0; d < kDays; ++d) {
        ++cells;
        const double raw = p.pv_actual.at(d, k);
        if (raw == 0.0) ++zeros;
        if (raw > 0.0 && raw <= 5.0) ++noise;
        const double cl = p.pv_clean.at(d, k);
        if (std::isnan(cl)) ++rejects;
        else {
          sum += cl;
          max_v = std::max(max_v, cl);
        }
      }
    }
    char hb[8];
    std::snprintf(hb, sizeof(hb), "%02d:00", hour);
    h_rows.push_back({hb, std::to_string(cells), io::fmt(static_cast<double>(zeros) / cells, 6),
                      io::fmt(static_cast<double>(noise) / cells, 6), std::to_string(rejects),
                      io::fmt(sum / cells, 8), io::fmt(max_v, 8)});
  }
  io::write_csv((fs::path(out_dir) / "quality_report.csv").string(), h_rows);

  // 共形尺度 sigma（NR8）
  std::vector<std::vector<std::string>> s_rows;
  s_rows.push_back({"slot_index", "slot_label", "sigma_raw", "sigma_used", "n_samples"});
  for (int k = 0; k < kSlots; ++k) {
    s_rows.push_back({std::to_string(k + 1), p.slots[k], io::fmt(p.sigma_raw[k], 8),
                      io::fmt(p.sigma_used[k], 8), std::to_string(p.sigma_n[k])});
  }
  io::write_csv((fs::path(out_dir) / "sigma.csv").string(), s_rows);

  // 窗口的月度汇总
  std::vector<std::vector<std::string>> m_rows;
  m_rows.push_back({"month", "days", "sunrise_median", "sunset_median", "night_core_hours_median"});
  int starts[13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
  for (int m = 0; m < 12; ++m) {
    std::vector<double> rise, set, night;
    for (int d = starts[m]; d < starts[m + 1]; ++d) {
      rise.push_back(p.windows[d].k_rise / 6.0);
      set.push_back(p.windows[d].k_set / 6.0);
      night.push_back(24.0 - (p.windows[d].k_set - p.windows[d].k_rise) / 6.0);
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    m_rows.push_back({std::to_string(m + 1), std::to_string(starts[m + 1] - starts[m]), io::fmt(med(rise), 4),
                      io::fmt(med(set), 4), io::fmt(med(night), 4)});
  }
  io::write_csv((fs::path(out_dir) / "window_monthly.csv").string(), m_rows);
}

int cmd_extract(const Options& o) {
  const model::Panel p = model::load_panel(make_paths(o), o.params);
  fs::create_directories(o.out_dir);
  const std::string raw_dir = (fs::path(o.out_dir) / "raw").string();
  fs::create_directories(raw_dir);
  io::write_npy_f64((fs::path(raw_dir) / "pv_actual.npy").string(), p.pv_actual.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(raw_dir) / "load.npy").string(), p.load.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(raw_dir) / "price.npy").string(), p.price.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(raw_dir) / "pv_forecast.npy").string(), p.pv_forecast.data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(model::kVintages),
                     static_cast<size_t>(model::kForecastHours)});
  io::write_npy_f64((fs::path(raw_dir) / "day1.npy").string(), p.day1.data(),
                    {static_cast<size_t>(kSlots), static_cast<size_t>(3)});
  io::write_text((fs::path(raw_dir) / "README.txt").string(),
                 "Raw (pre-rule) matrices exported by cleaning_cpp extract. Values are bit-identical to\n"
                 "the attachment cells; NaN marks unreadable cells. Use `clean` for the audited bundle.\n");
  std::cout << "[extract] wrote raw matrices to " << raw_dir << "\n";
  return 0;
}

int cmd_clean(const Options& o) {
  const auto t0 = std::chrono::steady_clock::now();
  const model::InputPaths paths = make_paths(o);
  const model::Panel p = model::load_panel(paths, o.params);
  fs::create_directories(o.out_dir);

  io::write_npy_f64((fs::path(o.out_dir) / "pv_actual.npy").string(), p.pv_clean.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(o.out_dir) / "pv_actual_raw.npy").string(), p.pv_actual.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(o.out_dir) / "load.npy").string(), p.load.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(o.out_dir) / "price.npy").string(), p.price.data().data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(kSlots)});
  io::write_npy_f64((fs::path(o.out_dir) / "pv_forecast.npy").string(), p.pv_forecast_clean.data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(model::kVintages),
                     static_cast<size_t>(model::kForecastHours)});
  io::write_npy_f64((fs::path(o.out_dir) / "pv_forecast_raw.npy").string(), p.pv_forecast.data(),
                    {static_cast<size_t>(kDays), static_cast<size_t>(model::kVintages),
                     static_cast<size_t>(model::kForecastHours)});
  io::write_npy_f64((fs::path(o.out_dir) / "day1.npy").string(), p.day1.data(),
                    {static_cast<size_t>(kSlots), static_cast<size_t>(3)});
  io::write_text((fs::path(o.out_dir) / "slots.json").string(), io::slots_json(p));
  io::write_text((fs::path(o.out_dir) / "dates.json").string(), io::dates_json(p));
  io::write_text((fs::path(o.out_dir) / "template_schema.json").string(),
                 template_schema_json(o.templates_dir));
  write_support_csv(p, o.out_dir);

  const auto t1 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();

  std::vector<std::pair<std::string, std::string>> inputs;
  for (const std::string& f : {paths.att1, paths.att2, paths.att3, paths.att4}) {
    inputs.emplace_back(fs::path(f).filename().string(), util::Sha256::file(f));
  }
  std::vector<std::string> outputs;
  for (const auto& e : fs::directory_iterator(o.out_dir)) {
    if (!e.is_regular_file()) continue;
    const std::string name = e.path().filename().string();
    // verify 阶段自己生成的文件不算进 manifest，否则每次校验都会让哈希对不上
    if (name == "invariants.json" || name == "cross_check.json" || name == "manifest.json" ||
        name == "meta.json")
      continue;
    outputs.push_back(e.path().string());
  }
  std::sort(outputs.begin(), outputs.end());
  std::vector<std::pair<std::string, std::string>> checksums;
  for (const std::string& f : outputs) {
    checksums.emplace_back(fs::path(f).filename().string(), util::Sha256::file(f));
  }
  // manifest 最后写：它记录除自己以外的所有产物
  io::write_text((fs::path(o.out_dir) / "manifest.json").string(),
                 io::manifest_json(p, seconds, peak_rss_kb(), inputs, checksums));

  std::string meta = "{\n  \"command\": \"cleaning_cpp clean --in " + o.in_dir + " --out " + o.out_dir + "\",\n";
  meta += "  \"wall_seconds\": " + io::fmt(seconds, 6) + ",\n";
  meta += "  \"peak_rss_kb\": " + std::to_string(peak_rss_kb()) + ",\n";
  meta += "  \"rules\": \"NR1..NR12 (prompt v1 4.3)\",\n";
  meta += "  \"params\": {\"noise_kw\": " + io::fmt(o.params.noise_kw) +
          ", \"shoulder_kw\": " + io::fmt(o.params.shoulder_kw) +
          ", \"theta_rel\": " + io::fmt(o.params.theta_rel) +
          ", \"theta_abs\": " + io::fmt(o.params.theta_abs) +
          ", \"window_jump_slots\": " + std::to_string(o.params.window_jump_slots) +
          ", \"interp_max_gap\": " + std::to_string(o.params.interp_max_gap) + "}\n}\n";
  io::write_text((fs::path(o.out_dir) / "meta.json").string(), meta);

  std::cout << "[clean] days=" << kDays << " slots=" << kSlots << " audit=" << p.audit.size()
            << " quarantine=" << p.quarantine.size() << " seconds=" << io::fmt(seconds, 3)
            << " peak_rss_kb=" << peak_rss_kb() << "\n";
  return 0;
}

// ---------- verify：不变量 I1–I8 + 和题面给的交叉校验基线对账 ----------
struct CheckResult {
  std::string id;
  std::string description;
  bool pass = false;
  std::string detail;
};

std::vector<CheckResult> verify_panel(const model::Panel& p, const std::string& out_dir) {
  std::vector<CheckResult> r;

  // I1：核夜间光伏要么是 0，要么是隔离清单里的 NaN
  {
    long core_cells = 0, nonzero = 0, nan_cells = 0;
    for (int d = 0; d < kDays; ++d) {
      const model::DayWindow& w = p.windows[d];
      for (int k = 0; k < kSlots; ++k) {
        const bool core = (k < w.k_rise - 1) || (k > w.k_set + 1);
        if (!core) continue;
        ++core_cells;
        const double v = p.pv_clean.at(d, k);
        if (std::isnan(v)) ++nan_cells;
        else if (v != 0.0) ++nonzero;
      }
    }
    CheckResult c{"I1", "core-night PV is exactly 0 (NaN only if quarantined)", nonzero == 0,
                  "core_cells=" + std::to_string(core_cells) + " nonzero=" + std::to_string(nonzero) +
                      " nan=" + std::to_string(nan_cells)};
    r.push_back(c);
  }
  // I2：08:00–16:00 没有零值、没有负值
  {
    long zeros = 0, neg = 0, cells = 0;
    for (int d = 0; d < kDays; ++d) {
      for (int k = 47; k <= 95; ++k) {  // right endpoints 08:00 .. 16:00
        ++cells;
        const double v = p.pv_clean.at(d, k);
        if (!std::isnan(v) && v == 0.0) ++zeros;
        if (!std::isnan(v) && v < 0.0) ++neg;
      }
    }
    r.push_back({"I2", "08:00-16:00 zero-free and positive", zeros == 0 && neg == 0,
                 "cells=" + std::to_string(cells) + " zeros=" + std::to_string(zeros) +
                     " negatives=" + std::to_string(neg)});
  }
  // I3/I4：日期序列和矩阵形状
  {
    bool ok = static_cast<int>(p.dates.size()) == kDays;
    for (int d = 1; d < kDays && ok; ++d) ok = p.dates[d] > p.dates[d - 1];
    ok = ok && p.dates.front() == "2025-01-01" && p.dates.back() == "2025-12-31";
    r.push_back({"I3", "365 strictly increasing dates 2025-01-01..2025-12-31", ok,
                 "first=" + p.dates.front() + " last=" + p.dates.back()});
    const bool shape_ok = p.pv_actual.rows() == kDays && p.pv_actual.cols() == kSlots &&
                          p.load.rows() == kDays && p.price.rows() == kDays;
    r.push_back({"I4", "matrix shape 365 x 144 in every panel", shape_ok, "ok"});
  }
  // I5：电价 > 0、负荷 > 0 且落在 [1500, 8000] kW
  {
    long bad = 0;
    double lo = 1e18, hi = -1e18, llo = 1e18, lhi = -1e18;
    for (int d = 0; d < kDays; ++d) {
      for (int k = 0; k < kSlots; ++k) {
        const double pr = p.price.at(d, k), ld = p.load.at(d, k);
        if (!(pr > 0.0)) ++bad;
        if (!(ld > 0.0) || ld < 1500.0 || ld > 8000.0) ++bad;
        lo = std::min(lo, pr); hi = std::max(hi, pr);
        llo = std::min(llo, ld); lhi = std::max(lhi, ld);
      }
    }
    r.push_back({"I5", "price>0 ; load in [1500,8000] kW", bad == 0,
                 "violations=" + std::to_string(bad) + " price=[" + io::fmt(lo, 6) + "," +
                     io::fmt(hi, 6) + "] load=[" + io::fmt(llo, 6) + "," + io::fmt(lhi, 6) + "]"});
  }
  // I6：NR1 的置零条数 = 核夜间“由非零变零”的格数
  {
    long nr1_audit = 0, nr1_delta = 0;
    for (const model::AuditRecord& a : p.audit)
      if (a.rule_id == "NR1") ++nr1_audit;
    for (int d = 0; d < kDays; ++d) {
      const model::DayWindow& w = p.windows[d];
      for (int k = 0; k < kSlots; ++k) {
        const bool core = (k < w.k_rise - 1) || (k > w.k_set + 1);
        if (!core) continue;
        const double raw = p.pv_actual.at(d, k);
        const double cl = p.pv_clean.at(d, k);
        if (!std::isnan(raw) && raw > 0.0 && cl == 0.0) ++nr1_delta;
      }
    }
    r.push_back({"I6", "NR1 audit count equals core-night nonzero -> zero cells", nr1_audit == nr1_delta,
                 "audit=" + std::to_string(nr1_audit) + " delta=" + std::to_string(nr1_delta)});
  }
  // I7：除被审计过的格子外，清洗前后逐位相同
  {
    std::vector<unsigned char> touched(static_cast<size_t>(kDays) * kSlots, 0);
    for (const model::AuditRecord& a : p.audit) {
      if (a.file != "附件2" || a.sheet != "光伏发电实际功率") continue;
      const int d = a.excel_row - 2, k = a.slot_index - 1;
      if (d >= 0 && d < kDays && k >= 0 && k < kSlots) touched[static_cast<size_t>(d) * kSlots + k] = 1;
    }
    long diffs = 0;
    for (int d = 0; d < kDays; ++d) {
      for (int k = 0; k < kSlots; ++k) {
        if (touched[static_cast<size_t>(d) * kSlots + k]) continue;
        const double raw = p.pv_actual.at(d, k), cl = p.pv_clean.at(d, k);
        if (raw != cl) ++diffs;
      }
    }
    r.push_back({"I7", "bitwise identity outside audited cells", diffs == 0,
                 "unexpected_differences=" + std::to_string(diffs)});
  }
  // I8：磁盘上的文件哈希和 manifest 里记的一致
  {
    bool ok = true;
    std::string detail = "manifest not found";
    const std::string mpath = (fs::path(out_dir) / "manifest.json").string();
    if (fs::exists(mpath)) {
      std::ifstream in(mpath);
      std::stringstream ss;
      ss << in.rdbuf();
      const std::string txt = ss.str();
      size_t pos = txt.find("\"outputs_sha256\"");
      int checked = 0, mismatched = 0;
      if (pos != std::string::npos) {
        const size_t block_b = txt.find('{', pos);
        const size_t block_e = txt.find('}', block_b);
        const std::string block = txt.substr(block_b + 1, block_e - block_b - 1);
        size_t p = 0;
        while ((p = block.find('"', p)) != std::string::npos) {
          const size_t key_e = block.find('"', p + 1);
          if (key_e == std::string::npos) break;
          const size_t val_b = block.find('"', block.find(':', key_e) + 1);
          if (val_b == std::string::npos) break;
          const size_t val_e = block.find('"', val_b + 1);
          if (val_e == std::string::npos) break;
          const std::string name = block.substr(p + 1, key_e - p - 1);
          const std::string want = block.substr(val_b + 1, val_e - val_b - 1);
          const std::string path = (fs::path(out_dir) / name).string();
          if (fs::exists(path)) {
            ++checked;
            if (util::Sha256::file(path) != want) ++mismatched;
          }
          p = val_e + 1;
        }
      }
      ok = (mismatched == 0 && checked > 0);
      detail = "checked=" + std::to_string(checked) + " mismatched=" + std::to_string(mismatched);
    }
    r.push_back({"I8", "output SHA-256 matches manifest", ok, detail});
  }
  return r;
}

std::string cross_check_json(const model::Panel& p) {
  long core_cells = 0, raw_core_zeros = 0, clean_core_zeros = 0, raw_core_big = 0;
  long day_zeros = 0, day1_small = 0, near_zero_price = 0, night_load_flags = 0, nr4_hits = 0;
  for (int d = 0; d < kDays; ++d) {
    const model::DayWindow& w = p.windows[d];
    for (int k = 0; k < kSlots; ++k) {
      const bool core = (k < w.k_rise - 1) || (k > w.k_set + 1);
      const double raw = p.pv_actual.at(d, k);
      const double cl = p.pv_clean.at(d, k);
      if (core) {
        ++core_cells;
        if (raw == 0.0) ++raw_core_zeros;
        if (cl == 0.0) ++clean_core_zeros;
        if (raw > 5.0) ++raw_core_big;
      }
      if (k >= 47 && k <= 95) {
        if (!std::isnan(cl) && cl == 0.0) ++day_zeros;
      }
      if (p.price.at(d, k) < 0.1) ++near_zero_price;
    }
  }
  for (const model::AuditRecord& a : p.audit) {
    if (a.rule_id == "NR4") ++nr4_hits;
    if (a.rule_id == "NR12") ++night_load_flags;
  }
  for (int k = 0; k < kSlots; ++k) {
    const double v = p.day1[3 * k + 2];
    if (v > 0.0 && v <= 5.0) ++day1_small;
  }
  std::string out = "{\n";
  out += "  \"att2_core_night_cells\": " + std::to_string(core_cells) + ",\n";
  out += "  \"att2_core_night_zero_raw\": " + std::to_string(raw_core_zeros) + ",\n";
  out += "  \"att2_core_night_zero_after\": " + std::to_string(clean_core_zeros) + ",\n";
  out += "  \"att2_core_night_over_5kw_raw\": " + std::to_string(raw_core_big) + ",\n";
  out += "  \"att2_daytime_0800_1600_zeros\": " + std::to_string(day_zeros) + ",\n";
  out += "  \"quarantine_records\": " + std::to_string(p.quarantine.size()) + ",\n";
  out += "  \"audit_records\": " + std::to_string(p.audit.size()) + ",\n";
  out += "  \"att1_small_pv_cells_0_5kw\": " + std::to_string(day1_small) + ",\n";
  out += "  \"att4_near_zero_price_cells\": " + std::to_string(near_zero_price) + ",\n";
  out += "  \"night_load_flag_records\": " + std::to_string(night_load_flags) + ",\n";
  out += "  \"nr4_shoulder_hits\": " + std::to_string(nr4_hits) + "\n}\n";
  return out;
}

int cmd_verify(const Options& o) {
  const model::Panel p = model::load_panel(make_paths(o), o.params);
  const std::vector<CheckResult> checks = verify_panel(p, o.out_dir);
  bool all = true;
  for (const CheckResult& c : checks) {
    std::cout << (c.pass ? "[PASS] " : "[FAIL] ") << c.id << " " << c.description << " :: " << c.detail
              << "\n";
    all = all && c.pass;
  }
  const std::string cc = cross_check_json(p);
  std::cout << "[cross-check]\n" << cc;
  fs::create_directories(o.out_dir);
  io::write_text((fs::path(o.out_dir) / "cross_check.json").string(), cc);
  std::string summary = "{\n  \"all_pass\": " + std::string(all ? "true" : "false") + ",\n  \"checks\": [\n";
  for (size_t i = 0; i < checks.size(); ++i) {
    summary += "    {\"id\": \"" + checks[i].id + "\", \"description\": \"" +
               io::json_escape(checks[i].description) + "\", \"pass\": " +
               std::string(checks[i].pass ? "true" : "false") + ", \"detail\": \"" +
               io::json_escape(checks[i].detail) + "\"}" + (i + 1 < checks.size() ? "," : "") + "\n";
  }
  summary += "  ]\n}\n";
  io::write_text((fs::path(o.out_dir) / "invariants.json").string(), summary);
  if (!all && o.strict) return 2;
  return all ? 0 : 1;
}

// ---------- selftest：规则的黄金用例 + 合成面板性质测试 ----------
bool check_action(const std::string& rule, const clean::PointResult& pr, clean::Action expect,
                  double expect_value) {
  const bool ok = pr.action == expect &&
                  (std::isnan(expect_value) ? std::isnan(pr.value)
                                            : std::fabs(pr.value - expect_value) < 1e-12);
  if (!ok) {
    std::cout << "   [FAIL] case " << rule << " expected action=" << static_cast<int>(expect)
              << " value=" << expect_value << " got action=" << static_cast<int>(pr.action)
              << " value=" << pr.value << "\n";
  }
  return ok;
}

int cmd_selftest(const Options&) {
  model::RuleParams p;
  int failures = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // --- 每条规则三个用例：触发 / 边界 / 不触发 ---
  failures += !check_action("NR1 trigger", clean::apply_pv_point(3.0, true, false, p), clean::Action::kSetZero, 0.0);
  failures += !check_action("NR1 boundary 5.0", clean::apply_pv_point(5.0, true, false, p), clean::Action::kSetZero, 0.0);
  failures += !check_action("NR1 boundary 0.0", clean::apply_pv_point(0.0, true, false, p), clean::Action::kNone, 0.0);
  failures += !check_action("NR2 trigger", clean::apply_pv_point(5.0001, true, false, p), clean::Action::kReject, nan);
  failures += !check_action("NR2 no-trigger", clean::apply_pv_point(4.9999, true, false, p), clean::Action::kSetZero, 0.0);
  failures += !check_action("NR3 negative", clean::apply_pv_point(-0.5, false, false, p), clean::Action::kReject, nan);
  failures += !check_action("NR4 sub-kW", clean::apply_pv_point(0.6, false, true, p), clean::Action::kSetZero, 0.0);
  failures += !check_action("NR4 boundary 1.0 -> NR5", clean::apply_pv_point(1.0, false, true, p), clean::Action::kFlag, 1.0);
  failures += !check_action("NR5 weak signal", clean::apply_pv_point(2.5, false, true, p), clean::Action::kFlag, 2.5);
  failures += !check_action("daylight keep", clean::apply_pv_point(1200.0, false, false, p), clean::Action::kNone, 1200.0);

  // --- 性质测试：200 块合成面板，故意注入缺陷 ---
  std::mt19937_64 rng(20250910);
  std::uniform_real_distribution<double> noise(0.01, 5.0);
  std::uniform_real_distribution<double> outlier(5.01, 12.0);
  std::uniform_real_distribution<double> daylight(10.0, 8000.0);
  int inject_noise = 0, inject_big = 0, hit_noise = 0, hit_big = 0, invariant_violations = 0;
  for (int trial = 0; trial < 200; ++trial) {
    // 造一天：日照窗口 [30, 110]，窗口内给正功率，窗口外为 0
    std::vector<double> day(144, 0.0);
    for (int k = 30; k <= 110; ++k) day[k] = daylight(rng);
    const int k_rise = 30, k_set = 110;
    std::vector<int> core_slots;
    for (int k = 0; k < 144; ++k)
      if (k < k_rise - 1 || k > k_set + 1) core_slots.push_back(k);
    std::shuffle(core_slots.begin(), core_slots.end(), rng);
    for (size_t i = 0; i < 6 && i < core_slots.size(); ++i) {
      day[static_cast<size_t>(core_slots[i])] = noise(rng);  // (0, 5] kW zero drift
      ++inject_noise;
    }
    for (size_t i = 6; i < 9 && i < core_slots.size(); ++i) {
      day[static_cast<size_t>(core_slots[i])] = outlier(rng);  // > 5 kW hard outlier
      ++inject_big;
    }
    // 跑规则
    for (int k = 0; k < 144; ++k) {
      const bool core = (k < k_rise - 1) || (k > k_set + 1);
      const bool shoulder = (k == k_rise - 1 || k == k_rise || k == k_set || k == k_set + 1);
      const clean::PointResult pr = clean::apply_pv_point(day[k], core, shoulder, p);
      if (pr.action == clean::Action::kSetZero && core) ++hit_noise;
      if (pr.action == clean::Action::kReject && core) ++hit_big;
      const double after = (pr.action == clean::Action::kNone || pr.action == clean::Action::kFlag)
                               ? day[k]
                               : pr.value;
      if (core && !(after == 0.0 || std::isnan(after))) ++invariant_violations;
      if (!core && !shoulder && after != day[k]) ++invariant_violations;
    }
  }
  const bool property_ok = (hit_noise == inject_noise) && (hit_big == inject_big) &&
                           (invariant_violations == 0);
  std::cout << "   [property] injected noise=" << inject_noise << " caught=" << hit_noise
            << " ; injected outliers=" << inject_big << " caught=" << hit_big
            << " ; invariant violations=" << invariant_violations << "\n";
  if (!property_ok) ++failures;

  // --- 确定性：同一批数据重复跑，结果必须一样 ---
  std::vector<double> a(144), b(144);
  for (int k = 0; k < 144; ++k) a[k] = (k % 7 == 0) ? 3.0 : 100.0 + k;
  b = a;
  bool det_ok = true;
  for (int rep = 0; rep < 2; ++rep) {
    for (int k = 0; k < 144; ++k) {
      const bool core = (k < 20) || (k > 120);
      const clean::PointResult pr = clean::apply_pv_point(a[k], core, false, p);
      const double v = (pr.action == clean::Action::kNone || pr.action == clean::Action::kFlag) ? a[k] : pr.value;
      if (rep == 0) b[k] = v;
      else if (!(v == b[k] || (std::isnan(v) && std::isnan(b[k])))) det_ok = false;
    }
  }
  if (!det_ok) ++failures;
  std::cout << "   [determinism] repeated rule application identical: " << (det_ok ? "yes" : "no") << "\n";
  std::cout << "[selftest] " << (failures == 0 ? "ALL PASS" : "FAILURES=" + std::to_string(failures))
            << " (golden cases + 200 synthetic panels)\n";
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: cleaning_cpp <inspect|extract|clean|export|verify|selftest> [--in DIR] [--out DIR]\n";
    return 1;
  }
  const std::string cmd = argv[1];
  try {
    const Options o = parse_options(argc, argv);
    if (cmd == "inspect") return cmd_inspect(o);
    if (cmd == "extract") return cmd_extract(o);
    if (cmd == "clean" || cmd == "export") return cmd_clean(o);
    if (cmd == "verify") return cmd_verify(o);
    if (cmd == "selftest") return cmd_selftest(o);
    std::cerr << "unknown subcommand: " << cmd << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 3;
  }
}
