#pragma once
#include <string>
#include <vector>

namespace model {

constexpr int kSlots = 144;       // 10-minute slots per day
constexpr int kDays = 365;        // 2025-01-01 .. 2025-12-31
constexpr int kVintages = 4;      // 0:00 / 6:00 / 12:00 / 18:00
constexpr int kForecastHours = 24;

struct DayWindow {
  int k_rise = 0;    // 0-based first slot above theta (data-side index)
  int k_set = 0;     // 0-based last slot above theta
  double theta = 1.0;
  double peak = 0.0;
  bool valid = false;
};

struct AuditRecord {
  std::string run_id;
  std::string file;
  std::string sheet;
  int excel_row = 0;
  std::string excel_col;
  std::string date;
  int slot_index = 0;      // 1-based (1..144)
  std::string slot_label;
  std::string hour_end;
  std::string rule_id;
  std::string action;
  std::string old_value;
  std::string new_value;
  std::string reason;
};

struct QuarantineRecord {
  std::string run_id;
  std::string file;
  std::string sheet;
  int excel_row = 0;
  std::string excel_col;
  std::string date;
  int slot_index = 0;
  std::string slot_label;
  std::string value;
  std::string rule_id;
  std::string reason;
};

// 行主序的矩阵包装：(d, k) 取第 d 天第 k 个时段。
class Matrix {
 public:
  Matrix() = default;
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(static_cast<size_t>(rows) * cols) {}
  double& at(int d, int k) { return data_.at(static_cast<size_t>(d) * cols_ + k); }
  double at(int d, int k) const { return data_.at(static_cast<size_t>(d) * cols_ + k); }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  const std::vector<double>& data() const { return data_; }
  std::vector<double>& data() { return data_; }

 private:
  int rows_ = 0;
  int cols_ = 0;
  std::vector<double> data_;
};

struct Panel {
  std::vector<std::string> dates;   // 365 ISO dates
  std::vector<std::string> slots;   // 144 data-side right-endpoint labels
  Matrix pv_actual;                 // 365 x 144 (raw, before rules)
  Matrix pv_clean;                  // 365 x 144 (after rules; NaN allowed per audit)
  Matrix load;                      // 365 x 144
  Matrix price;                     // 365 x 144
  std::vector<double> pv_forecast;  // 365*4*24 raw
  std::vector<double> pv_forecast_clean;
  std::vector<double> day1;         // 144*3 = price, load, pv_forecast
  std::vector<DayWindow> windows;   // 365
  std::vector<AuditRecord> audit;
  std::vector<QuarantineRecord> quarantine;
  double pv_global_max = 0.0;
  double sigma_floor = 1.0;
  std::vector<double> sigma_raw;    // 144
  std::vector<double> sigma_used;   // 144
  std::vector<int> sigma_n;         // 144
};

struct RuleParams {
  double noise_kw = 5.0;
  double shoulder_kw = 1.0;
  double theta_rel = 0.01;
  double theta_abs = 1.0;
  int window_jump_slots = 3;
  int interp_max_gap = 6;
};

// 输入路径：附件1–附件4，外加可选的模板目录。
struct InputPaths {
  std::string att1;
  std::string att2;
  std::string att3;
  std::string att4;
  std::string templates_dir;
};

Panel load_panel(const InputPaths& paths, const RuleParams& params);

}  // namespace model
