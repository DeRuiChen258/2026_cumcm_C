#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace xlsx {

// ---------- zip：只处理 deflate，靠中央目录定位（附件的 xlsx 基本都是 method 8） ----------
class ZipArchive {
 public:
  struct Entry {
    std::string name;
    uint16_t method = 0;
    uint64_t comp_size = 0;
    uint64_t uncomp_size = 0;
    uint64_t local_offset = 0;
  };

  static ZipArchive open(const std::string& path);
  bool has(const std::string& name) const { return index_.count(name) > 0; }
  std::string read(const std::string& name) const;
  std::vector<std::string> names() const;

 private:
  std::string raw_;
  std::map<std::string, Entry> index_;
};

// ---------- 单元格取值：附件1 的时间列混了数值和字符串，所以逐格判类型 ----------
enum class ValueKind { kEmpty, kNumber, kText, kBool };

struct CellValue {
  ValueKind kind = ValueKind::kEmpty;
  double number = 0.0;
  std::string text;

  bool empty() const { return kind == ValueKind::kEmpty; }
  bool is_number() const { return kind == ValueKind::kNumber; }
  bool is_text() const { return kind == ValueKind::kText; }
};

using Grid = std::vector<std::vector<CellValue>>;  // [row][col], 0-based, dense

struct Sheet {
  std::string name;
  Grid grid;
};

struct Workbook {
  std::vector<Sheet> sheets;
  const Sheet* find(const std::string& name) const;
  const Sheet& at(size_t i) const { return sheets.at(i); }
  size_t size() const { return sheets.size(); }
};

Workbook read_workbook(const std::string& path);

// 给清洗层用的小工具（属于纯解析，留在 xlsx 模块里）。
int col_from_ref(const std::string& ref);           // "B12" -> 1 (0-based)
int row_from_ref(const std::string& ref);           // "B12" -> 12 (1-based)
std::string excel_date_to_iso(double serial);       // 45658 -> "2025-01-01"
int minutes_of_day(double fraction);                // 0.006944 -> 10
std::string minutes_to_label(int minutes);          // 610 -> "10:10"

}  // namespace xlsx
