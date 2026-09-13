#include "xlsx/xlsx.h"

#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "tinyxml2.h"

namespace xlsx {
namespace {

uint16_t u16(const char* p) {
  return static_cast<uint16_t>(static_cast<uint8_t>(p[0]) | (static_cast<uint8_t>(p[1]) << 8));
}
uint32_t u32(const char* p) {
  return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

std::string inflate_raw(const std::string& in, uint64_t expected) {
  std::string out;
  out.resize(static_cast<size_t>(expected));
  z_stream zs;
  std::memset(&zs, 0, sizeof(zs));
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) throw std::runtime_error("inflateInit2 failed");
  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(in.data()));
  zs.avail_in = static_cast<uInt>(in.size());
  zs.next_out = reinterpret_cast<Bytef*>(&out[0]);
  zs.avail_out = static_cast<uInt>(out.size());
  const int rc = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (rc != Z_STREAM_END) throw std::runtime_error("inflate failed rc=" + std::to_string(rc));
  out.resize(zs.total_out);
  return out;
}

std::string si_text(const tinyxml2::XMLElement* si) {
  std::string out;
  for (const tinyxml2::XMLElement* t = si->FirstChildElement("t"); t != nullptr;
       t = t->NextSiblingElement("t")) {
    if (t->GetText() != nullptr) out += t->GetText();
  }
  if (out.empty()) {  // rich text: <r><t>..</t></r>
    for (const tinyxml2::XMLElement* r = si->FirstChildElement("r"); r != nullptr;
         r = r->NextSiblingElement("r")) {
      for (const tinyxml2::XMLElement* t = r->FirstChildElement("t"); t != nullptr;
           t = t->NextSiblingElement("t")) {
        if (t->GetText() != nullptr) out += t->GetText();
      }
    }
  }
  return out;
}

}  // namespace

ZipArchive ZipArchive::open(const std::string& path) {
  ZipArchive z;
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) throw std::runtime_error("cannot open " + path);
  std::fseek(fp, 0, SEEK_END);
  const long size = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  z.raw_.resize(static_cast<size_t>(size));
  if (size > 0 && std::fread(&z.raw_[0], 1, static_cast<size_t>(size), fp) != static_cast<size_t>(size)) {
    std::fclose(fp);
    throw std::runtime_error("short read " + path);
  }
  std::fclose(fp);

  // 从尾部往前找 EOCD（0x06054b50），注释区一般是空的
  const size_t max_back = std::min<size_t>(z.raw_.size(), 66000);
  size_t eocd = std::string::npos;
  for (size_t i = z.raw_.size() - 22 + 1; i-- > z.raw_.size() - max_back;) {
    if (u32(&z.raw_[i]) == 0x06054b50u) {
      eocd = i;
      break;
    }
    if (i == 0) break;
  }
  if (eocd == std::string::npos) throw std::runtime_error("EOCD not found: " + path);
  const uint16_t count = u16(&z.raw_[eocd + 10]);
  const uint32_t cd_off = u32(&z.raw_[eocd + 16]);

  size_t p = cd_off;
  for (uint16_t i = 0; i < count; ++i) {
    if (u32(&z.raw_[p]) != 0x02014b50u) throw std::runtime_error("bad central directory");
    Entry e;
    e.method = u16(&z.raw_[p + 10]);
    e.comp_size = u32(&z.raw_[p + 20]);
    e.uncomp_size = u32(&z.raw_[p + 24]);
    const uint16_t name_len = u16(&z.raw_[p + 28]);
    const uint16_t extra_len = u16(&z.raw_[p + 30]);
    const uint16_t comment_len = u16(&z.raw_[p + 32]);
    e.local_offset = u32(&z.raw_[p + 42]);
    e.name = z.raw_.substr(p + 46, name_len);
    z.index_[e.name] = e;
    p += 46 + name_len + extra_len + comment_len;
  }
  return z;
}

std::string ZipArchive::read(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) throw std::runtime_error("zip entry missing: " + name);
  const Entry& e = it->second;
  const size_t lh = static_cast<size_t>(e.local_offset);
  if (u32(&raw_[lh]) != 0x04034b50u) throw std::runtime_error("bad local header: " + name);
  const uint16_t name_len = u16(&raw_[lh + 26]);
  const uint16_t extra_len = u16(&raw_[lh + 28]);
  const size_t data_off = lh + 30 + name_len + extra_len;
  const std::string comp = raw_.substr(data_off, static_cast<size_t>(e.comp_size));
  if (e.method == 0) return comp;
  if (e.method != 8) throw std::runtime_error("unsupported compression method");
  return inflate_raw(comp, e.uncomp_size);
}

std::vector<std::string> ZipArchive::names() const {
  std::vector<std::string> out;
  out.reserve(index_.size());
  for (const auto& kv : index_) out.push_back(kv.first);
  return out;
}

const Sheet* Workbook::find(const std::string& name) const {
  for (const Sheet& s : sheets)
    if (s.name == name) return &s;
  return nullptr;
}

int col_from_ref(const std::string& ref) {
  int col = 0;
  for (char ch : ref) {
    if (ch >= 'A' && ch <= 'Z') {
      col = col * 26 + (ch - 'A' + 1);
    } else if (ch >= 'a' && ch <= 'z') {
      col = col * 26 + (ch - 'a' + 1);
    } else {
      break;
    }
  }
  return col - 1;
}

int row_from_ref(const std::string& ref) {
  size_t i = 0;
  while (i < ref.size() && !(ref[i] >= '0' && ref[i] <= '9')) ++i;
  if (i >= ref.size()) return -1;
  return std::atoi(ref.c_str() + i);
}

std::string excel_date_to_iso(double serial) {
  // 1900 日期系统；序列号 ≥ 61 时要跳过不存在的 1900-02-29
  long s = static_cast<long>(serial < 61 ? serial : serial - 1);
  long days = s - 25568;  // 25568 = serial of 1970-01-01 corrected
  // 用 civil-from-days 换算成日期（以 1970-01-01 为基准）
  long z = days + 719468;
  const long era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const long y = static_cast<long>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  const long year = y + (m <= 2 ? 1 : 0);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04ld-%02u-%02u", year, m, d);
  return std::string(buf);
}

int minutes_of_day(double fraction) {
  double mins = fraction * 1440.0;
  return static_cast<int>(mins + 0.5);
}

std::string minutes_to_label(int minutes) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d:%02d", minutes / 60, minutes % 60);
  return std::string(buf);
}

Workbook read_workbook(const std::string& path) {
  ZipArchive zip = ZipArchive::open(path);
  const std::string wb_xml = zip.read("xl/workbook.xml");
  const std::string rels_xml = zip.read("xl/_rels/workbook.xml.rels");

  std::map<std::string, std::string> rel_target;
  {
    tinyxml2::XMLDocument doc;
    if (doc.Parse(rels_xml.c_str()) != tinyxml2::XML_SUCCESS) throw std::runtime_error("bad rels xml");
    for (const tinyxml2::XMLElement* e = doc.RootElement()->FirstChildElement("Relationship");
         e != nullptr; e = e->NextSiblingElement("Relationship")) {
      rel_target[e->Attribute("Id")] = e->Attribute("Target");
    }
  }

  std::vector<std::pair<std::string, std::string>> sheet_refs;  // name, target
  {
    tinyxml2::XMLDocument doc;
    if (doc.Parse(wb_xml.c_str()) != tinyxml2::XML_SUCCESS) throw std::runtime_error("bad workbook xml");
    const tinyxml2::XMLElement* root = doc.RootElement();
    const tinyxml2::XMLElement* sheets = root->FirstChildElement("sheets");
    if (sheets == nullptr) throw std::runtime_error("no <sheets>");
    for (const tinyxml2::XMLElement* e = sheets->FirstChildElement("sheet"); e != nullptr;
         e = e->NextSiblingElement("sheet")) {
      const char* rid = e->Attribute("r:id");
      std::string target = rid != nullptr ? rel_target[rid] : std::string();
      if (!target.empty() && target.rfind("xl/", 0) != 0) target = "xl/" + target;
      sheet_refs.emplace_back(e->Attribute("name"), target);
    }
  }

  std::vector<std::string> shared;
  if (zip.has("xl/sharedStrings.xml")) {
    tinyxml2::XMLDocument doc;
    const std::string ss = zip.read("xl/sharedStrings.xml");
    if (doc.Parse(ss.c_str()) != tinyxml2::XML_SUCCESS) throw std::runtime_error("bad sharedStrings");
    for (const tinyxml2::XMLElement* si = doc.RootElement()->FirstChildElement("si"); si != nullptr;
         si = si->NextSiblingElement("si")) {
      shared.push_back(si_text(si));
    }
  }

  Workbook wb;
  for (const auto& sr : sheet_refs) {
    Sheet sheet;
    sheet.name = sr.first;
    const std::string xml = zip.read(sr.second);
    tinyxml2::XMLDocument doc;
    if (doc.Parse(xml.c_str()) != tinyxml2::XML_SUCCESS) throw std::runtime_error("bad sheet xml " + sr.second);
    const tinyxml2::XMLElement* data = doc.RootElement()->FirstChildElement("sheetData");
    int max_row = 0, max_col = -1;
    struct Raw {
      int r, c;
      CellValue v;
    };
    std::vector<Raw> raw_cells;
    if (data != nullptr) {
      for (const tinyxml2::XMLElement* row = data->FirstChildElement("row"); row != nullptr;
           row = row->NextSiblingElement("row")) {
        for (const tinyxml2::XMLElement* c = row->FirstChildElement("c"); c != nullptr;
             c = c->NextSiblingElement("c")) {
          const char* ref = c->Attribute("r");
          if (ref == nullptr) continue;
          Raw rc;
          rc.r = row_from_ref(ref);
          rc.c = col_from_ref(ref);
          max_row = std::max(max_row, rc.r);
          max_col = std::max(max_col, rc.c);
          const char* type = c->Attribute("t");
          const tinyxml2::XMLElement* v = c->FirstChildElement("v");
          if (type != nullptr && std::strcmp(type, "s") == 0 && v != nullptr && v->GetText() != nullptr) {
            const int idx = std::atoi(v->GetText());
            rc.v.kind = ValueKind::kText;
            rc.v.text = (idx >= 0 && idx < static_cast<int>(shared.size())) ? shared[idx] : std::string();
          } else if (type != nullptr && std::strcmp(type, "inlineStr") == 0) {
            const tinyxml2::XMLElement* is = c->FirstChildElement("is");
            rc.v.kind = ValueKind::kText;
            if (is != nullptr) rc.v.text = si_text(is);
          } else if (type != nullptr && std::strcmp(type, "str") == 0) {
            rc.v.kind = ValueKind::kText;
            if (v != nullptr && v->GetText() != nullptr) rc.v.text = v->GetText();
          } else if (type != nullptr && std::strcmp(type, "b") == 0) {
            rc.v.kind = ValueKind::kBool;
            rc.v.number = (v != nullptr && v->GetText() != nullptr) ? std::atof(v->GetText()) : 0.0;
          } else if (v != nullptr && v->GetText() != nullptr) {
            rc.v.kind = ValueKind::kNumber;
            rc.v.number = std::strtod(v->GetText(), nullptr);
          } else {
            rc.v.kind = ValueKind::kEmpty;
          }
          raw_cells.push_back(rc);
        }
      }
    }
    sheet.grid.assign(static_cast<size_t>(std::max(max_row, 0)),
                      std::vector<CellValue>(static_cast<size_t>(std::max(max_col + 1, 0))));
    for (const Raw& rc : raw_cells) {
      if (rc.r >= 1 && rc.c >= 0) sheet.grid[static_cast<size_t>(rc.r - 1)][static_cast<size_t>(rc.c)] = rc.v;
    }
    wb.sheets.push_back(std::move(sheet));
  }
  return wb;
}

}  // namespace xlsx
