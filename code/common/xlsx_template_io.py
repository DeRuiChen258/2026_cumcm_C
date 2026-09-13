"""用标准库读写 xlsx（不装 openpyxl）。

做法是复制附件5 模板、保留样式和共享字符串，只替换目标工作表的 <dimension> 和
<sheetData>，这样出表结果还能保持可复现。
"""
from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

CELL_RE = re.compile(rb"<c([^>]*?)(?:/>|>(.*?)</c>)", re.S)
ATTR_RE = re.compile(rb'([\w:]+)="([^"]*)"')
ROW_RE = re.compile(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', re.S)


def col_to_idx(ref: str) -> int:
    col = 0
    for ch in ref:
        if ch.isalpha():
            col = col * 26 + (ord(ch.upper()) - ord("A") + 1)
        else:
            break
    return col - 1


def idx_to_col(idx: int) -> str:
    out = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


@dataclass
class Cell:
    ref: str = ""
    style: str | None = None
    kind: str = "empty"      # empty | number | shared | inline | str | bool
    value: str = ""


@dataclass
class SheetData:
    name: str
    path: str
    xml: str
    dimension: str = "A1"
    rows: dict[int, dict[int, Cell]] = field(default_factory=dict)
    head_xml: str = ""
    tail_xml: str = ""


class Workbook:
    """Minimal reader that keeps raw XML so styles survive a rewrite."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        self.shared: list[str] = self._read_shared()
        self.new_strings: list[str] = []
        self.sheets: dict[str, SheetData] = {}
        self.sheet_order: list[str] = []
        self._read_workbook()

    def _read_shared(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.zip.namelist():
            return []
        xml = self.zip.read("xl/sharedStrings.xml").decode("utf-8")
        return ["".join(re.findall(r"<t[^>]*>(.*?)</t>", item, re.S))
                for item in re.findall(r"<si>(.*?)</si>", xml, re.S)]

    def shared_index(self, text: str) -> int:
        if text in self.shared:
            return self.shared.index(text)
        self.shared.append(text)
        self.new_strings.append(text)
        return len(self.shared) - 1

    def _read_workbook(self) -> None:
        wb = self.zip.read("xl/workbook.xml").decode("utf-8")
        rels = self.zip.read("xl/_rels/workbook.xml.rels").decode("utf-8")
        rel_map = {
            m.group(1): m.group(2)
            for m in re.finditer(r'<Relationship[^>]*Id="([^"]+)"[^>]*Target="([^"]+)"', rels)
        }
        for m in re.finditer(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
            name, rid = m.group(1), m.group(2)
            target = rel_map[rid]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            self._load_sheet(name, target)
            self.sheet_order.append(name)

    def _load_sheet(self, name: str, target: str) -> None:
        xml = self.zip.read(target).decode("utf-8")
        sheet = SheetData(name=name, path=target, xml=xml)
        dim = re.search(r'<dimension ref="([^"]*)"', xml)
        sheet.dimension = dim.group(1) if dim else "A1"
        data_match = re.search(r"<sheetData>.*?</sheetData>", xml, re.S)
        if data_match:
            sheet.head_xml = xml[: data_match.start()]
            sheet.tail_xml = xml[data_match.end():]
        else:
            sheet.head_xml, sheet.tail_xml = xml, ""
        for row_match in ROW_RE.finditer(xml):
            r = int(row_match.group(1))
            cells: dict[int, Cell] = {}
            for cell_match in CELL_RE.finditer(row_match.group(2).encode("utf-8")):
                attrs = {m.group(1).decode(): m.group(2).decode()
                         for m in ATTR_RE.finditer(cell_match.group(1))}
                ref = attrs.get("r", "")
                if not ref:
                    continue
                blob = cell_match.group(2) or b""
                kind, value = "empty", ""
                t = attrs.get("t")
                v = re.search(rb"<v>(.*?)</v>", blob, re.S)
                if t == "s":
                    kind, value = "shared", (v.group(1).decode() if v else "")
                elif t == "inlineStr":
                    it = re.search(rb"<t[^>]*>(.*?)</t>", blob, re.S)
                    kind, value = "inline", (it.group(1).decode() if it else "")
                elif t == "str":
                    kind, value = "str", (v.group(1).decode() if v else "")
                elif t == "b":
                    kind, value = "bool", (v.group(1).decode() if v else "0")
                elif v is not None:
                    kind, value = "number", v.group(1).decode()
                cells[col_to_idx(ref)] = Cell(ref=ref, style=attrs.get("s"), kind=kind, value=value)
            sheet.rows[r] = cells
        self.sheets[name] = sheet

    def text(self, sheet: str, row: int, col: int) -> str | None:
        cell = self.sheets[sheet].rows.get(row, {}).get(col)
        if cell is None or cell.kind == "empty":
            return None
        if cell.kind == "shared":
            return self.shared[int(cell.value)]
        return cell.value

    def number(self, sheet: str, row: int, col: int) -> float | None:
        cell = self.sheets[sheet].rows.get(row, {}).get(col)
        if cell is None or cell.kind != "number":
            return None
        return float(cell.value)

    def style(self, sheet: str, row: int, col: int) -> str | None:
        cell = self.sheets[sheet].rows.get(row, {}).get(col)
        return None if cell is None else cell.style

    def close(self) -> None:
        self.zip.close()


def render_cell(col: int, row: int, value, kind: str, style: str | None, shared_index) -> str:
    ref = f"{idx_to_col(col)}{row}"
    style_attr = f' s="{style}"' if style else ""
    if kind == "empty" or value is None:
        return f'<c r="{ref}"{style_attr}/>' if style else ""
    if kind == "text":
        return f'<c r="{ref}"{style_attr} t="s"><v>{shared_index(str(value))}</v></c>'
    if kind == "number":
        text = repr(float(value))
        if text.endswith(".0"):
            text = text[:-2]
        return f'<c r="{ref}"{style_attr}><v>{text}</v></c>'
    raise ValueError(f"unknown cell kind {kind}")


def build_sheet_xml(sheet: SheetData, rows: list[list[tuple]], shared_index) -> str:
    """rows: list of rows; each row is a list of (col_idx, value, kind, style_override)."""
    parts: list[str] = []
    max_col = 0
    for r_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for col, value, kind, style_override in row:
            style = style_override if style_override is not None else sheet.rows.get(1, {}).get(col, Cell()).style
            cells.append(render_cell(col, r_idx, value, kind, style, shared_index))
            max_col = max(max_col, col + 1)
        if cells:
            parts.append(f'<row r="{r_idx}" spans="1:{max_col}">' + "".join(cells) + "</row>")
    body = "".join(parts)
    dimension = f"A1:{idx_to_col(max(max_col - 1, 0))}{len(rows)}"
    head = re.sub(r'<dimension ref="[^"]*"', f'<dimension ref="{dimension}"', sheet.head_xml)
    return head + "<sheetData>" + body + "</sheetData>" + sheet.tail_xml


def write_workbook(template: Workbook, out_path: Path, sheet_rows: dict[str, list[list[tuple]]]) -> None:
    """Copy the template zip, replacing the sheet XML of the sheets in `sheet_rows`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with zipfile.ZipFile(template.path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            owner = next((s for s in template.sheets.values() if s.path == item.filename), None)
            if owner is not None and owner.name in sheet_rows:
                data = build_sheet_xml(owner, sheet_rows[owner.name], template.shared_index).encode("utf-8")
            elif item.filename == "xl/sharedStrings.xml" and template.new_strings:
                xml = data.decode("utf-8")
                additions = "".join(f"<si><t>{_escape(t)}</t></si>" for t in template.new_strings)
                xml = xml.replace("</sst>", additions + "</sst>")
                uniq = re.search(r'uniqueCount="(\d+)"', xml)
                if uniq:
                    xml = xml.replace(f'uniqueCount="{uniq.group(1)}"',
                                      f'uniqueCount="{int(uniq.group(1)) + len(template.new_strings)}"')
                cnt = re.search(r'count="(\d+)"', xml)
                if cnt:
                    xml = xml.replace(f'count="{cnt.group(1)}"',
                                      f'count="{int(cnt.group(1)) + len(template.new_strings)}"', 1)
                data = xml.encode("utf-8")
            dst.writestr(item, data)
    shutil.move(tmp, out_path)


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
