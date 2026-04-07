#!/usr/bin/env python3
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ICD_SOURCE = PROJECT_ROOT / "data" / "section111_valid_icd10_october2025.xlsx"
OUTPUT_JSON = PROJECT_ROOT / "reward_metrics_svc" / "icd10_tree.json"
EXCEL_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CODE_HEADER = "CODE"
COL_REF_RE = re.compile(r"[A-Z]+")
EXPECTED_HEADERS = (
    "CODE",
    "SHORT DESCRIPTION (VALID ICD-10 FY2026)",
    "LONG DESCRIPTION (VALID ICD-10 FY2026)",
    "NF EXCL",
)


def canonicalize(code: str) -> str:
    value = (code or "").strip().upper()
    if not value:
        return ""
    if len(value) > 3 and "." not in value:
        value = f"{value[:3]}.{value[3:]}"
    return value


def normalize_header(value: str) -> str:
    return " ".join(str(value or "").strip().upper().split())


def read_icd_codes(csv_or_xlsx_path: Path) -> list[str]:
    if not csv_or_xlsx_path.exists():
        raise FileNotFoundError(f"Missing ICD source file: {csv_or_xlsx_path}")

    suffix = csv_or_xlsx_path.suffix.lower()
    if suffix == ".xlsx":
        return read_icd_codes_from_xlsx(csv_or_xlsx_path)

    return read_icd_codes_from_csv(csv_or_xlsx_path)


def read_icd_codes_from_csv(csv_path: Path) -> list[str]:
    codes: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("ICD CSV has no header row")

        normalized_to_actual = {
            normalize_header(name): name
            for name in reader.fieldnames
            if name is not None
        }
        missing_headers = [h for h in EXPECTED_HEADERS if h not in normalized_to_actual]
        if missing_headers:
            raise ValueError(
                "ICD CSV is missing required headers. "
                f"Missing: {missing_headers}. "
                f"Found columns: {reader.fieldnames}"
            )

        code_key = normalized_to_actual[CODE_HEADER]
        for row in reader:
            code = canonicalize(row.get(code_key, ""))
            if code:
                codes.append(code)

    return codes


def read_icd_codes_from_xlsx(xlsx_path: Path) -> list[str]:
    with ZipFile(xlsx_path) as workbook:
        shared_strings = _load_shared_strings(workbook)
        worksheet_paths = list(_iter_worksheet_paths(workbook))

        if not worksheet_paths:
            raise ValueError("XLSX workbook has no worksheets")

        for worksheet_path in worksheet_paths:
            code_col = None
            codes: list[str] = []

            for row in _iter_worksheet_rows(workbook, worksheet_path, shared_strings):
                if code_col is None:
                    normalized_to_col = {
                        normalize_header(cell_value): col
                        for col, cell_value in row.items()
                        if cell_value
                    }
                    missing_headers = [h for h in EXPECTED_HEADERS if h not in normalized_to_col]
                    if missing_headers:
                        break

                    code_col = normalized_to_col.get(CODE_HEADER)
                    if code_col is None:
                        break
                    continue

                code = canonicalize(row.get(code_col, ""))
                if code:
                    codes.append(code)

            if code_col is not None:
                return codes

    raise ValueError(
        "No worksheet in XLSX source contains a 'CODE' header column"
    )


def _load_shared_strings(workbook: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    shared_strings: list[str] = []
    ns = f"{{{EXCEL_NS}}}"
    with workbook.open("xl/sharedStrings.xml") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != f"{ns}si":
                continue
            text_fragments = [text_node.text or "" for text_node in elem.findall(f".//{ns}t")]
            shared_strings.append("".join(text_fragments))
            elem.clear()

    return shared_strings


def _iter_worksheet_paths(workbook: ZipFile):
    rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_by_id: dict[str, str] = {}

    for rel in rels_root.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        rel_id = rel.get("Id")
        target = rel.get("Target")
        if not rel_id or not target:
            continue

        if target.startswith("/"):
            normalized_target = target.lstrip("/")
        elif target.startswith("xl/"):
            normalized_target = target
        else:
            normalized_target = f"xl/{target}"

        rel_by_id[rel_id] = normalized_target

    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    ns = f"{{{EXCEL_NS}}}"
    for sheet in workbook_root.findall(f"{ns}sheets/{ns}sheet"):
        rel_id = sheet.get(f"{{{OFFICE_REL_NS}}}id")
        if not rel_id:
            continue
        target = rel_by_id.get(rel_id)
        if target and target.startswith("xl/worksheets/"):
            yield target


def _iter_worksheet_rows(
    workbook: ZipFile,
    worksheet_path: str,
    shared_strings: list[str],
):
    ns = f"{{{EXCEL_NS}}}"
    with workbook.open(worksheet_path) as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != f"{ns}row":
                continue

            row_cells: dict[str, str] = {}
            for cell in elem.findall(f"{ns}c"):
                cell_ref = cell.get("r", "")
                match = COL_REF_RE.match(cell_ref)
                if not match:
                    continue

                col = match.group(0)
                row_cells[col] = _cell_value(cell, shared_strings)

            if row_cells:
                yield row_cells

            elem.clear()


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    ns = f"{{{EXCEL_NS}}}"
    cell_type = cell.get("t")

    if cell_type == "inlineStr":
        text_fragments = [text_node.text or "" for text_node in cell.findall(f".//{ns}t")]
        return "".join(text_fragments)

    raw_value = cell.findtext(f"{ns}v", default="")
    if not raw_value:
        return ""

    if cell_type == "s":
        try:
            idx = int(raw_value)
            if 0 <= idx < len(shared_strings):
                return shared_strings[idx]
            return ""
        except ValueError:
            return ""

    return raw_value


def read_expected_headers() -> list[str]:
    # CODE is the only column consumed for tree construction.
    return list(EXPECTED_HEADERS)


def build_tree(codes: list[str]) -> dict[str, set[str]]:
    tree: dict[str, set[str]] = defaultdict(set)

    for code in codes:
        canonicalized = canonicalize(code)
        if not canonicalized:
            continue

        if "." in canonicalized:
            parent = canonicalized.split(".")[0]
            tree[parent].add(canonicalized)

            block = parent[:2]
            tree[block].add(parent)

            chapter = parent[0]
            tree[chapter].add(block)
            tree["ROOT"].add(chapter)
        elif len(canonicalized) == 3:
            block = canonicalized[:2]
            tree[block].add(canonicalized)

            chapter = canonicalized[0]
            tree[chapter].add(block)
            tree["ROOT"].add(chapter)

        if canonicalized not in tree:
            tree[canonicalized] = set()

    return tree


def main() -> None:
    codes = read_icd_codes(ICD_SOURCE)
    unique_codes = sorted(set(codes))
    tree = build_tree(unique_codes)

    serializable = {key: sorted(children) for key, children in sorted(tree.items())}

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    total_edges = sum(len(children) for children in serializable.values())
    print(f"source_path={ICD_SOURCE}")
    print(f"expected_headers={read_expected_headers()}")
    print(f"codes_in_source={len(codes)}")
    print(f"unique_codes={len(unique_codes)}")
    print(f"parent_keys={len(serializable)}")
    print(f"total_edges={total_edges}")


if __name__ == "__main__":
    main()
