from __future__ import annotations

import csv
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile

EXCEL_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
COL_REF_RE = re.compile(r"[A-Z]+")

_CODE_HEADER_CANDIDATES = (
    "CODE",
    "ICD_CODE",
    "ICD10_CODE",
    "ICD10",
    "ICD-10",
)
_DESC_HEADER_CANDIDATES = (
    "LONG DESCRIPTION (VALID ICD-10 FY2026)",
    "SHORT DESCRIPTION (VALID ICD-10 FY2026)",
    "LONG_DESCRIPTION",
    "SHORT_DESCRIPTION",
    "DESCRIPTION",
    "DESC",
)

_DEFAULT_SOURCE_PATHS = (
    Path(__file__).resolve().parent / "icd10_descriptions.json",
    Path(__file__).resolve().parent / "icd10_descriptions.csv",
    Path(__file__).resolve().parent.parent / "data" / "icd10_descriptions.json",
    Path(__file__).resolve().parent.parent / "data" / "icd10_descriptions.csv",
    Path(__file__).resolve().parent.parent / "data" / "section111_valid_icd10_october2025.xlsx",
)

_DESCRIPTION_MAP: Optional[Dict[str, str]] = None
_DESCRIPTION_SOURCE: str = ""
_DESCRIPTION_LOCK = threading.Lock()


def canonicalize_code(code: str) -> str:
    normalized = str(code or "").strip().upper()
    if len(normalized) > 3 and "." not in normalized:
        normalized = f"{normalized[:3]}.{normalized[3:]}"
    return normalized


def normalize_description(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_header(value: str) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _resolve_source_path() -> Optional[Path]:
    configured = os.environ.get("ICD_DESCRIPTION_FILE", "").strip()
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent.parent / configured
        if candidate.exists():
            return candidate
        return None

    for candidate in _DEFAULT_SOURCE_PATHS:
        if candidate.exists():
            return candidate
    return None


def _pick_columns(headers: List[str]) -> tuple[Optional[str], Optional[str]]:
    normalized_to_actual = {
        _normalize_header(header): header
        for header in headers
        if header is not None
    }

    code_col = None
    for candidate in _CODE_HEADER_CANDIDATES:
        actual = normalized_to_actual.get(_normalize_header(candidate))
        if actual is not None:
            code_col = actual
            break

    desc_col = None
    for candidate in _DESC_HEADER_CANDIDATES:
        actual = normalized_to_actual.get(_normalize_header(candidate))
        if actual is not None:
            desc_col = actual
            break

    if desc_col is None:
        # Best-effort fallback for ad-hoc files with any "description" column name.
        for normalized, actual in normalized_to_actual.items():
            if "DESCRIPTION" in normalized:
                desc_col = actual
                break

    return code_col, desc_col


def _load_from_json(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}

    if isinstance(payload, dict):
        for raw_code, raw_value in payload.items():
            code = canonicalize_code(str(raw_code or ""))
            if not code:
                continue

            if isinstance(raw_value, dict):
                value = (
                    raw_value.get("description")
                    or raw_value.get("long_description")
                    or raw_value.get("short_description")
                    or ""
                )
            else:
                value = str(raw_value or "")

            normalized = normalize_description(value)
            if normalized:
                out[code] = normalized

    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_code = row.get("code") or row.get("icd_code") or row.get("icd10")
            code = canonicalize_code(str(raw_code or ""))
            if not code:
                continue
            raw_desc = row.get("description") or row.get("long_description") or row.get("short_description") or ""
            normalized = normalize_description(str(raw_desc or ""))
            if normalized:
                out[code] = normalized

    return out


def _load_from_csv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return out

        code_col, desc_col = _pick_columns(list(reader.fieldnames))
        if code_col is None or desc_col is None:
            return out

        for row in reader:
            code = canonicalize_code(str(row.get(code_col, "") or ""))
            if not code:
                continue
            normalized = normalize_description(str(row.get(desc_col, "") or ""))
            if normalized:
                out[code] = normalized

    return out


def _load_shared_strings(workbook: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    ns = f"{{{EXCEL_NS}}}"
    shared_strings: List[str] = []
    with workbook.open("xl/sharedStrings.xml") as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag != f"{ns}si":
                continue
            fragments = [node.text or "" for node in elem.findall(f".//{ns}t")]
            shared_strings.append("".join(fragments))
            elem.clear()
    return shared_strings


def _iter_worksheet_paths(workbook: ZipFile):
    rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_by_id: Dict[str, str] = {}

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


def _cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    ns = f"{{{EXCEL_NS}}}"
    cell_type = cell.get("t")

    if cell_type == "inlineStr":
        fragments = [node.text or "" for node in cell.findall(f".//{ns}t")]
        return "".join(fragments)

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


def _iter_worksheet_rows(workbook: ZipFile, worksheet_path: str, shared_strings: List[str]):
    ns = f"{{{EXCEL_NS}}}"
    with workbook.open(worksheet_path) as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag != f"{ns}row":
                continue

            row_cells: Dict[str, str] = {}
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


def _load_from_xlsx(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with ZipFile(path) as workbook:
        shared_strings = _load_shared_strings(workbook)
        worksheet_paths = list(_iter_worksheet_paths(workbook))
        if not worksheet_paths:
            return out

        for worksheet_path in worksheet_paths:
            code_col: Optional[str] = None
            desc_col: Optional[str] = None

            for row in _iter_worksheet_rows(workbook, worksheet_path, shared_strings):
                if code_col is None:
                    headers_by_col = {
                        col: _normalize_header(value)
                        for col, value in row.items()
                    }
                    normalized_to_col = {value: col for col, value in headers_by_col.items()}

                    code_col = None
                    for candidate in _CODE_HEADER_CANDIDATES:
                        col = normalized_to_col.get(_normalize_header(candidate))
                        if col is not None:
                            code_col = col
                            break

                    desc_col = None
                    for candidate in _DESC_HEADER_CANDIDATES:
                        col = normalized_to_col.get(_normalize_header(candidate))
                        if col is not None:
                            desc_col = col
                            break

                    if code_col is None or desc_col is None:
                        break
                    continue

                code = canonicalize_code(str(row.get(code_col, "") or ""))
                if not code:
                    continue
                normalized = normalize_description(str(row.get(desc_col, "") or ""))
                if normalized:
                    out[code] = normalized

            if code_col is not None and desc_col is not None and out:
                return out

    return out


def _load_description_map() -> Dict[str, str]:
    source_path = _resolve_source_path()
    if source_path is None:
        return {}

    suffix = source_path.suffix.lower()
    try:
        if suffix == ".json":
            mapping = _load_from_json(source_path)
        elif suffix == ".csv":
            mapping = _load_from_csv(source_path)
        elif suffix == ".xlsx":
            mapping = _load_from_xlsx(source_path)
        else:
            mapping = {}
    except Exception:
        mapping = {}

    global _DESCRIPTION_SOURCE
    _DESCRIPTION_SOURCE = str(source_path)
    return mapping


def _ensure_loaded() -> Dict[str, str]:
    global _DESCRIPTION_MAP
    if _DESCRIPTION_MAP is not None:
        return _DESCRIPTION_MAP

    with _DESCRIPTION_LOCK:
        if _DESCRIPTION_MAP is None:
            _DESCRIPTION_MAP = _load_description_map()
    return _DESCRIPTION_MAP


def get_description(icd_code: str) -> str:
    code = canonicalize_code(icd_code)
    if not code:
        return ""
    mapping = _ensure_loaded()
    return str(mapping.get(code, ""))


def get_descriptions(codes: List[str]) -> List[str]:
    return [get_description(code) for code in list(codes or [])]


def description_source() -> str:
    _ensure_loaded()
    return _DESCRIPTION_SOURCE


def description_count() -> int:
    return len(_ensure_loaded())
