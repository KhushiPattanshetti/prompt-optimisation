#!/usr/bin/env python3
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import datetime


class ICDNode:
    def __init__(self, code: str):
        self.code = code

        # Structure
        self.parent: "ICDNode | None" = None
        self.children: list["ICDNode"] = []

        # Metadata (to be filled later)
        self.description: str | None = None
        self.depth: int | None = None
        self.ancestors: list[str] = []

        # Classification (future use)
        self.chapter: str | None = None
        self.block: str | None = None

    def add_child(self, child: "ICDNode"):
        self.children.append(child)
        child.parent = self


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
CHAPTER_RANGES = [
    ("A00", "B99", "Chapter I"),
    ("C00", "D49", "Chapter II"),
    ("D50", "D89", "Chapter III"),
    ("E00", "E89", "Chapter IV"),
    ("F01", "F99", "Chapter V"),
    ("G00", "G99", "Chapter VI"),
    ("H00", "H59", "Chapter VII"),
    ("H60", "H95", "Chapter VIII"),
    ("I00", "I99", "Chapter IX"),
    ("J00", "J99", "Chapter X"),
    ("K00", "K95", "Chapter XI"),
    ("L00", "L99", "Chapter XII"),
    ("M00", "M99", "Chapter XIII"),
    ("N00", "N99", "Chapter XIV"),
    ("O00", "O9A", "Chapter XV"),
    ("P00", "P96", "Chapter XVI"),
    ("Q00", "Q99", "Chapter XVII"),
    ("R00", "R99", "Chapter XVIII"),
    ("S00", "T88", "Chapter XIX"),
    ("V00", "Y99", "Chapter XX"),
    ("Z00", "Z99", "Chapter XXI"),
]
BLOCK_RANGES = [
    ("A00", "A09", "A00-A09"),
    ("A15", "A19", "A15-A19"),
    ("A20", "A28", "A20-A28"),
    ("A30", "A49", "A30-A49"),
    ("E10", "E14", "E10-E14"),
    ("E15", "E16", "E15-E16"),
]
code_index: dict[str, ICDNode] = {}


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
                    missing_headers = [
                        h for h in EXPECTED_HEADERS if h not in normalized_to_col
                    ]
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

    raise ValueError("No worksheet in XLSX source contains a 'CODE' header column")


def _load_shared_strings(workbook: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    shared_strings: list[str] = []
    ns = f"{{{EXCEL_NS}}}"
    with workbook.open("xl/sharedStrings.xml") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != f"{ns}si":
                continue
            text_fragments = [
                text_node.text or "" for text_node in elem.findall(f".//{ns}t")
            ]
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
        text_fragments = [
            text_node.text or "" for text_node in cell.findall(f".//{ns}t")
        ]
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

            block = resolve_block(parent)
            if block:
                tree[block].add(parent)
                chapter = resolve_chapter(parent)
                if chapter:
                    tree[chapter].add(block)
                    tree["ROOT"].add(chapter)
        elif len(canonicalized) == 3:
            block = resolve_block(canonicalized)
            tree[block].add(canonicalized)

            chapter = canonicalized[0]
            tree[chapter].add(block)
            tree["ROOT"].add(chapter)

        if canonicalized not in tree:
            tree[canonicalized] = set()

    return tree


def code_to_tuple(code: str):
    """
    Converts ICD code into (letter, numeric_part, suffix)
    Example:
        'E11' → ('E', 11, '')
        'O9A' → ('O', 9, 'A')
    """
    code = code[:3]

    letter = code[0]
    rest = code[1:]

    match = re.match(r"(\d+)([A-Z]*)", rest)

    if match:
        number = int(match.group(1))
        suffix = match.group(2)
    else:
        number = 0
        suffix = rest

    return (letter, number, suffix)


def resolve_chapter(code: str) -> str | None:
    base = code[:3]

    for start, end, chapter in CHAPTER_RANGES:
        if code_to_tuple(start) <= code_to_tuple(base) <= code_to_tuple(end):
            return chapter

    # Fallback
    return f"{code[0]}-CHAPTER"


def resolve_block(code: str) -> str | None:
    base = code[:3]

    for start, end, block in BLOCK_RANGES:
        if code_to_tuple(start) <= code_to_tuple(base) <= code_to_tuple(end):
            return block

    # Fallback: group by first 3 characters range
    return f"{base}-BLOCK"


def assign_depth(node: ICDNode, depth: int):
    node.depth = depth
    for child in node.children:
        assign_depth(child, depth + 1)


def compute_ancestors(node: ICDNode):
    curr = node.parent
    while curr:
        node.ancestors.append(curr.code)
        curr = curr.parent


def find_first_ancestor(node: ICDNode, condition_fn):
    curr = node.parent
    while curr:
        if condition_fn(curr):
            return curr.code
        curr = curr.parent
    return None


def is_chapter(code: str):
    return code.startswith("Chapter")


def is_block(code: str):
    return "-" in code and not code.startswith("Chapter")


def build_json_schema(code_index: dict[str, ICDNode]) -> dict:
    index_json = {}

    for code, node in code_index.items():

        # ❌ Skip ROOT
        if code == "ROOT":
            continue

        index_json[code] = {
            "code": node.code,
            "description": node.description,  # may be None for now
            "parent": node.parent.code if node.parent else None,
            "children": [child.code for child in node.children],
            "ancestors": node.ancestors,
            "depth": node.depth,
            "chapter": node.chapter,
            "block": node.block,
            "is_leaf": len(node.children) == 0,
        }

    return {
        "version": "ICD-10",
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "index": index_json,
    }


def main() -> None:
    codes = read_icd_codes(ICD_SOURCE)
    unique_codes = sorted(set(codes))

    # Step 1: Create nodes
    for code in unique_codes:
        canonical = canonicalize(code)
        if canonical and canonical not in code_index:
            code_index[canonical] = ICDNode(canonical)

    # Step 2: Build existing tree (unchanged)
    tree = build_tree(unique_codes)

    # Step 2.1: Add extra safety check
    clean_tree = {}

    for key, children in tree.items():
        if key is None:
            continue
        clean_tree[key] = children

    tree = clean_tree

    # Link nodes using tree structure
    for parent_code, children_codes in tree.items():
        parent_node = code_index.get(parent_code)

        # Create parent node if not present (e.g., ROOT, blocks)
        if parent_node is None:
            parent_node = ICDNode(parent_code)
            code_index[parent_code] = parent_node

        for child_code in children_codes:
            child_node = code_index.get(child_code)

            if child_node is None:
                child_node = ICDNode(child_code)
                code_index[child_code] = child_node

            parent_node.add_child(child_node)

    # Depth Assignment
    root = code_index.get("ROOT")
    if root:
        assign_depth(root, 0)

    # Ancestor Computation
    for node in code_index.values():
        node.ancestors = []
        compute_ancestors(node)
        node.chapter = find_first_ancestor(node, lambda n: is_chapter(n.code))
        node.block = find_first_ancestor(node, lambda n: is_block(n.code))

    # Debug statements (to remove them later)
    """
    print(
        "Children of A00:",
        (
            [child.code for child in code_index.get("A00", []).children]
            if "A00" in code_index
            else "A00 not found"
        ),
    )
    print(
        "Parent of A00.0:",
        (
            code_index["A00.0"].parent.code
            if "A00.0" in code_index and code_index["A00.0"].parent
            else "Not found"
        ),
    )
    
    print(code_to_tuple("E11"))
    print(code_to_tuple("O9A"))
    print(code_to_tuple("A00"))
    print("Block of E11:", resolve_block("E11"))
    print("Chapter of E11:", resolve_chapter("E11"))
    
    sample = code_index.get("E11.9")
    if sample:
        print("Depth:", sample.depth)
        print("Ancestors:", sample.ancestors)
        print("Block:", sample.block)
        print("Chapter:", sample.chapter)
    """

    schema = build_json_schema(code_index)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Debug Check
    sample = schema["index"].get("E11.9")
    if sample:
        print(json.dumps(sample, indent=2))

    index = schema["index"]

    total_edges = sum(len(node["children"]) for node in index.values())

    print(f"source_path={ICD_SOURCE}")
    print(f"expected_headers={read_expected_headers()}")
    print(f"codes_in_source={len(codes)}")
    print(f"unique_codes={len(unique_codes)}")

    # Updated metrics
    print(f"total_nodes_in_index={len(index)}")
    print(f"total_edges={total_edges}")

    # Optional but VERY useful debug stats
    leaf_nodes = sum(1 for node in index.values() if node["is_leaf"])
    max_depth = max(
        (node["depth"] for node in index.values() if node["depth"] is not None),
        default=0,
    )

    print(f"leaf_nodes={leaf_nodes}")
    print(f"max_depth={max_depth}")


if __name__ == "__main__":
    main()
