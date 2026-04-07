from __future__ import annotations

from collections import Counter
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase

from rewriter_inference_svc.config import (
    BEST_PROMPT_CACHE_FILE,
    BEST_PROMPT_CACHE_THRESHOLD,
    COMMON_ICD_RELEVANT_TERMS,
    MEDICAL_DESCRIPTOR_TERMS,
    DO_SAMPLE,
    MAX_PROMPT_TOKENS,
    MAX_RAW_NOTE_CHARS,
    MAX_NEW_TOKENS,
    OUTPUT_PATH,
    REWRITER_ENABLE_POSTFILTER_GUARD,
    REWRITER_GUARD_MIN_KEYWORD_HITS,
    REWRITER_INPUT_MODE,
    REWRITER_MODEL_VARIANT,
    TEMPERATURE,
)
from rewriter_inference_svc.logger import get_logger
from rewriter_inference_svc.model_loader import load_model

log = get_logger(__name__)

REWRITER_SYSTEM_PROMPT = (
    "You are a clinical prompt optimizer for ICD-10 coding.\n"
    "You receive high-signal clinical content extracted from one note.\n\n"
    "Write one note-specific extraction instruction for a downstream coding model.\n"
    "Keep diagnosis intent faithful and preserve clinically relevant ambiguity.\n\n"
    "Rules:\n"
    "- Output an instruction prompt, not diagnosis codes\n"
    "- Never emit ICD-10 code strings\n"
    "- Use domain-specific clinical focus from this note (for example neuro, cardio, pulmonary, renal, GI/hepatic, endocrine, oncology, infectious, psychiatric)\n"
    "- Mention at least two note-specific clinical targets (diagnoses, complications, or procedures)\n"
    "- Reuse at least one concrete diagnostic descriptor from this note\n"
    "- Vary wording across notes and avoid fixed generic templates\n"
    "- Never begin with a boilerplate template such as 'Extract all ICD-10-CM diagnosis codes from the clinical information below'\n"
    "- Do not reuse generic phrasing like 'clinical information below' unless paired with explicit note focus\n"
    "- Keep output compact and JSON-extraction oriented\n"
    "- Output must be a single paragraph between 80 and 400 characters\n"
    "- Output must not contain any JSON arrays or code lists\n"
)

_ICD_CODE_REGEX = re.compile(r"\b[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?\b")
_HEALTHCHECK_LITERALS = {"test", "dummy", "healthcheck", "ping"}
_HEADER_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 /,&()\-']{1,88}:?$")
_INLINE_HEADER_RE = re.compile(r"^(?P<header>[A-Za-z][A-Za-z0-9 /,&()\-']{1,88}):\s*(?P<value>.+)$")

_HIGH_SIGNAL_SECTION_PATTERNS = [
    r"(?:final\s+)?diagnos(?:is|es)",
    r"assessment(?:\s+and\s+plan)?",
    r"impression",
    r"problem\s+list",
    r"brief\s+hospital\s+course",
    r"discharge\s+(?:diagnos(?:is|es)|condition|summary)",
    r"history\s+of\s+present\s+illness",
    r"chief\s+complaint",
    r"significant\s+(?:findings?|results?)",
    r"procedures?\s+performed",
    r"major\s+surgical\s+or\s+invasive\s+procedure",
    r"reason\s+for\s+(?:admission|visit)",
    r"past\s+(?:medical\s+)?history",
    r"complications?",
]

_LOW_SIGNAL_SECTION_PATTERNS = [
    r"name",
    r"unit\s+no",
    r"admission\s+date",
    r"discharge\s+date",
    r"date\s+of\s+birth",
    r"sex",
    r"service",
    r"attending",
    r"medications?\s+(?:on\s+(?:admission|discharge))?",
    r"(?:discharge\s+)?instructions?",
    r"discharge\s+disposition",
    r"discharge\s+condition",
    r"follow[\s\-]?up",
    r"social\s+history",
    r"family\s+history",
    r"review\s+of\s+systems",
    r"physical\s+exam(?:ination)?",
    r"vital\s+signs?",
    r"nursing\s+notes?",
    r"diet",
    r"activity",
    r"allergies?",
]

_SECTION_PRIORITY_ORDER = [
    "discharge diagnoses",
    "final diagnosis",
    "assessment and plan",
    "assessment",
    "impression",
    "problem list",
    "procedures performed",
    "reason for admission",
    "history of present illness",
    "chief complaint",
    "past medical history",
    "significant findings",
    "complications",
]

_DICTATION_NOISE_RE = re.compile(
    r"\b(?:dictated\s+by|transcribed\s+by|signed\s+by|electronically\s+signed|"
    r"cc:|attending:|resident:)[^\n]*",
    re.IGNORECASE,
)

_DEMOGRAPHIC_LINE_RE = re.compile(
    r"^(?:name|unit\s+no|admission\s+date|discharge\s+date|date\s+of\s+birth|sex|service|attending)\b",
    re.IGNORECASE,
)

_MEDICATION_LINE_HINT_RE = re.compile(
    r"\b(?:mg|mcg|tablet|tab|capsule|puff|neb|inhaler|patch|daily|bid|tid|q\d+h|prn|rx|po|iv|ih|disp|"
    r"lasix|aldactone|spironolactone|furosemide|truvada|raltegravir|albuterol|ipratropium|acetaminophen|"
    r"self-discontinuing|medication)\b",
    re.IGNORECASE,
)

_LAB_LINE_HINT_RE = re.compile(
    r"\b(?:glucose|urea|creat|creatinine|sodium|potassium|chloride|anion\s+gap|wbc|rbc|hgb|hct|mcv|mch|plt|platelet|neuts|lymphs|monos|eos|basos|inr|ptt|lipase|albumin|alk\s+phos|ast|alt|tot\s+bili)\b",
    re.IGNORECASE,
)

_DISEASE_SIGNAL_RE = re.compile(
    r"\b(?:cirrhosis|ascites|portal\s+htn|portal\s+hypertension|hiv|copd|hepatitis|"
    r"hypertension|diabetes|disorder|ptsd|bipolar|infection|failure|pain|distension|"
    r"dyspnea|encephalopathy|sepsis|cholelithiasis|splenomegaly|cancer|lesion|"
    r"thrombocytopenia|coagulopathy|confusion|hematuria|hematemesis|hemoptysis)\b",
    re.IGNORECASE,
)

_DIAGNOSIS_SECTION_HINTS = (
    "diagnosis",
    "assessment",
    "problem",
    "history of present illness",
    "past medical history",
    "brief hospital course",
    "impression",
    "significant findings",
    "procedures performed",
)

_SHORTHAND_REPLACEMENTS = [
    (r"\bc\/b\b", "complicated by"),
    (r"\bh\/o\b", "history of"),
    (r"\bp\/w\b", "presented with"),
    (r"\bs\/p\b", "status post"),
    (r"\bd\/t\b", "due to"),
]

_SEMANTIC_STOPWORDS = {
    "acute",
    "chronic",
    "likely",
    "possible",
    "history",
    "presented",
    "present",
    "with",
    "without",
    "from",
    "over",
    "past",
    "week",
    "weeks",
    "days",
    "months",
    "years",
    "patient",
    "reports",
    "report",
}

_DIAGNOSIS_HEADER_HINTS = ("diagnosis", "assessment", "impression", "problem list")
_PROCEDURE_HEADER_HINTS = (
    "major surgical or invasive procedure",
    "procedures performed",
    "procedure",
)

_PRIORITY_DIAGNOSIS_HEADERS = ("diagnosis", "assessment", "impression", "problem list")
_PRIORITY_CONTEXT_PATTERNS = (
    re.compile(
        r"(?:history\s+of|with|known|diagnosed\s+with|presenting\s+with) ([a-zA-Z\- ]{3,60})",
        re.IGNORECASE,
    ),
    re.compile(r"(?:status\s+post|s/p) ([a-zA-Z\- ]{3,60})", re.IGNORECASE),
)
_DESCRIPTOR_TRAILING_FRAGMENT_RE = re.compile(
    r"\b(?:who|which|that|when|while|because|after|before|during|"
    r"in\s+the\s+setting\s+of|secondary\s+to|due\s+to)\b",
    re.IGNORECASE,
)

_SECTION_TERM_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\n|\band\b)\s*", re.IGNORECASE)
_CONTEXT_TERM_RE = re.compile(
    r"\b(?:with|history\s+of|status\s+post|s\/p|presenting\s+with|known|confirmed)\s+([^.;:\n]{3,120})",
    re.IGNORECASE,
)
_ABBREV_EXPANSION_RE = re.compile(r"\b[A-Z]{2,6}\s*\(([^)]+)\)")
_PROCEDURE_TRIGGER_RE = re.compile(
    r"\b(?:s\/p|status\s+post|post-op)\s+([^.;,\n]{3,120})",
    re.IGNORECASE,
)
_SEMANTIC_NARRATIVE_RE = re.compile(
    r"\b(?:patients?|pt|says|said|reports?|reported|denies?|recommended|started|noted|"
    r"presented|presents|underwent|evaluated|transferred|discharged|"
    r"walking|fell|improved|admitted|sent\s+to)\b",
    re.IGNORECASE,
)

_DESCRIPTOR_LEADING_PREFIX_RE = re.compile(
    r"^(?:history\s+of|hx\s+of|h\/o|with|known|confirmed|presenting\s+with|"
    r"diagnosed\s+with|complaint\s+of|concern\s+for|secondary\s+to)\s+",
    re.IGNORECASE,
)
_DESCRIPTOR_TRAILING_SPLIT_RE = re.compile(
    r"\b(?:who|which|that|when|while|because|after|before|during|"
    r"reported|reports?|denies?|recommended|started|noted|presented|admitted|"
    r"in\s+the\s+setting\s+of|secondary\s+to|due\s+to)\b",
    re.IGNORECASE,
)
_DESCRIPTOR_EDGE_STOPWORDS = {
    "and",
    "or",
    "with",
    "without",
    "a",
    "an",
    "the",
    "of",
    "to",
    "for",
    "in",
    "on",
    "by",
    "from",
}
_SEMANTIC_DESCRIPTOR_STOPWORDS = _DESCRIPTOR_EDGE_STOPWORDS | _SEMANTIC_STOPWORDS | {
    "status",
    "post",
    "known",
    "diagnosed",
    "presenting",
    "complicated",
    "reason",
    "consult",
    "chief",
    "complaint",
}
_SEMANTIC_DESCRIPTOR_PREFIX_RE = re.compile(
    r"^(?:reason\s+for\s+consult|chief\s+complaint|hpi|history\s+of\s+present\s+illness|"
    r"past\s+medical\s+history|assessment(?:\s+and\s+plan)?|diagnosis|impression)\s*:?")
_INVALID_DESCRIPTOR_PHRASES = {
    "and a",
    "and an",
    "and the",
    "with a",
    "with an",
    "with the",
    "of a",
    "of the",
}
_DESCRIPTOR_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|\bor\b)\s*", re.IGNORECASE)
_DESCRIPTOR_GENERIC_HEADWORDS = {
    "disease",
    "disorder",
    "condition",
    "problem",
    "issue",
    "syndrome",
}
_DESCRIPTOR_STOPWORD_ONLY_RE = re.compile(
    r"^(?:and|or|with|without|of|to|for|in|on|by|a|an|the)(?:\s+(?:and|or|with|without|of|to|for|in|on|by|a|an|the))*$",
    re.IGNORECASE,
)
_FOCUS_CUE_RE = re.compile(r"\b(?:focus|focused|target(?:ing)?|prioriti[sz]e|for)\b", re.IGNORECASE)
_NON_MEDICAL_DESCRIPTOR_HINT_RE = re.compile(
    r"\b(?:follow[-\s]?up|appointment|therapy|rehab|medication|dose|tablet|capsule|daily|"
    r"bid|tid|home|hospital|admission|discharge|family|social|exam|vitals?|"
    r"phone|service|attending)\b",
    re.IGNORECASE,
)

_GENERIC_TEMPLATE_PHRASES = (
    "extract all icd-10-cm diagnosis codes",
    "extract all icd-10 diagnosis codes",
    "from the clinical information below",
    "output only a json array of code strings",
    "include diagnoses, complications, and relevant co-morbidities",
    "include diagnoses, complications, and relevant comorbidities",
)
_CLINICAL_FOCUS_KEYWORDS = (
    "neuro",
    "neurolog",
    "dementia",
    "parkinson",
    "stroke",
    "seizure",
    "hallucination",
    "gait",
    "cardio",
    "cardiac",
    "coronary",
    "myocardial",
    "heart",
    "atrial",
    "fibrillation",
    "arrhythm",
    "hypertension",
    "vascular",
    "pulmonary",
    "respiratory",
    "copd",
    "asthma",
    "pneumonia",
    "dyspnea",
    "embol",
    "renal",
    "kidney",
    "ckd",
    "aki",
    "neph",
    "hematuria",
    "hydronephrosis",
    "gastro",
    "hepatic",
    "liver",
    "cirrhosis",
    "ascites",
    "hepatitis",
    "gastr",
    "ulcer",
    "endocrine",
    "diabetes",
    "thyroid",
    "metabolic",
    "oncolog",
    "cancer",
    "tumor",
    "malignan",
    "carcinoma",
    "neoplasm",
    "infect",
    "sepsis",
    "hiv",
    "uti",
    "psychi",
    "depression",
    "anxiety",
    "bipolar",
    "ptsd",
)

_COMMON_ICD_TERM_SET = {term.lower() for term in COMMON_ICD_RELEVANT_TERMS}
_COMMON_ICD_TERM_PATTERNS = [
    (term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
    for term in COMMON_ICD_RELEVANT_TERMS
]

_STRUCTURED_PRIORITY_HEADERS = ("diagnosis", "assessment", "impression", "problem list")
_STRUCTURED_PRIORITY_LINE_RE = re.compile(
    r"(?:^|\n)\s*(diagnosis|assessment|impression|problem\s+list)\s*:?\s*([^\n]+)",
    re.IGNORECASE,
)
_STRUCTURED_CONTEXT_RE = re.compile(
    r"(?:history\s+of|with|known|presenting\s+with|s/p|status\s+post)\s+([A-Za-z][A-Za-z\-\' ]{2,50})",
    re.IGNORECASE,
)
_STRUCTURED_DESCRIPTOR_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\n)\s*")
_STRUCTURED_TRAILING_FRAGMENT_RE = re.compile(
    r"\b(?:who|which|that|when|while|because|after|before|during|"
    r"was|were|is|are|has|have|had|for|referred|admitted|discharged|"
    r"reported|denied|denies|presented|underwent|recommended|"
    r"in\s+the\s+setting\s+of|secondary\s+to|due\s+to)\b",
    re.IGNORECASE,
)
_STRUCTURED_LEADING_STOP_RE = re.compile(
    r"^(?:history\s+of|with|known|presenting\s+with|status\s+post|s/p|and|or)\s+",
    re.IGNORECASE,
)
_STRUCTURED_EDGE_STOPWORDS = {"history", "of", "with", "and", "or"}
_STRUCTURED_NON_NOUN_HINT_RE = re.compile(
    r"\b(?:referred|admitted|discharged|started|reported|denies?|"
    r"presented|underwent|recommended|follow(?:ed)?\s+up)\b",
    re.IGNORECASE,
)
_MEDICAL_DESCRIPTOR_TERM_SET = {term.lower() for term in MEDICAL_DESCRIPTOR_TERMS}
_MEDICAL_DESCRIPTOR_PATTERNS = [
    (term, re.compile(rf"{re.escape(term)}", re.IGNORECASE))
    for term in MEDICAL_DESCRIPTOR_TERMS
]

_inference_lock = threading.Lock()
_cache_lock = threading.Lock()
_best_prompt_cache: Optional[Dict[str, Dict[str, Any]]] = None

_LOGIT_CLAMP_MIN = -50.0
_LOGIT_CLAMP_MAX = 50.0
_MIN_RETRY_NEW_TOKENS = 32
_MIN_REWRITE_CHARS = 80
_MAX_REWRITE_CHARS = 420
_REWRITER_USER_TASK_PREFIX = (
    "Rewrite the following ICD-10-CM extraction prompt into a note-specific instruction. "
    "Preserve clinical focus, vary phrasing, and avoid generic boilerplate. "
    "Output one paragraph prompt only, and do not output diagnosis codes.\n\n"
)


class _FiniteClampLogitsProcessor(LogitsProcessor):
    """Force generation logits into finite, bounded values to prevent sampling/softmax blowups."""

    def __init__(self, min_logit: float = _LOGIT_CLAMP_MIN, max_logit: float = _LOGIT_CLAMP_MAX) -> None:
        self.min_logit = float(min_logit)
        self.max_logit = float(max_logit)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        finite_scores = torch.nan_to_num(
            scores,
            nan=0.0,
            posinf=self.max_logit,
            neginf=self.min_logit,
        )
        return torch.clamp(finite_scores, min=self.min_logit, max=self.max_logit)


_SAFE_LOGITS_PROCESSOR = LogitsProcessorList([_FiniteClampLogitsProcessor()])

_MIN_STAGE1_CHARS = 200
_STAGE1_MAX_CHARS = 2000
_STAGE2_MAX_CHARS = 1500
_STAGE3_MAX_CHARS = 1500
_STAGE4_MAX_CHARS = 1200


def _truncate_note(note: str, max_chars: int = MAX_RAW_NOTE_CHARS) -> str:
    text = (note or "").strip()
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars]
    boundaries = list(re.finditer(r"[.!?](?:\s|$)", clipped))
    if boundaries:
        return clipped[:boundaries[-1].end()].strip()
    return clipped.strip()


def _is_healthcheck_payload(note: str) -> bool:
    # Only treat exact probe literals as health checks to avoid bypassing
    # real clinical notes that naturally include words like "test".
    text = (note or "").strip().lower()
    return text in _HEALTHCHECK_LITERALS


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _render_rewriter_user_content(note_text: str) -> str:
    return f"{_REWRITER_USER_TASK_PREFIX}{str(note_text or '').strip()}"


def _truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value

    clipped = value[:max_chars]
    boundary = max(clipped.rfind("\n"), clipped.rfind(". "), clipped.rfind("; "))
    if boundary >= int(max_chars * 0.6):
        clipped = clipped[:boundary]
    return clipped.strip()


def _head_tail_fallback(note: str, max_chars: int = _STAGE2_MAX_CHARS) -> str:
    text = str(note or "").strip()
    if len(text) <= max_chars:
        return text

    sep = "\n...\n"
    head_chars = max(int(max_chars * 0.65), 1)
    tail_chars = max(max_chars - head_chars - len(sep), 1)
    return f"{text[:head_chars].rstrip()}{sep}{text[-tail_chars:].lstrip()}"


def _normalize_header(header: str) -> str:
    return re.sub(r"\s+", " ", str(header or "").strip().lower().rstrip(":"))


def _looks_like_section_header(line: str) -> bool:
    candidate = str(line or "").strip()
    if not candidate:
        return False
    if len(candidate) > 90:
        return False
    if len(candidate.split()) > 12:
        return False
    if candidate.count(":") > 1:
        return False
    if candidate.endswith((".", "?", "!")) and ":" not in candidate:
        return False
    return bool(_HEADER_LINE_RE.match(candidate))


def _classify_header(header: str) -> Optional[str]:
    normalized = _normalize_header(header)

    for pattern in _HIGH_SIGNAL_SECTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "high"

    for pattern in _LOW_SIGNAL_SECTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "low"

    return None


def _collect_relevant_sections(note: str) -> List[Tuple[str, str]]:
    lines = str(note or "").splitlines()
    sections: List[Tuple[str, str]] = []

    current_header = "Clinical Summary"
    current_relevant = False
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        if not current_relevant:
            current_lines = []
            return

        body = "\n".join(line.strip() for line in current_lines if line.strip()).strip()
        if body:
            sections.append((current_header, body))
        current_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        inline_match = _INLINE_HEADER_RE.match(stripped)
        if inline_match:
            header = inline_match.group("header").strip()
            inline_value = inline_match.group("value").strip()
            cls = _classify_header(header)
            if cls is not None:
                flush_current()
                current_header = header
                current_relevant = cls == "high"
                if current_relevant and inline_value:
                    current_lines.append(inline_value)
                continue

        if _looks_like_section_header(stripped):
            cls = _classify_header(stripped)
            if cls is not None:
                flush_current()
                current_header = stripped.rstrip(":").strip()
                current_relevant = cls == "high"
                continue

        if current_relevant:
            current_lines.append(stripped)

    flush_current()

    return sections


def extract_high_signal_sections(note: str, max_chars: int = _STAGE1_MAX_CHARS) -> str:
    sections = _collect_relevant_sections(note)

    blocks: List[str] = []
    chars_used = 0
    for header, body in sections:
        block = f"{header}:\n{body}".strip()
        if not block:
            continue

        remaining = max_chars - chars_used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = _truncate_text(block, remaining)
        blocks.append(block)
        chars_used += len(block) + 2

    extracted = "\n\n".join(blocks).strip()
    if not extracted:
        extracted = _head_tail_fallback(note, max_chars=min(max_chars, _STAGE2_MAX_CHARS))
    elif len(extracted) < _MIN_STAGE1_CHARS:
        extracted = _truncate_text(extracted, max_chars)
    return extracted


def compress_clinical_text(text: str, max_chars: int = _STAGE2_MAX_CHARS) -> str:
    value = str(text or "")
    value = _DICTATION_NOISE_RE.sub("", value)

    for pattern, replacement in _SHORTHAND_REPLACEMENTS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

    value = re.sub(
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b(?!\s+(?:diagnosis|procedure|surgery))",
        "[DATE]",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bMRN\s*:?\s*\d+\b", "[MRN]", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d{3}[\-\.]\d{3}[\-\.]\d{4}\b", "[PHONE]", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)

    cleaned_lines: List[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if len(stripped) <= 3:
            continue
        if _DEMOGRAPHIC_LINE_RE.match(stripped):
            continue
        if re.fullmatch(r"[\-_=*#\s]+", stripped):
            continue
        if _LAB_LINE_HINT_RE.search(stripped) and len(re.findall(r"\d", stripped)) >= 8:
            continue
        cleaned_lines.append(stripped)

    compressed = "\n".join(cleaned_lines).strip()
    if not compressed:
        compressed = _head_tail_fallback(text, max_chars=max_chars)
    return _truncate_text(compressed, max_chars)


def _parse_text_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_header = "Clinical Summary"
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        body = "\n".join(line.strip() for line in current_lines if line.strip()).strip()
        if body:
            sections.append((current_header, body))
        current_lines = []

    for line in str(text or "").splitlines():
        stripped = line.strip()
        if _looks_like_section_header(stripped) and stripped.endswith(":"):
            flush_current()
            current_header = stripped.rstrip(":").strip()
            continue
        current_lines.append(line)

    flush_current()
    return sections


def reorder_for_coding_priority(text: str, max_chars: int = _STAGE3_MAX_CHARS) -> str:
    sections = _parse_text_sections(text)
    if not sections:
        return _truncate_text(str(text or "").strip(), max_chars)

    remaining = list(sections)
    ordered: List[Tuple[str, str]] = []

    for priority in _SECTION_PRIORITY_ORDER:
        match_index: Optional[int] = None
        for idx, (header, _body) in enumerate(remaining):
            if priority in _normalize_header(header):
                match_index = idx
                break
        if match_index is not None:
            ordered.append(remaining.pop(match_index))

    ordered.extend(remaining)

    rendered = "\n\n".join(f"{header}:\n{body}" for header, body in ordered).strip()
    return _truncate_text(rendered, max_chars)


def _select_best_descriptor_fragment(value: str) -> str:
    source = str(value or "").strip()
    if not source:
        return ""

    fragments = [
        fragment.strip(" .;:-,")
        for fragment in _DESCRIPTOR_CLAUSE_SPLIT_RE.split(source)
        if fragment and fragment.strip(" .;:-,")
    ]
    if len(fragments) <= 1:
        return source

    ranked: List[Tuple[int, int, str]] = []
    for fragment in fragments:
        fragment_words = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", fragment)
        if not fragment_words:
            continue
        if len(fragment_words) > 5:
            continue

        fragment_norm = fragment.lower().strip()
        if _DESCRIPTOR_STOPWORD_ONLY_RE.fullmatch(fragment_norm):
            continue

        score = 0
        if _contains_common_icd_term(fragment_norm):
            score += 4
        if _DISEASE_SIGNAL_RE.search(fragment):
            score += 3
        if not _SEMANTIC_NARRATIVE_RE.search(fragment):
            score += 1
        if _NON_MEDICAL_DESCRIPTOR_HINT_RE.search(fragment):
            score -= 3
        score += min(len(fragment_words), 3)

        ranked.append((score, len(fragment_words), fragment))

    if not ranked:
        return fragments[0]

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def _normalize_descriptor_phrase(raw: str, category: str) -> str:
    value = str(raw or "").replace("___", " ").strip()
    value = re.sub(r"^[#>\-\*\s]+", "", value)
    value = re.sub(r"^\d+\.\s*", "", value)
    value = re.sub(
        r"^(?:diagnosis|assessment|impression|problem\s+list|major\s+surgical\s+or\s+invasive\s+procedure)\s*:?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    if category == "procedure":
        value = re.sub(r"^(?:status\s+post|s\/p|post-op)\s+", "", value, flags=re.IGNORECASE)
    else:
        value = re.sub(r"^the\s+above\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\((?:status\s+post|s\/p)[^)]+\)", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(?:history\s+of|presenting\s+with|with|known|confirmed)\s+", "", value, flags=re.IGNORECASE)
        value = re.split(r"\b(?:status\s+post|s\/p)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]

    value = re.split(
        r"\b(?:who|that|which|when|while|because|for|reports?|reported|denies?|"
        r"recommended|started|noted|complicated\s+by)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r"(?:\b(?:and|or|with|by|of|to|a|an|the)\b\s*)+$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" .;:-,")

    while value:
        reduced = _DESCRIPTOR_LEADING_PREFIX_RE.sub("", value, count=1).strip()
        if reduced == value:
            break
        value = reduced

    value = _DESCRIPTOR_TRAILING_SPLIT_RE.split(value, maxsplit=1)[0].strip(" .;:-,")
    value = _select_best_descriptor_fragment(value)

    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", value)]
    while tokens and tokens[0] in _DESCRIPTOR_EDGE_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _DESCRIPTOR_EDGE_STOPWORDS:
        tokens = tokens[:-1]

    if not tokens:
        return ""
    value = " ".join(tokens)
    return value


def _contains_common_icd_term(value_lower: str) -> bool:
    if value_lower in _COMMON_ICD_TERM_SET:
        return True
    return any(term in value_lower for term in _COMMON_ICD_TERM_SET)


def _is_clean_medical_term(candidate: str, category: str) -> bool:
    value = str(candidate or "").strip()
    if len(value) < 3 or len(value) > 80:
        return False

    words = value.split()
    if len(words) < 2 or len(words) > 5:
        return False

    lowered = value.lower()
    if _DESCRIPTOR_STOPWORD_ONLY_RE.fullmatch(lowered):
        return False
    if lowered in _INVALID_DESCRIPTOR_PHRASES:
        return False
    if re.search(r"\b(?:and|or)\b", lowered):
        return False
    if words[0].lower() in _DESCRIPTOR_GENERIC_HEADWORDS:
        return False
    if len(words) == 1 and lowered in _DESCRIPTOR_GENERIC_HEADWORDS:
        return False
    if any(word.lower() in _DESCRIPTOR_EDGE_STOPWORDS for word in (words[0], words[-1])):
        return False
    if len(words) <= 2 and all(word.lower() in _DESCRIPTOR_EDGE_STOPWORDS for word in words):
        return False

    if _SEMANTIC_NARRATIVE_RE.search(value):
        return False
    if re.search(r"\b(?:denies?|negative\s+for)\b", value, flags=re.IGNORECASE):
        return False
    if _MEDICATION_LINE_HINT_RE.search(value):
        return False
    if _LAB_LINE_HINT_RE.search(value):
        return False
    if _DEMOGRAPHIC_LINE_RE.match(value):
        return False

    if len(re.findall(r"\d", value)) > max(3, len(value) // 5):
        return False

    if _NON_MEDICAL_DESCRIPTOR_HINT_RE.search(value):
        return False

    if re.match(r"^(?:immediate|acute\s+onset)\b", lowered):
        return False
    if re.search(r"\b(?:immediate\s+pain|acute\s+onset)\b", lowered):
        return False
    if not _contains_common_icd_term(lowered) and not _DISEASE_SIGNAL_RE.search(value):
        return False

    return True


def _extract_section_terms(text: str, header_hints: Tuple[str, ...], max_items: int = 24) -> List[str]:
    out: List[str] = []
    for header, body in _parse_text_sections(text):
        header_norm = _normalize_header(header)
        if not any(hint in header_norm for hint in header_hints):
            continue

        for line in body.splitlines():
            for chunk in _SECTION_TERM_SPLIT_RE.split(line.strip()):
                if chunk:
                    out.append(chunk)
                    if len(out) >= max_items:
                        return out
    return out


def _extract_context_terms(text: str, max_items: int = 24) -> List[str]:
    out: List[str] = []
    for match in _CONTEXT_TERM_RE.finditer(str(text or "")):
        candidate = match.group(1).strip()
        for chunk in _SECTION_TERM_SPLIT_RE.split(candidate):
            chunk = chunk.strip()
            if len(chunk.split()) < 2:
                continue
            out.append(chunk)
            if len(out) >= max_items:
                break
        if len(out) >= max_items:
            break
    return out


def _sanitize_generated_rewrite(candidate: str) -> str:
    value = str(candidate or "").strip()
    if not value:
        return value

    value = value.strip('"').strip("'").strip()

    # Keep a single coherent paragraph.
    if "\n\n" in value:
        value = value.split("\n\n", 1)[0]

    # Remove array-like blocks and ICD-like codes that violate output rules.
    value = re.sub(r"\[[^\]]{0,240}\]", " ", value)
    value = _ICD_CODE_REGEX.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip(" .;:-,")
    return value


def _is_repetitive_rewrite(candidate: str) -> bool:
    normalized = _normalize_text(candidate)
    if not normalized:
        return False

    repeated_phrase = re.search(
        r"(\b[a-z0-9\-']+(?:\s+[a-z0-9\-']+){2,8})\s+\1\s+\1",
        normalized,
        flags=re.IGNORECASE,
    )
    if repeated_phrase:
        return True

    tokens = re.findall(r"[a-z0-9\-']+", normalized)
    if len(tokens) < 12:
        return False

    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / max(len(tokens), 1) > 0.24:
        return True

    return False


def _extract_abbreviation_terms(text: str, max_items: int = 24) -> List[str]:
    out: List[str] = []
    for match in _ABBREV_EXPANSION_RE.finditer(str(text or "")):
        expanded = match.group(1).strip()
        if 2 <= len(expanded.split()) <= 6:
            out.append(expanded)
            if len(out) >= max_items:
                break
    return out


def _extract_procedure_terms(text: str, max_items: int = 24) -> List[str]:
    out: List[str] = []
    for match in _PROCEDURE_TRIGGER_RE.finditer(str(text or "")):
        candidate = match.group(1).strip()
        if candidate:
            out.append(candidate)
            if len(out) >= max_items:
                return out

    for candidate in _extract_section_terms(text, _PROCEDURE_HEADER_HINTS, max_items=max_items):
        out.append(candidate)
        if len(out) >= max_items:
            break
    return out


def _extract_common_condition_terms(text: str, max_items: int = 24) -> List[str]:
    out: List[str] = []
    source = str(text or "")
    for term, pattern in _MEDICAL_DESCRIPTOR_PATTERNS:
        if pattern.search(source):
            out.append(term)
            if len(out) >= max_items:
                break
    return out


def _extract_priority_section_terms(text: str, max_items: int = 24) -> List[str]:
    out: List[str] = []

    for header, body in _parse_text_sections(text):
        normalized_header = _normalize_header(header)
        if not any(hint in normalized_header for hint in _STRUCTURED_PRIORITY_HEADERS):
            continue

        for chunk in _STRUCTURED_DESCRIPTOR_SPLIT_RE.split(body):
            candidate = chunk.strip()
            if not candidate:
                continue
            out.append(candidate)
            if len(out) >= max_items:
                return out

    source = str(text or "")
    for match in _STRUCTURED_PRIORITY_LINE_RE.finditer(source):
        candidate_line = str(match.group(2) or "").strip()
        if not candidate_line:
            continue
        for chunk in _STRUCTURED_DESCRIPTOR_SPLIT_RE.split(candidate_line):
            candidate = chunk.strip()
            if not candidate:
                continue
            out.append(candidate)
            if len(out) >= max_items:
                return out

    return out


def _extract_pattern_terms(text: str, max_items: int = 24) -> List[str]:
    out: List[str] = []
    source = str(text or "")
    for match in _STRUCTURED_CONTEXT_RE.finditer(source):
        candidate = str(match.group(1) or "").strip()
        if not candidate:
            continue

        for chunk in _STRUCTURED_DESCRIPTOR_SPLIT_RE.split(candidate):
            chunk = chunk.strip()
            if not chunk:
                continue
            out.append(chunk)
            if len(out) >= max_items:
                return out

    return out


def _clean_semantic_descriptor_phrase(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""

    value = re.sub(r"^[#>\-\*\d\.\s]+", "", value)
    value = _SEMANTIC_DESCRIPTOR_PREFIX_RE.sub("", value).strip()
    value = _STRUCTURED_TRAILING_FRAGMENT_RE.split(value, maxsplit=1)[0]

    while value:
        reduced = _STRUCTURED_LEADING_STOP_RE.sub("", value, count=1).strip()
        if reduced == value:
            break
        value = reduced

    value = re.sub(r"[^A-Za-z\-\'\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .;:-,")
    if not value:
        return ""

    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z\-']*", value)]
    while tokens and tokens[0] in _STRUCTURED_EDGE_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _STRUCTURED_EDGE_STOPWORDS:
        tokens = tokens[:-1]

    if not tokens:
        return ""
    if len(tokens) > 5:
        return ""

    phrase = " ".join(tokens)
    if len(phrase) < 3:
        return ""
    if _STRUCTURED_NON_NOUN_HINT_RE.search(phrase):
        return ""

    has_medical_term = any(term in phrase for term in _MEDICAL_DESCRIPTOR_TERM_SET)
    if not has_medical_term and not _DISEASE_SIGNAL_RE.search(phrase):
        return ""

    return phrase[:1].upper() + phrase[1:]


def extract_semantic_diagnosis_descriptors(text: str, max_items: int = 8) -> List[str]:
    max_items = max(int(max_items), 1)
    candidate_pool: List[str] = []

    # Step 1: pull diagnostic phrases from high-priority sections/headers.
    candidate_pool.extend(_extract_priority_section_terms(text, max_items=64))

    # Step 2: contextual noun phrases after medical context triggers.
    candidate_pool.extend(_extract_pattern_terms(text, max_items=64))

    # Step 3: dictionary terms found directly in the note text.
    candidate_pool.extend(_extract_common_condition_terms(text, max_items=64))

    descriptors: List[str] = []
    seen: set[str] = set()
    for raw_candidate in candidate_pool:
        cleaned = _normalize_descriptor_phrase(raw_candidate, category="diagnosis")
        if not cleaned:
            continue
        if not _is_clean_medical_term(cleaned, category="diagnosis"):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        descriptors.append(cleaned)
        if len(descriptors) >= max_items:
            break

    if len(descriptors) < 2:
        return []
    return descriptors


def _render_semantic_descriptors(descriptors: List[str], max_chars: int = 420) -> str:
    if not descriptors:
        return ""

    lines: List[str] = []
    chars_used = 0
    for descriptor in descriptors:
        line = f"- {descriptor}"
        if chars_used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        chars_used += len(line) + 1

    if not lines:
        return ""
    return "\n".join(lines)


def build_optimized_prompt(
    extracted_content: str,
    semantic_descriptors: Optional[List[str]] = None,
    max_content_chars: int = _STAGE4_MAX_CHARS,
) -> str:
    content = _truncate_text(str(extracted_content or "").strip(), max_content_chars)
    if not content:
        content = "No clinical details provided."

    descriptor_block = _render_semantic_descriptors(semantic_descriptors or [])

    extra_context = ""
    if descriptor_block:
        extra_context = (
            "--- POSSIBLE DIAGNOSTIC DESCRIPTORS (semantic cues, not codes) ---\n"
            f"{descriptor_block}\n\n"
        )

    return (
        "Extract all ICD-10-CM diagnosis codes from the clinical information below.\n"
        "Output ONLY a JSON array of code strings. No explanations.\n"
        "Include diagnoses, complications, and relevant co-morbidities.\n\n"
        f"{extra_context}"
        "--- CLINICAL INFORMATION ---\n"
        f"{content}\n"
        "--- END CLINICAL INFORMATION ---\n\n"
        "ICD-10-CM codes (JSON array):"
    )


def _build_guided_rewrite(semantic_descriptors: Optional[List[str]] = None) -> str:
    focus_terms = [term for term in (semantic_descriptors or []) if term][:5]
    focus_text = ", ".join(focus_terms) if focus_terms else "the key diagnoses and comorbidities"
    return (
        f"Extract ICD-10-CM diagnosis codes for this case with focus on {focus_text}. "
        "Prioritize principal diagnoses, active complications, and relevant comorbidities, "
        "then output only a JSON list of code strings."
    )


def _build_rule_based_prompt(clinical_note: str) -> Dict[str, Any]:
    stage1 = extract_high_signal_sections(clinical_note)
    stage2 = compress_clinical_text(stage1)
    stage3 = reorder_for_coding_priority(stage2)
    semantic_descriptors = extract_semantic_diagnosis_descriptors(stage3)
    stage4 = build_optimized_prompt(stage3, semantic_descriptors=semantic_descriptors)
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "semantic_descriptors": semantic_descriptors,
        "rule_prompt": stage4,
    }


def _semantic_guard_hit_count(rewritten: str, descriptors: List[str]) -> int:
    rewrite_tokens = set(re.findall(r"[a-z]{4,}", _normalize_text(rewritten)))
    descriptor_tokens: set[str] = set()

    for descriptor in descriptors:
        for token in re.findall(r"[a-z]{4,}", _normalize_text(descriptor)):
            if token in _SEMANTIC_STOPWORDS:
                continue
            descriptor_tokens.add(token)

    if not descriptor_tokens:
        return 0
    return len(rewrite_tokens & descriptor_tokens)


def _looks_like_generic_template(rewritten: str) -> bool:
    normalized = _normalize_text(rewritten)
    phrase_hits = sum(1 for phrase in _GENERIC_TEMPLATE_PHRASES if phrase in normalized)
    core_hits = 0
    template_specific_hits = 0
    if re.search(r"^(extract|identify)\s+(all\s+)?icd-?10(?:-cm)?\s+diagnos", normalized):
        core_hits += 1
    if "clinical information below" in normalized:
        template_specific_hits += 1
    if re.search(r"output\s+only\s+a\s+json\s+(?:array|list)\s+of\s+(?:code|diagnosis)\s+strings", normalized):
        core_hits += 1
    if re.search(r"include\s+diagnoses?,\s*complications?,?\s+and\s+relevant\s+co-?morbidit", normalized):
        template_specific_hits += 1

    if phrase_hits >= 2:
        return True

    if template_specific_hits >= 1 and core_hits >= 1:
        return True

    if template_specific_hits >= 1 and not any(kw in normalized for kw in _CLINICAL_FOCUS_KEYWORDS):
        return True

    if core_hits >= 2 and not re.search(r"\b(?:for|focused|focus|prioritiz|targeting|tailored|specific)\b", normalized):
        return True
    return False


def _has_clinical_focus_reference(
    rewritten: str,
    descriptors: List[str],
    rule_prompt: str,
) -> bool:
    candidate_norm = _normalize_text(rewritten)
    descriptor_norms = [_normalize_text(term) for term in descriptors if term]

    if any(term and term in candidate_norm for term in descriptor_norms):
        return True

    if any(kw in candidate_norm for kw in _CLINICAL_FOCUS_KEYWORDS):
        return True

    source_norm = _normalize_text(rule_prompt)
    source_focus_keywords = [kw for kw in _CLINICAL_FOCUS_KEYWORDS if kw in source_norm]
    if source_focus_keywords and any(kw in candidate_norm for kw in source_focus_keywords):
        return True

    descriptor_hit_count = _semantic_guard_hit_count(rewritten, descriptors)
    if descriptor_hit_count >= 2 and _FOCUS_CUE_RE.search(candidate_norm):
        return True

    return False


def _resolve_model_input(full_note: str, rule_prompt: str) -> Tuple[str, str]:
    mode = REWRITER_INPUT_MODE if REWRITER_INPUT_MODE in {"dynamic", "filtered", "raw"} else "dynamic"
    if mode != REWRITER_INPUT_MODE:
        log.warning("Invalid REWRITER_INPUT_MODE=%s; defaulting to dynamic", REWRITER_INPUT_MODE)

    if mode == "dynamic":
        chosen = "raw" if REWRITER_MODEL_VARIANT == "sft" else "filtered"
    else:
        chosen = mode

    if chosen == "raw":
        return _truncate_note(full_note), "raw"
    return rule_prompt, "filtered"


def _is_valid_rewrite(
    rewritten: str,
    rule_prompt: str,
    semantic_descriptors: Optional[List[str]] = None,
    model_input_source: str = "filtered",
) -> bool:
    candidate = (rewritten or "").strip()
    baseline = (rule_prompt or "").strip()

    if len(candidate) < _MIN_REWRITE_CHARS:
        log.info("rewrite_rejected reason=too_short chars=%d min=%d", len(candidate), _MIN_REWRITE_CHARS)
        return False
    if len(candidate) > _MAX_REWRITE_CHARS:
        log.info("rewrite_rejected reason=too_long chars=%d max=%d", len(candidate), _MAX_REWRITE_CHARS)
        return False
    if _normalize_text(candidate) == _normalize_text(baseline):
        log.info("rewrite_rejected reason=identical_to_rule_prompt")
        return False
    if _is_repetitive_rewrite(candidate):
        log.info("rewrite_rejected reason=repetitive_output")
        return False
    if _ICD_CODE_REGEX.search(candidate):
        log.info("rewrite_rejected reason=contains_icd_like_token")
        return False
    if _looks_like_generic_template(candidate):
        log.info("rewrite_rejected reason=generic_template")
        return False

    candidate_norm = _normalize_text(candidate)
    descriptor_terms = [_normalize_text(term) for term in list(semantic_descriptors or []) if term]
    if descriptor_terms:
        exact_descriptor_hit = any(term and term in candidate_norm for term in descriptor_terms)
        token_overlap = _semantic_guard_hit_count(candidate, descriptor_terms)
        if not exact_descriptor_hit and token_overlap < 2:
            log.info("rewrite_rejected reason=descriptor_reference_missing")
            return False

    keywords = ["extract", "icd", "code", "diagnos", "identify", "clinical", "json", "list", "output", "format"]
    hits = sum(1 for token in keywords if token in candidate.lower())
    if hits < 1:
        log.info("rewrite_rejected reason=keyword_hits hits=%d threshold=1", hits)
        return False

    if not _has_clinical_focus_reference(candidate, list(semantic_descriptors or []), rule_prompt):
        log.info("rewrite_rejected reason=clinical_focus_missing")
        return False

    if REWRITER_ENABLE_POSTFILTER_GUARD and model_input_source == "raw":
        guard_descriptors = list(semantic_descriptors or [])
        if guard_descriptors:
            hit_count = _semantic_guard_hit_count(candidate, guard_descriptors)
            if hit_count < max(REWRITER_GUARD_MIN_KEYWORD_HITS, 1):
                log.info(
                    "rewrite_rejected reason=semantic_guard hit_count=%d min_required=%d",
                    hit_count,
                    max(REWRITER_GUARD_MIN_KEYWORD_HITS, 1),
                )
                return False

    return True


def _load_best_prompt_cache() -> Dict[str, Dict[str, Any]]:
    global _best_prompt_cache

    if _best_prompt_cache is not None:
        return _best_prompt_cache

    cache_path = Path(BEST_PROMPT_CACHE_FILE)
    if cache_path.exists():
        try:
            _best_prompt_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _best_prompt_cache = {}
    else:
        _best_prompt_cache = {}
    return _best_prompt_cache


def _persist_best_prompt_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    cache_path = Path(BEST_PROMPT_CACHE_FILE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _build_prompt_inputs(
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    user_content = _render_rewriter_user_content(note_text)
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if prompt_ids.shape[1] > MAX_PROMPT_TOKENS:
            prompt_ids = prompt_ids[:, -MAX_PROMPT_TOKENS:]
        attention_mask = torch.ones_like(prompt_ids)
        return prompt_ids.to(device), attention_mask.to(device)

    prompt = f"{REWRITER_SYSTEM_PROMPT}\n\n{user_content}"
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def _build_full_sequence(
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    rewritten_prompt: str,
    device: torch.device,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    user_content = _render_rewriter_user_content(note_text)
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": rewritten_prompt}],
            add_generation_prompt=False,
            return_tensors="pt",
        )
        if prompt_ids.shape[1] > MAX_PROMPT_TOKENS:
            prompt_ids = prompt_ids[:, -MAX_PROMPT_TOKENS:]
        if full_ids.shape[1] > MAX_PROMPT_TOKENS:
            full_ids = full_ids[:, -MAX_PROMPT_TOKENS:]
        prompt_ids = prompt_ids.to(device)
        full_ids = full_ids.to(device)
        return full_ids, prompt_ids.shape[1], prompt_ids

    prompt = f"{REWRITER_SYSTEM_PROMPT}\n\n{user_content}"
    full_text = f"{prompt}{rewritten_prompt}"
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )["input_ids"].to(device)
    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )["input_ids"].to(device)
    return full_ids, prompt_ids.shape[1], prompt_ids


def _build_full_sequence_safe(
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    rewritten_prompt: str,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    try:
        return _build_full_sequence(tokenizer, note_text, rewritten_prompt, device)
    except Exception as exc:
        log.exception("full_sequence_build_failed | error=%s", exc)
        return None, 0, None


def _generate_rewrite(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
) -> Optional[Tuple[str, torch.Tensor, int, torch.Tensor]]:
    device = next(model.parameters()).device
    input_ids, attention_mask = _build_prompt_inputs(tokenizer, note_text, device)
    input_length = input_ids.shape[1]

    if DO_SAMPLE:
        log.warning(
            "Sampling requested via DO_SAMPLE=true but temporarily forcing greedy decode for stability"
        )

    token_budgets = [MAX_NEW_TOKENS, max(_MIN_RETRY_NEW_TOKENS, MAX_NEW_TOKENS // 2)]
    seen_budgets: set[int] = set()

    for attempt_idx, token_budget in enumerate(token_budgets, start=1):
        if token_budget in seen_budgets:
            continue
        seen_budgets.add(token_budget)

        try:
            with _inference_lock:
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=token_budget,
                        do_sample=False,
                        use_cache=False,
                        logits_processor=_SAFE_LOGITS_PROCESSOR,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            if generated_ids.shape[1] <= input_length:
                log.warning(
                    "generate_empty_output | attempt=%d token_budget=%d input_tokens=%d",
                    attempt_idx,
                    token_budget,
                    input_length,
                )
                continue

            rewritten = tokenizer.decode(
                generated_ids[0, input_length:],
                skip_special_tokens=True,
            ).strip()
            return rewritten, generated_ids, input_length, input_ids
        except Exception as exc:
            log.exception(
                "generate_attempt_failed | attempt=%d token_budget=%d temp=%.3f error=%s",
                attempt_idx,
                token_budget,
                TEMPERATURE,
                exc,
            )

    return None


def _generate_rewrite_with_base_adapter_disabled(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
) -> Optional[Tuple[str, torch.Tensor, int, torch.Tensor]]:
    if not hasattr(model, "disable_adapter"):
        return None

    try:
        with model.disable_adapter():
            return _generate_rewrite(model, tokenizer, note_text)
    except Exception as exc:
        log.exception("base_adapter_disabled_generate_failed | error=%s", exc)
        return None


def _compute_log_prob(
    model: PreTrainedModel,
    full_input_ids: torch.Tensor,
    input_length: int,
) -> float:
    outputs = model(full_input_ids)
    logits = outputs.logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    generated_token_ids = full_input_ids[:, input_length:]
    generated_positions = log_probs[:, input_length - 1 : input_length - 1 + generated_token_ids.shape[1], :]
    token_log_probs = generated_positions.gather(
        dim=-1,
        index=generated_token_ids.unsqueeze(-1),
    ).squeeze(-1)
    return float(token_log_probs.sum().item())


def _compute_value_estimate(
    model: PreTrainedModel,
    value_head: torch.nn.Module,
    input_ids: torch.Tensor,
) -> float:
    with _inference_lock:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
    value_head_dtype = next(value_head.parameters()).dtype
    state = outputs.hidden_states[-1][:, -1, :].to(value_head_dtype)
    value = value_head(state)
    return float(value.squeeze().item())


def _save_output(payload: Dict[str, Any]) -> Path:
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filepath = output_dir / f"{timestamp.replace(':', '-')}.json"

    record = dict(payload)
    record["timestamp"] = timestamp
    filepath.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return filepath


def _fallback_prompt(note_text: str) -> str:
    pipeline = _build_rule_based_prompt(note_text)
    return pipeline["rule_prompt"]


def run_inference(clinical_note: str, note_id: Optional[str] = None) -> Dict[str, Any]:
    full_note = str(clinical_note or "").strip()
    pipeline = _build_rule_based_prompt(full_note)
    rule_prompt = pipeline["rule_prompt"]
    semantic_descriptors = list(pipeline.get("semantic_descriptors", []))
    model_input_text, model_input_source = _resolve_model_input(full_note, rule_prompt)
    input_note_preview = _truncate_note(full_note)
    generation_source = "model"

    if _is_healthcheck_payload(clinical_note):
        rewritten_prompt = _fallback_prompt(full_note)
        generation_source = "healthcheck_fallback"
        result = {
            "note_id": note_id,
            "input_note": input_note_preview,
            "rule_prompt": rule_prompt,
            "model_input_source": "healthcheck",
            "semantic_descriptors": pipeline.get("semantic_descriptors", []),
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
        }
        _save_output(result)
        return {
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
        }

    try:
        model, tokenizer, value_head = load_model()
        device = next(model.parameters()).device
    except Exception as exc:
        log.exception("model_load_failed | using fallback prompt | error=%s", exc)
        guided = _build_guided_rewrite(semantic_descriptors)
        if _is_valid_rewrite(
            guided,
            rule_prompt,
            semantic_descriptors=semantic_descriptors,
            model_input_source=model_input_source,
        ):
            rewritten_prompt = guided
            generation_source = "guided_fallback_model_load_error"
        else:
            rewritten_prompt = rule_prompt
            generation_source = "rule_fallback_model_load_error"

        output = {
            "note_id": note_id,
            "input_note": input_note_preview,
            "model_variant": REWRITER_MODEL_VARIANT,
            "model_input_mode": REWRITER_INPUT_MODE,
            "model_input_source": model_input_source,
            "model_input": model_input_text,
            "stage1_excerpt": pipeline["stage1"],
            "stage2_excerpt": pipeline["stage2"],
            "stage3_excerpt": pipeline["stage3"],
            "semantic_descriptors": pipeline.get("semantic_descriptors", []),
            "rule_prompt": rule_prompt,
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
        }
        _save_output(output)
        return {
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
        }

    cached_rewrite: Optional[str] = None
    if note_id:
        cache = _load_best_prompt_cache()
        entry = cache.get(note_id)
        if entry and float(entry.get("reward", -1.0)) >= 0.0:
            cached_rewrite = str(entry.get("rewritten_prompt", "")).strip()

    rewritten_prompt = ""
    full_ids: Optional[torch.Tensor] = None
    input_length = 0
    prompt_ids: Optional[torch.Tensor] = None

    if cached_rewrite and _is_valid_rewrite(
        _sanitize_generated_rewrite(cached_rewrite),
        rule_prompt,
        semantic_descriptors=semantic_descriptors,
        model_input_source=model_input_source,
    ):
        rewritten_prompt = _sanitize_generated_rewrite(cached_rewrite)
        generation_source = "cache"
        full_ids, input_length, prompt_ids = _build_full_sequence_safe(
            tokenizer,
            model_input_text,
            rewritten_prompt,
            device,
        )

    if not rewritten_prompt:
        first_result = _generate_rewrite(model, tokenizer, model_input_text)
        if first_result is not None:
            first_raw, full_1, in_len_1, prompt_1 = first_result
            first = _sanitize_generated_rewrite(first_raw)
            if first != first_raw:
                full_1, in_len_1, prompt_1 = _build_full_sequence_safe(
                    tokenizer,
                    model_input_text,
                    first,
                    device,
                )
            if _is_valid_rewrite(
                first,
                rule_prompt,
                semantic_descriptors=semantic_descriptors,
                model_input_source=model_input_source,
            ):
                rewritten_prompt = first
                generation_source = "model_first"
                full_ids = full_1
                input_length = in_len_1
                prompt_ids = prompt_1

    if not rewritten_prompt:
        second_result = _generate_rewrite(model, tokenizer, model_input_text)
        if second_result is not None:
            second_raw, full_2, in_len_2, prompt_2 = second_result
            second = _sanitize_generated_rewrite(second_raw)
            if second != second_raw:
                full_2, in_len_2, prompt_2 = _build_full_sequence_safe(
                    tokenizer,
                    model_input_text,
                    second,
                    device,
                )
            if _is_valid_rewrite(
                second,
                rule_prompt,
                semantic_descriptors=semantic_descriptors,
                model_input_source=model_input_source,
            ):
                rewritten_prompt = second
                generation_source = "model_second"
                full_ids = full_2
                input_length = in_len_2
                prompt_ids = prompt_2

    if not rewritten_prompt:
        base_result = _generate_rewrite_with_base_adapter_disabled(
            model,
            tokenizer,
            model_input_text,
        )
        if base_result is not None:
            base_raw, full_b, in_len_b, prompt_b = base_result
            base = _sanitize_generated_rewrite(base_raw)
            if base != base_raw:
                full_b, in_len_b, prompt_b = _build_full_sequence_safe(
                    tokenizer,
                    model_input_text,
                    base,
                    device,
                )
            if _is_valid_rewrite(
                base,
                rule_prompt,
                semantic_descriptors=semantic_descriptors,
                model_input_source=model_input_source,
            ):
                rewritten_prompt = base
                generation_source = "model_base_adapter_disabled"
                full_ids = full_b
                input_length = in_len_b
                prompt_ids = prompt_b

    if not rewritten_prompt:
        guided = _build_guided_rewrite(semantic_descriptors)
        if _is_valid_rewrite(
            guided,
            rule_prompt,
            semantic_descriptors=semantic_descriptors,
            model_input_source=model_input_source,
        ):
            rewritten_prompt = guided
            generation_source = "guided_fallback"
            full_ids, input_length, prompt_ids = _build_full_sequence_safe(
                tokenizer,
                model_input_text,
                rewritten_prompt,
                device,
            )
        else:
            rewritten_prompt = rule_prompt
            generation_source = "rule_fallback"
            full_ids, input_length, prompt_ids = _build_full_sequence_safe(
                tokenizer,
                model_input_text,
                rewritten_prompt,
                device,
            )

    if not rewritten_prompt:
        rewritten_prompt = rule_prompt
        generation_source = "rule_fallback"
        full_ids, input_length, prompt_ids = _build_full_sequence_safe(
            tokenizer,
            model_input_text,
            rewritten_prompt,
            device,
        )

    log_prob_old = -1e-6
    if full_ids is not None:
        try:
            log_prob_old = _compute_log_prob(model, full_ids, input_length)
            if log_prob_old == 0.0:
                log_prob_old = -1e-6
        except Exception:
            log_prob_old = -1e-6

    value_estimate = 0.0
    if prompt_ids is not None:
        try:
            value_estimate = _compute_value_estimate(model, value_head, prompt_ids)
        except Exception:
            value_estimate = 0.0

    output = {
        "note_id": note_id,
        "input_note": input_note_preview,
        "model_variant": REWRITER_MODEL_VARIANT,
        "model_input_mode": REWRITER_INPUT_MODE,
        "model_input_source": model_input_source,
        "model_input": model_input_text,
        "stage1_excerpt": pipeline["stage1"],
        "stage2_excerpt": pipeline["stage2"],
        "stage3_excerpt": pipeline["stage3"],
        "semantic_descriptors": pipeline.get("semantic_descriptors", []),
        "rule_prompt": rule_prompt,
        "rewritten_prompt": rewritten_prompt,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
        "generation_source": generation_source,
    }
    _save_output(output)
    return {
        "rewritten_prompt": rewritten_prompt,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
        "generation_source": generation_source,
    }


def update_best_prompt_cache(note_id: str, rewritten: str, reward: float) -> None:
    if reward <= BEST_PROMPT_CACHE_THRESHOLD:
        return

    with _cache_lock:
        cache = _load_best_prompt_cache()
        existing = cache.get(note_id)
        if existing is not None and float(existing.get("reward", -1.0)) >= float(reward):
            return

        cache[note_id] = {
            "rewritten_prompt": rewritten,
            "reward": float(reward),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _persist_best_prompt_cache(cache)

# """Inference engine for the rewriter inference service.

# Responsible for:
# - Text generation
# - Log probability computation
# - Value estimation
# - Saving inference outputs to disk
# """

# from __future__ import annotations

# import json
# import time
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any, Dict

# import torch
# import torch.nn.functional as F
# from transformers import PreTrainedModel, PreTrainedTokenizerBase

# from config import MAX_NEW_TOKENS, OUTPUT_PATH, TEMPERATURE, DO_SAMPLE
# from logger import get_logger
# from model_loader import load_model

# log = get_logger(__name__)


# def _get_device(model: PreTrainedModel) -> torch.device:
#     """Return the device the model parameters live on."""
#     return next(model.parameters()).device


# def _compute_log_prob(
#     model: PreTrainedModel,
#     full_input_ids: torch.Tensor,
#     input_length: int,
# ) -> float:
#     """Compute the log probability of the generated tokens.

#     Implements log π_old(a|s):
#         1. Forward pass on the full sequence (input + generated).
#         2. log-softmax over the vocabulary dimension.
#         3. Gather the log-prob for each *generated* token.
#         4. Sum to obtain the scalar log_prob_old.

#     Args:
#         model: The loaded causal LM.
#         full_input_ids: Tensor of shape (1, seq_len) containing
#             input tokens concatenated with generated tokens.
#         input_length: Number of tokens that belong to the input prompt.

#     Returns:
#         Scalar log probability (float).
#     """
#     outputs = model(full_input_ids)
#     logits = outputs.logits  # (1, seq_len, vocab_size)

#     log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len, vocab_size)

#     # For each generated position t, the prediction comes from logits at t-1
#     # Generated tokens start at index `input_length`
#     generated_token_ids = full_input_ids[:, input_length:]  # (1, gen_len)
#     # Corresponding logit predictions are at positions [input_length-1 .. -2]
#     prediction_logits = log_probs[:, input_length - 1 : -1, :]  # (1, gen_len, V)

#     token_log_probs = prediction_logits.gather(
#         dim=-1, index=generated_token_ids.unsqueeze(-1)
#     ).squeeze(-1)  # (1, gen_len)

#     total_log_prob: float = token_log_probs.sum().item()
#     return total_log_prob


# def _compute_value_estimate(
#     model: PreTrainedModel,
#     input_ids: torch.Tensor,
# ) -> float:
#     """Extract V(s) from the value head attached to the actor model.

#     The value head is expected to be available as ``model.value_head``
#     or via a ``score`` / ``v_head`` attribute depending on how the
#     checkpoint was saved during SFT/RL training.

#     Falls back to using the mean of the last hidden state projected
#     through any available value head layer.

#     Args:
#         model: The loaded model with a value head.
#         input_ids: Tokenised input (1, seq_len).

#     Returns:
#         Scalar value estimate (float).
#     """
#     outputs = model(input_ids, output_hidden_states=True)
#     last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

#     # Try known value-head attribute names
#     for attr in ("value_head", "v_head", "score"):
#         head = getattr(model, attr, None)
#         if head is not None:
#             value = head(last_hidden[:, -1, :])  # (1, 1) or (1,)
#             return value.squeeze().item()

#     # Fallback: use the mean-pooled hidden state norm as a proxy
#     log.warning("No explicit value head found; using mean hidden-state norm as proxy.")
#     value_proxy = last_hidden[:, -1, :].mean().item()
#     return value_proxy


# def _save_output(result: Dict[str, Any]) -> Path:
#     """Persist inference output to disk as a timestamped JSON file.

#     Args:
#         result: Dictionary containing inference results.

#     Returns:
#         Path to the saved file.
#     """
#     output_dir = Path(OUTPUT_PATH)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
#     # Use a filename-safe version of the timestamp
#     filename = timestamp.replace(":", "-") + ".json"
#     filepath = output_dir / filename

#     payload = {
#         "timestamp": timestamp,
#         "input_note": result["input_note"],
#         "rewritten_prompt": result["rewritten_prompt"],
#         "log_prob_old": result["log_prob_old"],
#         "value_estimate": result["value_estimate"],
#     }

#     filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
#     log.info("output_saved | path=%s", filepath)
#     return filepath


# def run_inference(clinical_note: str) -> Dict[str, Any]:
#     """Run the full inference pipeline for a given clinical note.

#     Steps:
#         1. Load model & tokenizer (cached).
#         2. Tokenize the clinical note.
#         3. Generate rewritten prompt via model.generate().
#         4. Compute log_prob_old (log π_old(a|s)).
#         5. Compute value_estimate (V(s)).
#         6. Save output to disk.
#         7. Return results.

#     All inference runs inside ``torch.no_grad()``.

#     Args:
#         clinical_note: Raw clinical note text.

#     Returns:
#         Dictionary with keys: rewritten_prompt, log_prob_old, value_estimate.
#     """
#     t_start = time.perf_counter()

#     model, tokenizer = load_model()
#     device = _get_device(model)

#     with torch.no_grad():
#         # Step 2: Tokenize input
#         inputs = tokenizer(clinical_note, return_tensors="pt").to(device)
#         input_ids = inputs["input_ids"]
#         input_length = input_ids.shape[1]

#         # Step 3: Generate rewritten prompt
#         gen_output = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=TEMPERATURE,
#             do_sample=DO_SAMPLE,
#         )
#         generated_ids = gen_output  # (1, input_len + gen_len)

#         # Decode only the newly generated tokens
#         new_token_ids = generated_ids[:, input_length:]
#         rewritten_prompt: str = tokenizer.decode(
#             new_token_ids[0], skip_special_tokens=True
#         )

#         # Step 4-6: Compute log_prob_old
#         log_prob_old: float = _compute_log_prob(model, generated_ids, input_length)

#         # Step 7: Compute value_estimate
#         value_estimate: float = _compute_value_estimate(model, input_ids)

#     generation_time = time.perf_counter() - t_start
#     log.info("generation_time | seconds=%.4f", generation_time)

#     result: Dict[str, Any] = {
#         "input_note": clinical_note,
#         "rewritten_prompt": rewritten_prompt,
#         "log_prob_old": log_prob_old,
#         "value_estimate": value_estimate,
#     }

#     # Step 8: Save to disk
#     _save_output(result)

#     return {
#         "rewritten_prompt": rewritten_prompt,
#         "log_prob_old": log_prob_old,
#         "value_estimate": value_estimate,
#     }
