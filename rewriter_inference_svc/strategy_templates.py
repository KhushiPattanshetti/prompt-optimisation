"""Predefined instruction templates for policy-based ICD prompting."""

TEMPLATES = {
    "balanced": (
        "You are an expert ICD-10-CM coding instruction writer. Read the full clinical note and craft one concise "
        "extraction instruction that captures all possible supported diagnosis codes, while explicitly prioritizing "
        "the principal diagnosis first, then active complications and relevant comorbidities. Keep wording specific "
        "to this note and avoid generic boilerplate."
    ),
    "high_recall": (
        "Write one note-specific ICD extraction instruction optimized for comprehensive capture of all possible "
        "supported diagnosis codes. Prioritize principal diagnoses first, then include secondary active conditions, "
        "chronic comorbidity burden, and documented status/history conditions when clearly supported in the note."
    ),
    "high_precision": (
        "Write one note-specific ICD extraction instruction optimized for precision while still aiming to capture all "
        "supported diagnosis codes. Prioritize principal diagnosis extraction first, include only diagnoses with clear "
        "textual support and coding relevance, and avoid speculative additions or redundant overlaps."
    ),
    "neuro_focus": (
        "Write one note-specific ICD extraction instruction with neurologic emphasis when supported. Capture all "
        "possible supported diagnoses, prioritizing principal neurologic diagnoses first, then cognitive and movement "
        "findings, fall/gait-related morbidity, and relevant systemic comorbidities."
    ),
    "evidence_first": (
        "Write one note-specific ICD extraction instruction that requires explicit evidence anchoring. Ask for all "
        "possible supported diagnosis codes with the principal diagnosis prioritized first, and require each returned "
        "diagnosis to be supported by direct note wording without inferred conditions lacking textual evidence."
    ),
    "differential_expansion": (
        "Write one note-specific ICD extraction instruction that captures all possible supported diagnosis codes, "
        "including diagnostically similar but distinct conditions where the note differentiates them. Prioritize the "
        "principal diagnosis first and preserve ambiguity only when documentation remains inconclusive."
    ),
    "comorbidity_focus": (
        "Write one note-specific ICD extraction instruction that captures all possible supported diagnosis codes while "
        "prioritizing principal diagnoses first, and emphasizing clinically meaningful comorbidity capture including "
        "chronic disease interactions, acute-on-chronic decompensation, and documented complication burden."
    ),
}

VALID_STRATEGIES = frozenset(TEMPLATES.keys())
