from typing import List, Optional

from pydantic import BaseModel


class CodeRequest(BaseModel):
    note_id: str
    run_id: Optional[str] = None
    group_id: Optional[str] = None
    original_prompt: str
    rewritten_prompt: str
    generation_source: Optional[str] = None
    log_prob_old: Optional[float] = None
    value_estimate: Optional[float] = None


class CodeResponse(BaseModel):
    note_id: str
    enh_codes: List[str]
    org_codes: List[str]
    gt_codes: List[str]
    enh_raw_output: str
    org_raw_output: str
    parsing_success: bool
