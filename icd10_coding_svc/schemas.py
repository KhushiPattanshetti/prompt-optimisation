from typing import List

from pydantic import BaseModel


class CodeRequest(BaseModel):
    note_id: str
    original_prompt: str
    rewritten_prompt: str


class CodeResponse(BaseModel):
    note_id: str
    enh_codes: List[str]
    org_codes: List[str]
    gt_codes: List[str]
    enh_raw_output: str
    org_raw_output: str
    parsing_success: bool
