import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from code_parser import parse_icd10_codes, validate_code


class TestValidateCode:
    def test_validate_code_valid(self):
        assert validate_code("I21.9") is True

    def test_validate_code_valid_no_decimal(self):
        assert validate_code("I21") is True

    def test_validate_code_invalid(self):
        assert validate_code("not-a-code") is False

    def test_validate_code_invalid_lowercase(self):
        assert validate_code("i21.9") is False

    def test_validate_code_invalid_empty(self):
        assert validate_code("") is False


class TestParseICD10Codes:
    def test_parse_json_success(self):
        raw = '["R07.9", "I20.9", "R07.9"]'
        result = parse_icd10_codes(raw)
        assert result == ["R07.9", "I20.9"]

    def test_parse_regex_fallback(self):
        raw = "The patient has R07.9 and also I20.9."
        result = parse_icd10_codes(raw)
        assert result == ["R07.9", "I20.9"]

    def test_parse_empty_fallback(self):
        raw = "No codes found in this output."
        result = parse_icd10_codes(raw)
        assert result == []

    def test_parse_deduplication(self):
        raw = '["E11.9", "Z79.4", "E11.9"]'
        result = parse_icd10_codes(raw)
        assert result == ["E11.9", "Z79.4"]

    def test_parse_json_embedded_in_text(self):
        raw = 'Here are the codes: ["R07.9", "I20.9"] extracted from the note.'
        result = parse_icd10_codes(raw)
        assert result == ["R07.9", "I20.9"]

    def test_parse_single_code_regex(self):
        raw = "Primary diagnosis is E11.9"
        result = parse_icd10_codes(raw)
        assert result == ["E11.9"]

    def test_parse_codes_without_decimal(self):
        raw = '["I21", "E11"]'
        result = parse_icd10_codes(raw)
        assert result == ["I21", "E11"]
