"""Unit tests for pilot module: privacy."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pilot.privacy import (
    PrivacySanitizer,
    strip_sensitive_data,
    strip_sensitive_dict,
    contains_local_only_content,
)


class TestStripSensitiveData(unittest.TestCase):
    def test_strip_openai_key(self):
        text = "My key is sk-abc123def456ghi789jklmno"
        result = strip_sensitive_data(text)
        self.assertNotIn("sk-abc123def456ghi789jklmno", result)
        self.assertIn("[REDACTED]", result)

    def test_strip_api_key_pattern(self):
        text = 'api_key="abcdef1234567890abcdef12"'
        result = strip_sensitive_data(text)
        self.assertIn("[REDACTED]", result)

    def test_strip_authorization_header(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = strip_sensitive_data(text)
        self.assertIn("[REDACTED]", result)

    def test_keep_normal_text(self):
        text = "Hello, this is a normal question about 3DGS rendering."
        result = strip_sensitive_data(text)
        self.assertEqual(result, text)


class TestStripSensitiveDict(unittest.TestCase):
    def test_strip_api_key_field(self):
        data = {"api_key": "sk-test1234567890abcdef", "query": "hello"}
        result = strip_sensitive_dict(data)
        self.assertEqual(result["api_key"], "[REDACTED]")
        self.assertEqual(result["query"], "hello")

    def test_redact_mixed_case_fields(self):
        data = {"API_KEY": "secret123", "ApiKey": "secret456"}
        result = strip_sensitive_dict(data)
        self.assertEqual(result["API_KEY"], "[REDACTED]")
        self.assertEqual(result["ApiKey"], "[REDACTED]")

    def test_nested_dict_redaction(self):
        data = {
            "config": {
                "secret": "my-secret-value",
                "token": "abc123",
            }
        }
        result = strip_sensitive_dict(data)
        self.assertEqual(result["config"]["secret"], "[REDACTED]")
        self.assertEqual(result["config"]["token"], "[REDACTED]")

    def test_normal_dict_unchanged(self):
        data = {"name": "test", "scores": [1, 2, 3]}
        result = strip_sensitive_dict(data)
        self.assertEqual(result, data)


class TestContainsLocalOnly(unittest.TestCase):
    def test_detect_local_only(self):
        data = {"content": "This is LOCAL_ONLY data"}
        self.assertTrue(contains_local_only_content(data))

    def test_no_local_only(self):
        data = {"content": "This is normal data"}
        self.assertFalse(contains_local_only_content(data))

    def test_nested_local_only(self):
        data = {"outer": {"inner": "contains LOCAL_ONLY marker"}}
        self.assertTrue(contains_local_only_content(data))


class TestPrivacySanitizer(unittest.TestCase):
    def setUp(self):
        self.sanitizer = PrivacySanitizer()

    def test_sanitize_event_removes_sensitive_fields(self):
        event_data = {
            "id": "evt_001",
            "full_prompt": "What is 3DGS?",
            "full_response": "3DGS is...",
            "api_key": "sk-test123",
            "conversation_id": "conv_001",
        }
        result = self.sanitizer.sanitize_event(event_data)
        self.assertNotIn("full_prompt", result)
        self.assertNotIn("full_response", result)
        self.assertIn("conversation_id", result)

    def test_is_safe_to_export(self):
        safe = {"data": "normal"}
        unsafe = {"data": "LOCAL_ONLY content"}
        self.assertTrue(self.sanitizer.is_safe_to_export(safe))
        self.assertFalse(self.sanitizer.is_safe_to_export(unsafe))


if __name__ == "__main__":
    unittest.main()
