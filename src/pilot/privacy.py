"""Privacy module — ensures pilot telemetry never leaks sensitive data.

Rules:
- Never exposes credentials
- Never sends LOCAL_ONLY content to telemetry cloud
- Strips API keys from event data
- Strips authorization headers
- Limits prompt/response storage to IDs and metadata
"""
from __future__ import annotations

import re
from typing import Any


SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),       # OpenAI-style keys
    re.compile(r"api[_-]?key[=:]\s*['\"]?[a-zA-Z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"authorization:\s*(bearer|token)\s+[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"x-api-key:\s*[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"[a-fA-F0-9]{32,}"),  # Long hex strings (potential tokens/keys)
]

LOCAL_ONLY_CONTENT_MARKERS: list[str] = [
    "LOCAL_ONLY",
    "local-only",
    "LOCAL ONLY",
]


def strip_sensitive_data(text: str) -> str:
    """Remove or redact sensitive patterns from a text string."""
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def strip_sensitive_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive values from a dictionary."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        # Skip known sensitive fields
        if any(
            marker in key_lower
            for marker in ["api_key", "apikey", "api-key", "secret", "token",
                           "authorization", "auth_header", "password", "credential"]
        ):
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            result[key] = strip_sensitive_data(value)
        elif isinstance(value, dict):
            result[key] = strip_sensitive_dict(value)
        elif isinstance(value, list):
            result[key] = [
                strip_sensitive_data(item) if isinstance(item, str)
                else strip_sensitive_dict(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def contains_local_only_content(data: dict[str, Any]) -> bool:
    """Check if any value contains LOCAL_ONLY markers."""
    def _check(value: Any) -> bool:
        if isinstance(value, str):
            return any(marker in value for marker in LOCAL_ONLY_CONTENT_MARKERS)
        if isinstance(value, dict):
            return any(_check(v) for v in value.values())
        if isinstance(value, list):
            return any(_check(item) for item in value)
        return False
    return _check(data)


class PrivacySanitizer:
    """Sanitizes pilot data before export or storage."""

    def sanitize_event(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize a usage event dict for safe storage/export."""
        # Remove entire fields that should never be stored
        data.pop("full_prompt", None)
        data.pop("full_response", None)
        data.pop("raw_api_response", None)
        data.pop("headers", None)

        # Strip sensitive data from text fields
        for key in ("user_input", "query", "free_text", "user_note",
                     "description", "original_text", "context"):
            if key in data and isinstance(data[key], str):
                data[key] = strip_sensitive_data(data[key])

        return data

    def sanitize_feedback(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize feedback dict for safe storage/export."""
        if "free_text" in data and isinstance(data["free_text"], str):
            data["free_text"] = strip_sensitive_data(data["free_text"])
        return data

    def is_safe_to_export(self, data: dict[str, Any]) -> bool:
        """Check if data is safe for telemetry export."""
        return not contains_local_only_content(data)
