"""CopilotUsageEvent — minimal, low-friction usage telemetry for the real-user pilot.

Every interaction in PAI-OS generates one CopilotUsageEvent.
It records what happened, how, and with which providers/models —
without storing full prompts, responses, or credentials.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskType(Enum):
    LEARNING = "learning"
    RESEARCH = "research"
    KNOWLEDGE = "knowledge"
    PLANNING = "planning"
    GENERAL = "general"
    DAILY_REVIEW = "daily_review"
    ROUTER_DECISION = "router_decision"
    MEMORY_OPERATION = "memory_operation"
    RECOMMENDATION = "recommendation"


class UsageMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class RetrievalMode(Enum):
    NONE = "none"
    VECTOR = "vector"
    HYBRID = "hybrid"
    FULL_TEXT = "full_text"
    MEMORY = "memory"
    CONTEXT = "context"


class DataSource(Enum):
    """Tag for REAL_USER / DEMO / EVAL separation."""
    REAL_USER = "REAL_USER"
    DEMO = "DEMO"
    EVAL = "EVAL"


@dataclass
class CopilotUsageEvent:
    """Immutable usage event for one PAI-OS interaction.

    Does NOT store:
    - full API keys
    - authorization headers
    - unnecessary full prompts
    - unnecessary full responses
    """

    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    conversation_id: str = ""
    task_type: str = TaskType.GENERAL.value
    mode: str = UsageMode.LOCAL.value
    selected_provider: str = ""
    selected_model: str = ""
    retrieval_mode: str = RetrievalMode.NONE.value
    context_types: list[str] = field(default_factory=list)
    response_id: str = ""
    data_source: str = DataSource.REAL_USER.value

    # Context quality (set after feedback)
    personal_context_used: bool = False
    relevant_context_ratio: float = 0.0
    irrelevant_context_ratio: float = 0.0

    # Actual context items used (kept as metadata, never full content)
    context_ids: list[str] = field(default_factory=list)
    retrieval_score: float | None = None
    latency_ms: float | None = None
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_type"] = self.task_type
        d["mode"] = self.mode
        d["retrieval_mode"] = self.retrieval_mode
        d["data_source"] = self.data_source
        return d


class UsageEventStore:
    """Append-only store for CopilotUsageEvent records."""

    def __init__(self) -> None:
        self._events: list[CopilotUsageEvent] = []

    def record(self, event: CopilotUsageEvent) -> None:
        self._events.append(event)

    def list_all(self) -> list[CopilotUsageEvent]:
        return list(self._events)

    def find_by_conversation(
        self, conversation_id: str
    ) -> list[CopilotUsageEvent]:
        return [
            e for e in self._events
            if e.conversation_id == conversation_id
        ]

    def find_by_task_type(self, task_type: str) -> list[CopilotUsageEvent]:
        return [
            e for e in self._events if e.task_type == task_type
        ]

    def find_by_data_source(self, source: str) -> list[CopilotUsageEvent]:
        return [
            e for e in self._events if e.data_source == source
        ]

    def count(self) -> int:
        return len(self._events)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self._events],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> UsageEventStore:
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            event = CopilotUsageEvent(
                id=item.get("id", ""),
                timestamp=item.get("timestamp", ""),
                conversation_id=item.get("conversation_id", ""),
                task_type=item.get("task_type", TaskType.GENERAL.value),
                mode=item.get("mode", UsageMode.LOCAL.value),
                selected_provider=item.get("selected_provider", ""),
                selected_model=item.get("selected_model", ""),
                retrieval_mode=item.get(
                    "retrieval_mode", RetrievalMode.NONE.value
                ),
                context_types=item.get("context_types", []),
                response_id=item.get("response_id", ""),
                data_source=item.get(
                    "data_source", DataSource.REAL_USER.value
                ),
                personal_context_used=item.get(
                    "personal_context_used", False
                ),
                relevant_context_ratio=item.get(
                    "relevant_context_ratio", 0.0
                ),
                irrelevant_context_ratio=item.get(
                    "irrelevant_context_ratio", 0.0
                ),
                context_ids=item.get("context_ids", []),
                retrieval_score=item.get("retrieval_score"),
                latency_ms=item.get("latency_ms"),
                token_count=item.get("token_count"),
            )
            store._events.append(event)
        return store

    def clear(self) -> None:
        """Clear all events (test helper)."""
        self._events.clear()
