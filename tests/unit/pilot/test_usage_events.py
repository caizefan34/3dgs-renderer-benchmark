"""Unit tests for pilot module: usage events."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pilot.usage_event import (
    CopilotUsageEvent,
    UsageEventStore,
    TaskType,
    UsageMode,
    RetrievalMode,
    DataSource,
)


class TestCopilotUsageEvent(unittest.TestCase):
    def test_create_event_minimal(self):
        event = CopilotUsageEvent()
        self.assertTrue(event.id.startswith("evt_"))
        self.assertIsNotNone(event.timestamp)
        self.assertEqual(event.task_type, TaskType.GENERAL.value)
        self.assertEqual(event.mode, UsageMode.LOCAL.value)
        self.assertEqual(event.retrieval_mode, RetrievalMode.NONE.value)
        self.assertEqual(event.data_source, DataSource.REAL_USER.value)

    def test_create_event_full(self):
        event = CopilotUsageEvent(
            conversation_id="conv_001",
            task_type=TaskType.LEARNING.value,
            mode=UsageMode.CLOUD.value,
            selected_provider="openai",
            selected_model="gpt-4",
            retrieval_mode=RetrievalMode.VECTOR.value,
            context_types=["personal", "research"],
            response_id="resp_001",
            personal_context_used=True,
            relevant_context_ratio=0.8,
            irrelevant_context_ratio=0.2,
            latency_ms=1200.5,
            token_count=450,
        )
        self.assertEqual(event.conversation_id, "conv_001")
        self.assertEqual(event.task_type, TaskType.LEARNING.value)
        self.assertEqual(event.mode, UsageMode.CLOUD.value)
        self.assertEqual(event.selected_provider, "openai")
        self.assertEqual(event.selected_model, "gpt-4")
        self.assertEqual(event.retrieval_mode, RetrievalMode.VECTOR.value)
        self.assertEqual(event.context_types, ["personal", "research"])
        self.assertTrue(event.personal_context_used)
        self.assertEqual(event.relevant_context_ratio, 0.8)
        self.assertEqual(event.latency_ms, 1200.5)
        self.assertEqual(event.token_count, 450)

    def test_to_dict(self):
        event = CopilotUsageEvent(
            conversation_id="conv_001",
            task_type=TaskType.RESEARCH.value,
            mode=UsageMode.HYBRID.value,
        )
        d = event.to_dict()
        self.assertEqual(d["conversation_id"], "conv_001")
        self.assertEqual(d["task_type"], TaskType.RESEARCH.value)
        self.assertEqual(d["mode"], UsageMode.HYBRID.value)
        self.assertEqual(d["data_source"], DataSource.REAL_USER.value)

    def test_no_sensitive_fields_in_data(self):
        """Ensure sensitive fields are never part of the event."""
        event = CopilotUsageEvent()
        d = event.to_dict()
        self.assertNotIn("full_prompt", d)
        self.assertNotIn("full_response", d)
        self.assertNotIn("api_key", d)
        self.assertNotIn("authorization", d)
        self.assertNotIn("headers", d)


class TestUsageEventStore(unittest.TestCase):
    def setUp(self):
        self.store = UsageEventStore()

    def test_record_and_count(self):
        self.store.record(CopilotUsageEvent())
        self.store.record(CopilotUsageEvent())
        self.assertEqual(self.store.count(), 2)

    def test_find_by_conversation(self):
        e1 = CopilotUsageEvent(conversation_id="conv_001")
        e2 = CopilotUsageEvent(conversation_id="conv_002")
        e3 = CopilotUsageEvent(conversation_id="conv_001")
        self.store.record(e1)
        self.store.record(e2)
        self.store.record(e3)
        results = self.store.find_by_conversation("conv_001")
        self.assertEqual(len(results), 2)

    def test_find_by_task_type(self):
        e1 = CopilotUsageEvent(task_type=TaskType.LEARNING.value)
        e2 = CopilotUsageEvent(task_type=TaskType.RESEARCH.value)
        self.store.record(e1)
        self.store.record(e2)
        results = self.store.find_by_task_type(TaskType.LEARNING.value)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].task_type, TaskType.LEARNING.value)

    def test_find_by_data_source(self):
        e1 = CopilotUsageEvent(data_source=DataSource.REAL_USER.value)
        e2 = CopilotUsageEvent(data_source=DataSource.DEMO.value)
        self.store.record(e1)
        self.store.record(e2)
        results = self.store.find_by_data_source(DataSource.REAL_USER.value)
        self.assertEqual(len(results), 1)

    def test_save_and_load_json(self):
        self.store.record(CopilotUsageEvent(conversation_id="conv_001"))
        self.store.record(CopilotUsageEvent(conversation_id="conv_002"))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
            self.store.save_json(path)

        loaded = UsageEventStore.load_json(path)
        self.assertEqual(loaded.count(), 2)
        os.unlink(path)

    def test_clear(self):
        self.store.record(CopilotUsageEvent())
        self.store.clear()
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
