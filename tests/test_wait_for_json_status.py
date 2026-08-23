import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.wait_for_json_status import wait_for_status


class WaitForJsonStatusTest(unittest.TestCase):
    def test_waits_through_missing_partial_and_malformed_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.json"
            states = [None, "{", json.dumps({"status": "partial"}), json.dumps({"status": "complete"})]

            def advance(_):
                state = states.pop(0)
                if state is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(state, encoding="utf-8")

            with mock.patch("scripts.wait_for_json_status.time.sleep", side_effect=advance):
                advance(0)
                result = wait_for_status(path, "complete", 0.01)

        self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
