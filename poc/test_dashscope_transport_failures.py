import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.application.vision_usage import VisionSessionUsageLedger
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider


class DashScopeTransportFailureTests(unittest.TestCase):
    @staticmethod
    def response(payload):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_empty_final_content_is_failed_and_preserves_reasoning_diagnostic(self):
        provider = DashScopeVisionProvider(api_key="test-key", max_attempts=1)
        ledger = VisionSessionUsageLedger(session_id="empty-content")
        payload = {
            "id": "request-empty",
            "model": "qwen3.7-plus",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"finish_reason": "stop", "message": {
                "reasoning_content": "分析完成，但最终字段为空", "content": ""
            }}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "abc_model_request.json"
            with patch("agent.infrastructure.dashscope_vision_provider.httpx.post",
                       return_value=self.response(payload)):
                with self.assertRaisesRegex(VisionAgentError, "reasoning_content"):
                    with provider.session_usage_scope(ledger):
                        with provider.call_scope(stage="single_step_observation",
                                                fingerprint="frame",
                                                request_evidence_path=request_path):
                            provider._chat([{"role": "user", "content": "observe"}], max_tokens=None)
            response_path = Path(tmp) / "abc_model_response.json"
            self.assertTrue(response_path.is_file())
            saved = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual("empty_message_content", saved["reason"])
            self.assertEqual("分析完成，但最终字段为空", saved["raw_response"]["choices"][0]["message"]["reasoning_content"])
        event = ledger.to_dict()["events"][0]
        self.assertEqual("failed", event["outcome"])
        self.assertEqual(0, ledger.to_dict()["totals"]["successful_requests"])
        self.assertEqual(30, provider.status()["usage_totals"]["total_tokens"])

    def test_missing_content_is_failed_without_promoting_reasoning(self):
        provider = DashScopeVisionProvider(api_key="test-key", max_attempts=1)
        payload = {"id": "request-missing", "model": "qwen3.7-plus",
                   "choices": [{"finish_reason": "stop", "message": {
                       "reasoning_content": "仅有推理"
                   }}]}
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "abc_model_request.json"
            with patch("agent.infrastructure.dashscope_vision_provider.httpx.post",
                       return_value=self.response(payload)):
                with self.assertRaisesRegex(VisionAgentError, "缺少 message.content"):
                    with provider.call_scope(stage="single_step_observation",
                                            request_evidence_path=request_path):
                        provider._chat([{"role": "user", "content": "observe"}], max_tokens=None)
            saved = json.loads((Path(tmp) / "abc_model_response.json").read_text(encoding="utf-8"))
            self.assertEqual("missing_message_content", saved["reason"])


if __name__ == "__main__":
    unittest.main()
