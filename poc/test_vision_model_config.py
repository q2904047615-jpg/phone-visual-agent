import unittest
from unittest.mock import patch

import httpx

from vision_agent import DashScopeVisionProvider
from vision_model_config import (
    DEFAULT_VISION_BASE_URL,
    DEFAULT_VISION_MODEL,
    VISION_MODEL_CONFIG_VERSION,
    VisionModelConfig,
    load_vision_model_config,
    public_model_identity,
)


class VisionModelConfigTests(unittest.TestCase):
    def test_default_is_qwen37_plus_non_thinking(self) -> None:
        config = load_vision_model_config(environ={})
        self.assertEqual(DEFAULT_VISION_MODEL, "qwen3.7-plus")
        self.assertEqual(config.model, "qwen3.7-plus")
        self.assertEqual(config.base_url, DEFAULT_VISION_BASE_URL)
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.coordinate_scale, 1000)

    def test_generic_environment_names_override_legacy_names(self) -> None:
        config = load_vision_model_config(
            environ={
                "VISION_MODEL": "qwen3.7-plus",
                "QWEN_VL_MODEL": "legacy-model",
                "VISION_MODEL_BASE_URL": "https://new.example/v1/",
                "DASHSCOPE_BASE_URL": "https://legacy.example/v1",
            }
        )
        self.assertEqual(config.model, "qwen3.7-plus")
        self.assertEqual(config.base_url, "https://new.example/v1")

    def test_legacy_environment_names_remain_compatible(self) -> None:
        config = load_vision_model_config(
            environ={
                "QWEN_VL_MODEL": "qwen3-vl-plus",
                "DASHSCOPE_BASE_URL": "https://legacy.example/v1/",
            }
        )
        self.assertEqual(config.model, "qwen3-vl-plus")
        self.assertEqual(config.base_url, "https://legacy.example/v1")

    def test_invalid_model_or_remote_http_url_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "模型 ID"):
            VisionModelConfig(model="bad model", base_url=DEFAULT_VISION_BASE_URL)
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            VisionModelConfig(model="safe-model", base_url="http://remote.example/v1")

    def test_public_identity_excludes_endpoint_and_usage(self) -> None:
        identity = public_model_identity(
            {
                "model_config_version": VISION_MODEL_CONFIG_VERSION,
                "provider": "aliyun_model_studio",
                "model": "qwen3.7-plus",
                "thinking_enabled": False,
                "coordinate_scale": 1000,
                "response_model": "qwen3.7-plus-2026-05-26",
                "base_url": "https://secret-workspace.example/v1",
                "last_usage": {"total_tokens": 10},
            }
        )
        self.assertNotIn("base_url", identity)
        self.assertNotIn("last_usage", identity)
        self.assertEqual(identity["model"], "qwen3.7-plus")


class DashScopeVisionModelRequestTests(unittest.TestCase):
    @staticmethod
    def _response() -> httpx.Response:
        request = httpx.Request(
            "POST",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "request-37",
                "model": "qwen3.7-plus-2026-05-26",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok":true}'},
                    }
                ],
                "usage": {"total_tokens": 12},
            },
        )

    def test_default_request_uses_replaceable_model_and_disables_thinking(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VISION_MODEL": "qwen3.7-plus",
                "DASHSCOPE_API_KEY": "test-key",
            },
            clear=True,
        ):
            provider = DashScopeVisionProvider(max_attempts=1)
        with patch("vision_agent.httpx.post", return_value=self._response()) as mocked:
            self.assertEqual(
                provider._chat([{"role": "user", "content": "json"}], max_tokens=1200),
                '{"ok":true}',
            )
        body = mocked.call_args.kwargs["json"]
        self.assertEqual(body["model"], "qwen3.7-plus")
        self.assertIs(body["enable_thinking"], False)
        self.assertEqual(body["max_tokens"], 1200)
        status = provider.status()
        self.assertEqual(status["model_config_version"], VISION_MODEL_CONFIG_VERSION)
        self.assertFalse(status["thinking_enabled"])
        self.assertEqual(status["response_model"], "qwen3.7-plus-2026-05-26")
        self.assertEqual(status["last_finish_reason"], "stop")

    def test_explicit_model_config_is_atomic(self) -> None:
        config = VisionModelConfig(
            model="qwen3.7-plus-2026-05-26",
            base_url=DEFAULT_VISION_BASE_URL,
        )
        provider = DashScopeVisionProvider(api_key="test-key", model_config=config)
        self.assertIs(provider.model_config, config)
        with self.assertRaisesRegex(ValueError, "不能与"):
            DashScopeVisionProvider(
                api_key="test-key",
                model="qwen3.7-plus",
                model_config=config,
            )

    def test_usage_totals_accumulate_only_valid_token_counts(self) -> None:
        provider = DashScopeVisionProvider(api_key="test-key", max_attempts=1)
        first = self._response()
        request = httpx.Request(
            "POST",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        second = httpx.Response(
            200,
            request=request,
            json={
                "id": "request-38",
                "model": "qwen3.7-plus",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "{}"}}
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 3,
                    "total_tokens": 23,
                    "invalid": "ignored",
                },
            },
        )
        with patch("vision_agent.httpx.post", side_effect=[first, second]):
            provider._chat([{"role": "user", "content": "one"}], max_tokens=10)
            provider._chat([{"role": "user", "content": "two"}], max_tokens=10)
        status = provider.status()
        self.assertEqual(status["successful_call_count"], 2)
        self.assertEqual(
            status["usage_totals"],
            {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 35},
        )


if __name__ == "__main__":
    unittest.main()
