import unittest
import base64
from unittest.mock import patch
import httpx
from agent.infrastructure.dashscope_vision_provider import (
    DashScopeVisionProvider,
)
from agent.domain.vision_model import (
    DEFAULT_VISION_BASE_URL,
    DEFAULT_VISION_MODEL,
    VISION_MODEL_CONFIG_VERSION,
    VisionModelConfig,
    public_model_identity,
)
from agent.infrastructure.environment_vision_model_config import (
    load_vision_model_config,
)


class VisionModelConfigTests(unittest.TestCase):
    def test_default_is_qwen3_vl_plus_non_thinking(self) -> None:
        config = load_vision_model_config(environ={})
        self.assertEqual(DEFAULT_VISION_MODEL, "qwen3-vl-plus")
        self.assertEqual(config.model, "qwen3-vl-plus")
        self.assertEqual(config.base_url, DEFAULT_VISION_BASE_URL)
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.coordinate_scale, 1000)

    def test_current_environment_names_override_defaults(self) -> None:
        config = load_vision_model_config(
            environ={
                "VISION_MODEL": "qwen3-vl-plus",
                "VISION_MODEL_BASE_URL": "https://new.example/v1/",
            }
        )
        self.assertEqual(config.model, "qwen3-vl-plus")
        self.assertEqual(config.base_url, "https://new.example/v1")

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
        with patch(
            "agent.infrastructure.dashscope_vision_provider.httpx.post",
            return_value=self._response(),
        ) as mocked:
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

    def test_json_object_mode_is_forwarded_only_when_requested(self) -> None:
        provider = DashScopeVisionProvider(api_key="test-key", max_attempts=1)
        with patch(
            "agent.infrastructure.dashscope_vision_provider.httpx.post",
            return_value=self._response(),
        ) as mocked:
            provider._chat(
                [{"role": "user", "content": "json"}],
                max_tokens=500,
                response_format={"type": "json_object"},
            )
        self.assertEqual(
            {"type": "json_object"},
            mocked.call_args.kwargs["json"]["response_format"],
        )

    def test_strict_json_schema_is_forwarded_and_max_tokens_can_be_omitted(self) -> None:
        provider = DashScopeVisionProvider(api_key="test-key", max_attempts=1)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "pending_transition",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "enum": ["action"]}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        }
        with patch(
            "agent.infrastructure.dashscope_vision_provider.httpx.post",
            return_value=self._response(),
        ) as mocked:
            provider._chat(
                [{"role": "user", "content": "json"}],
                max_tokens=None,
                response_format=response_format,
            )
        body = mocked.call_args.kwargs["json"]
        self.assertNotIn("max_tokens", body)
        self.assertEqual(response_format, body["response_format"])

    def test_malformed_json_schema_is_rejected_before_network(self) -> None:
        provider = DashScopeVisionProvider(api_key="test-key", max_attempts=1)
        with patch("agent.infrastructure.dashscope_vision_provider.httpx.post") as mocked:
            with self.assertRaisesRegex(Exception, "严格 json_schema"):
                provider._chat(
                    [{"role": "user", "content": "json"}],
                    max_tokens=None,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "unsafe_schema",
                            "strict": True,
                            "schema": {"type": "object"},
                        },
                    },
                )
        mocked.assert_not_called()

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
        with patch(
            "agent.infrastructure.dashscope_vision_provider.httpx.post",
            side_effect=[first, second],
        ):
            provider._chat([{"role": "user", "content": "one"}], max_tokens=10)
            provider._chat([{"role": "user", "content": "two"}], max_tokens=10)
        status = provider.status()
        self.assertEqual(status["successful_call_count"], 2)
        self.assertEqual(
            status["usage_totals"],
            {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 35},
        )

    def test_retries_known_dashscope_inline_url_rejection_once(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=2,
            retry_base_delay=0,
        )
        request = httpx.Request(
            "POST",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        rejected = httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": (
                        "<400> InternalError.Algo.InvalidParameter: The provided "
                        "URL does not appear to be valid. Ensure it is correctly "
                        "formatted."
                    )
                }
            },
        )
        inline_jpeg = "data:image/jpeg;base64," + base64.b64encode(
            b"\xff\xd8valid-test-jpeg\xff\xd9"
        ).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "json"},
                    {"type": "image_url", "image_url": {"url": inline_jpeg}},
                ],
            }
        ]
        with patch(
            "agent.infrastructure.dashscope_vision_provider.httpx.post",
            side_effect=[rejected, self._response()],
        ) as mocked:
            self.assertEqual(provider._chat(messages, max_tokens=10), '{"ok":true}')
        self.assertEqual(2, mocked.call_count)
        self.assertEqual(2, provider.status()["last_network_attempts"])

    def test_does_not_retry_malformed_inline_url_rejection(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=2,
            retry_base_delay=0,
        )
        request = httpx.Request(
            "POST",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        rejected = httpx.Response(
            400,
            request=request,
            text=(
                "InternalError.Algo.InvalidParameter: The provided URL does not "
                "appear to be valid."
            ),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,not-base64!"},
                    }
                ],
            }
        ]
        with patch(
            "agent.infrastructure.dashscope_vision_provider.httpx.post",
            return_value=rejected,
        ) as mocked:
            with self.assertRaisesRegex(Exception, "HTTP 400"):
                provider._chat(messages, max_tokens=10)
        self.assertEqual(1, mocked.call_count)


if __name__ == "__main__":
    unittest.main()
