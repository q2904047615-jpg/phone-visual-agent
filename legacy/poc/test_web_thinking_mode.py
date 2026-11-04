import unittest
import web_app


class WebThinkingModeTests(unittest.TestCase):
    def test_runtime_shares_thinking_provider(self):
        provider = web_app.runtime.vision_provider
        self.assertTrue(provider.model_config.enable_thinking)
        self.assertIs(web_app.runtime.generic_scene_observer.provider, provider)
        self.assertIs(web_app.runtime.qwen_visual_decision_observer.provider, provider)
        self.assertTrue(provider.status()['thinking_enabled'])


if __name__ == '__main__':
    unittest.main()
