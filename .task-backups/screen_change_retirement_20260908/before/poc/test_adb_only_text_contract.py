"""The current text path: same-frame field facts -> canonical -> ADB, never keys."""
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError, bind_same_response_action,
)
from agent.domain.device_execution import DeviceActionRequest, DeviceExecutionError
from agent.domain.universal_action_controller import UniversalActionController, UniversalActionError
from agent.domain.ui_scene import UIScene
from agent.domain.text_transport import text_digest
from agent.infrastructure.generic_scene_observer import (
    INPUT_STRUCTURE_AUDIT_VERSION, _apply_input_structure_audit,
)
from agent.infrastructure.robot_controller import RobotController, MockRobotController
from agent.infrastructure.device_executor import RobotDeviceExecutor
from test_canonical_action_protocol import ime_profile
from test_generic_action_adapter import FakeAdbKeyboardTextTransport


def current_field(text='', preedit='', focused=True, *, app='sample.notes', multiline=True):
    raw = {'protocol_version': INPUT_STRUCTURE_AUDIT_VERSION, 'application_inputs': [
        {'bounds': [100, 200, 800, 350], 'text': text, 'preedit_text': preedit, 'focused': focused, 'multiline': multiline}]}
    ctx = {'objective': '编辑当前字段', 'entities': {'history': []}}
    scene = UIScene(app_id=app, screen_id='editor', summary='当前编辑框', elements=(), stable=True,
        confidence=1.0, fingerprint='current-frame')
    return _apply_input_structure_audit(scene, json.dumps(raw, ensure_ascii=False),
        fingerprint='current-frame', goal_context=ctx)


def bind(kind, scene, target='', *, profile=True):
    ctx = SimpleNamespace(device_id='device-local-01',revision=1,exact_input_text=None)
    return bind_same_response_action({'status': 'action', 'action': kind, 'text': target if kind == 'input_verified_text' else None}, context=ctx,
        observation=SimpleNamespace(scene=scene), available_action_kinds={kind},
        text_transport_profile=ime_profile() if profile else None)


class AdbOnlyTextContractTests(unittest.TestCase):
    def test_current_field_keeps_body_focus_and_preedit_independent_across_apps(self):
        for app in ('sample.notes', 'sample.search', 'sample.chat'):
            for focus in (True, False, None):
                with self.subTest(app=app, focus=focus):
                    scene = current_field('已提交', '未提交', focus, app=app)
                    states = scene.elements[0].states
                    self.assertEqual('已提交', states['value'])
                    self.assertEqual('未提交', states['ime_preedit_text'])
                    self.assertIs(focus, states.get('focused'))
                    self.assertNotIn('keyboard_geometry', states)
                    self.assertEqual(1, len(scene.elements))

    def test_input_clear_and_newline_resolve_without_a_keyboard_or_a_point(self):
        for kind, prior, target in [('input_verified_text', '前缀', '前缀🙂\r\n中A1！'),
            ('clear_verified_text', '草稿', ''), ('press_enter', '两行', '两行\n')]:
            with self.subTest(kind=kind):
                scene = current_field(prior)
                action = bind(kind, scene, target)
                resolved = UniversalActionController().resolve_one(action, scene)
                self.assertEqual('adb_keyboard', resolved.text_transport)
                self.assertIsNone(resolved.normalized_point)
                self.assertEqual(target, resolved.expected_input_value)
                transport = FakeAdbKeyboardTextTransport()
                fragment = resolved.input_fragment or ''
                scope = transport.mint_action_scope(session_id='session', task_id='task', revision=1,
                    action_id=action.node_id, input_field_id='body', observation_fingerprint=scene.fingerprint,
                    prior_text_digest=text_digest(prior), fragment_text_digest=text_digest(fragment),
                    expected_text_digest=text_digest(target))
                robot = SimpleNamespace()  # No mechanical text method exists.
                request = DeviceActionRequest(kind=kind, input_fragment=resolved.input_fragment,
                    text_transport='adb_keyboard', text_scope=scope)
                result = RobotDeviceExecutor(robot, text_transport=transport).execute(request)
                self.assertEqual(1, result.physical_actions)
                self.assertEqual(1, len(transport.calls))
                after = replace(current_field(target), fingerprint='after-frame')
                UniversalActionController().verify_after_action(resolved, scene, after)

    def test_newline_is_exactly_one_character_and_requires_multiline(self):
        with self.assertRaisesRegex(CanonicalActionProtocolError, '多行'):
            bind('press_enter', current_field('one', multiline=False))
        scene = current_field('one')
        resolved = UniversalActionController().resolve_one(bind('press_enter', scene), scene)
        self.assertEqual('\n', resolved.input_fragment)
        with self.assertRaisesRegex(UniversalActionError, '文字不匹配'):
            UniversalActionController().verify_after_action(resolved, scene,
                replace(current_field('one'), fingerprint='after'))

    def test_unknown_or_false_focus_never_authorizes_nonempty_text_action(self):
        for focus in (None, False):
            for kind in ('input_verified_text', 'clear_verified_text', 'press_enter'):
                with self.subTest(focus=focus, kind=kind), self.assertRaisesRegex(CanonicalActionProtocolError, '聚焦'):
                    bind(kind, current_field('old', focused=focus), 'oldnew')

    def test_preedit_prevents_append_but_can_be_cleared_without_committed_text(self):
        scene = current_field('', 'pending')
        with self.assertRaisesRegex(UniversalActionError, '组合'):
            UniversalActionController().resolve_one(bind('input_verified_text', scene, 'hello'), scene)
        clear = UniversalActionController().resolve_one(bind('clear_verified_text', scene), scene)
        self.assertEqual('', clear.expected_input_value)
        with self.assertRaisesRegex(UniversalActionError, '预编辑'):
            UniversalActionController().verify_after_action(clear, scene,
                replace(current_field('', 'pending'), fingerprint='after'))

    def test_unconfigured_text_never_falls_back_to_mechanical(self):
        with self.assertRaisesRegex(CanonicalActionProtocolError, 'ADB Keyboard'):
            bind('input_verified_text', current_field(), 'hello', profile=False)
        with self.assertRaisesRegex(DeviceExecutionError, 'ADB Keyboard'):
            DeviceActionRequest(kind='input_verified_text', input_fragment='hello').validate()

    def test_retired_modules_methods_and_request_fields_cannot_return(self):
        root = Path(__file__).resolve().parent
        for relative in ('agent/domain/verified_text_transaction.py',
            'agent/infrastructure/windows_ocr_runtime.py', 'windows_ocr.ps1'):
            self.assertFalse((root / relative).exists(), relative)
        for cls in (RobotController, MockRobotController):
            for method in ('vision_type_text_with_layout', 'vision_type_pinyin', 'vision_clear_text', 'validate_verified_text'):
                self.assertFalse(hasattr(cls, method), method)
        names = {item.name for item in fields(DeviceActionRequest)}
        self.assertFalse(names & {'input_method', 'input_pinyin', 'keyboard_geometry', 'delete_count'})


if __name__ == '__main__':
    unittest.main()
