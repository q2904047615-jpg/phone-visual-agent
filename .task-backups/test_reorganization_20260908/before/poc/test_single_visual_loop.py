"""Offline whole-task regression: real canonical binder, no remote model or device."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from PIL import Image

from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.application.universal_agent_orchestrator import UniversalAgentOrchestrator
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.text_transport import TextTransportProfile
from agent.domain.ui_scene import UIScene, UIElement
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.infrastructure.generic_action_adapter import GenericActionExecutionResult, GenericSingleActionAdapter
from agent.infrastructure import DeviceTaskRegistry, FileSystemAgentEvidenceStore


def scene(number, *, text=None, focused=True, app='notes'):
    elements = () if text is None else (UIElement(element_id='local_audited_input_1', role='input',
        meaning='application_text_input', label='输入框', bounds=(.1,.6,.8,.8), confidence=1.,
        states={'input_field_id':'current_input','value':text,'focused':focused,
            'ime_preedit_text':'','input_multiline':True,'fully_visible':True,'enabled':True}, evidence=()),)
    return UIScene(app_id=app, screen_id=f'page_{number}', summary=f'当前第{number}页',
        elements=elements, stable=True, confidence=1., fingerprint=f'frame_{number}')


def decision(kind=None, *, meaning='open_details', role='button', text=None, outcome='matched'):
    value = {'status':'finish' if kind is None else 'action', 'action':kind,
        'reason':'当前新图与本次执行记录证明结果', 'previous_action_outcome':outcome}
    if kind in {'tap_semantic','dismiss_overlay','double_tap','long_press'}:
        value.update(target={'role':role,'meaning':meaning},tap_point=[230,670])
    if text is not None:
        value['text'] = text
    return value


class Observation:
    def __init__(self, scene, device_id, observation_id='initial-observation', **_):
        self.scene, self.device_id, self.observation_id = scene, device_id, observation_id
        self.fingerprint = scene.fingerprint
    def get_candidate(self, key):
        return self.scene.get_element(key)
    def to_dict(self):
        return {'scene':self.scene.to_dict(),'device_id':self.device_id,
            'observation_id':self.observation_id,'fingerprint':self.fingerprint}


class Adapter:
    def __init__(self, rows):
        self.rows, self.position, self.calls, self.contexts = rows, 0, [], []
        self.frames = [Image.new('RGB',(100,200),'white') for _ in range(4)]
    def supported_action_kinds(self):
        return CANONICAL_ACTION_KINDS - {'launch_app'}
    def text_transport_profile(self):
        return TextTransportProfile(protocol_version='2026-09-02-adb-keyboard-v1',
            profile_id='offline-profile', device_id='device-1', adb_serial='offline-device', enabled=True,
            capabilities=('append_text','clear_text'), command_timeout_seconds=10)
    def capture_scene(self, goal, **_):
        self.contexts.append(goal.entities)
        current, payload = self.rows[self.position]
        return current, self.frames, tuple(f'frame_{i}' for i in range(4)), payload
    def execute(self, *, requested_action, planned_scene, goal, action_authority, **_):
        self.calls.append(requested_action)
        resolved = ResolvedSemanticAction(node_id=requested_action.node_id, kind=requested_action.action,
            **{k:v for k,v in requested_action.params.items() if k in {'text','text_transport','input_fragment',
                'input_field_id','prior_input_value','expected_input_value'}})
        post_goal = GenericSingleActionAdapter._post_action_goal(goal, authority=action_authority, requested=requested_action,
            resolved=resolved, physical_actions=1)
        self.contexts.append(post_goal.entities)
        self.position += 1
        after, payload = self.rows[self.position]
        return GenericActionExecutionResult(requested_action=requested_action, rebound_action=requested_action,
            resolved_action=resolved,before_scene=planned_scene,after_scene=after,
            planned_scene_fingerprint=planned_scene.fingerprint,confirmation_frame_identity_verified=True,
            confirmation_frame_delta=0.,physical_actions=1,robot_result=True,evidence=(),
            after_model_decision=payload,after_frames=tuple(self.frames),
            after_frame_paths=tuple(f'after_{self.position}_{i}' for i in range(4)))


class LoopHarness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
    def start(self, rows, goal='打开笔记，清空输入框，然后回主屏幕', **kwargs):
        adapter = kwargs.pop('adapter', None) or Adapter(rows)
        observer = kwargs.pop('observer', None) or QwenVisualDecisionObserver(SimpleNamespace(status=lambda:{'model':'offline'}),
            trusted_observation_frame_validator=lambda *_a,**_k:None)
        registry = DeviceTaskRegistry(lease_directory=Path(self.temp.name)/'leases')
        self.addCleanup(lambda: registry.release('device-1','session-1'))
        loop = UniversalAgentOrchestrator(required_action_kind=kwargs.pop('required_action_kind', ''),
            qwen_observer=observer, adapter_factory=lambda _:adapter,
            evidence_store_factory=FileSystemAgentEvidenceStore, trusted_observation_factory=Observation,
            device_registry=registry)
        session = loop.start(session_id='session-1',raw_goal=goal,device_id='device-1',
            run_dir=Path(self.temp.name)/'run',**kwargs)
        return loop,session,adapter


class WholeTaskLoopTests(LoopHarness):
    def test_navigation_clear_home_is_one_task(self):
        rows=[(scene(0),decision('tap_semantic')),(scene(1,text='旧内容'),decision('clear_verified_text')),
            (scene(2,text='',focused=None),decision('home')),(scene(3,app='launcher'),decision())]
        loop,s,a=self.start(rows)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(s.status,'succeeded')
        self.assertEqual([x.action for x in a.calls],['tap_semantic','clear_verified_text','home'])
        self.assertEqual([len(x['history']) for x in a.contexts],[0,1,2,3])
        self.assertNotIn('task_graph',s.snapshot())
    def test_input_send_home_variation(self):
        text='aaazjie？你好\n  保留空格 '
        rows=[(scene(0,text=''),decision('input_verified_text',text=text)),
            (scene(1,text=text),decision('tap_semantic',meaning='send_message')),
            (scene(2,text=''),decision('home')),(scene(3,app='launcher'),decision())]
        loop,s,a=self.start(rows,goal='将指定内容发给目标联系人，然后返回桌面',exact_input_text=text)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(s.status,'succeeded')
        self.assertEqual(a.calls[0].params['input_fragment'],text)
        self.assertEqual(s.physical_actions,3)
    def test_uncertain_effect_stops_before_repeated_send(self):
        rows=[(scene(0),decision('tap_semantic',meaning='send_message')),
            (scene(1),decision('tap_semantic',meaning='send_message',outcome='uncertain'))]
        loop,s,a=self.start(rows,goal='发送一条消息')
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(s.status,'failed')
        self.assertEqual(len(a.calls),1)
    def test_unknown_focus_cannot_type(self):
        with self.assertRaisesRegex(Exception,'聚焦'):
            self.start([(scene(0,text='',focused=None),decision('input_verified_text',text='正文'))])
    def test_body_must_match_explicit_api_text(self):
        with self.assertRaisesRegex(Exception,'逐字正文'):
            self.start([(scene(0,text=''),decision('input_verified_text',text='篡改'))],exact_input_text='原文')
    def test_stale_scope_not_executed(self):
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision())])
        scope=s.confirmation_authority.scope()
        scope['fingerprint']='old-frame'
        with self.assertRaisesRegex(Exception,'scope'):
            loop.confirm_one(s,scope)
        self.assertEqual(a.calls,[])
    def test_authentication_waits_for_exact_action_confirmation(self):
        loop,s,a=self.start([(scene(0),decision('tap_semantic',meaning='authentication')),
            (scene(1),decision())],goal='登录账号')
        self.assertEqual(s.status,'awaiting_effect_confirmation')
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(a.calls,[])
        loop.approve_effects(s,s.effect_confirmation_authority.scope())
        self.assertEqual(s.status,'succeeded')
        self.assertEqual(len(a.calls),1)
    def test_observation_budget_pause_does_not_succeed(self):
        loop,s,a=self.start([(scene(0),decision('home'))],max_observations=1)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(s.status,'budget_paused')
        self.assertEqual(a.calls,[])

if __name__ == '__main__':
    unittest.main()
