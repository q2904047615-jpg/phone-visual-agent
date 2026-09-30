"""Full raw-Qwen reply -> observer -> canonical binder -> Controller loop, offline."""
from dataclasses import replace
import unittest
from agent.domain.universal_action_controller import UniversalActionController
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from test_single_visual_loop import Adapter, LoopHarness, scene, decision
from test_support.generic_action_adapter import (
    RawSceneProvider,
)
from test_support.generic_scene_observer import (
    audited_application_input,
    input_audit_payload,
)
from contract_tests.observation.test_point_scene_projection import wire
from test_qwen_visual_decision import patterned_frames


class RawAdapter(Adapter):
    def __init__(self, rows):
        super().__init__(rows)
        self.frames = patterned_frames()
        self.model_calls = 0
        self.scopes = []

    def observe(self, goal, entities, allowed):
        source, choice = self.rows[self.position]
        choice = dict(choice)
        if choice.get('tap_point'):
            choice['tap_point'] = [choice['tap_point'][0], round(choice['tap_point'][1] * .48)]
        payload = wire(choice)
        payload['scene'].update(foreground_app_id=source.app_id, screen_id=source.screen_id,
            summary=source.summary, elements=[])
        for element in source.elements:
            x1,y1,x2,y2=element.bounds
            if element.role == 'input':
                item=audited_application_input(bounds=[round(x1*1000),round(y1*480),
                    round(x2*1000),round(y2*480)], text=element.states['value'],
                    focused=element.states.get('focused'), visible_editable_cues=[])
                item['multiline']=True
                payload['input_structure']=input_audit_payload(application_inputs=[item])
        provider=RawSceneProvider([payload])
        result=SingleStepGenericSceneObserver(provider).observe_with_decision(frames=self.frames,
            goal_context={'objective':goal.objective,'entities':entities}, device_id='device-1',
            available_action_kinds=allowed)
        self.model_calls+=provider.calls
        return result

    def capture_scene(self, goal, *, available_action_kinds, **_):
        self.contexts.append(goal.entities)
        current,payload=self.observe(goal,goal.entities,available_action_kinds)
        return current,self.frames,(),payload

    def execute(self, **kwargs):
        self.scopes.append((kwargs['available_action_kinds'],kwargs['post_action_available_action_kinds']))
        controller=UniversalActionController()
        resolved=controller.resolve_one(kwargs['requested_action'],kwargs['planned_scene'])
        result=super().execute(**kwargs)
        after,payload=self.observe(kwargs['goal'],self.contexts[-1],kwargs['post_action_available_action_kinds'])
        controller.verify_after_action(resolved,kwargs['planned_scene'],after)
        return replace(result,resolved_action=resolved,after_scene=after,after_model_decision=payload)


class RawWholeTaskLoopTests(LoopHarness):
    def test_raw_draft_focus_clear_type_send_home_variations(self):
        for app,old,focused in [('messenger','草稿',True),('browser','old',False),('notes','',False)]:
            with self.subTest(app=app):
                def frame(n,text=None,focused=True):
                    return scene(n,text=text,focused=focused,app=app)
                rows=[(frame(0),decision('tap_semantic'))]
                if not focused:
                    rows.append((frame(1,old,False),decision('tap_semantic',role='input',meaning='application_text_input')))
                if old:
                    rows.append((frame(2,old),decision('clear_verified_text')))
                rows.extend([(frame(3,''),decision('input_verified_text',text='aaazjie？你好')),
                    (frame(4,'aaazjie？你好'),decision('tap_semantic',meaning='send_message')),
                    (frame(5,''),decision('home')),(scene(6,app='launcher'),decision())])
                adapter=RawAdapter(rows)
                loop,s,a=self.start([],adapter=adapter,goal='打开目标编辑页，输入并发送指定文字，然后回到主屏幕')
                loop.run_autonomous_safe_loop(s)
                self.assertEqual('succeeded',s.status)
                self.assertEqual(len(a.calls)+1,a.model_calls)
                self.assertEqual(list(range(a.model_calls)),[len(x['history']) for x in a.contexts])
                self.assertEqual('home',a.calls[-1].action)
                self.assertTrue(all(before is after for before,after in a.scopes))

    def test_raw_clear_home_needs_no_finish_between_steps(self):
        a=RawAdapter([(scene(0,text='旧草稿'),decision('clear_verified_text')),
            (scene(1,text='',focused=None),decision('home')),(scene(2,app='launcher'),decision())])
        loop,s,a=self.start([],adapter=a)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual('succeeded',s.status)
        self.assertEqual(3,a.model_calls)
        self.assertEqual(2,s.physical_actions)


if __name__ == '__main__':
    unittest.main()
