"""Explicit, bounded recognition-only comparison. No hardware or project API writes.

prepare: build paired requests entirely offline.
recognize CASE baseline|candidate --allow-remote: one request, no retry.
The remote flag must only be used following the user's new sampling permission.
"""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments.history_dialogue_candidate import DialogueCandidateObserver, CANDIDATE_VERSION
from agent.domain.qwen_task_context import execution_history_entry
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver, SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes

CAMPAIGN = POC / 'output/history_dialogue_20260907'
OUT = CAMPAIGN / 'clean_fixture_v2'
FIXTURE_VERSION = '2026-09-07-current-history-fixture-v2'
SUCCESS = POC / 'output/web/generic_supervised_20260823_131844_e5ef5dce'
CASES = ('old_unsent', 'sent', 'empty_clear', 'sent_then_home')
VARIANTS = {'baseline': SingleStepGenericSceneObserver, 'candidate': DialogueCandidateObserver}
RETIRED_METADATA = frozenset({'formal_report_digest', 'formal_transition',
    'formal_candidate_id', 'expected_effect'})
LEGACY_NULL_FIELDS = frozenset({'input_method', 'input_pinyin', 'delete_count'})


def project_historical_point_receipt(receipt):
    """Offline fixture projection, not a production history filter or compatibility path.

    Keep actual tap semantics and physical facts; never turn historical expected
    effects into outcomes. Unsupported non-null legacy transport facts require
    separate review rather than silently dropping them.
    """
    old = receipt['requested_action']
    if old['action'] != 'tap_semantic' or receipt['resolved_action']['kind'] != old['action']:
        raise ValueError('Fixture projection only audited for an actual point action')
    params = old['params']
    unknown = set(params) - RETIRED_METADATA - {'target', 'role', 'label', 'element_id', 'states', 'tap_point'}
    if unknown:
        raise ValueError('Unaudited historical request fields: ' + ','.join(sorted(unknown)))
    requested = {'node_id': old['node_id'], 'action': old['action'],
        'params': {key: deepcopy(params[key]) for key in ('target', 'role', 'label')}}
    current_fields = {field.name for field in fields(ResolvedSemanticAction)}
    resolved = {}
    for key, value in receipt['resolved_action'].items():
        if key in current_fields:
            resolved[key] = deepcopy(value)
        elif key in RETIRED_METADATA:
            continue
        elif key in LEGACY_NULL_FIELDS and value is None:
            continue
        else:
            raise ValueError('Unaudited historical resolved field: ' + key)
    ResolvedSemanticAction(**resolved)
    return requested, resolved


def assert_clean_history(value):
    """Inspect keys, never rewrite a user's literal message body."""
    if isinstance(value, dict):
        if set(value) & (RETIRED_METADATA | {'goal_complete_on_success'}):
            raise ValueError('Retired planning/completion metadata in experiment history')
        for child in value.values():
            assert_clean_history(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_clean_history(child)


def budget_precheck(name, variant):
    assert not (OUT / 'stopped.json').exists(), 'Comparison stopped; read stopped.json before designing a new validation'
    if name == 'empty_clear':
        permission = read(OUT / 'empty_clear_authorization.json')
        assert permission == {'case': 'empty_clear', 'additional_calls': 2, 'variants': ['baseline', 'candidate'], 'phone_actions': 0}
        assert variant in permission['variants']
        assert len(list(CAMPAIGN.rglob('*_attempt.json'))) < 10, 'Extended authorization budget exhausted'
        assert len(list(OUT.glob('empty_clear_*_attempt.json'))) < 2, 'Empty-clear authorization budget exhausted'
    else:
        assert len(list(CAMPAIGN.rglob('*_attempt.json'))) < 8, 'Original authorization budget exhausted'
        assert len(list(OUT.glob('*_attempt.json'))) < 4, 'Remaining comparison budget exhausted'
    assert not (OUT / f'{name}_{variant}_attempt.json').exists(), 'No resampling of a case/variant'


def save(name, value):
    return atomic_replace_bytes(OUT / name, json_bytes(value))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cases():
    # Reuse only the previously audited pure evidence reconstruction, never its CLI.
    path = POC / 'output/action_comparison_causal_20260906/verify.py'
    spec = importlib.util.spec_from_file_location('causal_evidence_reader', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    record, original, frames, _, allowed, device_id = helper.inputs()
    assert original['entities']['history'][-1]['visual_outcome'] is None
    assert original['entities']['history'][-1]['after_scene'] == ''
    output = {}
    for name, goal, expected in [
        ('old_unsent', '进入微信的文件传输助手，发一条消息，正文是aaazjie？你好。', 'action'),
        ('empty_clear', '打开微信，打开文件传输助手，把输入框里的内容清空。', 'finish'),
    ]:
        context = deepcopy(original)
        context['objective'] = goal
        output[name] = dict(context=context, frames=frames, allowed=allowed,
            device_id=device_id, expected_status=expected, source=str(helper.SOURCE),
            latest_result_withheld=True)
    session = read(SUCCESS / 'session.json')
    entry = session['history'][0]
    receipt = entry['execution']
    old_action = receipt['requested_action']
    assert old_action['params']['target'] == 'send_message' and receipt['physical_actions'] == 1
    # Project actual historical command semantics to today's history; retired formal
    # expectations/goal_complete flags are not executed facts and must not leak in.
    requested, resolved = project_historical_point_receipt(receipt)
    factual = execution_history_entry(step=1, requested_action=requested,
        resolved_action=resolved, physical_actions=1,
        transport_outcome='executed', visual_outcome=None, after_scene='')
    sent_frames = []
    for value in receipt['after_frame_paths']:
        with Image.open(SUCCESS / Path(value).name) as image:
            sent_frames.append(image.convert('RGB'))
    for name, goal, expected in [
        ('sent', '把当前输入框里的 loopok 发送给文件传输助手。', 'finish'),
        ('sent_then_home', '把当前输入框里的 loopok 发送给文件传输助手，然后回到主屏幕。', 'action'),
    ]:
        output[name] = dict(context={'objective': goal, 'entities': {
            'task_id': session['session_id'], 'history': [factual]}}, frames=sent_frames,
            allowed=allowed, device_id=device_id, expected_status=expected,
            source=str(SUCCESS), latest_result_withheld=True)
    for case in output.values():
        assert_clean_history(case['context']['entities']['history'])
    return record, output


def redacted_wire(body):
    safe = deepcopy(body)
    for message in safe['messages']:
        for part in message['content'] if isinstance(message['content'], list) else []:
            if part.get('type') == 'image_url':
                url = part['image_url']['url']
                part['image_url']['url'] = '[CURRENT_IMAGE_SHA256:' + hashlib.sha256(url.encode()).hexdigest() + ']'
    return safe


def observe(observer, case, **extra):
    return observer.observe_with_decision(frames=case['frames'], goal_context=case['context'],
        device_id=case['device_id'], available_action_kinds=case['allowed'], **extra)


def comparison_config(provider):
    # Freeze only non-secret model settings; never persist API keys or headers.
    return dict(model=provider.model, base_url=provider.base_url,
        request_options=provider.model_config.request_options())


def prepare():
    assert not list(OUT.glob('*_attempt.json')), 'Do not overwrite a started comparison'
    record, inputs = cases()
    reply = json.loads(record['redacted_raw_response'])
    reply['protocol_version'] = SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
    response = Mock()
    response.json.return_value = {'choices': [{'message': {'content': json.dumps(reply)},
        'finish_reason': 'stop'}], 'model': 'offline', 'usage': {}}
    summaries = []
    for name, case in inputs.items():
        bodies = {}
        for variant, observer_type in VARIANTS.items():
            with patch('agent.infrastructure.dashscope_vision_provider.httpx.post', return_value=response) as post:
                observe(observer_type(DashScopeVisionProvider(api_key='offline-dummy')), case)
                assert post.call_count == 1
                body = redacted_wire(post.call_args.kwargs['json'])
                bodies[variant] = body
                save(f'{name}_{variant}_wire.json', body)
        outside = lambda body: {k: v for k, v in body.items() if k != 'messages'}
        assert outside(bodies['baseline']) == outside(bodies['candidate'])
        images = lambda body: [p['image_url']['url'] for m in body['messages']
            if isinstance(m['content'], list) for p in m['content'] if p['type'] == 'image_url']
        assert images(bodies['baseline']) == images(bodies['candidate'])
        assert len(images(bodies['candidate'])) == 3
        summaries.append(dict(case=name, expected_status=case['expected_status'], source=case['source'],
            roles={key: [m['role'] for m in body['messages']] for key, body in bodies.items()},
            identical_settings_schema_images=True, latest_result_withheld=True))
    save('preflight.json', dict(candidate=CANDIDATE_VERSION, cases=summaries,
        fixture_version=FIXTURE_VERSION,
        model_config=comparison_config(DashScopeVisionProvider(api_key='offline-dummy')),
        remote_calls=0, phone_actions=0, production_changed=False))
    print(json.dumps({'prepared_pairs': len(summaries), 'remote_calls': 0, 'phone_actions': 0}))


def recognize(name, variant):
    budget_precheck(name, variant)
    assert name in {'sent', 'sent_then_home', 'empty_clear'}, 'Case not authorized for this comparison'
    assert (OUT / 'preflight.json').is_file(), 'Run offline prepare first'
    assert read(OUT / 'preflight.json')['fixture_version'] == FIXTURE_VERSION
    _, inputs = cases()
    case = inputs[name]
    provider = DashScopeVisionProvider(max_attempts=1)
    assert provider.configured, 'Current Qwen configuration unavailable'
    assert comparison_config(provider) == read(OUT / 'preflight.json')['model_config'], 'Model settings changed; stop the comparison'
    observer = VARIANTS[variant](provider)
    # Record a consumed attempt BEFORE any network request; failures also consume it.
    save(f'{name}_{variant}_attempt.json', dict(started_at=datetime.now().astimezone().isoformat(),
        model=provider.model, context=case['context'], source=case['source'], max_attempts=1,
        candidate=CANDIDATE_VERSION, fixture_version=FIXTURE_VERSION, phone_actions=0))
    result = {'case': name, 'variant': variant, 'expected_status': case['expected_status'], 'phone_actions': 0}
    start = time.perf_counter()
    try:
        scene, choice = observe(observer, case, response_evidence_dir=OUT / f'{name}_{variant}')
        result.update(decision=choice, scene=scene.to_dict(), parse_succeeded=True,
            status_matches_expected=choice['status'] == case['expected_status'])
        # A matching status alone does not establish semantic/action correctness.
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), parse_succeeded=False)
    result.update(seconds=round(time.perf_counter() - start, 3), usage=provider.last_usage,
        response_model=provider.last_response_model, network_attempts=provider.last_network_attempts,
        response_file=observer.last_response_evidence_path)
    save(f'{name}_{variant}_result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'scene'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize'))
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--allow-remote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.case and args.variant:
        recognize(args.case, args.variant)
    else:
        parser.error('Recognition requires new user authorization, --allow-remote, --case and --variant')
