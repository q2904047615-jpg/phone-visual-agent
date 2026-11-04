"""Authorized recognition-only campaign: ten calls maximum, no retries or device access.

prepare constructs and fingerprints paired requests offline. recognize sends one
case/variant once. A stopped campaign cannot be resumed by this script.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments.bound_action_result_candidate import prepare as candidate_request, project_result_for_evaluation
from experiments.probe_history_dialogue import cases, comparison_config, redacted_wire, read
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider, _extract_json_object, _image_request_size
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver

OUT = POC / 'output/bound_action_result_20260907'
SOURCE = POC / 'output/web/generic_supervised_20260907_165714_719dd258'
CASE_NAMES = ('failed_send', 'sent', 'old_unsent', 'empty_clear', 'sent_then_home')
VARIANTS = ('baseline', 'candidate')


def save(name, data):
    atomic_replace_bytes(OUT / name, json_bytes(data))


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def fixtures():
    _, result = cases()
    record = read(SOURCE / '0df6ccf437914203937c50b78d9c7721_model_response.json')
    receipt = read(SOURCE / 'session.json')['history'][3]['execution']
    assert receipt['requested_action']['params']['target'] == 'send_message'
    assert record['goal_context']['entities']['history'][-1]['visual_outcome'] is None
    frames = []
    for path in receipt['after_frame_paths']:
        with Image.open(SOURCE / Path(path).name) as image:
            frames.append(image.convert('RGB'))
    result['failed_send'] = dict(context=record['goal_context'], frames=frames,
        allowed=result['old_unsent']['allowed'], device_id=record['device_id'],
        source=str(SOURCE), expected_status='action', latest_result_withheld=True)
    return result


class Captured(Exception):
    pass


class ProbeObserver(SingleStepGenericSceneObserver):
    def __init__(self, provider, *, case, variant, remote=False):
        super().__init__(provider)
        self.case, self.variant, self.remote = case, variant, remote

    def _provider_chat(self, messages, *, max_tokens, response_format=None):
        messages = deepcopy(messages)
        width, height = _image_request_size(self.case['frames'][-1])
        image_count = sum(p['type'] == 'image_url' for p in messages[1]['content'])
        props = response_format['json_schema']['schema']['properties']
        kinds = tuple(k for k in props['decision']['properties']['action']['enum'] if k is not None)
        options = dict(request_width=width, request_height=height, image_count=image_count,
            available_action_kinds=kinds, include_input_structure=props['input_structure']['type'] != 'null')
        baseline = candidate_request(self.case['context'], bind_result=False, **options)
        assert baseline['prompt'] == messages[1]['content'][0]['text'], 'Baseline prompt drift'
        assert baseline['response_format'] == response_format, 'Baseline schema drift'
        prepared = candidate_request(self.case['context'], bind_result=self.variant == 'candidate', **options)
        messages[1]['content'][0]['text'] = prepared['prompt']
        response_format = prepared['response_format']
        body = dict(model=self.provider.model, messages=messages, temperature=0.0,
            response_format=response_format, **self.provider.model_config.request_options())
        self.wire = redacted_wire(deepcopy(body))
        if not self.remote:
            raise Captured()
        name = self.case['name'] + '_' + self.variant
        assert digest(self.wire) == read(OUT / 'preflight.json')['wire_hashes'][name], 'Request drift'
        reserve_attempt(name)
        raw = super()._provider_chat(messages, max_tokens=max_tokens, response_format=response_format)
        # Keep exact model output before any experimental projection or parsing.
        save(name + '_raw.json', {'raw': raw, 'expected_action_id': prepared['expected_action_id']})
        if self.variant == 'candidate':
            payload = _extract_json_object(raw, reject_duplicate_keys=True, unwrap_singleton_object_array=True)
            raw = json.dumps(project_result_for_evaluation(payload,
                expected_action_id=prepared['expected_action_id']), ensure_ascii=False)
        return raw


def reserve_attempt(name):
    assert name in {f'{c}_{v}' for c in CASE_NAMES for v in VARIANTS}
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert len(list(OUT.glob('*_attempt.json'))) < 10, 'Authorization exhausted'
    # Exclusive create makes a repeated invocation consume no extra remote call.
    with (OUT / (name + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump({'authorized_at': '2026-09-07 user:允许', 'started_at': datetime.now().astimezone().isoformat(),
            'max_attempts': 1, 'phone_actions': 0}, stream, ensure_ascii=False)


def observe(observer, case):
    return observer.observe_with_decision(frames=case['frames'], goal_context=case['context'],
        device_id=case['device_id'], available_action_kinds=case['allowed'])


def prepare():
    assert not list(OUT.glob('*_attempt.json')), 'Do not overwrite a started campaign'
    hashes = {}
    for name, case in fixtures().items():
        wires = []
        for variant in VARIANTS:
            observer = ProbeObserver(DashScopeVisionProvider(api_key='offline-dummy', max_attempts=1),
                case={**case, 'name': name}, variant=variant)
            try:
                observe(observer, case)
            except Captured:
                pass
            wire = observer.wire
            wires.append(wire)
            hashes[name + '_' + variant] = digest(wire)
            save(name + '_' + variant + '_wire.json', wire)
        # Only text/schema differs. Image marker lists and provider settings match.
        assert wires[0]['messages'][0] == wires[1]['messages'][0]
        assert wires[0]['messages'][1]['content'][1:] == wires[1]['messages'][1]['content'][1:]
        for key in ('model', 'temperature', 'enable_thinking'):
            assert wires[0][key] == wires[1][key]
    save('preflight.json', dict(wire_hashes=hashes, cases=list(CASE_NAMES), max_calls=10,
        model_config=comparison_config(DashScopeVisionProvider(api_key='offline-dummy')),
        paired_current_images_identical=True, uniform_xy=False, phone_actions=0))
    print(json.dumps({'prepared_pairs': len(CASE_NAMES), 'remote_calls': 0, 'phone_actions': 0}))


def recognize(name, variant):
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert not (OUT / f'{name}_{variant}_attempt.json').exists(), 'No resampling or overwriting results'
    case = {**fixtures()[name], 'name': name}
    provider = DashScopeVisionProvider(max_attempts=1)
    assert provider.configured, 'Qwen credential unavailable'
    assert comparison_config(provider) == read(OUT / 'preflight.json')['model_config'], 'Provider config changed'
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'Unexpected destination'
    observer = ProbeObserver(provider, case=case, variant=variant, remote=True)
    started = time.perf_counter()
    result = dict(case=name, variant=variant, phone_actions=0)
    try:
        scene, decision = observe(observer, case)
        result.update(parse_succeeded=True, decision=decision, scene=scene.to_dict())
    except Exception as exc:
        result.update(parse_succeeded=False, error_type=type(exc).__name__, error=str(exc))
    result.update(seconds=round(time.perf_counter()-started, 3), network_attempts=provider.last_network_attempts,
        usage=provider.last_usage, response_model=provider.last_response_model)
    save(f'{name}_{variant}_result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'scene'}, ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('prepare', 'recognize'))
    cli.add_argument('--case', choices=CASE_NAMES)
    cli.add_argument('--variant', choices=VARIANTS)
    cli.add_argument('--allow-remote', action='store_true')
    args = cli.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.case and args.variant:
        recognize(args.case, args.variant)
    else:
        cli.error('Recognition requires --allow-remote, --case and --variant')
