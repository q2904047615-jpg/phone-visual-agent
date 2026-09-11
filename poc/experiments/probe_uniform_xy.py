"""Eight-call authorized XY-only experiment. No device or project API access."""
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
from experiments.bound_action_result_candidate import prepare as candidate_request
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider, _image_request_size
from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver, _parse_single_step_observation_envelope,
)

OUT = POC / 'output/uniform_xy_20260907'
CURRENT = POC / 'output/web/generic_supervised_20260907_200548_9e56e1e2'
SUCCESS = POC / 'output/web/generic_supervised_20260907_185750_5748fa48'
CASES = ('failed_point', 'successful_point', 'input_focus', 'scroll_variation')
VARIANTS = ('baseline', 'uniform')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, data):
    atomic_replace_bytes(OUT / name, json_bytes(data))


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def production_hashes():
    paths = list((POC / 'agent').rglob('*.py')) + list((POC / 'agent').rglob('*.txt'))
    paths += [POC / 'web_app.py', POC / 'tap_calibration.json']
    return {str(p.relative_to(POC)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def fixture(name):
    specs = {
        'failed_point': (CURRENT, '3baacd4002744532ad5b3c8280bf577e_model_response.json', 3),
        'successful_point': (SUCCESS, 'e6bae83a80fc41279032c5a691024f4f_model_response.json', 1),
        'input_focus': (CURRENT, '8059e12e3b07452cb554fb6c287758b2_model_response.json', 1),
        'scroll_variation': (CURRENT, '8059e12e3b07452cb554fb6c287758b2_model_response.json', 1),
    }
    directory, filename, step = specs[name]
    record = read(directory / filename)
    paths = [directory / f"{record['response_evidence_prefix']}_{i}.jpg" for i in range(1, 5)]
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert('RGB'))
    context = deepcopy(record['goal_context'])
    if name == 'scroll_variation':
        # Explicit offline goal variation, not a claim that this was the original task.
        context['objective'] = '将当前聊天页面向上滚动一次，不输入或发送消息。'
        context['entities']['history'] = []
    return dict(name=name, context=context, frames=frames, device_id=record['device_id'],
        allowed=read(directory / f'model_binding_step_{step}.json')['available_action_kinds'],
        source_record=str(directory / filename), frame_paths=[str(p) for p in paths],
        modified_goal=name == 'scroll_variation')


class Captured(Exception):
    pass


class CaptureObserver(SingleStepGenericSceneObserver):
    def _provider_chat(self, messages, *, max_tokens, response_format=None):
        self.captured = (deepcopy(messages), deepcopy(response_format))
        raise Captured()


def model_config(provider):
    return dict(model=provider.model, base_url=provider.base_url,
                request_options=provider.model_config.request_options())


def redacted_wire(body):
    value = deepcopy(body)
    for msg in value['messages']:
        for part in msg['content'] if isinstance(msg['content'], list) else []:
            if part.get('type') == 'image_url':
                url = part['image_url']['url']
                part['image_url']['url'] = '[IMAGE_SHA256:' + hashlib.sha256(url.encode()).hexdigest() + ']'
    return value


def build(case, variant, provider):
    observer = CaptureObserver(provider)
    try:
        observer.observe_with_decision(frames=case['frames'], goal_context=case['context'],
            device_id=case['device_id'], available_action_kinds=case['allowed'])
    except Captured:
        pass
    messages, schema = observer.captured
    width, height = _image_request_size(case['frames'][-1])
    props = schema['json_schema']['schema']['properties']
    options = dict(request_width=width, request_height=height,
        image_count=sum(p['type'] == 'image_url' for p in messages[1]['content']),
        available_action_kinds=tuple(k for k in props['decision']['properties']['action']['enum'] if k),
        include_input_structure=props['input_structure']['type'] != 'null', bind_result=False)
    original = candidate_request(case['context'], **options)
    assert original['prompt'] == messages[1]['content'][0]['text'], 'Original prompt drift'
    assert original['response_format'] == schema, 'Original schema drift'
    selected = candidate_request(case['context'], uniform_xy=variant == 'uniform', **options)
    messages[1]['content'][0]['text'] = selected['prompt']
    body = dict(model=provider.model, temperature=0.0, messages=messages,
        response_format=selected['response_format'], **provider.model_config.request_options())
    return body, (width, height), options['include_input_structure']


def preflight():
    assert not list(OUT.glob('*_attempt.json')), 'Campaign already started'
    provider = DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    hashes, fixtures = {}, {}
    for name in CASES:
        case = fixture(name)
        pair = []
        for variant in VARIANTS:
            body, _, _ = build(case, variant, provider)
            wire = redacted_wire(body)
            pair.append(wire)
            key = name + '_' + variant
            hashes[key] = digest(wire)
            save(key + '_wire.json', wire)
        assert pair[0]['messages'][0] == pair[1]['messages'][0]
        assert pair[0]['messages'][1]['content'][1:] == pair[1]['messages'][1]['content'][1:]
        for key in ('model', 'temperature', 'enable_thinking'):
            assert pair[0][key] == pair[1][key]
        fixtures[name] = {k: case[k] for k in ('source_record', 'frame_paths', 'modified_goal')}
    save('preflight.json', dict(wire_hashes=hashes, fixtures=fixtures, max_calls=8,
        model_config=model_config(provider), production_hashes=production_hashes(),
        only_xy_contract_changed=True, phone_actions=0,
        stop_condition='Stop after a completed pair if uniform still misses or regresses; no resampling.'))
    print(json.dumps(dict(prepared_pairs=4, remote_calls=0, phone_actions=0)))


def reserve(name):
    assert name in {c + '_' + v for c in CASES for v in VARIANTS}
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert len(list(OUT.glob('*_attempt.json'))) < 8, 'Eight-call authorization exhausted'
    # Exclusive creation prevents a command retry from making a second model call.
    with (OUT / (name + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump(dict(authorization='2026-09-07 user approved at most eight recognition calls',
            started_at=datetime.now().astimezone().isoformat(), max_attempts=1, phone_actions=0), stream)


def parse_reply(raw, *, uniform, actual_size, include_input):
    # No image is resized here. The production pure parser takes the wire Y extent
    # via request_image_size; for this isolated candidate that extent is 1000.
    parse_size = (actual_size[0], 1000 if uniform else actual_size[1])
    return _parse_single_step_observation_envelope(raw,
        input_structure_required=include_input, request_image_size=parse_size)


def recognize(name, variant):
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    pre = read(OUT / 'preflight.json')
    assert production_hashes() == pre['production_hashes'], 'Production source drift'
    provider = DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    assert provider.configured, 'Credential unavailable'
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert model_config(provider) == pre['model_config'], 'Model settings drift'
    case = fixture(name)
    body, size, include_input = build(case, variant, provider)
    key = name + '_' + variant
    assert digest(redacted_wire(body)) == pre['wire_hashes'][key], 'Frozen request drift'
    reserve(key)
    started = time.perf_counter()
    result = dict(case=name, variant=variant, actual_request_image_size=size, phone_actions=0)
    try:
        raw = provider._chat(body['messages'], max_tokens=None, timeout=60.0, max_attempts=1,
            response_format=body['response_format'])
        save(key + '_raw.json', dict(raw=raw))
        parsed = parse_reply(raw, uniform=variant == 'uniform', actual_size=size, include_input=include_input)
        point = parsed['decision'].get('tap_point')
        result.update(parse_succeeded=True, parsed=parsed,
            frame_point=([round(point[0] * (case['frames'][-1].width - 1) / 1000),
                          round(point[1] * (case['frames'][-1].height - 1) / 1000)] if point else None))
    except Exception as exc:
        result.update(parse_succeeded=False, error_type=type(exc).__name__, error=str(exc))
    result.update(seconds=round(time.perf_counter() - started, 3), usage=provider.last_usage,
        network_attempts=provider.last_network_attempts, response_model=provider.last_response_model)
    save(key + '_result.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('prepare', 'recognize'))
    cli.add_argument('--case', choices=CASES)
    cli.add_argument('--variant', choices=VARIANTS)
    cli.add_argument('--allow-remote', action='store_true')
    args = cli.parse_args()
    if args.mode == 'prepare':
        preflight()
    elif args.allow_remote and args.case and args.variant:
        recognize(args.case, args.variant)
    else:
        cli.error('Explicit --allow-remote, case, and variant required')
