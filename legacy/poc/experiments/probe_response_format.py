"""Approved two-call parameter isolation on saved images; no device/API actions."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_grounding_isolation as base

OUT = POC / 'output/response_format_isolation_20260907'
REFERENCE = POC / 'output/grounding_isolation_20260907/approved_supplement_6/successful_send_grounding_wire.json'
ORDER = ('schema', 'omitted')


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def code_hashes():
    return {**base.code_hashes(), str(Path(__file__).relative_to(POC)):
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def build(provider, variant):
    if variant not in ORDER:
        raise ValueError('Unknown variant')
    case = base.fixture('successful_send')
    body, size, include_input = base.build(case, 'grounding', provider)
    if variant == 'omitted':
        del body['response_format']
    return body, size, include_input, case


def prepare():
    if (OUT / 'preflight.json').exists() or list(OUT.glob('*_attempt.json')):
        raise RuntimeError('Already frozen; do not reset campaign')
    provider = base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    bodies = {}
    for variant in ORDER:
        body, size, _, case = build(provider, variant)
        bodies[variant] = base.redacted_wire(body)
    if bodies['schema'] != base.read(REFERENCE):
        raise RuntimeError('Baseline differs from the actual schema-echo request')
    without_format = deepcopy(bodies['schema'])
    del without_format['response_format']
    if without_format != bodies['omitted']:
        raise RuntimeError('More than response_format changed')
    for variant, wire in bodies.items():
        save(variant + '_wire.json', wire)
    save('preflight.json', {
        'created_at': datetime.now().astimezone().isoformat(),
        'max_calls': 2, 'order': ORDER, 'max_attempts_per_call': 1,
        'authorization': 'User approved one schema and one omitted-format saved-image recognition; no phone actions.',
        'wire_hashes': {key: base.digest(value) for key, value in bodies.items()},
        'reference': str(REFERENCE), 'model_config': base.model_config(provider),
        'production_hashes': base.production_hashes(), 'code_hashes': code_hashes(),
        'image_size': size, 'frame_paths': case['frame_paths'],
        'offline_scoring_region': base.REGIONS['successful_send'],
        'only_variable': 'Presence versus absence of top-level response_format; thinking stays enabled.',
        'limitations': 'One paired sample, no causal certainty or cross-control/device acceptance. Region never sent to model.',
        'stop': 'No retries. A received malformed answer is a diagnostic result, so still run the other variant. Stop on transport failure or after two calls.',
        'phone_actions': 0, 'production_switch': False})
    print(json.dumps({'prepared_calls': 2, 'remote_calls': 0, 'phone_actions': 0}))


def reserve(variant):
    if (OUT / 'stopped.json').exists():
        raise RuntimeError('Campaign stopped')
    index = len(list(OUT.glob('*_attempt.json')))
    if index >= 2 or ORDER[index] != variant:
        raise RuntimeError('Duplicate, out-of-order or exhausted budget')
    if index:
        previous = OUT / (ORDER[index-1] + '_result.json')
        if not previous.exists() or not base.read(previous).get('transport_succeeded'):
            raise RuntimeError('Previous call unresolved or transport failed')
    with (OUT / f'{index+1:02d}_{variant}_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': datetime.now().astimezone().isoformat(), 'max_attempts': 1}, stream)


def chat(provider, body):
    # Direct provider call: observer._provider_chat would substitute json_object
    # for None, which would invalidate the omitted-field comparison.
    return provider._chat(body['messages'], max_tokens=None, timeout=60.0,
        max_attempts=1, response_format=body.get('response_format'))


def recognize(variant):
    pre = base.read(OUT / 'preflight.json')
    if base.production_hashes() != pre['production_hashes'] or code_hashes() != pre['code_hashes']:
        raise RuntimeError('Frozen code drift')
    provider = base.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    if not provider.configured or provider.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise RuntimeError('Approved endpoint or credential unavailable')
    if base.model_config(provider) != pre['model_config']:
        raise RuntimeError('Model configuration drift')
    body, size, include_input, case = build(provider, variant)
    if base.digest(base.redacted_wire(body)) != pre['wire_hashes'][variant]:
        raise RuntimeError('Frozen wire drift')
    reserve(variant)
    started = time.perf_counter()
    result = {'variant': variant, 'phone_actions': 0, 'transport_succeeded': False,
        'parse_succeeded': False, 'inside_target_region': None}
    try:
        raw = chat(provider, body)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    else:
        # Preserve returned content before interpretation. No schema constants
        # or offline scoring bounds can be used to synthesize an answer.
        save(variant + '_raw.json', {'raw': raw})
        result['transport_succeeded'] = True
        try:
            point, parsed = base.parse_reply(raw, 'grounding', size, include_input)
            frame = ([round(point[0]*(case['frames'][-1].width-1)/1000),
                round(point[1]*(case['frames'][-1].height-1)/1000)] if point else None)
            result.update(parse_succeeded=True, parsed=parsed, canonical_point=point,
                frame_point=frame, inside_target_region=base.in_region(frame, pre['offline_scoring_region']))
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
    result.update(seconds=round(time.perf_counter()-started, 3), usage=provider.last_usage,
        network_attempts=provider.last_network_attempts, response_model=provider.last_response_model,
        provider_response_id=provider.last_request_id, finish_reason=provider.last_finish_reason)
    save(variant + '_result.json', result)
    used = len(list(OUT.glob('*_attempt.json')))
    if not result['transport_succeeded'] or used == 2:
        save('stopped.json', {'reason': 'Two-call comparison complete' if used == 2 else 'Transport failure',
            'calls_used': used, 'remaining_calls_cancelled': 2-used, 'resume_allowed': False,
            'phone_actions': 0, 'production_switch': False})
    print(json.dumps({key: value for key, value in result.items() if key != 'parsed'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize'))
    parser.add_argument('--variant', choices=ORDER)
    parser.add_argument('--allow-remote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.variant:
        recognize(args.variant)
    else:
        parser.error('Recognition requires explicit --allow-remote and --variant')
