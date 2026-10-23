"""One approved saved-image diagnosis; capture response body, never device actions."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_grounding_isolation as base
from agent.infrastructure import dashscope_vision_provider as transport

OUT = POC / 'output/full_response_diagnosis_20260907'
SOURCE = POC / 'output/web/generic_supervised_20260907_230957_7175680c'
RECORD = SOURCE / '8bc1ca22779946409ef1e6663247df9f_model_response.json'


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def build(provider):
    record = base.read(RECORD)
    paths = [SOURCE / f"{record['response_evidence_prefix']}_{i}.jpg" for i in range(1, 5)]
    frames = []
    for path in paths:
        with base.Image.open(path) as image:
            frames.append(image.convert('RGB'))
    case = dict(context=record['goal_context'], frames=frames, device_id=record['device_id'],
        allowed=base.read(SOURCE / 'model_binding_step_1.json')['available_action_kinds'])
    if case['context']['entities']['history'] or case['allowed'] != ['tap_semantic']:
        raise RuntimeError('Unexpected source task')
    body, size, include_input = base.build(case, 'full', provider)
    return body, size, include_input, frames[-1].size


def fingerprints():
    paths = [RECORD, SOURCE / 'model_binding_step_1.json', Path(__file__),
        POC / 'experiments/probe_grounding_isolation.py', POC / 'experiments/probe_uniform_xy.py']
    paths.extend(SOURCE.glob('before_step_1_frame_attempt_1_*.jpg'))
    return {str(path.relative_to(POC)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def reserve():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': datetime.now().astimezone().isoformat(), 'max_calls': 1,
            'max_attempts': 1, 'phone_actions': 0}, stream)


def capture_chat(provider, body):
    original_post = transport.httpx.post
    calls = 0

    def capture(url, **kwargs):
        nonlocal calls
        if calls or url != provider.base_url + '/chat/completions' or kwargs['json'] != body:
            raise RuntimeError('Unexpected or repeated wire request')
        calls += 1
        response = original_post(url, **kwargs)
        # Save only response bytes, never request headers or authentication.
        base.atomic_replace_bytes(OUT / 'response_body.json', response.content)
        save('response_metadata.json', {'status_code': response.status_code, 'calls': calls})
        return response

    with patch.object(transport.httpx, 'post', capture):
        return provider._chat(body['messages'], max_tokens=None, timeout=60,
            max_attempts=1, response_format=body['response_format'])


def prepare():
    if (OUT / 'preflight.json').exists() or (OUT / 'attempt.json').exists():
        raise RuntimeError('Do not reset this campaign')
    provider = base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    body, size, _, original = build(provider)
    wire = base.redacted_wire(body)
    save('wire.json', wire)
    save('preflight.json', dict(model_config=base.model_config(provider), wire_hash=base.digest(wire),
        production_hashes=base.production_hashes(), fingerprints=fingerprints(), size=size,
        original_size=original, max_calls=1, max_attempts=1, phone_actions=0,
        limitation='Current production request rebuilt from saved JPEGs and task, not recovered historical wire bytes.',
        authorization='One isolated Qwen recognition with complete response capture; no retry or device actions.',
        offline_region=base.REGIONS['successful_send']))
    print(json.dumps({'prepared_calls': 1, 'remote_calls': 0}))


def recognize():
    pre = base.read(OUT / 'preflight.json')
    if fingerprints() != pre['fingerprints'] or base.production_hashes() != pre['production_hashes']:
        raise RuntimeError('Frozen code or input drift')
    provider = base.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    if not provider.configured or provider.model != 'qwen3-vl-plus' or provider.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise RuntimeError('Unexpected endpoint/model or missing credentials')
    body, size, include_input, original = build(provider)
    if base.model_config(provider) != pre['model_config'] or base.digest(base.redacted_wire(body)) != pre['wire_hash']:
        raise RuntimeError('Frozen request drift')
    reserve()
    result = dict(phone_actions=0, transport_succeeded=False, parse_succeeded=False)
    try:
        raw = capture_chat(provider, body)
        result['transport_succeeded'] = True
        save('content.json', {'raw': raw})
        payload = base.read(OUT / 'response_body.json')
        reasoning = payload['choices'][0]['message'].get('reasoning_content')
        result['reasoning_characters'] = len(reasoning) if isinstance(reasoning, str) else 0
        point, parsed = base.parse_reply(raw, 'full', size, include_input)
        frame = [round(point[0]*(original[0]-1)/1000), round(point[1]*(original[1]-1)/1000)] if point else None
        save('parsed.json', parsed)
        result.update(parse_succeeded=True, canonical_point=point, frame_point=frame,
            inside_target_region=base.in_region(frame, pre['offline_region']))
    except Exception as exc:
        # Do not serialize arbitrary exception text which might contain request data.
        result['error_type'] = type(exc).__name__
    finally:
        result.update(usage=provider.last_usage, network_attempts=provider.last_network_attempts,
            response_id=provider.last_request_id, finish_reason=provider.last_finish_reason,
            production_unchanged=base.production_hashes() == pre['production_hashes'])
        save('result.json', result)
        save('stopped.json', {'calls_reserved': 1, 'resume_allowed': False, 'phone_actions': 0})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize'))
    parser.add_argument('--allow-remote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote:
        recognize()
    else:
        parser.error('Requires explicit --allow-remote')
