"""Approved at most four saved-image calls: three frames versus the last frame."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_full_response as capture
from experiments import probe_grounding_isolation as base

OUT = POC / 'output/frame_count_comparison_20260908'
ORDER = ('send_three', 'send_single', 'recent_three', 'recent_single')
TEMPORAL_THREE = '共有3张同一稳定手机画面的时间对齐帧。只把它们合并为一个当前状态；闪烁光标可从任一帧读取，其他瞬态不得合并。'
TEMPORAL_SINGLE = '只有一张当前稳定手机画面。'


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def single_frame(body):
    result = deepcopy(body)
    parts = result['messages'][1]['content']
    if len(parts) != 7 or parts[0]['text'].count(TEMPORAL_THREE) != 1:
        raise RuntimeError('Unexpected three-frame structure')
    if [p['type'] for p in parts] != ['text', 'text', 'image_url', 'text', 'image_url', 'text', 'image_url']:
        raise RuntimeError('Unexpected image ordering')
    parts[0]['text'] = parts[0]['text'].replace(TEMPORAL_THREE, TEMPORAL_SINGLE)
    result['messages'][1]['content'] = [parts[0],
        {'type': 'text', 'text': 'IMAGE 1 - CURRENT STABLE PHONE SURFACE'}, parts[-1]]
    return result


def build(provider, key):
    if key not in ORDER:
        raise ValueError('Unknown case')
    if key.startswith('send_'):
        body, size, required, original = capture.build(provider)
    else:
        case = base.fixture('recent_clear')
        body, size, required = base.build(case, 'full', provider)
        original = case['frames'][-1].size
    if key.endswith('_single'):
        body = single_frame(body)
    return body, size, required, original


def hashes():
    result = capture.fingerprints()
    case = base.fixture('recent_clear')
    paths = [Path(__file__), Path(capture.__file__), Path(case['source_record']),
        Path(case['source_record']).parent / 'model_binding_step_3.json']
    paths.extend(Path(p) for p in case['frame_paths'])
    for path in paths:
        result[str(path.relative_to(POC))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def prepare():
    if (OUT / 'preflight.json').exists() or list(OUT.glob('*_attempt.json')):
        raise RuntimeError('Campaign already frozen')
    provider = base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    wires = {}
    for key in ORDER:
        body, size, _, original = build(provider, key)
        wires[key] = base.redacted_wire(body)
        save(key + '_wire.json', wires[key])
    for case in ('send', 'recent'):
        if single_frame(wires[case + '_three']) != wires[case + '_single']:
            raise RuntimeError('Unexpected paired difference')
    save('preflight.json', dict(created_at=datetime.now().astimezone().isoformat(),
        order=ORDER, max_calls=4, max_attempts_per_call=1,
        authorization='User approved at most four saved-image recognitions, no phone actions or retry.',
        stop='Stop on transport failure, unresolved attempt, any single-frame miss/nonpoint/parse failure, or four calls.',
        model_config=base.model_config(provider), wire_hashes={k: base.digest(v) for k,v in wires.items()},
        production_hashes=base.production_hashes(), fingerprints=hashes(),
        regions={'send': base.REGIONS['successful_send'], 'recent': base.REGIONS['recent_clear']},
        limitations='Pairs differ only in frame count, corresponding temporal description and image numbering. Last stable JPEG is selected independently of outcome. Saved JPEG reconstruction is not historical wire recovery. One pair cannot prove causal reliability or focus safety.',
        phone_actions=0, production_switch=False))
    print(json.dumps({'prepared_calls': 4, 'remote_calls': 0}))


def reserve(key):
    if (OUT / 'stopped.json').exists():
        raise RuntimeError('Campaign stopped')
    for prior in ORDER[:ORDER.index(key)]:
        if not (OUT / (prior + '_result.json')).exists():
            raise RuntimeError('Missing or unresolved previous attempt')
        previous = base.read(OUT / (prior + '_result.json'))
        if not previous['transport_succeeded'] or (prior.endswith('_single') and not previous.get('inside_target_region')):
            raise RuntimeError('Previous result requires stop')
    with (OUT / (key + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump({'started_at': datetime.now().astimezone().isoformat(), 'max_attempts': 1}, stream)


def recognize(key):
    pre = base.read(OUT / 'preflight.json')
    if hashes() != pre['fingerprints'] or base.production_hashes() != pre['production_hashes']:
        raise RuntimeError('Frozen source drift')
    provider = base.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    if not provider.configured or provider.model != 'qwen3-vl-plus' or provider.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise RuntimeError('Unexpected endpoint/model or unavailable credentials')
    body, size, required, original = build(provider, key)
    if base.model_config(provider) != pre['model_config'] or base.digest(base.redacted_wire(body)) != pre['wire_hashes'][key]:
        raise RuntimeError('Frozen request drift')
    reserve(key)
    result = dict(key=key, transport_succeeded=False, parse_succeeded=False, phone_actions=0,
        inside_target_region=False)
    try:
        with patch.object(capture, 'OUT', OUT / key):
            raw = capture.capture_chat(provider, body)
        result['transport_succeeded'] = True
        save(key + '_raw.json', {'raw': raw})
        point, parsed = base.parse_reply(raw, 'full', size, required)
        frame = [round(point[0]*(original[0]-1)/1000), round(point[1]*(original[1]-1)/1000)] if point else None
        save(key + '_parsed.json', parsed)
        result.update(parse_succeeded=True, canonical_point=point, frame_point=frame,
            inside_target_region=base.in_region(frame, pre['regions'][key.split('_')[0]]))
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    finally:
        result.update(network_attempts=provider.last_network_attempts, usage=provider.last_usage,
            response_id=provider.last_request_id, finish_reason=provider.last_finish_reason,
            production_unchanged=base.production_hashes() == pre['production_hashes'])
        save(key + '_result.json', result)
        stop = (not result['transport_succeeded'] or (key.endswith('_single') and not result['inside_target_region'])
            or key == ORDER[-1])
        if stop:
            used = len(list(OUT.glob('*_attempt.json')))
            save('stopped.json', {'calls_reserved': used, 'remaining_calls_cancelled': 4-used,
                'reason': 'Single-frame failure' if key.endswith('_single') and not result['inside_target_region'] else 'Transport failure or campaign complete',
                'resume_allowed': False, 'phone_actions': 0})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize'))
    parser.add_argument('--key', choices=ORDER)
    parser.add_argument('--allow-remote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.key:
        recognize(args.key)
    else:
        parser.error('Recognition requires --allow-remote and --key')
