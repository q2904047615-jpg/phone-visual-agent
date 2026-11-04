"""Two-call saved-image old/new prompt comparison, no device or production changes."""
import argparse
from copy import deepcopy
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_grounding_isolation as base
from experiments import probe_full_response as capture
from agent.infrastructure import generic_scene_observer as observer

OUT = POC / 'output/prompt_delta_comparison_20260908'
OLD = POC / 'output/recent_navigation_20260907/before/poc/agent/infrastructure/prompts/single_step_observation.txt'
ORDER = ('old', 'current')


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def build(provider, variant):
    if variant not in ORDER:
        raise ValueError('Unknown prompt variant')
    case = base.fixture('failed_send')
    original = observer._prompt_template
    def template(name):
        return OLD.read_text(encoding='utf-8') if variant == 'old' and name == 'single_step_observation.txt' else original(name)
    with patch.object(observer, '_prompt_template', side_effect=template):
        body, size, required = base.build(case, 'full', provider)
    return body, size, required, case


def hashes():
    case = base.fixture('failed_send')
    paths = [Path(__file__), Path(capture.__file__), OLD, POC / 'experiments/probe_grounding_isolation.py',
        POC / 'experiments/probe_uniform_xy.py', Path(case['source_record']),
        Path(case['source_record']).parent / 'model_binding_step_3.json']
    paths.extend(Path(path) for path in case['frame_paths'])
    return {str(path.relative_to(POC)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare():
    if (OUT / 'preflight.json').exists() or list(OUT.glob('*_attempt.json')):
        raise RuntimeError('Campaign already frozen')
    provider = base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    wires = {}
    for variant in ORDER:
        body, size, _, case = build(provider, variant)
        wires[variant] = base.redacted_wire(body)
    a, b = deepcopy(wires['old']), deepcopy(wires['current'])
    old_prompt = a['messages'][1]['content'][0].pop('text')
    new_prompt = b['messages'][1]['content'][0].pop('text')
    if a != b or old_prompt == new_prompt:
        raise RuntimeError('Expected only prompt text to differ')
    for variant in ORDER:
        save(variant + '_wire.json', wires[variant])
    delta = '\n'.join(difflib.unified_diff(old_prompt.splitlines(), new_prompt.splitlines(),
        fromfile='before_recent_navigation', tofile='current', n=2))
    base.atomic_replace_bytes(OUT / 'prompt.diff', delta.encode('utf-8'))
    save('preflight.json', dict(created_at=datetime.now().astimezone().isoformat(), max_calls=2,
        max_attempts_per_call=1, order=ORDER, model_config=base.model_config(provider),
        production_hashes=base.production_hashes(), fingerprints=hashes(),
        wire_hashes={k: base.digest(v) for k,v in wires.items()}, source=case['source_record'],
        frame_paths=case['frame_paths'], image_size=size, history_count=len(case['context']['entities']['history']),
        prompt_characters={'old': len(old_prompt), 'current': len(new_prompt)},
        offline_region=base.REGIONS['failed_send'], phone_actions=0,
        authorization='User requested the proposed fixed-image/task/history/model old/new prompt investigation.',
        stop='One attempt per variant; malformed response is retained for comparison, transport failure stops all; no retries.',
        limitations='Saved JPEG reconstruction, not recovered historical wire. Only outer prompt version differs; one pair cannot establish stable causality or certify rollback. Old prompt used only in isolated inference.'))
    print(json.dumps({'prepared_calls': 2, 'remote_calls': 0}))


def reserve(variant):
    if (OUT / 'stopped.json').exists():
        raise RuntimeError('Campaign stopped')
    if variant == 'current':
        prior = OUT / 'old_result.json'
        if not prior.exists() or not base.read(prior)['transport_succeeded']:
            raise RuntimeError('Previous call unresolved or failed')
    with (OUT / (variant + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump({'started_at': datetime.now().astimezone().isoformat(), 'max_attempts': 1}, stream)


def recognize(variant):
    pre = base.read(OUT / 'preflight.json')
    if hashes() != pre['fingerprints'] or base.production_hashes() != pre['production_hashes']:
        raise RuntimeError('Frozen code/input drift')
    provider = base.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    if not provider.configured or provider.model != 'qwen3-vl-plus' or provider.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise RuntimeError('Unexpected endpoint/model or unavailable credentials')
    body, size, required, case = build(provider, variant)
    if base.model_config(provider) != pre['model_config'] or base.digest(base.redacted_wire(body)) != pre['wire_hashes'][variant]:
        raise RuntimeError('Frozen request drift')
    reserve(variant)
    result = dict(variant=variant, transport_succeeded=False, parse_succeeded=False, phone_actions=0)
    try:
        with patch.object(capture, 'OUT', OUT / variant):
            raw = capture.capture_chat(provider, body)
        result['transport_succeeded'] = True
        save(variant + '_raw.json', {'raw': raw})
        point, parsed = base.parse_reply(raw, 'full', size, required)
        original = case['frames'][-1].size
        frame = [round(point[0]*(original[0]-1)/1000), round(point[1]*(original[1]-1)/1000)] if point else None
        save(variant + '_parsed.json', parsed)
        result.update(parse_succeeded=True, canonical_point=point, frame_point=frame,
            inside_target_region=base.in_region(frame, pre['offline_region']))
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    finally:
        result.update(usage=provider.last_usage, network_attempts=provider.last_network_attempts,
            response_id=provider.last_request_id, finish_reason=provider.last_finish_reason,
            production_unchanged=base.production_hashes() == pre['production_hashes'])
        save(variant + '_result.json', result)
        if variant == 'current' or not result['transport_succeeded']:
            used = len(list(OUT.glob('*_attempt.json')))
            save('stopped.json', {'calls_reserved': used, 'remaining_calls_cancelled': 2-used,
                'resume_allowed': False, 'phone_actions': 0})
    print(json.dumps(result, ensure_ascii=False))


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
        parser.error('Recognition requires --allow-remote and --variant')
