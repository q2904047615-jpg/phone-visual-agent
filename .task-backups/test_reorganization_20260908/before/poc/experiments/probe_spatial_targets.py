"""Four saved-image target localizations; no device access or production changes."""
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
from experiments import probe_image_pair as images
from experiments import probe_grounding_isolation as base
from experiments import probe_full_response as capture

OUT = POC / 'output/spatial_targets_20260908'
TARGETS = {
    'upper_avatar': '消息“18766”右侧的灰色人形头像',
    'middle_avatar': '以“VXIE6s_Tu_”开头的绿色消息气泡右侧的灰色人形头像',
    'voice': '输入栏左侧圆形语音切换图标',
    'send': '绿色发送按钮',
}
ORDER = tuple(TARGETS)
# Human-inspected visible icon extents, original 810x1440; never sent to model.
REGIONS = {
    'upper_avatar': [677, 160, 725, 203],
    'middle_avatar': [678, 638, 727, 682],
    'voice': [82, 1230, 132, 1281],
    'send': [641, 1228, 745, 1283],
}


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def build(provider, name):
    label = TARGETS[name]
    case = images.fixture('current')
    case['context']['objective'] = (f'仅定位当前截图中的目标：{label}。'
        '给出目标自身可点击区域内的明确点击点。只生成位置建议，不点击、不输入、不发送消息；目标不可见时如实说明。')
    case['context']['entities']['task_id'] = 'isolated-spatial-targets-20260908'
    case['context']['entities']['exact_target_label'] = label
    body, size, required = base.build(case, 'full', provider)
    return body, size, required, case


def neutral(body, name):
    value = deepcopy(body)
    value['messages'][1]['content'][0]['text'] = value['messages'][1]['content'][0]['text'].replace(TARGETS[name], '[TARGET]')
    return value


def hashes():
    case = images.fixture('current')
    paths = [Path(__file__), Path(images.__file__), Path(base.__file__), Path(capture.__file__),
        POC / 'experiments/probe_uniform_xy.py', capture.RECORD, Path(case['source_record'])]
    paths.extend(Path(p) for p in case['frame_paths'])
    return {str(p.relative_to(POC)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare():
    if (OUT / 'preflight.json').exists() or list(OUT.glob('*_attempt.json')):
        raise RuntimeError('Campaign already frozen')
    provider = base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    wires, invariant = {}, None
    for name in ORDER:
        body, size, required, case = build(provider, name)
        candidate = neutral(body, name)
        if invariant is not None and candidate != invariant:
            raise RuntimeError('Unexpected non-target request difference')
        invariant = candidate
        if size != (720, 1280) or not required:
            raise RuntimeError('Unexpected request shape')
        wires[name] = base.redacted_wire(body)
        save(name + '_wire.json', wires[name])
    save('preflight.json', dict(created_at=datetime.now().astimezone().isoformat(), max_calls=4,
        max_attempts=1, order=ORDER, targets=TARGETS, regions=REGIONS,
        model_config=base.model_config(provider), fingerprints=hashes(), production_hashes=base.production_hashes(),
        wire_hashes={k: base.digest(v) for k, v in wires.items()}, image_size=size,
        source=case['source_record'], frame_paths=case['frame_paths'], history_count=0, phone_actions=0,
        authorization='User approved spatially distributed icon comparison on the same saved image, localization only.',
        stop='Four single attempts; retain misses; transport failure or two consecutive malformed responses stop. No retries.',
        limitation='One sample per target, visible icon extents not actual touch hitboxes; cannot establish success rate or physical contact.'))
    print(json.dumps({'prepared_calls':4, 'remote_calls':0, 'only_target_description_differs':True}))


def reserve(name):
    if name not in ORDER or (OUT / 'stopped.json').exists():
        raise RuntimeError('Unknown target or stopped campaign')
    index = len(list(OUT.glob('*_attempt.json')))
    if index >= 4 or ORDER[index] != name:
        raise RuntimeError('Duplicate, exhausted or out-of-order request')
    if index:
        prior = OUT / (ORDER[index-1] + '_result.json')
        if not prior.exists() or not base.read(prior)['transport_succeeded']:
            raise RuntimeError('Prior request unresolved or failed')
    with (OUT / (name + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump({'started_at':datetime.now().astimezone().isoformat(), 'max_attempts':1}, stream)


def recognize(name):
    pre = base.read(OUT / 'preflight.json')
    if hashes() != pre['fingerprints'] or base.production_hashes() != pre['production_hashes']:
        raise RuntimeError('Frozen source drift')
    provider = base.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    if not provider.configured or provider.model != 'qwen3.7-plus' or provider.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise RuntimeError('Unconfigured or unexpected provider')
    body, size, required, case = build(provider, name)
    if base.model_config(provider) != pre['model_config'] or base.digest(base.redacted_wire(body)) != pre['wire_hashes'][name]:
        raise RuntimeError('Frozen request drift')
    reserve(name)
    result = dict(target=name, transport_succeeded=False, parse_succeeded=False, phone_actions=0)
    try:
        with patch.object(capture, 'OUT', OUT / name):
            raw = capture.capture_chat(provider, body)
        result['transport_succeeded'] = True
        save(name + '_raw.json', {'raw':raw})
        point, parsed = base.parse_reply(raw, 'full', size, required)
        original = case['frames'][-1].size
        frame = [round(point[0]*(original[0]-1)/1000), round(point[1]*(original[1]-1)/1000)] if point else None
        save(name + '_parsed.json', parsed)
        bounds = pre['regions'][name]
        center = [(bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2]
        result.update(parse_succeeded=True, canonical_point=point, frame_point=frame,
            reference_center=center, center_delta=[round(frame[i]-center[i],1) for i in (0,1)] if frame else None,
            inside_visible_region=base.in_region(frame, {'kind':'rectangle','bounds':bounds}))
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    finally:
        result.update(usage=provider.last_usage, network_attempts=provider.last_network_attempts,
            response_id=provider.last_request_id, response_model=provider.last_response_model,
            production_unchanged=base.production_hashes()==pre['production_hashes'])
        save(name + '_result.json', result)
        index = ORDER.index(name)
        repeated_invalid = index > 0 and not result['parse_succeeded'] and not base.read(OUT / (ORDER[index-1]+'_result.json'))['parse_succeeded']
        if index == 3 or not result['transport_succeeded'] or repeated_invalid:
            save('stopped.json', {'calls_reserved':index+1,'remaining_calls_cancelled':3-index,'resume_allowed':False,'phone_actions':0})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('prepare','recognize'))
    cli.add_argument('--target', choices=ORDER)
    cli.add_argument('--allow-remote', action='store_true')
    args = cli.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.target:
        recognize(args.target)
    else:
        cli.error('Requires --target and --allow-remote')
