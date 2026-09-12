"""Two authorized saved-frame calls: identical localization context, images only differ."""
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
from experiments import probe_grounding_isolation as base
from experiments import probe_full_response as capture
from experiments import probe_prompt_delta as driver

OUT = POC / 'output/image_pair_comparison_20260908'
CURRENT = POC / 'output/web/generic_supervised_20260908_002512_af19abb8/cc4f1522529b463f92b690415d34a033_model_response.json'
ORDER = ('old', 'current')  # old successful image, latest failed image; NOT prompt versions


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def fixture(variant):
    if variant not in ORDER:
        raise ValueError('Unknown image variant')
    if variant == 'old':
        case = base.fixture('successful_send')
    else:
        record = base.read(CURRENT)
        paths = [CURRENT.parent / f"{record['response_evidence_prefix']}_{i}.jpg" for i in range(1, 5)]
        frames = []
        for path in paths:
            with base.Image.open(path) as image:
                frames.append(image.convert('RGB'))
        case = dict(frames=frames, frame_paths=[str(p) for p in paths], source_record=str(CURRENT),
            device_id=record['device_id'])
    # Identical prior localization-only task; no old/new message body injected into the goal.
    context = deepcopy(base.read(capture.RECORD)['goal_context'])
    context['entities']['task_id'] = 'isolated-image-pair-20260908'
    if context['entities']['history'] or context['entities']['exact_target_label'] != '发送':
        raise RuntimeError('Unexpected shared localization context')
    case.update(context=context, allowed=['tap_semantic'])
    return case


def build(provider, variant):
    case = fixture(variant)
    body, size, required = base.build(case, 'full', provider)
    return body, size, required, case


def hashes():
    paths = {Path(__file__), Path(driver.__file__), Path(capture.__file__), Path(base.__file__),
        POC / 'experiments/probe_uniform_xy.py', capture.RECORD}
    for variant in ORDER:
        case = fixture(variant)
        paths.add(Path(case['source_record']))
        paths.update(Path(p) for p in case['frame_paths'])
    return {str(p.relative_to(POC)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def without_images(body):
    value = deepcopy(body)
    for part in value['messages'][1]['content']:
        if part.get('type') == 'image_url':
            part['image_url']['url'] = '[SAME_IMAGE_SLOT]'
    return value


def prepare():
    if (OUT / 'preflight.json').exists() or list(OUT.glob('*_attempt.json')):
        raise RuntimeError('Campaign already frozen')
    provider = base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
    wires, cases, bodies = {}, {}, {}
    for variant in ORDER:
        body, size, required, case = build(provider, variant)
        bodies[variant] = body
        wires[variant] = base.redacted_wire(body)
        cases[variant] = {k: case[k] for k in ('source_record', 'frame_paths', 'context')}
        if not required:
            raise RuntimeError('Unexpected dimensions/input contract')
    if without_images(bodies['old']) != without_images(bodies['current']) or wires['old'] == wires['current']:
        raise RuntimeError('Comparison must differ only in image bytes')
    for variant in ORDER:
        save(variant + '_wire.json', wires[variant])
    save('preflight.json', dict(created_at=datetime.now().astimezone().isoformat(), max_calls=2,
        max_attempts_per_call=1, order=ORDER, model_config=base.model_config(provider),
        production_hashes=base.production_hashes(), fingerprints=hashes(),
        wire_hashes={k: base.digest(v) for k, v in wires.items()}, cases=cases,
        image_size=size, history_count=0, offline_region=base.REGIONS['successful_send'], phone_actions=0,
        authorization='User approved identical localization instruction on past successful and latest failed frames, once each; no phone actions.',
        stop='One attempt each; retain malformed/missed results for the pair; transport failure stops; no retry.',
        limitation='Saved images, not live task. Different image contents and frame sets are bundled; one pair cannot establish individual visual feature causality or accuracy rates.'))
    print(json.dumps({'prepared_calls': 2, 'remote_calls': 0, 'only_images_differ': True}))


def recognize(variant):
    if variant not in ORDER:
        raise ValueError('Unknown image variant')
    # Reuse tested two-call budget, exact wire check, full response capture, and terminal stop.
    # Overrides exist only in this isolated process, never in the running project API.
    with patch.object(driver, 'OUT', OUT), patch.object(driver, 'build', build), patch.object(driver, 'hashes', hashes):
        driver.recognize(variant)


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
