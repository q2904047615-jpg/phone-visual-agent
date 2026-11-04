"""Approved isolated A/B: move task text after unchanged current image blocks."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_minimal_wire_grounding as base
from agent.infrastructure import dashscope_vision_provider as transport
from agent.infrastructure.generic_scene_observer import _parse_single_step_observation_envelope

OUT = POC / 'output/content_order_20260908'
SOURCE = base.SOURCE
ORDER = (('send', False), ('send', True), ('voice', False), ('voice', True))


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def build(target, image_first):
    body = base.read(SOURCE)
    parts = body['messages'][1]['content']
    assert len(parts) == 7 and parts[0]['type'] == 'text'
    assert [p['type'] for p in parts[1:]] == ['text', 'image_url'] * 3
    if target == 'voice':
        lines = parts[0]['text'].splitlines(keepends=True)
        prefix = '当前任务与实际执行历史：'
        found = 0
        for n, line in enumerate(lines):
            if line.startswith(prefix):
                context = json.loads(line[len(prefix):])
                assert context['entities']['history'] == []
                context['objective'] = '定位当前输入栏左侧的圆形语音切换图标，并给出一次点击动作。'
                context['entities']['exact_target_label'] = '语音切换图标'
                lines[n] = prefix + json.dumps(context, ensure_ascii=False, separators=(',', ':')) + '\n'
                found += 1
        assert found == 1
        parts[0]['text'] = ''.join(lines)
    else:
        assert target == 'send'
    if image_first:
        body['messages'][1]['content'] = parts[1:] + parts[:1]
    return body


def score(point, target):
    # Offline assessment only: original wire X 0..1000, Y 0..1280.
    xy = [point[0] * 719 / 1000, point[1] * 1279 / 1280]
    left, top, right, bottom = base.REGIONS[target]
    return {'image_point': xy,
            'inside_visible_region': left <= xy[0] <= right and top <= xy[1] <= bottom}


def evaluate(raw, target):
    wire = transport._extract_json_object(raw, reject_duplicate_keys=True,
                                         unwrap_singleton_object_array=True)
    envelope = _parse_single_step_observation_envelope(raw, input_structure_required=True,
                                                      request_image_size=(720, 1280))
    decision = envelope['decision']
    if decision['status'] != 'action' or decision['action'] != 'tap_semantic':
        raise ValueError('Expected one tap_semantic suggestion')
    point = wire['decision']['tap_point']
    return {'parse_ok': True, 'raw_point': point, 'canonical_point': decision['tap_point'],
            'target': decision['target'], **score(point, target)}


def improved(a, b):
    return (a.get('parse_ok') is True and b.get('parse_ok') is True
            and a.get('inside_visible_region') is False and b.get('inside_visible_region') is True)


class OfflineTests(unittest.TestCase):
    def test_baseline_is_actual_request(self):
        self.assertEqual(build('send', False), base.read(SOURCE))

    def test_only_prompt_position_changes(self):
        for target in ('send', 'voice'):
            a, b = build(target, False), build(target, True)
            p = b['messages'][1]['content']
            b['messages'][1]['content'] = p[-1:] + p[:-1]
            self.assertEqual(a, b)

    def test_image_bytes_sizes_and_order(self):
        original = [p for p in build('send', False)['messages'][1]['content'] if p['type'] == 'image_url']
        for target, first in ORDER:
            images = [p for p in build(target, first)['messages'][1]['content'] if p['type'] == 'image_url']
            self.assertEqual(images, original)
            for p in images:
                data = base64.b64decode(p['image_url']['url'].split(',', 1)[1], validate=True)
                with Image.open(io.BytesIO(data)) as im:
                    self.assertEqual(im.size, (720, 1280))
                    self.assertEqual(im.format, 'JPEG')

    def test_same_protocol_and_settings(self):
        original = build('send', False)
        for target, first in ORDER:
            candidate = build(target, first)
            self.assertEqual({k: v for k, v in candidate.items() if k != 'messages'},
                             {k: v for k, v in original.items() if k != 'messages'})
            self.assertEqual(candidate['messages'][0], original['messages'][0])

    def test_saved_failure_parses_and_misses(self):
        raw = base.read(SOURCE.with_name(SOURCE.name.replace('_model_request', '_model_response')))['redacted_raw_response']
        result = evaluate(raw, 'send')
        self.assertEqual(result['raw_point'], [880, 1170])
        self.assertFalse(result['inside_visible_region'])

    def test_scoring(self):
        self.assertTrue(score([855, 1118], 'send')['inside_visible_region'])
        self.assertFalse(score([880, 1170], 'send')['inside_visible_region'])
        self.assertTrue(score([130, 1118], 'voice')['inside_visible_region'])

    def test_continuation_requires_valid_miss_to_hit(self):
        miss = {'parse_ok': True, 'inside_visible_region': False}
        hit = {'parse_ok': True, 'inside_visible_region': True}
        self.assertTrue(improved(miss, hit))
        for a, b in ((hit, hit), (miss, miss), (hit, miss), ({}, hit), (miss, {})):
            self.assertFalse(improved(a, b))


def prepare():
    if OUT.exists():
        raise RuntimeError('Campaign exists; never reset')
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests))
    if not result.wasSuccessful():
        raise RuntimeError('Offline preflight failed')
    OUT.mkdir()
    bodies = [build(*args) for args in ORDER]
    for i, body in enumerate(bodies, 1):
        save(f'{i}_request.json', body)
    images = [p for p in bodies[0]['messages'][1]['content'] if p['type'] == 'image_url']
    image_hashes = []
    for i, part in enumerate(images, 1):
        data = base64.b64decode(part['image_url']['url'].split(',', 1)[1], validate=True)
        base.atomic_replace_bytes(OUT / f'current_{i}.jpg', data)
        image_hashes.append(hashlib.sha256(data).hexdigest())
    save('preflight.json', {'created_at': base.datetime.now().astimezone().isoformat(),
        'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_hashes': base.hashes(), 'request_hashes': [base.digest(b) for b in bodies],
        'image_hashes': image_hashes, 'regions': base.REGIONS, 'order': ORDER,
        'max_calls': 4, 'max_attempts_per_call': 1, 'phone_actions': 0,
        'change': 'Move only first user text block after all three unchanged labelled image blocks.',
        'stop': 'Transport error; two consecutive contract failures; first pair not valid miss-to-hit; or four calls.',
        'limitation': 'One pair per target is screening, not statistical causality or real-device acceptance.',
        'tests_run': result.testsRun})
    print('PREPARED: tests=7, network_calls=0, phone_actions=0')


def recognize(index):
    pre = base.read(OUT / 'preflight.json')
    if ((OUT / 'stopped.json').exists() or index not in range(1, 5)
            or index != len(list(OUT.glob('*_attempt.json'))) + 1):
        raise RuntimeError('Stopped, repeated, or out of order')
    if index > 1 and not (OUT / f'{index-1}_result.json').exists():
        raise RuntimeError('Previous attempt unresolved')
    if index > 2 and not improved(base.read(OUT / '1_result.json'), base.read(OUT / '2_result.json')):
        raise RuntimeError('First pair did not justify second target')
    assert base.hashes() == pre['production_hashes']
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == pre['source_sha256']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == pre['script_sha256']
    body = base.read(OUT / f'{index}_request.json')
    assert base.digest(body) == pre['request_hashes'][index-1]
    provider = transport.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    assert provider.configured and provider.model == body['model']
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert provider.model_config.request_options() == {'enable_thinking': True}
    with (OUT / f'{index}_attempt.json').open('x', encoding='utf-8') as handle:
        json.dump({'started_at': base.datetime.now().astimezone().isoformat(), 'max_attempts': 1}, handle)
    post = transport.httpx.post
    calls = 0

    def captured_post(url, **kwargs):
        nonlocal calls
        assert calls == 0 and url == provider.base_url + '/chat/completions' and kwargs['json'] == body
        calls += 1
        response = post(url, **kwargs)
        base.atomic_replace_bytes(OUT / f'{index}_response_body.json', response.content)
        save(f'{index}_http.json', {'status_code': response.status_code, 'calls': calls})
        return response

    result = {'index': index, 'target_name': ORDER[index-1][0], 'image_first': ORDER[index-1][1],
              'transport_ok': False, 'parse_ok': False, 'phone_actions': 0}
    started = time.perf_counter()
    try:
        with patch.object(transport.httpx, 'post', captured_post):
            raw = provider._chat(body['messages'], max_tokens=None, timeout=60, max_attempts=1,
                                 response_format=body['response_format'])
        result['transport_ok'] = True
        save(f'{index}_content.json', {'raw': raw})
        result.update(evaluate(raw, ORDER[index-1][0]))
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    finally:
        result.update(elapsed_seconds=round(time.perf_counter()-started, 3),
            network_attempts=provider.last_network_attempts, usage=provider.last_usage,
            response_id=provider.last_request_id, response_model=provider.last_response_model,
            finish_reason=provider.last_finish_reason, production_unchanged=base.hashes() == pre['production_hashes'])
        save(f'{index}_result.json', result)
        reason = None
        if not result['transport_ok']:
            reason = 'transport_failure_no_retry'
        elif index > 1 and not result['parse_ok'] and not base.read(OUT / f'{index-1}_result.json')['parse_ok']:
            reason = 'consecutive_contract_failures'
        elif index == 2 and not improved(base.read(OUT / '1_result.json'), result):
            reason = 'first_pair_no_clear_improvement'
        elif index == 4:
            reason = 'budget_complete'
        if reason:
            save('stopped.json', {'reason': reason, 'calls_reserved': index,
                'remaining_calls_cancelled': 4-index, 'resume_allowed': False, 'phone_actions': 0})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize', 'selftest'))
    parser.add_argument('--index', type=int)
    parser.add_argument('--allow-remote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'selftest':
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests)
        sys.exit(not unittest.TextTestRunner().run(suite).wasSuccessful())
    elif args.allow_remote:
        recognize(args.index)
    else:
        parser.error('Explicit --allow-remote required')
