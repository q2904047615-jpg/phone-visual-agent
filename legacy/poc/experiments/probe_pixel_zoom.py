"""Four newly authorized saved-image calls; explicit-pixel ROI and crop point."""
import argparse
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import pixel_zoom_candidate as candidate
from experiments import probe_zoom_grounding as previous

base = previous.base
OUT = POC / 'output/pixel_zoom_20260908'
ORDER = ('voice', 'more')


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def frozen():
    paths = (Path(__file__), Path(candidate.__file__), Path(previous.__file__), Path(base.__file__), base.SOURCE)
    return {str(p.relative_to(POC)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def stop(reason):
    save('stopped.json', {'reason': reason, 'reserved_calls': len(list(OUT.glob('*_attempt.json'))),
                         'max_calls': 4, 'resume_allowed': False, 'phone_actions': 0})


def guard(index, pre):
    if (OUT / 'stopped.json').exists() or index not in range(1, 5):
        raise RuntimeError('Stopped or budget exhausted')
    if index != len(list(OUT.glob('*_attempt.json'))) + 1:
        raise RuntimeError('No repeats or out of order calls')
    if frozen() != pre['frozen'] or base.hashes() != pre['production_hashes']:
        raise RuntimeError('Frozen input or production changed')
    if index > 1 and not (OUT / f'{index-1}_result.json').exists():
        raise RuntimeError('Previous attempt unresolved')
    if index % 2 == 0:
        review = base.read(OUT / f'{index-1}_crop_review.json')
        if review.get('complete_target_visible') is not True:
            raise RuntimeError('Crop not visually verified; do not send')


def prepare():
    if OUT.exists():
        raise RuntimeError('Existing campaign must not be reset')
    url, _, sha = previous.source_image()
    OUT.mkdir()
    bodies = [candidate.build_request(url, previous.TARGETS[t], region=True) for t in ORDER]
    for index, body in zip((1, 3), bodies):
        save(f'{index}_request.json', body)
    save('preflight.json', {'created_at': datetime.now().astimezone().isoformat(),
        'authorization': 'User approved new max 4 calls for voice/more ROI+point; no phone/production.',
        'frozen': frozen(), 'production_hashes': base.hashes(), 'jpeg_sha256': sha,
        'roi_request_hashes': [base.digest(b) for b in bodies], 'order': ORDER,
        'max_calls': 4, 'max_attempts': 1, 'phone_actions': 0,
        'offline_reference_regions': {t: previous.REGIONS[t] for t in ORDER},
        'stop': 'Any null/contract/transport failure, cropped target incomplete, final point outside reference, drift or 4 calls.',
        'method': 'Frozen explicit image_pixels candidate. Model selects ROI; fixed 2x Lanczos PNG; independent crop-only point request.',
        'limitations': 'Not official native Qwen coordinate protocol. Two targets once, no success-control or live validation.'})
    print('PREPARED: four NEW calls maximum; zero remote/device calls')


def review(index, visible):
    if index not in (1, 3) or (OUT / 'stopped.json').exists():
        raise RuntimeError('Invalid review or campaign stopped')
    result = base.read(OUT / f'{index}_result.json')
    if not result['ok'] or result['stage'] != 'roi':
        raise RuntimeError('No valid crop to review')
    review_path = OUT / f'{index}_crop_review.json'
    if review_path.exists():
        raise RuntimeError('Review already recorded; do not overwrite')
    save(review_path.name, {'complete_target_visible': visible,
        'method': 'Assistant visually inspected saved model-chosen crop; no crop/point modifications.',
        'png_sha256': result['png_sha256']})
    if not visible:
        stop('model_roi_cropped_out_target')
    print(json.dumps({'crop_review': index, 'complete_target_visible': visible}))


def recognize(index):
    pre = base.read(OUT / 'preflight.json')
    guard(index, pre)
    target = ORDER[(index-1)//2]
    region = index % 2 == 1
    body = base.read(OUT / f'{index}_request.json')
    if region:
        assert base.digest(body) == pre['roi_request_hashes'][(index-1)//2]
        _, original_image, _ = previous.source_image()
        size = original_image.size
    else:
        prior = base.read(OUT / f'{index-1}_result.json')
        assert base.digest(body) == prior['next_request_sha256']
        size = prior['geometry']['zoom_size']
    provider = base.transport.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    assert provider.configured and provider.model == body['model']
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert provider.model_config.request_options() == {'enable_thinking': True}
    with (OUT / f'{index}_attempt.json').open('x', encoding='utf-8') as handle:
        json.dump({'started_at': datetime.now().astimezone().isoformat(), 'index': index, 'target': target}, handle)
    original_post = base.transport.httpx.post
    calls = 0

    def captured_post(url, **kwargs):
        nonlocal calls
        assert calls == 0 and url == provider.base_url + '/chat/completions' and kwargs['json'] == body
        calls += 1
        response = original_post(url, **kwargs)
        base.atomic_replace_bytes(OUT / f'{index}_response_body.json', response.content)
        save(f'{index}_http.json', {'status_code': response.status_code, 'calls': calls})
        return response

    result = {'index': index, 'target': target, 'stage': 'roi' if region else 'point', 'ok': False, 'phone_actions': 0}
    reason = None
    transport_returned = False
    try:
        with patch.object(base.transport.httpx, 'post', captured_post):
            raw = provider._chat(body['messages'], max_tokens=None, timeout=60, max_attempts=1)
        transport_returned = True
        save(f'{index}_content.json', {'raw': raw})
        parsed = candidate.parse(raw, *size, region=region)
        result['parsed'] = parsed
        if parsed is None:
            reason = 'model_reports_no_unique_target'
        elif region:
            zoom, geometry = candidate.crop_and_enlarge(original_image, parsed)
            buffer = io.BytesIO()
            zoom.save(buffer, format='PNG')
            data = buffer.getvalue()
            base.atomic_replace_bytes(OUT / f'{target}_zoom.png', data)
            url = 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
            next_body = candidate.build_request(url, previous.TARGETS[target], region=False)
            save(f'{index+1}_request.json', next_body)
            result.update(ok=True, geometry=geometry, png_sha256=hashlib.sha256(data).hexdigest(),
                          next_request_sha256=base.digest(next_body))
        else:
            xy = candidate.to_original(raw, prior['geometry'])
            result.update(ok=True, original_image_point=xy, **previous.score(xy, target))
            if not result['inside_visible_region']:
                reason = 'candidate_point_missed_reference'
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        reason = 'model_contract_or_experiment_error' if transport_returned else 'transport_or_provider_error'
    finally:
        result.update(network_attempts=provider.last_network_attempts, usage=provider.last_usage,
            response_id=provider.last_request_id, finish_reason=provider.last_finish_reason,
            response_model=provider.last_response_model, production_unchanged=base.hashes() == pre['production_hashes'])
        save(f'{index}_result.json', result)
        if reason or index == 4 or not result['production_unchanged']:
            stop(reason or ('completed_screening_budget' if index == 4 else 'production_drift'))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize', 'review'))
    parser.add_argument('--index', type=int)
    parser.add_argument('--allow-remote', action='store_true')
    parser.add_argument('--target-visible', choices=('yes', 'no'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'review' and args.target_visible:
        review(args.index, args.target_visible == 'yes')
    elif args.mode == 'recognize' and args.allow_remote:
        recognize(args.index)
    else:
        parser.error('Explicit remote authorization or crop review required')
