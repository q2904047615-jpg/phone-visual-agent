"""One approved thinking-off replay of the latest actual request, no hardware."""
import argparse
import base64
from copy import deepcopy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_minimal_wire_grounding as base
from agent.infrastructure.generic_scene_observer import _parse_single_step_observation_envelope

SOURCE = POC / 'output/web/generic_supervised_20260908_025152_f7104fd4/13bb304a66eb46d3bf5835ab91af8c15_model_request.json'
OUT = POC / 'output/thinking_off_exact_wire_20260908'


def save(name, data):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(data))


def source_hash():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def image_facts(body):
    facts = []
    for message in body['messages']:
        if not isinstance(message['content'], list):
            continue
        for content in message['content']:
            if content['type'] == 'image_url':
                raw = base64.b64decode(content['image_url']['url'].split(',',1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    facts.append({'size':list(image.size), 'sha256':hashlib.sha256(raw).hexdigest()})
    return facts


def prepare():
    if OUT.exists():
        raise RuntimeError('Never overwrite or reset an existing campaign')
    original = base.read(SOURCE)
    assert original['enable_thinking'] is True
    body = deepcopy(original)
    body['enable_thinking'] = False
    assert [key for key in body if body[key] != original[key]] == ['enable_thinking']
    assert set(body) == {'model','messages','temperature','enable_thinking','response_format'}
    facts = image_facts(body)
    assert len(facts) == 3 and all(f['size'] == [720,1280] for f in facts)
    assert facts == image_facts(original)
    OUT.mkdir()
    save('request.json',body)
    save('preflight.json',{'created_at':datetime.now().astimezone().isoformat(),
        'authorization':'User requested trying thinking mode off; one exact saved-request recognition, no phone.',
        'source':str(SOURCE),'source_sha256':source_hash(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_hashes':base.hashes(),'request_sha256':base.digest(body),'images':facts,
        'only_changed_field':'enable_thinking','max_calls':1,'max_attempts':1,'phone_actions':0,
        'limits':'Historical on response vs one fresh off response; not repeated randomized A/B or live acceptance.'})
    print('PREFLIGHT_PASS: only enable_thinking changed; three exact JPEGs; one request maximum')


def recognize():
    pre = base.read(OUT/'preflight.json')
    if (OUT/'attempt.json').exists() or (OUT/'stopped.json').exists():
        raise RuntimeError('Already used or stopped')
    assert source_hash() == pre['source_sha256']
    assert base.hashes() == pre['production_hashes']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == pre['script_sha256']
    body = base.read(OUT/'request.json')
    assert base.digest(body) == pre['request_sha256']
    provider = base.transport.DashScopeVisionProvider(enable_thinking=False, max_attempts=1)
    assert provider.configured and provider.model == body['model']
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert provider.model_config.request_options() == {'enable_thinking':False}
    with (OUT/'attempt.json').open('x',encoding='utf-8') as handle:
        json.dump({'started_at':datetime.now().astimezone().isoformat(),'max_attempts':1},handle)
    real_post = base.transport.httpx.post
    calls = 0

    def capture(url, **kwargs):
        nonlocal calls
        assert calls == 0 and kwargs['json'] == body and url == provider.base_url+'/chat/completions'
        calls += 1
        response = real_post(url,**kwargs)
        base.atomic_replace_bytes(OUT/'response_body.json',response.content)
        save('http.json',{'status_code':response.status_code,'calls':calls})
        return response

    result = {'phone_actions':0,'thinking_enabled':False,'envelope_parse_ok':False}
    started = time.perf_counter()
    try:
        with patch.object(base.transport.httpx,'post',capture):
            raw = provider._chat(body['messages'],max_tokens=None,timeout=60,max_attempts=1,
                                 response_format=body['response_format'])
        save('content.json',{'raw':raw})
        parsed = _parse_single_step_observation_envelope(raw,input_structure_required=True,
                                                        request_image_size=(720,1280))
        save('parsed_envelope.json',parsed)
        decision = parsed['decision']
        result.update(envelope_parse_ok=True,decision=decision)
        point = decision.get('tap_point')
        if point is not None:
            result['original_810x1440_point'] = [round(point[0]*809/1000),round(point[1]*1439/1000)]
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    finally:
        result.update(seconds=round(time.perf_counter()-started,3),usage=provider.last_usage,
            network_attempts=provider.last_network_attempts,response_id=provider.last_request_id,
            response_model=provider.last_response_model,finish_reason=provider.last_finish_reason,
            production_unchanged=base.hashes()==pre['production_hashes'])
        save('result.json',result)
        save('stopped.json',{'reserved_calls':1,'max_calls':1,'resume_allowed':False,'phone_actions':0})
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','recognize'))
    parser.add_argument('--allow-remote',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':
        prepare()
    elif args.allow_remote:
        recognize()
    else:
        parser.error('Explicit remote authorization required')
