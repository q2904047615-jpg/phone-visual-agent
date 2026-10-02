"""Six-call, recognition-only validation. No resampling, hardware or API reload."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments.visible_effect_candidate import prepare as candidate_request, project_for_evaluation
from experiments.probe_bound_action_result import fixtures as prior_fixtures, Captured, digest, observe
from experiments.probe_history_dialogue import comparison_config, redacted_wire, read
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider, _extract_json_object, _image_request_size
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver

OUT = POC / 'output/visible_effect_20260907'
NAMES = ('failed_send', 'sent', 'empty_clear', 'sent_then_home', 'old_unsent', 'failed_send_reworded')


def fixtures():
    result = prior_fixtures()
    result['failed_send_reworded'] = deepcopy(result['failed_send'])
    result['failed_send_reworded']['context']['objective'] = '在微信的文件传输助手里发送这句话：新消息验证：蓝鲸背包7392，你好。'
    return result


def save(name, value):
    atomic_replace_bytes(OUT / name, json_bytes(value))


def reserve(name):
    assert name in NAMES, 'Unauthorized case'
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert len(list(OUT.glob('*_attempt.json'))) < 6, 'Six-call authorization exhausted'
    with (OUT / (name + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump({'authorized_by': 'user:允许, six saved-image recognition calls',
            'started_at': datetime.now().astimezone().isoformat(), 'max_attempts': 1, 'phone_actions': 0}, stream)


class Observer(SingleStepGenericSceneObserver):
    def __init__(self, provider, case, remote=False):
        super().__init__(provider)
        self.case, self.remote = case, remote

    def _provider_chat(self, messages, *, max_tokens, response_format=None):
        source = deepcopy(messages)
        props = response_format['json_schema']['schema']['properties']
        kinds = tuple(k for k in props['decision']['properties']['action']['enum'] if k is not None)
        width, height = _image_request_size(self.case['frames'][-1])
        prepared = candidate_request(self.case['context'], request_width=width, request_height=height,
            image_count=sum(p['type']=='image_url' for p in source[1]['content']),
            available_action_kinds=kinds, include_input_structure=props['input_structure']['type'] != 'null')
        source[1]['content'][0]['text'] = prepared['prompt']
        assert source[0] == messages[0] and source[1]['content'][1:] == messages[1]['content'][1:]
        self.wire = redacted_wire(dict(model=self.provider.model, messages=source, temperature=0.0,
            response_format=prepared['response_format'], **self.provider.model_config.request_options()))
        if not self.remote:
            raise Captured()
        name = self.case['name']
        assert digest(self.wire) == read(OUT / 'preflight.json')['wire_hashes'][name], 'Request changed'
        reserve(name)
        raw = super()._provider_chat(source, max_tokens=max_tokens, response_format=prepared['response_format'])
        save(name + '_raw.json', {'raw': raw})
        payload = _extract_json_object(raw, reject_duplicate_keys=True, unwrap_singleton_object_array=True)
        projected = project_for_evaluation(payload, has_history=bool(self.case['context']['entities']['history']))
        return json.dumps(projected, ensure_ascii=False)


def prepare():
    assert not list(OUT.glob('*_attempt.json')), 'Campaign already started'
    hashes = {}
    for name, case in fixtures().items():
        observer = Observer(DashScopeVisionProvider(api_key='offline-dummy', max_attempts=1), case)
        try:
            observe(observer, case)
        except Captured:
            pass
        save(name + '_wire.json', observer.wire)
        hashes[name] = digest(observer.wire)
    save('preflight.json', dict(wire_hashes=hashes, max_calls=6, current_images_unchanged=True,
        model_config=comparison_config(DashScopeVisionProvider(api_key='offline-dummy')),
        phone_actions=0, uniform_xy=False))
    print('{"prepared":6,"remote_calls":0,"phone_actions":0}')


def recognize(name):
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert not (OUT / (name + '_attempt.json')).exists(), 'No resampling'
    provider = DashScopeVisionProvider(max_attempts=1)
    assert provider.configured, 'Qwen credential unavailable'
    assert comparison_config(provider) == read(OUT / 'preflight.json')['model_config'], 'Provider changed'
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'Destination changed'
    case = {**fixtures()[name], 'name': name}
    observer = Observer(provider, case, remote=True)
    result = dict(case=name, phone_actions=0)
    started = time.perf_counter()
    try:
        scene, decision = observe(observer, case)
        result.update(parse_succeeded=True, decision=decision, scene=scene.to_dict())
    except Exception as exc:
        result.update(parse_succeeded=False, error_type=type(exc).__name__, error=str(exc))
    result.update(seconds=round(time.perf_counter()-started, 3), usage=provider.last_usage,
        network_attempts=provider.last_network_attempts, response_model=provider.last_response_model)
    save(name + '_result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'scene'}, ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('prepare', 'recognize'))
    cli.add_argument('--case', choices=NAMES)
    cli.add_argument('--allow-remote', action='store_true')
    args = cli.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.case:
        recognize(args.case)
    else:
        cli.error('Explicit six-call permission required before --allow-remote')
