"""Isolated production-request thinking A/B. No device or project API access.

Six cases, one request per mode/case, no retry or resampling. Only enable_thinking
changes. Semantic assessment is manual; schema success is not task success.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments.probe_bound_action_result import fixtures as source_fixtures, Captured, digest, observe
from experiments.probe_history_dialogue import comparison_config, redacted_wire, read
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver, OBSERVATION_TIMEOUT_SECONDS

OUT = POC / 'output/thinking_mode_20260907'
NAMES = ('failed_send', 'sent', 'empty_clear', 'sent_then_home', 'old_unsent', 'failed_send_reworded')
MODES = ('off', 'on')


def fixtures():
    cases = source_fixtures()
    cases['failed_send_reworded'] = deepcopy(cases['failed_send'])
    cases['failed_send_reworded']['context']['objective'] = '在微信的文件传输助手里发送这句话：新消息验证：蓝鲸背包7392，你好。'
    return cases


def save(name, value):
    atomic_replace_bytes(OUT / name, json_bytes(value))


def reserve(name, mode):
    assert name in NAMES and mode in MODES, 'Unknown case/mode'
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert len(list(OUT.glob('*_attempt.json'))) < 12, 'Twelve-call limit exhausted'
    with (OUT / f'{name}_{mode}_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'authorized_by': '2026-09-07 user: 那就用思考模式; approved isolated comparison',
            'started_at': datetime.now().astimezone().isoformat(), 'enable_thinking': mode == 'on',
            'max_attempts': 1, 'phone_actions': 0}, stream, ensure_ascii=False)


class Observer(SingleStepGenericSceneObserver):
    def __init__(self, provider, *, name, mode, remote=False):
        super().__init__(provider)
        self.name, self.mode, self.remote = name, mode, remote

    def _provider_chat(self, messages, *, max_tokens, response_format=None):
        assert max_tokens is None, 'Production token policy changed'
        self.wire = redacted_wire(dict(model=self.provider.model, messages=messages,
            temperature=0.0, response_format=response_format, **self.provider.model_config.request_options()))
        if not self.remote:
            raise Captured()
        key = f'{self.name}_{self.mode}'
        assert digest(self.wire) == read(OUT / 'preflight.json')['wire_hashes'][key], 'Request changed'
        reserve(self.name, self.mode)
        raw = super()._provider_chat(messages, max_tokens=max_tokens, response_format=response_format)
        # Exact final answer before production parsing; no experimental projection.
        save(key + '_raw.json', {'raw': raw})
        return raw


def prepare():
    assert not (OUT / 'preflight.json').exists(), 'Do not overwrite a frozen campaign'
    assert not list(OUT.glob('*_attempt.json')), 'Campaign already started'
    hashes, configurations, sources = {}, {}, {}
    for name, case in fixtures().items():
        wires = []
        for mode in MODES:
            provider = DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=mode == 'on', max_attempts=1)
            observer = Observer(provider, name=name, mode=mode)
            try:
                observe(observer, case)
            except Captured:
                pass
            wires.append(observer.wire)
            key = f'{name}_{mode}'
            hashes[key] = digest(observer.wire)
            configurations[mode] = comparison_config(provider)
            save(key + '_wire.json', observer.wire)
        assert wires[0]['enable_thinking'] is False and wires[1]['enable_thinking'] is True
        assert {k:v for k,v in wires[0].items() if k != 'enable_thinking'} == {
            k:v for k,v in wires[1].items() if k != 'enable_thinking'}, 'More than one variable changed'
        sources[name] = case['source']
    save('preflight.json', {'wire_hashes': hashes, 'model_configs': configurations, 'sources': sources,
        'max_calls': 12, 'max_attempts': 1, 'timeout_seconds': OBSERVATION_TIMEOUT_SECONDS,
        'only_changed_field': 'enable_thinking', 'phone_actions': 0,
        'evidence_limit': 'Saved-image screening; no live task or reliability-rate proof'})
    print(json.dumps({'prepared_pairs': len(NAMES), 'remote_calls': 0, 'phone_actions': 0}))


def recognize(name, mode):
    assert name in NAMES and mode in MODES
    assert not (OUT / 'stopped.json').exists(), 'Campaign stopped'
    assert not (OUT / f'{name}_{mode}_attempt.json').exists(), 'No resampling'
    provider = DashScopeVisionProvider(enable_thinking=mode == 'on', max_attempts=1)
    assert provider.configured, 'Qwen credential unavailable'
    preflight = read(OUT / 'preflight.json')
    assert comparison_config(provider) == preflight['model_configs'][mode], 'Model config changed'
    assert provider.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'Destination changed'
    assert OBSERVATION_TIMEOUT_SECONDS == preflight['timeout_seconds'], 'Timeout changed'
    observer = Observer(provider, name=name, mode=mode, remote=True)
    result = dict(case=name, mode=mode, phone_actions=0)
    start = time.perf_counter()
    try:
        scene, decision = observe(observer, fixtures()[name])
        result.update(parse_succeeded=True, decision=decision, scene=scene.to_dict())
    except Exception as exc:
        result.update(parse_succeeded=False, error_type=type(exc).__name__, error=str(exc))
    result.update(seconds=round(time.perf_counter()-start, 3), usage=provider.last_usage,
        network_attempts=provider.last_network_attempts, response_model=provider.last_response_model,
        finish_reason=provider.last_finish_reason)
    save(f'{name}_{mode}_result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'scene'}, ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('operation', choices=('prepare', 'recognize'))
    cli.add_argument('--case', choices=NAMES)
    cli.add_argument('--mode', choices=MODES)
    cli.add_argument('--allow-remote', action='store_true')
    args = cli.parse_args()
    if args.operation == 'prepare':
        prepare()
    elif args.allow_remote and args.case and args.mode:
        recognize(args.case, args.mode)
    else:
        cli.error('Recognition requires approved --allow-remote, --case and --mode')
