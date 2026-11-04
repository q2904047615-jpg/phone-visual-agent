"""Isolated GUI-Owl upstream-API screen recognition; never dispatches actions.

Only this campaign's four cases are authorized. One network attempt per case,
including errors. The existing stopped Qwen campaign is never resumed.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.dashscope_vision_provider import _image_data_url
from experiments.probe_history_dialogue import cases, assert_clean_history

OUT = POC / 'output/gui_owl_probe_20260907'
MODEL = 'gui-plus-2026-02-26'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
SOURCE_PATH = 'Mobile-Agent-v3.5/mobile_use/utils.py'
CASE_NAMES = ('old_unsent', 'sent', 'empty_clear', 'sent_then_home')


def save(name, value):
    atomic_replace_bytes(OUT / name, json_bytes(value))


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def extract_prompt(source):
    """Read a literal only; never import or execute downloaded upstream code."""
    values = [ast.literal_eval(node.value) for node in ast.parse(source).body
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets)]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError('Expected one literal upstream SYSTEM_PROMPT')
    return values[0]


def request_body(case, prompt):
    history = deepcopy(case['context']['entities']['history'])
    assert_clean_history(history)
    # No injected expected answer, no last action's future visual judgment.
    assert history[-1]['visual_outcome'] is None and history[-1]['after_scene'] == ''
    text = ('Please generate the next move according to the current UI screenshots, '
            'instruction and previous executed actions. The attached images are '
            'consecutive stable frames of the same current observation.\n\nInstruction: '
            + case['context']['objective'] + '\n\nPrevious executed actions and results:\n'
            + json.dumps(history, ensure_ascii=False))
    frames = case['frames'][-3:]
    assert len(frames) == 3
    return {'model': MODEL, 'enable_thinking': False, 'temperature': 0,
            'max_tokens': 2048, 'stream': False,
            'messages': [{'role': 'system', 'content': prompt},
                         {'role': 'user', 'content': [{'type': 'text', 'text': text}]
                          + [{'type': 'image_url', 'image_url': {'url': _image_data_url(frame)}}
                             for frame in frames]}]}


def redact_images(body):
    result = deepcopy(body)
    for message in result['messages']:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part['type'] == 'image_url':
                    url = part['image_url']['url']
                    part['image_url']['url'] = 'SHA256:' + hashlib.sha256(url.encode()).hexdigest()
    return result


def parse_action(text):
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
    if len(blocks) != 1:
        raise ValueError('Expected exactly one upstream tool_call; no repair or retry')
    call = json.loads(blocks[0])
    if call.get('name') != 'mobile_use' or not isinstance(call.get('arguments'), dict):
        raise ValueError('Expected upstream mobile_use arguments')
    args = call['arguments']
    if args.get('action') not in {'key', 'click', 'long_press', 'swipe', 'type',
                                 'system_button', 'open', 'wait', 'answer', 'interact', 'terminate'}:
        raise ValueError('Unknown native action')
    if args['action'] == 'terminate' and args.get('status') not in {'success', 'failure'}:
        raise ValueError('Invalid terminate status')
    for key in ('coordinate', 'coordinate2'):
        if key in args and (not isinstance(args[key], list) or len(args[key]) != 2
                or any(type(v) not in (float, int) or not 0 <= v <= 1000 for v in args[key])):
            raise ValueError('Invalid normalized coordinate')
    return call


def prepare():
    if (OUT / 'preflight.json').exists():
        raise RuntimeError('Prepared campaign is immutable')
    import httpx
    # Public source download only, no credentials and no upstream execution.
    commit = httpx.get('https://api.github.com/repos/X-PLUG/MobileAgent/commits/main',
                       timeout=30, follow_redirects=False)
    commit.raise_for_status()
    sha = commit.json()['sha']
    source_url = f'https://raw.githubusercontent.com/X-PLUG/MobileAgent/{sha}/{SOURCE_PATH}'
    response = httpx.get(source_url, timeout=30, follow_redirects=False)
    response.raise_for_status()
    prompt = extract_prompt(response.text)
    save('upstream.json', {'commit': sha, 'source_url': source_url,
         'source_sha256': hashlib.sha256(response.content).hexdigest(), 'system_prompt': prompt})
    _, samples = cases()
    summaries = []
    for name in CASE_NAMES:
        case = samples[name]
        body = request_body(case, prompt)
        save(f'{name}_request.json', redact_images(body))
        summaries.append({'case': name, 'source': case['source'],
                          'frame_sizes': [list(f.size) for f in case['frames'][-3:]],
                          'expected_status': case['expected_status']})
    save('preflight.json', {'model': MODEL, 'base_url': BASE_URL, 'max_calls': 4,
         'user_authorization': '先验证 GUI-Owl-1.5', 'cases': summaries,
         'phone_actions': 0, 'production_changed': False,
         'design': 'Upstream native action prompt; actual full text history + 3 current stable frames; no historical images; no execution; not a pure model-only A/B test against the production schema.'})
    print(json.dumps({'prepared': 4, 'remote_model_calls': 0,
                      'upstream_commit': sha, 'phone_actions': 0}))


def recognize(name):
    if name not in CASE_NAMES or (OUT / 'stopped.json').exists():
        raise RuntimeError('Campaign stopped or case not authorized')
    preflight = read('preflight.json')
    if preflight['model'] != MODEL or len(list(OUT.glob('*_attempt.json'))) >= 4:
        raise RuntimeError('Campaign settings changed or budget exhausted')
    key = os.environ.get('DASHSCOPE_API_KEY', '')
    if not key:
        raise RuntimeError('No configured DashScope credential')
    _, samples = cases()
    body = request_body(samples[name], read('upstream.json')['system_prompt'])
    if redact_images(body) != read(f'{name}_request.json'):
        raise RuntimeError('Frozen case changed')
    OUT.mkdir(parents=True, exist_ok=True)
    attempt = {'case': name, 'model': MODEL, 'started_at': datetime.now().astimezone().isoformat(),
               'network_attempts': 1, 'phone_actions': 0, 'retry': False}
    # Exclusive creation reserves the attempt before sending, even on interruption.
    with (OUT / f'{name}_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump(attempt, stream, ensure_ascii=False, indent=2)
    from openai import OpenAI
    started = time.perf_counter()
    result = {'case': name, 'requested_model': MODEL, 'phone_actions': 0}
    try:
        with OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=55) as client:
            options = {k: v for k, v in body.items() if k != 'enable_thinking'}
            reply = client.chat.completions.create(**options, extra_body={'enable_thinking': False})
        raw = reply.model_dump(mode='json')
        save(f'{name}_response.json', raw)
        result.update(response_model=reply.model, usage=raw.get('usage'),
                      finish_reason=reply.choices[0].finish_reason)
        content = reply.choices[0].message.content or ''
        result['response_text'] = content
        if result['finish_reason'] != 'stop':
            raise ValueError('Incomplete model response')
        result['native_action'] = parse_action(content)
        result['parsed'] = True
    except Exception as exc:
        result.update(parsed=False, error_type=type(exc).__name__,
                      error=str(exc).replace(key, '[REDACTED_SECRET]'))
        # Service/parse failure stops this run; never resample to get a legal answer.
        save('stopped.json', {'reason': 'service_or_contract_error', 'case': name})
    result['seconds'] = round(time.perf_counter() - started, 3)
    save(f'{name}_result.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recognize'))
    parser.add_argument('--case', choices=CASE_NAMES)
    parser.add_argument('--allow-remote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.case:
        recognize(args.case)
    else:
        parser.error('Recognition requires --allow-remote and --case')
