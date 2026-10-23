"""Authorized historical-image replay with upstream pure message/image functions.

No device, project API, executable upstream runner, or production integration.
Historical assistant actions are factual receipt projections, not original
GUI-Owl outputs. Never inject a later visual judgment into a past action.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from experiments.probe_gui_owl import MODEL, BASE_URL, parse_action, redact_images, extract_prompt

OUT = POC / 'output/gui_owl_native_history_20260907'
PREVIOUS = POC / 'output/gui_owl_probe_20260907'
SOURCE_HASH = '1cd5414943f379ddd6a46e279fd11b4cf6317e7e568cf1e94cb7749fc5732e3d'
SOURCE_URL = ('https://raw.githubusercontent.com/X-PLUG/MobileAgent/'
              '11cea575561fb7800b5fb6b6cafa56f7a91de11f/Mobile-Agent-v3.5/mobile_use/utils.py')
FUNCTION_NAMES = {'build_messages', 'smart_resize', 'pil_to_base64', 'image_to_base64'}
CASE_NAMES = ('old_unsent', 'sent', 'empty_clear', 'sent_then_home')
GOALS = {
    'old_unsent': '进入微信的文件传输助手，发一条消息，正文是aaazjie？你好。',
    'sent': '把当前输入框里的 loopok 发送给文件传输助手。',
    'empty_clear': '打开微信，打开文件传输助手，把输入框里的内容清空。',
    'sent_then_home': '把当前输入框里的 loopok 发送给文件传输助手，然后回到主屏幕。',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name, value):
    atomic_replace_bytes(OUT / name, json_bytes(value))


def load_pure_upstream(source):
    # Execute ONLY four reviewed pure functions from the exact audited source.
    # No downloaded imports, classes, initializers, ADB methods or runner execute.
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != SOURCE_HASH:
        raise ValueError('Upstream source differs from the audited snapshot')
    functions = [n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name in FUNCTION_NAMES]
    if {n.name for n in functions} != FUNCTION_NAMES:
        raise ValueError('Missing audited pure function')
    namespace = {'datetime': datetime, 'math': math, 'Image': Image,
                 'BytesIO': BytesIO, 'base64': base64, 'SYSTEM_PROMPT': extract_prompt(source)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<audited-upstream-pure-functions>', 'exec'), namespace)
    return namespace


def receipt_action(receipt):
    if receipt['physical_actions'] != 1:
        raise ValueError('Only audited single executed actions can be replayed')
    requested, resolved = receipt['requested_action'], receipt['resolved_action']
    if requested['action'] != resolved['kind']:
        raise ValueError('Receipt action disagreement')
    if requested['action'] == 'launch_app':
        name = requested['params']['target_app_name']
        args = {'action': 'open', 'text': name}
        description = '打开' + name
    elif requested['action'] == 'tap_semantic':
        point = resolved['normalized_point']
        if len(point) != 2 or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in point):
            raise ValueError('Invalid historical normalized point')
        args = {'action': 'click', 'coordinate': [p * 1000 for p in point]}
        description = '点击' + requested['params']['label']
    else:
        raise ValueError('Receipt type not audited for this offline projection')
    return 'Action: ' + description + '\n<tool_call>\n' + json.dumps(
        {'name': 'mobile_use', 'arguments': args}, ensure_ascii=False) + '\n</tool_call>'


def evidence(name):
    source = POC / 'output/web' / (
        'generic_supervised_20260906_213706_7c6b9602' if name in ('old_unsent', 'empty_clear')
        else 'generic_supervised_20260823_131844_e5ef5dce')
    session = read(source / 'session.json')
    history = []
    audit = []
    for entry in session['history']:
        receipt = entry['execution']
        before = source / Path(receipt['before_frame_paths'][-1]).name
        after = source / Path(receipt['after_frame_paths'][-1]).name
        if not before.is_file() or not after.is_file():
            raise ValueError('Missing original evidence frame')
        output = receipt_action(receipt)
        history.append({'output': output, 'image': str(before)})
        audit.append({'step': entry['step_number'], 'before': str(before), 'after': str(after),
                      'native_receipt_projection': output, 'physical_actions': 1})
    return str(after), history, audit


def build_request(name, upstream):
    current, history, audit = evidence(name)
    # Call unmodified upstream builder, default history_n=4, one current image.
    original = upstream['build_messages'](current, GOALS[name], history, MODEL)
    converted, images = [], []
    for message in original:
        content = []
        for part in message['content']:
            if 'text' in part:
                content.append({'type': 'text', 'text': part['text']})
            elif 'image' in part:
                # SDK/wire adaptation only: PIL expects a path, not a file:// URI.
                path = part['image'].removeprefix('file://')
                url = upstream['image_to_base64'](path)
                content.append({'type': 'image_url', 'image_url': {'url': url}})
                with Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
                    encoded_size = list(image.size)
                images.append({'source': path, 'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               'encoded_size': encoded_size})
            else:
                raise ValueError('Unknown upstream message part')
        converted.append({'role': message['role'], 'content': content})
    return ({'model': MODEL, 'messages': converted, 'temperature': 0,
             'max_tokens': 2048, 'stream': False, 'enable_thinking': False},
            {'case': name, 'images': images, 'history_receipts': audit,
             'roles': [m['role'] for m in original]})


def prepare():
    if (OUT / 'preflight.json').exists():
        raise RuntimeError('Campaign already frozen')
    import httpx
    response = httpx.get(SOURCE_URL, timeout=30, follow_redirects=False)
    response.raise_for_status()
    upstream = load_pure_upstream(response.text)
    save('upstream_snapshot.json', {'source_url': SOURCE_URL, 'sha256': SOURCE_HASH,
                                   'source': response.text})
    audits = []
    for name in CASE_NAMES:
        body, audit = build_request(name, upstream)
        save(name + '_request.json', redact_images(body))
        audits.append(audit)
    save('preflight.json', {'model': MODEL, 'base_url': BASE_URL, 'max_calls': 4,
         'user_authorization': '可以：隔离实验允许历史截图，生产不变',
         'date': datetime.today().strftime('%Y-%m-%d'), 'phone_actions': 0,
         'production_changed': False, 'cases': audits,
         'limitations': ['Recorded non-GUI-Owl actions projected to native assistant turns',
                        'Camera photos rather than direct device screenshots',
                        'Historical replay, not end-to-end native device run',
                        'Online gui-plus service, not a specified open-weight size',
                        'No true paired model-only A/B; no success-rate estimate']})
    print(json.dumps({'prepared': len(audits), 'model_calls': 0,
                      'roles': {a['case']: a['roles'] for a in audits}}, ensure_ascii=False))


def guard(name):
    if name not in CASE_NAMES or (OUT / 'stopped.json').exists():
        raise RuntimeError('Invalid case or stopped campaign')
    if len(list(OUT.glob('*_attempt.json'))) >= 4 or (OUT / (name + '_attempt.json')).exists():
        raise RuntimeError('Attempt budget exhausted; no resampling')
    if read(OUT / 'preflight.json')['date'] != datetime.today().strftime('%Y-%m-%d'):
        raise RuntimeError('Upstream date context changed')


def recognize(name):
    guard(name)
    key = os.getenv('DASHSCOPE_API_KEY', '')
    if not key:
        raise RuntimeError('Missing configured credential')
    upstream = load_pure_upstream(read(OUT / 'upstream_snapshot.json')['source'])
    body, audit = build_request(name, upstream)
    if redact_images(body) != read(OUT / (name + '_request.json')):
        raise RuntimeError('Frozen request or evidence changed')
    with (OUT / (name + '_attempt.json')).open('x', encoding='utf-8') as stream:
        json.dump({'case': name, 'started_at': datetime.now().astimezone().isoformat(),
                   'network_attempts': 1, 'retries': 0, 'phone_actions': 0}, stream)
    from openai import OpenAI
    result = {'case': name, 'phone_actions': 0}
    started = time.perf_counter()
    try:
        with OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=55) as client:
            reply = client.chat.completions.create(
                **{k: v for k, v in body.items() if k != 'enable_thinking'},
                extra_body={'enable_thinking': False})
        raw = reply.model_dump(mode='json')
        save(name + '_response.json', raw)
        content = reply.choices[0].message.content or ''
        result.update(model=reply.model, usage=raw.get('usage'), response_text=content,
                      finish_reason=reply.choices[0].finish_reason)
        if result['finish_reason'] != 'stop':
            raise ValueError('Incomplete response')
        result['native_action'] = parse_action(content)
        result['parsed'] = True
    except Exception as exc:
        result.update(parsed=False, error_type=type(exc).__name__,
                      error=str(exc).replace(key, '[REDACTED_SECRET]'))
        save('stopped.json', {'reason': 'service_or_contract_error', 'case': name})
    result['seconds'] = round(time.perf_counter() - started, 3)
    save(name + '_result.json', result)
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
        parser.error('Recognition requires --case and --allow-remote')
