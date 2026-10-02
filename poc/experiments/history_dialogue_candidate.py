"""Isolated request-layout experiment. Never imported by the production runtime.

Keep the current observer, images, prompt rules, schema, model and parser intact.
Only move its embedded task/history JSON into factual conversation turns.
This is NOT an alternate finish judge and cannot execute a phone action.
"""
from copy import deepcopy
import json

from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver


CONTEXT_PREFIX = '当前任务与实际执行历史：'
CANDIDATE_VERSION = '2026-09-07-factual-dialogue-layout-experiment-v1'


def reframe_as_dialogue(messages):
    source = deepcopy(messages)
    if len(source) != 2 or source[0]['role'] != 'system' or source[1]['role'] != 'user':
        raise ValueError('Experiment expects the current two-message production request')
    content = source[1]['content']
    prompt = content[0]['text']
    start = prompt.index(CONTEXT_PREFIX) + len(CONTEXT_PREFIX)
    context, length = json.JSONDecoder().raw_decode(prompt[start:])
    entities = context.get('entities', {})
    history = entities.pop('history', [])
    if not isinstance(history, list):
        raise ValueError('Experiment requires factual history as a list')
    # Replace only the exact rendered context occurrence, never task substrings.
    content[0]['text'] = (prompt[:start - len(CONTEXT_PREFIX)]
        + '当前任务在首条用户消息中；本会话实际执行历史按前序动作/结果对话给出。'
        + prompt[start + length:])
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    result = [source[0], {'role': 'user', 'content': [
        {'type': 'text', 'text': '本会话用户任务与上下文：' + encode(context)}]}]
    for entry in history:
        # Never add model thoughts, selected-but-unexecuted proposals or future outcomes.
        action = {key: entry[key] for key in ('step', 'canonical_action', 'action') if key in entry}
        outcome = {key: value for key, value in entry.items()
            if key not in {'canonical_action', 'action'}}
        result.append({'role': 'assistant', 'content': encode({'已执行动作': action})})
        result.append({'role': 'user', 'content': [
            {'type': 'text', 'text': '实际执行结果：' + encode(outcome)}]})
    # Only the final user message carries this observation's current frame group.
    result[-1]['content'].extend(content)
    return result


class DialogueCandidateObserver(SingleStepGenericSceneObserver):
    def _provider_chat(self, messages, *, max_tokens, response_format=None):
        return super()._provider_chat(reframe_as_dialogue(messages),
            max_tokens=max_tokens, response_format=response_format)
