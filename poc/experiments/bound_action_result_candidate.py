"""Offline-only candidate: explicit last-action result, with a separate XY experiment.

Not imported by the runtime. No device, network, session or service operations.
An action ID proves which receipt was referenced, NOT that the visual claim is true.
The existing canonical parser remains the sole next-action parser during evaluation.
"""
from copy import deepcopy
import hashlib
import json

from agent.infrastructure.generic_scene_observer import (
    _single_step_observation_prompt, _single_step_response_format,
)

VERSION = '2026-09-07-bound-action-result-experiment-v1'
PREFIX = '当前任务与实际执行历史：'
RESULT_FIELDS = ('evaluated_action_id', 'action_result', 'action_result_reason')


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _receipt_context(context):
    """Only for histories from the audited GenericSingleActionAdapter producer.

    Its action_outcome=matched is a transport receipt. visual_outcome is separate.
    Do not use this projection for unrelated producers or historical plan metadata.
    """
    result = deepcopy(context)
    history = result.setdefault('entities', {}).setdefault('history', [])
    if not isinstance(history, list):
        raise ValueError('history must contain actual receipts')
    for entry in history:
        if not isinstance(entry, dict) or not isinstance(entry.get('canonical_action'), dict):
            raise ValueError('missing canonical action receipt')
        if entry.get('transport_outcome') == 'matched':
            entry['transport_outcome'] = 'executed'
    if not history:
        return result, None
    latest = history[-1]
    if latest.get('transport_outcome') != 'executed':
        raise ValueError('latest receipt is not an executed action')
    identity = {key: latest.get(key) for key in ('step', 'canonical_action', 'action')}
    identity['task_id'] = result['entities'].get('task_id')
    action_id = hashlib.sha256(_encode(identity).encode('utf-8')).hexdigest()
    return result, action_id


def prepare(context, *, request_width, request_height, image_count,
            available_action_kinds, include_input_structure=True,
            bind_result=True, uniform_xy=False):
    """Prepare a paired text/schema candidate; never load or transmit images.

    Use bind_result=True, uniform_xy=False to isolate the result experiment;
    use bind_result=False, uniform_xy=True to isolate the coordinate experiment.
    Both false returns the current production text/schema unchanged.
    """
    projected, action_id = (_receipt_context(context) if bind_result
                            else (deepcopy(context), None))
    grid_height = 1000 if uniform_xy else request_height
    prompt = _single_step_observation_prompt(projected,
        request_image_size=(request_width, grid_height),
        image_count=image_count, include_input_structure=include_input_structure,
        available_action_kinds=available_action_kinds)
    schema = _single_step_response_format(projected,
        input_structure_required=include_input_structure, request_height=grid_height,
        available_action_kinds=available_action_kinds)
    # Exclude the JSON task/history span from ALL contract rewriting. User text
    # may literally contain field names, dimensions or instruction-like strings.
    start = prompt.index(PREFIX) + len(PREFIX)
    _, length = json.JSONDecoder().raw_decode(prompt[start:])
    prefix, literal, contract = prompt[:start], prompt[start:start+length], prompt[start+length:]
    if uniform_xy:
        contract = contract.replace(
            f'CURRENT组每张JPEG为{request_width}×1000',
            f'CURRENT组每张JPEG为{request_width}×{request_height}', 1)
    if bind_result:
        contract = contract.replace('previous_action_outcome', 'action_result')
        contract = contract.replace('"action_result":null,',
            '"evaluated_action_id":null,"action_result":null,"action_result_reason":null,', 1)
        contract += ('\n本候选把刚执行动作的视觉结果与下一步理由分开。'
            '\nLAST_EXECUTED_ACTION_ID=' + _encode(action_id)
            + '\n它只指history最后一项的canonical_action与action，不指更早步骤。'
            '\n有历史时evaluated_action_id原样返回上面的ID；action_result_reason只说明'
            '当前图如何支持最后一次动作的matched/unmatched/uncertain，不能用较早动作成功作理由。'
            '\ntransport_outcome=executed只表示执行调用已返回，不证明目标触达或页面效果。'
            '\n无历史时evaluated_action_id、action_result、action_result_reason均为null。'
            '\nreason仅说明下一action或整任务finish；不能代替最后动作结果理由。')
        decision = schema['json_schema']['schema']['properties']['decision']
        props = decision['properties']
        props['action_result'] = props.pop('previous_action_outcome')
        props['evaluated_action_id'] = {'type': ['string', 'null'], 'enum': [action_id]}
        props['action_result_reason'] = {'type': ['string', 'null']}
        decision['required'] = list(props)
    return {'version': VERSION, 'prompt': prefix + literal + contract,
        'response_format': schema, 'expected_action_id': action_id,
        'grid_height': grid_height}


def project_result_for_evaluation(payload, *, expected_action_id):
    """Check reference/type only, then evaluate with the unchanged canonical parser.

    This is an isolated fixture adapter, not a production compatibility branch.
    In particular, matched + a contradictory reason is NOT locally relabelled.
    """
    result = deepcopy(payload)
    decision = result['decision']
    if 'previous_action_outcome' in decision or any(k not in decision for k in RESULT_FIELDS):
        raise ValueError('candidate requires the new result fields only')
    if decision['evaluated_action_id'] != expected_action_id:
        raise ValueError('result references a different action')
    outcome, reason = decision['action_result'], decision['action_result_reason']
    if expected_action_id is None:
        if outcome is not None or reason is not None:
            raise ValueError('initial observation has no action result')
    elif outcome not in ('matched', 'unmatched', 'uncertain') or not isinstance(reason, str) or not reason.strip():
        raise ValueError('missing visual result or result reason')
    for field in RESULT_FIELDS:
        decision.pop(field)
    decision['previous_action_outcome'] = outcome
    return result
