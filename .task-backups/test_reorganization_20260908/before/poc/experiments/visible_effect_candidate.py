"""Isolated semantic contract experiment, not a production judge.

Ask whether the last action's effect is already visible, not whether its choice
matches the task. No action ID, local task checklist, extra model or old image.
"""
from copy import deepcopy
import json

from experiments.bound_action_result_candidate import prepare as baseline_request, PREFIX

FIELD = 'previous_action_effect_visible'


def prepare(context, **options):
    result = baseline_request(context, bind_result=False, uniform_xy=False, **options)
    prompt = result['prompt']
    start = prompt.index(PREFIX) + len(PREFIX)
    _, length = json.JSONDecoder().raw_decode(prompt[start:])
    prefix, literal, contract = prompt[:start], prompt[start:start+length], prompt[start+length:]
    contract = contract.replace('previous_action_outcome', FIELD)
    old = ('- ' + FIELD + '：有执行历史时，根据本轮图判断最后一次动作是否达到预期，填matched/unmatched/uncertain；\n'
        '  初始无历史时为null。不能仅因transport_outcome是executed或matched就填写matched。')
    new = ('- ' + FIELD + '：只判断history最后一次实际动作的效果是否已经在当前图中出现。\n'
        '  true=该动作所要造成的结果已经有当前画面证据；false=当前画面明确显示该结果尚未出现；'
        'null=无法从当前画面确定，或初始没有动作历史。\n'
        '  不判断动作选得对不对、按钮是否可点或准备是否就绪；这些均不等于效果已经出现。'
        '不得把执行回执、较早动作成功、原有内容或猜测将来会出现的结果当作true。')
    if contract.count(old) != 1:
        raise ValueError('Production result definition changed; re-audit before sampling')
    contract = contract.replace(old, new, 1)
    result['prompt'] = prefix + literal + contract
    decision = result['response_format']['json_schema']['schema']['properties']['decision']
    decision['properties'].pop('previous_action_outcome')
    decision['properties'][FIELD] = {'type': ['boolean', 'null']}
    decision['required'] = [FIELD if name == 'previous_action_outcome' else name for name in decision['required']]
    result['version'] = '2026-09-07-visible-effect-contract-experiment-v1'
    return result


def project_for_evaluation(payload, *, has_history):
    result = deepcopy(payload)
    decision = result['decision']
    if 'previous_action_outcome' in decision or FIELD not in decision:
        raise ValueError('Use the candidate visible-effect field only')
    visible = decision.pop(FIELD)
    if visible is not None and type(visible) is not bool:
        raise ValueError('Visible effect must be boolean or null')
    if not has_history and visible is not None:
        raise ValueError('Initial observation has no executed action to evaluate')
    # Value translation only. This cannot verify the truth of a visual assertion.
    decision['previous_action_outcome'] = ('matched' if visible is True else
        'unmatched' if visible is False else 'uncertain' if has_history else None)
    return result
