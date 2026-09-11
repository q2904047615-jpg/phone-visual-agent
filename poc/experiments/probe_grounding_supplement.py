"""New explicit six-call authorization; old stopped two-call campaign is read-only."""
import argparse
from pathlib import Path
import sys

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_grounding_isolation as base

PARENT = POC / 'output/grounding_isolation_20260907'
OUT = PARENT / 'approved_supplement_6'
CASES = ('successful_send', 'input_focus', 'recent_clear')


def configure():
    base.OUT = OUT
    base.CASES = CASES
    base.ORDER = tuple((name, variant) for name in CASES for variant in base.VARIANTS)


def prepare():
    old = base.read(PARENT / 'preflight.json')
    assert base.read(PARENT / 'stopped.json')['calls_used'] == 2
    assert base.production_hashes() == old['production_hashes'], 'Production changed since paired diagnosis'
    configure()
    base.preflight()
    new = base.read(OUT / 'preflight.json')
    for key, value in new['wire_hashes'].items():
        assert old['wire_hashes'][key] == value, 'Previously frozen request changed'
    # Preserve original raw/result files; reparse both saved replies OFFLINE.
    reviewed = {}
    for variant in base.VARIANTS:
        raw = base.read(PARENT / f'failed_send_{variant}_raw.json')['raw']
        point, parsed = base.parse_reply(raw, variant, (720, 1280), True)
        frame = [round(point[0]*809/1000), round(point[1]*1439/1000)]
        reviewed[variant] = {'canonical_point': point, 'frame_point': frame,
            'inside_target_region': base.in_region(frame, old['regions']['failed_send']),
            'parsed': parsed, 'network_calls': 0}
    base.save('prior_pair_reparsed_offline.json', reviewed)
    base.save('authorization.json', {
        'user_decision': 'User explicitly allowed correcting isolated parser and at most six additional recognitions.',
        'parent_campaign': str(PARENT), 'parent_stopped_preserved': True,
        'max_new_calls': 6, 'excluded_cases': ['failed_send'], 'allowed_cases': CASES,
        'old_requests_not_repeated': True, 'all_six_wire_hashes_match_parent': True,
        'only_parser_fix': 'Use existing production singleton-object-array normalization; no prompt/schema/image changes.',
        'parent_code_hashes': old['code_hashes'], 'phone_actions': 0,
    })


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('prepare', 'recognize'))
    cli.add_argument('--case', choices=CASES)
    cli.add_argument('--variant', choices=base.VARIANTS)
    cli.add_argument('--allow-remote', action='store_true')
    args = cli.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.allow_remote and args.case and args.variant:
        configure()
        assert base.read(OUT / 'authorization.json')['max_new_calls'] == 6
        base.recognize(args.case, args.variant)
    else:
        cli.error('Explicit --allow-remote, case and variant required')
