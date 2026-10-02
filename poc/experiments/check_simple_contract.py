"""Offline-only compact unittest report; does not import runtime composition roots."""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    names = sys.argv[1:]
    suite = (unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]), pattern='test_*.py')
        if names == ['--discover'] else unittest.defaultTestLoader.loadTestsFromNames(names))
    started = time.perf_counter()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = unittest.TextTestRunner(stream=captured, verbosity=0).run(suite)
    problems = [{'test': test.id(), 'trace': '\n'.join(line[:350] for line in trace.splitlines()[-8:])}
        for test, trace in result.errors + result.failures]
    report = {'tests': result.testsRun, 'seconds': round(time.perf_counter() - started, 3),
        'errors': len(result.errors), 'failures': len(result.failures), 'problems': problems}
    output = Path(__file__).resolve().parents[1] / 'output' / 'simple_contract_offline'
    output.mkdir(parents=True, exist_ok=True)
    (output / ('full.json' if names == ['--discover'] else 'related.json')).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return not result.wasSuccessful()


if __name__ == '__main__':
    sys.exit(main())
