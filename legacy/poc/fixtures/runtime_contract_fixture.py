"""Emit real lifecycle/repository output for browser tests; no network or hardware."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.infrastructure.in_memory_session_repository import InMemoryAgentSessionRepository
from agent.domain.action_capabilities import unverified_promotable_actions
from test_single_visual_loop import LoopHarness, scene, decision


def snapshots():
    result = {'capabilities': unverified_promotable_actions(['tap_semantic', 'dismiss_overlay', 'swipe', 'back'])}
    for mode in ('paused', 'budget_paused'):
        harness = LoopHarness()
        harness.setUp()
        try:
            loop, session, _ = harness.start([(scene(0), decision('home')), (scene(1), decision())],
                max_observations=1 if mode == 'budget_paused' else 200)
            if mode == 'paused':
                loop.pause(session)
            else:
                loop.run_autonomous_safe_loop(session)
            repo = InMemoryAgentSessionRepository()
            repo.add(session)
            result[mode] = {'session': session.snapshot(), 'active': repo.active_snapshots()}
            loop.run_autonomous_safe_loop(session, max_observations=200)
            result[mode]['resumed'] = session.snapshot()
        finally:
            harness.doCleanups()
    return result


if __name__ == '__main__':
    print(json.dumps(snapshots(), ensure_ascii=False))
