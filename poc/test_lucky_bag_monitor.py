"""Lifecycle regression for the new dedicated monitor, without a real phone."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from agent.infrastructure.device_task_registry import DeviceTaskRegistry
from features.lucky_bag.monitor import LuckyBagMonitor, Record
from features.lucky_bag.flow import Flow, Page
from features.lucky_bag.profile import LuckyBagProfile

def wait_until(predicate):
    end = time.monotonic()+3
    while not predicate() and time.monotonic() < end:
        time.sleep(.01)
    assert predicate(), 'worker did not settle'

class MonitorTests(unittest.TestCase):
    def make(self,root,sink=None):
        runtime = SimpleNamespace(device_task_registry=DeviceTaskRegistry(lease_directory=Path(root)/'leases'))
        return LuckyBagMonitor(runtime=runtime,hardware_lock=lambda _:nullcontext(),output_root=Path(root),notification_sink=sink or SimpleNamespace(publish=lambda _:None,remote=None))

    def test_cancel_paused_releases_same_device_for_generic_task(self):
        with TemporaryDirectory() as tmp:
            monitor = self.make(tmp)
            with patch.object(monitor,'_submit'):
                r = monitor.start(profile=LuckyBagProfile('test-device','a@example.com'),goal='test')
            monitor.pause(r.monitor_id)
            with self.assertRaises(Exception):
                monitor.runtime.device_task_registry.reserve('test-device','generic')
            monitor.cancel(r.monitor_id)
            monitor.runtime.device_task_registry.reserve('test-device','generic')
            monitor.runtime.device_task_registry.release('test-device','generic')
            monitor.shutdown()

    def test_restart_keeps_send_attempted_and_requires_resume(self):
        with TemporaryDirectory() as tmp:
            first = self.make(tmp)
            with patch.object(first,'_submit'):
                r = first.start(profile=LuckyBagProfile('test-device','a@example.com'),goal='test')
            r.flow.send_attempted = True
            first.pause(r.monitor_id)
            first.runtime.device_task_registry.release('test-device','lucky-'+r.monitor_id)
            second = self.make(tmp)
            restored = second.get(r.monitor_id)
            self.assertEqual('recovery_required',restored.status)
            self.assertTrue(restored.flow.send_attempted)
            self.assertFalse(second.workers)
            first.shutdown()
            second.cancel(r.monitor_id)
            second.shutdown()

    def test_corrupt_state_does_not_crash_generic_service(self):
        with TemporaryDirectory() as tmp:
            state = Path(tmp)/'dedicated_monitors.json'
            state.write_text('broken',encoding='utf-8')
            monitor = self.make(tmp)
            with self.assertRaisesRegex(RuntimeError,'状态文件'):
                monitor.start(profile=LuckyBagProfile('test-device','a@example.com'),goal='test')
            self.assertEqual('broken',state.read_text())
            monitor.shutdown()

    def test_no_remote_mail_is_visible_failure_and_still_halted(self):
        with TemporaryDirectory() as tmp:
            sent=[]
            monitor = self.make(tmp,SimpleNamespace(publish=sent.append,remote=None))
            r = Record('test',LuckyBagProfile('test-device','a@example.com'),'test',Path(tmp),flow=Flow(halted=True))
            monitor.records[r.monitor_id] = r
            monitor._notify(r)
            self.assertEqual('local_queue_only',r.notification_status)
            self.assertEqual('failed',r.status)
            self.assertEqual('疑似中奖',sent[0].subject)
            self.assertTrue(r.flow.halted)
            monitor.shutdown()

    def test_mail_error_never_clears_stop_latch(self):
        with TemporaryDirectory() as tmp:
            def fail(_):
                raise OSError('offline')
            monitor = self.make(tmp,SimpleNamespace(publish=fail,remote=object()))
            r = Record('test',LuckyBagProfile('test-device','a@example.com'),'test',Path(tmp),flow=Flow(halted=True))
            monitor.records[r.monitor_id] = r
            monitor._notify(r)
            self.assertEqual('failed',r.status)
            self.assertTrue(r.flow.halted)
            monitor.shutdown()

    def test_send_state_saved_before_transport_error_and_no_retry(self):
        with TemporaryDirectory() as tmp:
            monitor = self.make(tmp)
            monitor.config['mode'] = 'local'
            calls=[]
            class Transport:
                def __init__(self,*_):
                    self.controller=SimpleNamespace(begin_new_task=lambda:None)
                def preflight(self): pass
                def capture(self): return Image.new('RGB',(100,200),'white')
                def tap(self,point,frame):
                    state = json.loads(monitor.state_path.read_text(encoding='utf-8'))
                    assert state[0]['flow']['send_attempted']
                    calls.append(point)
                    raise OSError('uncertain send')
            with patch('features.lucky_bag.monitor.ArmTransport',Transport),patch('features.lucky_bag.monitor.LocalOcr') as ocr,patch('features.lucky_bag.monitor.TemplateDetector') as detector,patch('features.lucky_bag.monitor.interpret',return_value=Page(send_button=(20,30),prefilled=True)):
                ocr.return_value.read.return_value=[]
                detector.return_value.locate.return_value=None
                with patch.object(monitor,'_submit'):
                    r = monitor.start(profile=LuckyBagProfile('test-device','a@example.com'),goal='test')
                r.flow.clicked_comment=True
                monitor._submit(r)
                wait_until(lambda:r.status=='failed' and not monitor.workers)
            self.assertEqual([(20,30)],calls)
            self.assertTrue(r.flow.send_attempted)
            self.assertIsNone(monitor.runtime.device_task_registry.active_session('test-device'))
            monitor.shutdown()

if __name__ == '__main__':
    unittest.main()
