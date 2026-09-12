from __future__ import annotations
from agent.domain import EvidenceStoreError
from PIL import Image
from agent.infrastructure import InterProcessLease
from pathlib import Path
from agent.infrastructure.generic_scene_observer import SINGLE_STEP_SCENE_OBSERVER_VERSION
from types import SimpleNamespace
from agent.domain.ui_scene import UIScene
from agent.domain.vision_model import VisionAgentError
import json
from contextlib import nullcontext
from unittest.mock import patch
import tempfile
import threading
import unittest
import web_app
from test_support.web_platform import (
    _BaseApiEndToEndTests,
)


class ApiEndToEndTests(_BaseApiEndToEndTests):
    def test_actual_pause_and_budget_state_survive_api_listing(self):
        from agent.domain.action_capabilities import unverified_promotable_actions
        for mode in ('paused', 'budget_paused'):
            with self.subTest(mode=mode):
                orchestrator, _, _, adapter = self._universal_api_orchestrator()
                session = orchestrator.start(session_id='integration-' + mode, raw_goal='查看页面',
                    device_id='phone-01', run_dir=web_app.WEB_OUTPUT_DIR / mode,
                    max_observations=1 if mode == 'budget_paused' else 200)
                web_app.runtime.agent_session_repository.add(session)
                try:
                    with patch.object(web_app.runtime, 'universal_agent_orchestrator', orchestrator):
                        if mode == 'paused':
                            response = self.client.post(
                                f'/api/agent/generic-supervised/{session.session_id}/pause',
                                headers=self.headers, json={'device_id': 'phone-01'})
                            self.assertEqual(200, response.status_code, response.text)
                        else:
                            orchestrator.run_autonomous_safe_loop(session)
                        self.assertEqual(mode, session.status)
                        status = self.client.get('/api/device').json()
                        self.assertIn(session.session_id, [x['session_id'] for x in status['active_tasks']])
                        for device in status['devices']:
                            self.assertEqual(unverified_promotable_actions(device['verified_actions']),
                                device['capability_acceptance_actions'])
                        self.assertEqual(session.session_id, orchestrator.device_registry.active_session('phone-01'))
                finally:
                    orchestrator.cancel(session)

    def test_machine_position_endpoint_updates_selected_position(self) -> None:
        response = self.client.post(
            "/api/device/device-local-01/machine-position",
            headers=self.headers,
            json={"machine_position": 7},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["machine_position"], 7)
        self.assertEqual(response.json()["status"]["controller_online"], True)

    def test_home_and_device_are_available(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        with patch.object(
            web_app.runtime,
            "app_launcher_for_device",
            return_value=SimpleNamespace(enabled=False),
        ):
            device = self.client.get("/api/device").json()
        self.assertTrue(device["controller_online"])
        self.assertTrue(device["camera_online"])
        self.assertEqual(device["default_device_id"], "device-local-01")
        self.assertEqual(device["devices"][0]["device_id"], "device-local-01")
        self.assertNotIn("legacy_free_agent", device)
        architecture = dict(device["execution_architecture"])
        universal = dict(architecture["universal_agent"])
        observer = universal.pop("observer")
        self.assertEqual(
            observer["observer_version"],
            SINGLE_STEP_SCENE_OBSERVER_VERSION,
        )
        self.assertEqual(observer["max_online_calls_per_observation"], 1)
        self.assertEqual(
            observer["model_role"],
            "single_step_current_scene_observation",
        )
        self.assertEqual(observer["supported_app_scope"], "dynamic")
        architecture["universal_agent"] = universal
        self.assertEqual(architecture["active_orchestrator"], "universal_agent")
        self.assertEqual(architecture["controller"], "universal_action_controller")
        self.assertTrue(architecture["fixed_app_workflows_retired"])
        self.assertNotIn("background_compatibility_worker", architecture)
        self.assertTrue(universal["automatic_loop_enabled"])
        self.assertEqual(universal["task_budget_default_actions"], 100)
        self.assertEqual(universal["task_budget_default_observations"], 200)
        self.assertEqual(universal["supported_app_scope"], "dynamic")
        self.assertEqual(
            universal["action_protocol"],
            "2026-09-06-canonical-whole-task-v10",
        )
        self.assertEqual(
            universal["controller_protocol"],
            web_app.UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
        )
        self.assertEqual(
            universal["protocol_physical_actions"],
            sorted(web_app.CANONICAL_ACTION_KINDS - {"wait_for_change"}),
        )
        self.assertNotIn("launch_app", universal["enabled_physical_actions"])
        self.assertNotIn("swipe", universal["enabled_physical_actions"])
        self.assertIn("scroll", universal["enabled_physical_actions"])
        self.assertIn("swipe_element", universal["enabled_physical_actions"])
        semantic_authority = universal["typed_effect_authority"]
        self.assertEqual(
            semantic_authority["authority_scope"],
            "qwen_current_action_effect_kind",
        )
        self.assertFalse(
            semantic_authority["retired_remote_risk_diagnostics_enabled"]
        )
        self.assertEqual(
            semantic_authority["canonical_action_protocol"],
            "2026-09-06-canonical-whole-task-v10",
        )
        self.assertEqual(
            universal["hardware_capability_profile"]["protocol_version"],
            "2026-08-18-device-capability-profile-v1",
        )
        self.assertTrue(
            universal["hardware_capability_profile"]["actions"]["tap_semantic"]
            ["fresh_visual_postcondition_required"]
        )

    def test_device_status_exposes_enabled_trusted_package_launch(self) -> None:
        default_device_id = web_app.runtime.device_controllers.default_device_id
        with patch.object(
            web_app.runtime,
            "app_launcher_for_device",
            return_value=SimpleNamespace(enabled=True),
        ) as launcher_for_device:
            device = self.client.get("/api/device").json()

        universal = device["execution_architecture"]["universal_agent"]
        self.assertIn("launch_app", universal["enabled_physical_actions"])
        self.assertEqual(
            universal["protocol_physical_actions"],
            sorted(web_app.CANONICAL_ACTION_KINDS - {"wait_for_change"}),
        )
        launcher_for_device.assert_called_once_with(default_device_id)

    def test_device_status_only_exposes_current_task_budget_authority(self) -> None:
        device = self.client.get("/api/device").json()
        execution = device["generic_supervised_execution"]
        for retired in ("max_safe_loop_physical_actions", "max_safe_loop_iterations",
                        "external_effect_confirmation_count"):
            self.assertNotIn(retired, execution)
        self.assertEqual(1, execution["max_physical_actions_per_confirmation"])
        universal = device["execution_architecture"]["universal_agent"]
        self.assertEqual(web_app.DEFAULT_DEVICE_ACTION_BUDGET, universal["task_budget_default_actions"])
        self.assertEqual(web_app.DEFAULT_OBSERVATION_BUDGET, universal["task_budget_default_observations"])

    def test_home_uses_generic_supervised_single_step_endpoints(self) -> None:
        home = self.client.get("/")
        script = self.client.get("/assets/app.js")
        protocol_adapter = self.client.get("/assets/protocol_adapter.js")
        styles = self.client.get("/assets/styles.css")
        self.assertEqual(home.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertEqual(protocol_adapter.status_code, 200)
        self.assertEqual(styles.status_code, 200)
        self.assertIn("你希望手机完成什么", home.text)
        self.assertIn("动态计划", home.text)
        self.assertIn("当前画面", home.text)
        self.assertIn("步骤记录", home.text)
        self.assertIn('id="pauseButton"', home.text)
        self.assertIn('id="stopButton"', home.text)
        self.assertIn('id="riskDialog"', home.text)
        self.assertIn('id="capabilityAcceptancePanel"', home.text)
        self.assertIn('id="promotionDialog"', home.text)
        self.assertIn('id="deviceId"', home.text)
        self.assertNotIn("微信工作流", home.text)
        self.assertNotIn("抖音工作流", home.text)
        self.assertIn('/assets/protocol_adapter.js', home.text)
        self.assertIn("/api/agent/generic-supervised/start", script.text)
        self.assertIn("/api/capability-acceptance/start", script.text)
        self.assertIn("createPromotionGrant", protocol_adapter.text)
        self.assertTrue("/api/agent/generic-supervised/${view.sessionId}/auto" in script.text)
        self.assertNotIn('id="confirmSafeLoop"', home.text)
        self.assertIn("nextSupervisedAgent", script.text)
        self.assertIn("togglePause", script.text)
        self.assertNotIn('api("/api/agent/supervised/start"', script.text)
        self.assertNotIn("wechatView", script.text)
        self.assertNotIn("douyinView", script.text)

    def test_capability_revision_must_match_loaded_service_code(self) -> None:
        runtime = web_app.Runtime.__new__(web_app.Runtime)
        runtime.loaded_code_revision = "loaded-revision"

        with patch.object(web_app, "current_code_revision", return_value="loaded-revision"):
            self.assertEqual(runtime.capability_code_revision(), "loaded-revision")
        with (
            patch.object(web_app, "current_code_revision", return_value="new-revision"),
            self.assertRaisesRegex(
                web_app.CapabilityAcceptanceError,
                "服务启动后代码状态发生变化",
            ),
        ):
            runtime.capability_code_revision()

    def test_generic_supervised_auto_request_is_strict_and_bounded(self) -> None:
        request = web_app.GenericSupervisedAutoRequest(device_id="phone-01")
        self.assertFalse(request.confirmed)
        self.assertIsNone(request.confirmation)
        self.assertIsNone(request.max_physical_actions)
        self.assertIsNone(request.max_observations)
        bounded = web_app.GenericSupervisedAutoRequest(
            device_id="phone-01",
            max_physical_actions=20,
            max_observations=400,
        )
        self.assertEqual(bounded.max_physical_actions, 20)
        self.assertEqual(bounded.max_observations, 400)
        with self.assertRaises(ValueError):
            web_app.GenericSupervisedAutoRequest(
                device_id="phone-01",
                max_physical_actions=0,
            )
        with self.assertRaises(ValueError):
            web_app.GenericSupervisedAutoRequest(
                device_id="phone-01",
                max_physical_actions="1",
            )

    def test_capability_acceptance_requests_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            web_app.CapabilityAcceptanceStartRequest(
                device_id="device-a",
                action="drag",
                text="拖动安全控件",
                unexpected="forbidden",
            )
        with self.assertRaises(ValueError):
            web_app.CapabilityActionConfirmationRequest(confirmed="true")
        with self.assertRaises(ValueError):
            web_app.CapabilityPromotionRequest(
                confirmed=True,
                trial_id="trial-a",
                device_id="device-a",
                action="drag",
                report_sha256="not-a-sha",
                registry_sha256="b" * 64,
            )

    def test_capability_acceptance_api_starts_at_zero_and_binds_action_scope(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app,
                "_supervised_hardware_lock",
                side_effect=lambda _device_id: nullcontext(),
            ),
        ):
            missing_token = self.client.post(
                "/api/capability-acceptance/start",
                json={
                    "device_id": trial.device_id,
                    "action": "drag",
                    "text": "拖动安全控件",
                },
            )
            started = self.client.post(
                "/api/capability-acceptance/start",
                headers=self.headers,
                json={
                    "device_id": trial.device_id,
                    "action": "drag",
                    "text": "拖动安全控件",
                },
            )

        self.assertEqual(missing_token.status_code, 403, missing_token.text)
        self.assertEqual(started.status_code, 200, started.text)
        payload = started.json()
        self.assertEqual(payload["physical_actions"], 0)
        scope = payload["trial"]["action_confirmation_scope"]
        self.assertEqual(scope["trial_id"], trial.trial_id)
        self.assertEqual(scope["action"], "drag")
        self.assertEqual([call[0] for call in calls], ["start"])
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_action_confirmation_is_exact_once(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        scope = {
            **trial.session.snapshot()["confirmation_scope"],
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        path = f"/api/capability-acceptance/{trial.trial_id}/confirm"
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app,
                "_supervised_hardware_lock",
                side_effect=lambda _device_id: nullcontext(),
            ),
        ):
            first = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
            replay = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
            extra = self.client.post(
                path,
                headers=self.headers,
                json={
                    "confirmed": True,
                    "confirmation": {**scope, "unexpected": "forbidden"},
                },
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["physical_actions"], 1)
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(extra.status_code, 422, extra.text)
        self.assertEqual([call[0] for call in calls].count("confirm"), 2)
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_outer_scope_mismatch_is_terminal_without_action(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        scope = {
            **trial.session.snapshot()["confirmation_scope"],
            "trial_id": trial.trial_id,
            "action": "long_press",
        }
        path = f"/api/capability-acceptance/{trial.trial_id}/confirm"
        with patch.object(web_app.runtime, "capability_acceptance_manager", manager):
            wrong = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )

        self.assertEqual(wrong.status_code, 409, wrong.text)
        self.assertEqual(wrong.json()["detail"]["physical_actions"], 0)
        self.assertEqual(trial.session.physical_actions, 0)
        self.assertEqual(trial.session.status, "cancelled")
        self.assertEqual([call[0] for call in calls].count("confirm"), 0)
        self.assertEqual([call[0] for call in calls].count("cancel"), 1)

    def test_capability_promotion_is_separate_zero_action_and_requires_restart(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        path = f"/api/capability-acceptance/{trial.trial_id}"
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                side_effect=AssertionError("promotion must not inspect Qwen"),
            ),
        ):
            preview = self.client.get(
                f"{path}/promotion-preview",
                headers=self.headers,
            )
            scope = preview.json()["promotion_scope"]
            refused = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": False, **scope},
            )
            promoted = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": True, **scope},
            )
            replay = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": True, **scope},
            )

        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["physical_actions"], 0)
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(promoted.status_code, 200, promoted.text)
        self.assertEqual(promoted.json()["physical_actions"], 0)
        self.assertTrue(promoted.json()["promotion"]["requires_restart"])
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual([call[0] for call in calls].count("promote"), 2)
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_evidence_endpoint_is_token_and_trial_bound(self) -> None:
        manager, trial, _calls = self._fake_capability_manager()
        run_dir = Path(self.temp_dir.name) / "capability-evidence"
        run_dir.mkdir(exist_ok=True)
        frame = run_dir / "before-1.jpg"
        Image.new("RGB", (8, 8), "white").save(frame, format="JPEG")
        outside = Path(self.temp_dir.name) / "outside.jpg"
        Image.new("RGB", (8, 8), "black").save(outside, format="JPEG")
        trial.run_dir = run_dir
        trial.report_path = run_dir / "acceptance_report.json"
        trial.report_path.write_text(
            json.dumps(
                {
                    "before_frame_paths": [str(frame)],
                    "after_frame_paths": [],
                }
            ),
            encoding="utf-8",
        )
        path = f"/api/capability-acceptance/{trial.trial_id}/evidence/before/0"

        with patch.object(web_app.runtime, "capability_acceptance_manager", manager):
            forbidden = self.client.get(path)
            accepted = self.client.get(path, headers=self.headers)
            missing = self.client.get(
                f"/api/capability-acceptance/{trial.trial_id}/evidence/before/1",
                headers=self.headers,
            )
            trial.report_path.write_text(
                json.dumps(
                    {
                        "before_frame_paths": [str(outside)],
                        "after_frame_paths": [],
                    }
                ),
                encoding="utf-8",
            )
            escaped = self.client.get(path, headers=self.headers)

        self.assertEqual(forbidden.status_code, 403, forbidden.text)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.headers["content-type"], "image/jpeg")
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(escaped.status_code, 404, escaped.text)

    def test_v3_confirm_request_requires_scope_and_forbids_extra_fields(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-scope",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-scope",
        )
        web_app.runtime.agent_session_repository.add(session)
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        scope = session.snapshot()["confirmation_scope"]

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            missing = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True},
            )
        self.assertEqual(missing.status_code, 409, missing.text)
        self.assertEqual(missing.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 0)

        with self.assertRaises(ValueError):
            web_app.GenericSupervisedStepRequest(
                confirmed="true",
                confirmation=scope,
            )

        for payload in (
            {
                "confirmed": True,
                "confirmation": scope,
                "unexpected": "forbidden",
            },
            {
                "confirmed": True,
                "confirmation": {
                    **scope,
                    "unexpected": "forbidden",
                },
            },
        ):
            with self.subTest(payload=payload):
                with patch.object(web_app, "_require_supervised_device_ready"):
                    rejected = self.client.post(
                        path,
                        headers=self.headers,
                        json=payload,
                    )
                self.assertEqual(rejected.status_code, 422, rejected.text)
                self.assertEqual(adapter.execute_calls, 0)

    def test_v3_confirm_api_atomically_consumes_one_scope(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-confirm-once",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-confirm-once",
        )
        web_app.runtime.agent_session_repository.add(session)
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        payload = {
            "confirmed": True,
            "confirmation": session.snapshot()["confirmation_scope"],
        }

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            first = self.client.post(path, headers=self.headers, json=payload)
            replay = self.client.post(path, headers=self.headers, json=payload)

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["execution"]["physical_actions"], 1)
        # Executing one action no longer invokes DeepSeek replan or mutates the
        # high-level graph; only a later same-frame Qwen finish advances it.
        self.assertEqual(first.json()["session"]["revision"], 2)
        self.assertEqual(len(first.json()["execution"]["after_frame_paths"]), 4)
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(replay.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 1)

    def test_device_disconnect_invalidates_pending_confirmation(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-disconnect-invalidates",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-disconnect-invalidates",
        )
        web_app.runtime.agent_session_repository.add(session)
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        payload = {
            "confirmed": True,
            "confirmation": session.snapshot()["confirmation_scope"],
        }

        def offline(_device_id=None) -> None:
            raise web_app.HTTPException(status_code=409, detail="控制端或摄像头离线。")

        with (
            patch.object(web_app, "_require_supervised_device_ready", offline),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            disconnected = self.client.post(
                path, headers=self.headers, json=payload
            )

        self.assertEqual(409, disconnected.status_code, disconnected.text)
        self.assertIsNone(session.snapshot()["confirmation_scope"])
        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(0, adapter.execute_calls)

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            stale = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(409, stale.status_code, stale.text)
        self.assertEqual(0, adapter.execute_calls)

    def test_hardware_lock_rejects_another_process_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            lease_dir = Path(temp)
            external = InterProcessLease(
                lease_dir / "physical_hardware_action.lease",
                owner_id="other-service",
                metadata={"purpose": "physical_hardware_action"},
            )
            self.assertTrue(external.acquire())
            try:
                with (
                    patch.object(web_app, "SHARED_DEVICE_LEASE_DIR", lease_dir),
                    self.assertRaisesRegex(
                        web_app.HTTPException, "另一进程已占用"
                    ),
                ):
                    with web_app._supervised_hardware_lock():
                        self.fail("cross-process lease must block the hardware lock")
            finally:
                external.release()

    def test_hardware_locks_allow_different_devices_but_reject_same_device(self) -> None:
        first_controller = SimpleNamespace(operation_lock=threading.Lock())
        second_controller = SimpleNamespace(operation_lock=threading.Lock())
        controllers = {
            "phone-a": first_controller,
            "phone-b": second_controller,
        }
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(web_app, "SHARED_DEVICE_LEASE_DIR", Path(temp)),
                patch.object(
                    web_app.runtime,
                    "controller_for_device",
                    side_effect=lambda device_id: controllers[device_id],
                ),
            ):
                with web_app._supervised_hardware_lock("phone-a"):
                    with web_app._supervised_hardware_lock("phone-b"):
                        self.assertTrue(first_controller.operation_lock.locked())
                        self.assertTrue(second_controller.operation_lock.locked())
                    with self.assertRaisesRegex(
                        web_app.HTTPException,
                        "占用|正在进行",
                    ):
                        with web_app._supervised_hardware_lock("phone-a"):
                            self.fail("同一设备不能取得第二个硬件锁")

        self.assertFalse(first_controller.operation_lock.locked())
        self.assertFalse(second_controller.operation_lock.locked())

    def test_v3_confirm_api_rejects_cross_device_scope(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-cross-device",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-cross-device",
        )
        web_app.runtime.agent_session_repository.add(session)
        scope = session.snapshot()["confirmation_scope"]
        scope["device_id"] = "phone-02"

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            response = self.client.post(
                f"/api/agent/generic-supervised/{session.session_id}/confirm",
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 0)

    def test_generic_scene_preview_is_read_only(self) -> None:
        scene = UIScene(
            app_id="calculator",
            screen_id="app_home",
            summary="计算器首页",
            stable=True,
            confidence=0.96,
            fingerprint="local-frame",
        )
        before_executions = len(web_app.runtime.controller.executions)
        before_failures = set(
            web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*")
        )
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(
                web_app.runtime.generic_scene_observer,
                "observe",
                return_value=scene,
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "在计算器输入7"}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["executed"])
        self.assertFalse(payload["physical_action_requested"])
        self.assertEqual(payload["scene"]["app_id"], "calculator")
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )
        self.assertEqual(
            before_failures,
            set(web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*")),
        )

    def test_generic_scene_preview_failure_persists_redacted_raw_response(self) -> None:
        observer = web_app.runtime.generic_scene_observer
        raw = (
            '{"api_key":"preview-secret",'
            '"image":"data:image/jpeg;base64,QUJD",'
            '"unexpected":true}'
        )
        before_failures = set(
            web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*")
        )
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(observer, "last_raw_response", raw),
            patch.object(
                observer,
                "last_diagnostics",
                {
                    "failed_stage": "parsing_targeted_refinement",
                    "error_type": "schema_validation",
                },
            ),
            patch.object(
                observer,
                "observe",
                side_effect=VisionAgentError("目标精查结果不符合最小增量协议"),
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "只读核对当前输入区域"}},
            )

        self.assertEqual(422, response.status_code, response.text)
        new_failures = (
            set(web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*"))
            - before_failures
        )
        self.assertEqual(1, len(new_failures))
        artifacts = list(next(iter(new_failures)).glob("*_qwen_failure.json"))
        self.assertEqual(1, len(artifacts))
        artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
        serialized = json.dumps(artifact, ensure_ascii=False)
        self.assertNotIn("preview-secret", serialized)
        self.assertNotIn("data:image", serialized)
        self.assertIn("[REDACTED_SECRET]", serialized)
        self.assertIn("[REDACTED_IMAGE_DATA_URL]", serialized)

    def test_generic_scene_preview_diagnostic_failure_preserves_original_422(self) -> None:
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(
                web_app.runtime.generic_scene_observer,
                "observe",
                side_effect=VisionAgentError("原始只读观察错误"),
            ),
            patch.object(
                web_app,
                "persist_observer_failure_diagnostic",
                side_effect=OSError("disk unavailable"),
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "只读核对当前输入区域"}},
            )

        self.assertEqual(422, response.status_code, response.text)
        self.assertIn("原始只读观察错误", response.text)

    def test_generic_supervised_api_starts_and_enters_safe_auto_loop(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        before_executions = len(web_app.runtime.controller.executions)
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "run_autonomous_safe_loop",
                return_value={
                    "physical_actions": 0,
                    "iterations": 0,
                    "status": "awaiting_confirmation",
                    "pause_reason": "测试保留待执行安全动作",
                },
            ) as auto_loop,
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "查看当前页面的详情", "device_id": "phone-01"},
            )
            self.assertEqual(started.status_code, 200, started.text)
            payload = started.json()
            session_id = payload["session"]["session_id"]
            self.assertEqual(payload["physical_actions"], 0)
            self.assertTrue(payload["automatic_loop_enabled"])
            self.assertEqual(
                payload["session"]["status"], "awaiting_confirmation"
            )
            self.assertEqual(len(qwen.calls), 1)
            self.assertEqual(len(qwen.calls), 1)
            self.assertEqual(adapter.capture_calls, 1)
            self.assertEqual(adapter.execute_calls, 0)
            auto_loop.assert_called_once()

            rejected = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/confirm",
                headers=self.headers,
                json={"confirmed": False},
            )
            self.assertEqual(rejected.status_code, 409, rejected.text)
            self.assertEqual(rejected.json()["detail"]["physical_actions"], 0)

            cancelled = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertEqual(cancelled.json()["session"]["status"], "cancelled")
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_start_auto_loop_failure_returns_persisted_failed_session(self) -> None:
        from test_universal_agent_orchestrator import CountingQwen

        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        qwen = CountingQwen(
            2,
            error=VisionAgentError("第二步 Qwen 当前截图解析失败"),
        )
        orchestrator.qwen_observer = qwen
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "连续查看当前页面", "device_id": "phone-01"},
            )

        self.assertEqual(409, response.status_code, response.text)
        failure = response.json()["detail"]
        session = failure["session"]
        self.assertEqual("failed", session["status"])
        self.assertEqual(
            "第二步 Qwen 当前截图解析失败",
            session["failed_reason"],
        )
        self.assertFalse(session["automatic_loop_enabled"])
        self.assertEqual(1, session["physical_actions"])
        self.assertEqual(1, adapter.execute_calls)
        self.assertIsNotNone(failure["report"])
        self.assertTrue(Path(failure["report"]).is_file())
        self.assertIsNone(
            orchestrator.device_registry.active_session(session["device_id"])
        )

    def test_generic_supervised_evidence_failure_remains_http_409(self) -> None:
        orchestrator, _planner, _qwen, _adapter = self._universal_api_orchestrator()
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "start",
                side_effect=EvidenceStoreError("simulated evidence disk failure"),
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={
                    "text": "查看当前页面的详情",
                    "device_id": "phone-01",
                    "auto_advance": False,
                },
            )

        self.assertEqual(409, response.status_code, response.text)
        self.assertEqual(0, response.json()["detail"]["physical_actions"])
        self.assertIn("simulated evidence disk failure", response.text)

    def test_new_generic_session_clears_stop_from_an_earlier_task(self) -> None:
        orchestrator, _planner, _qwen, _adapter = self._universal_api_orchestrator()
        controller = web_app.runtime.controller
        original_start = orchestrator.start
        stop_state_at_task_boundary = []

        def start_after_boundary(**kwargs):
            stop_state_at_task_boundary.append(controller.stop_event.is_set())
            return original_start(**kwargs)

        try:
            stopped = self.client.post(
                "/api/stop",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
            self.assertEqual(200, stopped.status_code, stopped.text)
            self.assertTrue(controller.stop_event.is_set())

            with (
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
                patch.object(orchestrator, "start", side_effect=start_after_boundary),
            ):
                started = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={
                        "text": "查看当前页面的详情",
                        "device_id": "phone-01",
                        "auto_advance": False,
                    },
                )

            self.assertEqual(200, started.status_code, started.text)
            self.assertEqual([False], stop_state_at_task_boundary)
            self.assertFalse(controller.stop_event.is_set())
            session_id = started.json()["session"]["session_id"]
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        finally:
            controller.stop_event.clear()

    def test_stop_requested_after_new_task_boundary_is_not_cleared(self) -> None:
        orchestrator, _planner, _qwen, _adapter = self._universal_api_orchestrator()
        controller = web_app.runtime.controller
        original_start = orchestrator.start
        controller.stop_event.clear()

        def start_then_request_stop(**kwargs):
            self.assertFalse(controller.stop_event.is_set())
            controller.request_stop()
            return original_start(**kwargs)

        try:
            with (
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
                patch.object(
                    orchestrator,
                    "start",
                    side_effect=start_then_request_stop,
                ),
            ):
                started = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={
                        "text": "查看当前页面的详情",
                        "device_id": "phone-01",
                        "auto_advance": False,
                    },
                )

            self.assertEqual(200, started.status_code, started.text)
            self.assertTrue(controller.stop_event.is_set())
            session_id = started.json()["session"]["session_id"]
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        finally:
            controller.stop_event.clear()

    def test_generic_supervised_api_can_start_in_explicit_single_step_mode(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        before_executions = len(web_app.runtime.controller.executions)
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "run_autonomous_safe_loop",
                side_effect=AssertionError("single-step start must not run safe loop"),
            ) as auto_loop,
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={
                    "text": "查看当前页面的详情",
                    "device_id": "phone-01",
                    "auto_advance": False,
                },
            )
            self.assertEqual(started.status_code, 200, started.text)
            payload = started.json()
            session_id = payload["session"]["session_id"]
            self.assertEqual("generic_supervised_single_step", payload["mode"])
            self.assertEqual(0, payload["physical_actions"])
            self.assertFalse(payload["automatic_loop_enabled"])
            self.assertEqual("awaiting_confirmation", payload["session"]["status"])
            self.assertEqual(1, len(qwen.calls))
            self.assertEqual(1, len(qwen.calls))
            self.assertEqual(1, adapter.capture_calls)
            self.assertEqual(0, adapter.execute_calls)
            auto_loop.assert_not_called()

            cancelled = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )

        self.assertEqual(200, cancelled.status_code, cancelled.text)
        self.assertEqual("cancelled", cancelled.json()["session"]["status"])
        self.assertEqual(
            before_executions,
            len(web_app.runtime.controller.executions),
        )

    def test_same_device_second_generic_session_returns_409(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(web_app, "WEB_OUTPUT_DIR", Path(temp)),
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
                patch.object(
                    orchestrator,
                    "run_autonomous_safe_loop",
                    return_value={
                        "physical_actions": 0,
                        "iterations": 0,
                        "status": "awaiting_confirmation",
                        "pause_reason": "测试保留活动会话",
                    },
                ),
            ):
                first = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={"text": "查看详情", "device_id": "phone-01"},
                )
                directories_after_first = sorted(Path(temp).iterdir())
                second = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={"text": "返回上一页", "device_id": "phone-01"},
                )
                directories_after_second = sorted(Path(temp).iterdir())
                session_id = first.json()["session"]["session_id"]
                self.client.post(
                    f"/api/agent/generic-supervised/{session_id}/cancel",
                    headers=self.headers,
                    json={"device_id": "phone-01"},
                )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["detail"]["physical_actions"], 0)
        self.assertEqual(directories_after_second, directories_after_first)
        self.assertEqual(len(directories_after_first), 1)
        self.assertEqual(len(qwen.calls), 1)
        self.assertEqual(len(qwen.calls), 1)
        self.assertEqual(adapter.capture_calls, 1)
        self.assertEqual(adapter.execute_calls, 0)

    def test_safe_auto_executes_one_verified_action_without_confirmation(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "run_autonomous_safe_loop",
                return_value={
                    "physical_actions": 0,
                    "iterations": 0,
                    "status": "awaiting_confirmation",
                    "pause_reason": "测试把执行留给 /auto",
                },
            ),
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "查看详情", "device_id": "phone-01"},
            )
            session_id = started.json()["session"]["session_id"]

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            automatic = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/auto",
                headers=self.headers,
                json={
                    "device_id": "phone-01",
                    "confirmed": False,
                    "max_physical_actions": 1,
                    "max_observations": 2,
                },
            )
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )

        self.assertEqual(automatic.status_code, 200, automatic.text)
        self.assertEqual(automatic.json()["execution"]["physical_actions"], 1)
        self.assertEqual(automatic.json()["session"]["physical_actions"], 1)
        self.assertGreaterEqual(adapter.capture_calls, 1)
        self.assertEqual(adapter.execute_calls, 1)

    def test_preview_requires_and_uses_exact_registered_device(self) -> None:
        class PreviewController:
            def __init__(self, marker: bytes):
                self.marker = marker
                self.calls = []

            def capture_preview(self, quality=76):
                self.calls.append(quality)
                return b"jpeg-" + self.marker

        phone_a = PreviewController(b"phone-a")
        phone_b = PreviewController(b"phone-b")

        def controller_for_device(device_id):
            controllers = {"phone-a": phone_a, "phone-b": phone_b}
            if device_id not in controllers:
                raise web_app.UniversalAgentOrchestratorError(
                    f"device_id 未登记或未启用：{device_id}。"
                )
            return controllers[device_id]

        with patch.object(
            web_app.runtime,
            "controller_for_device",
            side_effect=controller_for_device,
        ):
            missing = self.client.get("/api/preview.jpg")
            unknown = self.client.get("/api/preview.jpg?device_id=phone-x")
            first = self.client.get("/api/preview.jpg?device_id=phone-a")
            second = self.client.get("/api/preview.jpg?device_id=phone-b")

        self.assertEqual(422, missing.status_code)
        self.assertEqual(404, unknown.status_code)
        self.assertEqual(b"jpeg-phone-a", first.content)
        self.assertEqual(b"jpeg-phone-b", second.content)
        self.assertEqual([72], phone_a.calls)
        self.assertEqual([72], phone_b.calls)

    def test_preview_reports_live_capture_failure_instead_of_mock_frame(self) -> None:
        with patch.object(
            web_app.runtime,
            "capture_preview",
            side_effect=RuntimeError("控制端窗口不可用"),
        ):
            response = self.client.get(
                "/api/preview.jpg?device_id=device-local-01"
            )

        self.assertEqual(503, response.status_code)
        self.assertIn("实时相机预览不可用", response.json()["detail"])
        self.assertIn("控制端窗口不可用", response.json()["detail"])

    def test_preview_uses_one_cached_frame_while_device_is_coordinated(self) -> None:
        class PreviewController:
            def __init__(self):
                self.calls = []

            def capture_preview(self, quality=76):
                self.calls.append(quality)
                return f"jpeg-live-{len(self.calls)}".encode("ascii")

        controller = PreviewController()
        device_id = "phone-camera-lease"
        with patch.object(
            web_app.runtime,
            "controller_for_device",
            return_value=controller,
        ):
            first = self.client.get(f"/api/preview.jpg?device_id={device_id}")
            coordination = (
                web_app.runtime.device_runtime_resources.coordination_lock(
                    device_id
                )
            )
            self.assertTrue(coordination.acquire(blocking=False))
            try:
                cached = [
                    self.client.get(f"/api/preview.jpg?device_id={device_id}")
                    for _index in range(3)
                ]
            finally:
                coordination.release()

        self.assertEqual(200, first.status_code)
        self.assertEqual("live", first.headers["X-Camera-Source"])
        self.assertEqual([72], controller.calls)
        self.assertTrue(all(item.content == first.content for item in cached))
        self.assertTrue(
            all(item.headers["X-Camera-Source"] == "cache" for item in cached)
        )

    def test_device_status_identifies_each_active_generic_session_device(self) -> None:
        def active_session(session_id: str, device_id: str):
            payload = {
                "session_id": session_id,
                "device_id": device_id,
                "status": "awaiting_confirmation",
                "step_number": 1,
                "proposal": {"status": "action"},
            }
            return SimpleNamespace(
                session_id=session_id,
                device_id=device_id,
                status="awaiting_confirmation",
                snapshot=lambda payload=payload: dict(payload),
            )

        web_app.runtime.agent_session_repository.add(
            active_session("session-phone-a", "phone-a")
        )
        web_app.runtime.agent_session_repository.add(
            active_session("session-phone-b", "phone-b")
        )

        active = self.client.get("/api/device").json()[
            "generic_supervised_execution"
        ]["active_sessions"]

        self.assertEqual(
            {
                (item["session_id"], item["device_id"])
                for item in active
            },
            {
                ("session-phone-a", "phone-a"),
                ("session-phone-b", "phone-b"),
            },
        )

    def test_device_status_and_doctor_share_observing_session_lease_state(self) -> None:
        device_id = web_app.runtime.device_controllers.default_device_id
        payload = {
            "session_id": "session-observing",
            "device_id": device_id,
            "status": "observing",
            "step_number": 2,
            "proposal": None,
        }
        session = SimpleNamespace(
            session_id=payload["session_id"],
            device_id=device_id,
            status="observing",
            snapshot=lambda: dict(payload, status=session.status),
        )
        web_app.runtime.agent_session_repository.add(session)
        web_app.runtime.device_task_registry.reserve(device_id, session.session_id)

        with patch.object(
            web_app.runtime,
            "serial_camera_session",
            return_value=nullcontext(),
        ), patch.object(
            web_app,
            "run_runtime_doctor",
            side_effect=lambda **kwargs: {
                "device": {"active_session": kwargs["active_session"]}
            },
        ):
            active_device = self.client.get("/api/device").json()
            active_doctor = self.client.get(f"/api/doctor/{device_id}").json()

            active_sessions = active_device["generic_supervised_execution"][
                "active_sessions"
            ]
            self.assertEqual(active_sessions, active_device["active_tasks"])
            self.assertEqual("observing", active_sessions[0]["status"])
            self.assertEqual(
                session.session_id,
                active_doctor["device"]["active_session"],
            )

            session.status = "failed"
            web_app.runtime.device_task_registry.release(device_id, session.session_id)
            failed_device = self.client.get("/api/device").json()
            failed_doctor = self.client.get(f"/api/doctor/{device_id}").json()

        self.assertEqual([], failed_device["active_tasks"])
        self.assertEqual(
            [],
            failed_device["generic_supervised_execution"]["active_sessions"],
        )
        self.assertIsNone(failed_doctor["device"]["active_session"])


if __name__ == "__main__":
    unittest.main()
