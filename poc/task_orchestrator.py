from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any


GENERIC_ORCHESTRATOR_PROTOCOL_VERSION = "2026-08-06-generic-plan-v1"
GOAL_SPEC_PROTOCOL_VERSION = "2026-08-06-goal-spec-v1"
GENERIC_SOURCE_OPERATION = "agent.dynamic"

SUPPORTED_OPERATION_APPS = {
    "wechat.send_text": "wechat",
    "wechat.send_text_to_file_transfer": "wechat",
    "wechat.send_album_image": "wechat",
    "douyin.search": "douyin",
    "douyin.batch_interact": "douyin",
}


# The planner may compose only semantic, device-independent primitives.  Raw
# coordinates, Python/Shell commands and vendor-control calls are deliberately
# absent from this vocabulary.
ALLOWED_ACTIONS = {
    "ensure_app",
    "observe",
    "tap_semantic",
    "input_verified_text",
    "swipe",
    "back",
    "dismiss_overlay",
    "wait_for_change",
    "verify",
    "recover_unknown",
    "record_verified_result",
    "finish",
}

ALLOWED_CONDITIONS = {
    "page_is",
    "property_equals",
    "counter_less_than",
}

ALLOWED_NODE_KINDS = {"action", "branch", "repeat", "sequence"}
APP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class TaskPlanError(ValueError):
    pass


@dataclass(frozen=True)
class GoalSpec:
    """Model-independent task goal passed from intent parsing to planning."""

    app_id: str
    objective: str
    source_operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    needs_confirmation: bool = True
    protocol_version: str = GOAL_SPEC_PROTOCOL_VERSION

    def validate(self) -> None:
        expected_app = SUPPORTED_OPERATION_APPS.get(self.source_operation)
        if self.source_operation == GENERIC_SOURCE_OPERATION:
            if not APP_ID_PATTERN.fullmatch(self.app_id):
                raise TaskPlanError(f"通用目标 App ID 无效：{self.app_id!r}")
        elif expected_app is None:
            raise TaskPlanError(f"目标操作不受支持：{self.source_operation}")
        elif self.app_id != expected_app:
            raise TaskPlanError(
                f"目标 App 与操作不一致：{self.app_id} != {expected_app}"
            )
        if not self.objective.strip():
            raise TaskPlanError("任务目标不能为空。")
        if self.needs_confirmation is not True:
            raise TaskPlanError("第一阶段目标必须经过人工确认。")
        _reject_unsafe_values(self.parameters, path="parameters")
        _reject_unsafe_values(self.success_criteria, path="success_criteria")
        _reject_unsafe_values(self.limits, path="limits")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_operation(
        cls,
        operation: str,
        params: dict[str, Any],
        *,
        objective: str | None = None,
    ) -> "GoalSpec":
        app_id = SUPPORTED_OPERATION_APPS.get(operation)
        if app_id is None:
            raise TaskPlanError(f"目标操作不受支持：{operation}")
        parameters = dict(params)

        if operation in {"wechat.send_text", "wechat.send_text_to_file_transfer"}:
            chat_name = str(parameters.get("chat_name") or "文件传输助手").strip()
            text = str(parameters.get("text") or "").strip()
            if not chat_name or not text:
                raise TaskPlanError("微信聊天名称和文字不能为空。")
            default_objective = f"向{chat_name}发送文字"
            success_criteria = {
                "type": "message_sent",
                "chat_name": chat_name,
                "text": text,
            }
            limits = {"full_retype_attempts": 2, "wrong_chat_tolerance": 0}
        elif operation == "wechat.send_album_image":
            chat_name = str(parameters.get("chat_name") or "").strip()
            image_index = parameters.get("image_index")
            if (
                not chat_name
                or isinstance(image_index, bool)
                or not isinstance(image_index, int)
                or not 1 <= image_index <= 20
            ):
                raise TaskPlanError("微信图片任务需要聊天名称和1～20的图片序号。")
            default_objective = f"向{chat_name}发送相册第{image_index}张图片"
            success_criteria = {
                "type": "album_image_sent",
                "chat_name": chat_name,
                "image_index": image_index,
            }
            limits = {"selected_image_count": 1, "wrong_chat_tolerance": 0}
        elif operation == "douyin.search":
            keyword = str(parameters.get("keyword") or "").strip()
            if not keyword:
                raise TaskPlanError("抖音搜索关键词不能为空。")
            default_objective = f"在抖音搜索{keyword}"
            success_criteria = {"type": "search_results_visible", "keyword": keyword}
            limits = {"full_retype_attempts": 2}
        else:
            target_count = parameters.get("target_count")
            like = parameters.get("like") is True
            comment = parameters.get("comment") is True
            if (
                isinstance(target_count, bool)
                or not isinstance(target_count, int)
                or not 1 <= target_count <= 10
            ):
                raise TaskPlanError("抖音批量目标数量必须是1～10。")
            if not like and not comment:
                raise TaskPlanError("抖音批量任务至少需要点赞或评论之一。")
            if comment and not str(parameters.get("comment_text") or "").strip():
                raise TaskPlanError("评论任务缺少评论内容。")
            default_objective = f"处理接下来的{target_count}个抖音普通视频"
            success_criteria = {
                "type": "processed_video_count",
                "target_count": target_count,
                "like": like,
                "comment": comment,
            }
            limits = {
                "max_pages": target_count + 5,
                "uncertain_action_tolerance": 0,
            }

        goal = cls(
            app_id=app_id,
            objective=(objective or default_objective).strip(),
            source_operation=operation,
            parameters=parameters,
            success_criteria=success_criteria,
            limits=limits,
        )
        goal.validate()
        return goal

    @classmethod
    def from_dynamic(
        cls,
        *,
        app_id: str,
        objective: str,
        parameters: dict[str, Any] | None = None,
        success_criteria: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
    ) -> "GoalSpec":
        """Build an App-independent goal produced by the text-understanding layer."""
        goal = cls(
            app_id=app_id.strip().lower(),
            objective=objective.strip(),
            source_operation=GENERIC_SOURCE_OPERATION,
            parameters=dict(parameters or {}),
            success_criteria=dict(success_criteria or {}),
            limits=dict(limits or {}),
            needs_confirmation=True,
        )
        goal.validate()
        return goal


@dataclass(frozen=True)
class PlanCondition:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in ALLOWED_CONDITIONS:
            raise TaskPlanError(f"不允许的计划条件：{self.kind}")
        _reject_unsafe_values(self.params)


@dataclass(frozen=True)
class PlanNode:
    kind: str
    node_id: str
    action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    condition: PlanCondition | None = None
    children: tuple["PlanNode", ...] = ()
    otherwise: tuple["PlanNode", ...] = ()
    max_iterations: int | None = None

    def validate(self, *, seen_ids: set[str] | None = None) -> None:
        seen_ids = seen_ids if seen_ids is not None else set()
        if self.kind not in ALLOWED_NODE_KINDS:
            raise TaskPlanError(f"不允许的计划节点：{self.kind}")
        if not self.node_id or self.node_id in seen_ids:
            raise TaskPlanError(f"计划节点ID为空或重复：{self.node_id!r}")
        seen_ids.add(self.node_id)
        _reject_unsafe_values(self.params)

        if self.kind == "action":
            if self.action not in ALLOWED_ACTIONS:
                raise TaskPlanError(f"不允许的通用动作：{self.action}")
            if self.condition is not None or self.children or self.otherwise:
                raise TaskPlanError("动作节点不能包含条件或子节点。")
        elif self.kind == "branch":
            if self.condition is None or not self.children:
                raise TaskPlanError("分支节点必须包含条件和真分支。")
            if self.action is not None or self.max_iterations is not None:
                raise TaskPlanError("分支节点不能包含动作或循环次数。")
            self.condition.validate()
        elif self.kind == "repeat":
            if not self.children:
                raise TaskPlanError("循环节点必须包含循环体。")
            if (
                isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, int)
                or not 1 <= self.max_iterations <= 50
            ):
                raise TaskPlanError("循环最大次数必须是1～50之间的整数。")
            if self.condition is None:
                raise TaskPlanError("循环节点必须包含退出条件。")
            if self.action is not None or self.otherwise:
                raise TaskPlanError("循环节点不能包含动作或否则分支。")
            self.condition.validate()
        else:  # sequence
            if not self.children:
                raise TaskPlanError("顺序节点必须包含子节点。")
            if (
                self.action is not None
                or self.condition is not None
                or self.otherwise
                or self.max_iterations is not None
            ):
                raise TaskPlanError("顺序节点只能包含子节点。")

        for child in (*self.children, *self.otherwise):
            child.validate(seen_ids=seen_ids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["children"] = [child.to_dict() for child in self.children]
        value["otherwise"] = [child.to_dict() for child in self.otherwise]
        if self.condition is not None:
            value["condition"] = asdict(self.condition)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanNode":
        if not isinstance(value, dict):
            raise TaskPlanError("计划节点必须是对象。")
        condition_value = value.get("condition")
        condition = None
        if condition_value is not None:
            if not isinstance(condition_value, dict):
                raise TaskPlanError("计划条件必须是对象。")
            condition = PlanCondition(
                kind=str(condition_value.get("kind") or ""),
                params=dict(condition_value.get("params") or {}),
            )
        children_value = value.get("children") or []
        otherwise_value = value.get("otherwise") or []
        if not isinstance(children_value, list) or not isinstance(otherwise_value, list):
            raise TaskPlanError("计划子节点必须是数组。")
        node = cls(
            kind=str(value.get("kind") or ""),
            node_id=str(value.get("node_id") or ""),
            action=(
                str(value.get("action"))
                if value.get("action") is not None
                else None
            ),
            params=dict(value.get("params") or {}),
            condition=condition,
            children=tuple(cls.from_dict(item) for item in children_value),
            otherwise=tuple(cls.from_dict(item) for item in otherwise_value),
            max_iterations=value.get("max_iterations"),
        )
        node.validate()
        return node


@dataclass(frozen=True)
class TaskPlan:
    app_id: str
    objective: str
    source_operation: str
    root: PlanNode
    protocol_version: str = GENERIC_ORCHESTRATOR_PROTOCOL_VERSION
    goal: GoalSpec | None = None

    def validate(self) -> None:
        if not APP_ID_PATTERN.fullmatch(self.app_id):
            raise TaskPlanError(f"App ID 无效：{self.app_id!r}")
        if not self.objective.strip():
            raise TaskPlanError("任务目标不能为空。")
        if not self.source_operation.strip():
            raise TaskPlanError("来源操作不能为空。")
        if self.goal is not None:
            self.goal.validate()
            if (
                self.goal.app_id != self.app_id
                or self.goal.source_operation != self.source_operation
            ):
                raise TaskPlanError("计划与目标结构不一致。")
        self.root.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = {
            "protocol_version": self.protocol_version,
            "app_id": self.app_id,
            "objective": self.objective,
            "source_operation": self.source_operation,
            "root": self.root.to_dict(),
        }
        if self.goal is not None:
            value["goal"] = self.goal.to_dict()
        return value

    @classmethod
    def from_model_dict(cls, value: dict[str, Any], *, goal: GoalSpec) -> "TaskPlan":
        """Validate a model-authored semantic plan without accepting raw controls."""
        goal.validate()
        if not isinstance(value, dict):
            raise TaskPlanError("模型计划必须是对象。")
        unexpected = set(value) - {"root", "objective"}
        if unexpected:
            raise TaskPlanError(
                "模型计划包含不允许的顶层字段：" + ", ".join(sorted(unexpected))
            )
        root_value = value.get("root")
        if not isinstance(root_value, dict):
            raise TaskPlanError("模型计划缺少 root。")
        plan = cls(
            app_id=goal.app_id,
            objective=str(value.get("objective") or goal.objective).strip(),
            source_operation=goal.source_operation,
            root=PlanNode.from_dict(root_value),
            goal=goal,
        )
        plan.validate()
        return plan


def _reject_unsafe_values(value: Any, *, path: str = "params") -> None:
    """Reject planner output that could bypass semantic control boundaries."""

    forbidden_keys = {
        "coordinate",
        "coordinates",
        "x",
        "y",
        "shell",
        "command",
        "python",
        "powershell",
        "main_exe",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).strip().lower()
            if name in forbidden_keys:
                raise TaskPlanError(f"计划包含禁止字段：{path}.{key}")
            _reject_unsafe_values(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_unsafe_values(item, path=f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise TaskPlanError(f"计划参数类型不受支持：{path}")


class GenericTaskOrchestrator:
    """Dormant parallel orchestrator boundary.

    Phase one intentionally compiles and validates plans only.  It has no
    RobotController reference and therefore cannot move hardware.  Activation
    will be a separate, user-approved phase after a single-step executor has
    offline tests.
    """

    execution_enabled = False

    def status(self) -> dict[str, Any]:
        return {
            "available": True,
            "execution_enabled": self.execution_enabled,
            "protocol_version": GENERIC_ORCHESTRATOR_PROTOCOL_VERSION,
            "allowed_actions": sorted(ALLOWED_ACTIONS),
        }

    def compile(self, operation: str, params: dict[str, Any]) -> TaskPlan:
        return self.compile_goal(GoalSpec.from_operation(operation, params))

    def compile_goal(self, goal: GoalSpec) -> TaskPlan:
        goal.validate()
        if goal.source_operation == GENERIC_SOURCE_OPERATION:
            raise TaskPlanError(
                "动态目标必须提供经过白名单验证的语义计划，不能套用旧 App 模板。"
            )
        operation = goal.source_operation
        params = goal.parameters
        if operation == "douyin.batch_interact":
            plan = self._compile_douyin_batch(params)
        elif operation == "douyin.search":
            plan = self._compile_douyin_search(params)
        elif operation in {"wechat.send_text", "wechat.send_text_to_file_transfer"}:
            plan = self._compile_wechat_text(operation, params)
        elif operation == "wechat.send_album_image":
            plan = self._compile_wechat_album(params)
        else:
            raise TaskPlanError(f"通用编排器第一阶段尚未支持：{operation}")
        compiled = replace(
            plan,
            app_id=goal.app_id,
            objective=goal.objective,
            source_operation=goal.source_operation,
            goal=goal,
        )
        compiled.validate()
        return compiled

    def compile_dynamic(
        self,
        goal: GoalSpec,
        model_plan: dict[str, Any],
    ) -> TaskPlan:
        if goal.source_operation != GENERIC_SOURCE_OPERATION:
            raise TaskPlanError("compile_dynamic 只接受通用动态目标。")
        return TaskPlan.from_model_dict(model_plan, goal=goal)

    @staticmethod
    def _action(node_id: str, action: str, **params: Any) -> PlanNode:
        return PlanNode(kind="action", node_id=node_id, action=action, params=params)

    def _compile_douyin_search(self, params: dict[str, Any]) -> TaskPlan:
        keyword = str(params.get("keyword") or "").strip()
        if not keyword:
            raise TaskPlanError("抖音搜索关键词不能为空。")
        root = PlanNode(
            kind="sequence",
            node_id="root",
            children=(
                self._action("open_app", "ensure_app", app_id="douyin"),
                self._action("observe_home", "observe"),
                self._action("open_search", "tap_semantic", target="open_search"),
                self._action(
                    "input_keyword",
                    "input_verified_text",
                    field="active_input",
                    text=keyword,
                ),
                self._action("submit", "tap_semantic", target="submit_search"),
                self._action("verify_results", "observe"),
                self._action("finish", "finish", expected_state="douyin_search_results"),
            ),
        )
        plan = TaskPlan(
            app_id="douyin",
            objective=f"搜索{keyword}",
            source_operation="douyin.search",
            root=root,
        )
        plan.validate()
        return plan

    def _compile_douyin_batch(self, params: dict[str, Any]) -> TaskPlan:
        target_count = params.get("target_count")
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or not 1 <= target_count <= 10
        ):
            raise TaskPlanError("抖音批量目标数量必须是1～10。")
        like = params.get("like") is True
        comment = params.get("comment") is True
        if not like and not comment:
            raise TaskPlanError("抖音批量任务至少需要点赞或评论之一。")

        video_actions: list[PlanNode] = []
        if like:
            video_actions.append(
                PlanNode(
                    kind="branch",
                    node_id="like_if_needed",
                    condition=PlanCondition(
                        "property_equals",
                        {"name": "heart_state", "value": "unliked"},
                    ),
                    children=(
                        self._action("tap_heart", "tap_semantic", target="heart"),
                        self._action("verify_heart", "observe"),
                    ),
                )
            )
        if comment:
            comment_text = str(params.get("comment_text") or "").strip()
            if not comment_text:
                raise TaskPlanError("评论任务缺少评论内容。")
            video_actions.extend(
                (
                    self._action("open_comments", "tap_semantic", target="comments"),
                    self._action(
                        "input_comment",
                        "input_verified_text",
                        field="active_input",
                        text=comment_text,
                    ),
                    self._action("send_comment", "tap_semantic", target="comment_send"),
                    self._action("verify_comment", "observe"),
                )
            )

        verified_requirements: dict[str, Any] = {}
        if like:
            verified_requirements["heart_state"] = "liked"
        if comment:
            verified_requirements["comment_sent"] = True

        continue_if_needed = PlanNode(
            kind="branch",
            node_id="continue_if_needed",
            condition=PlanCondition(
                "counter_less_than",
                {"counter": "processed_videos", "value": target_count},
            ),
            children=(self._action("next_video", "swipe", direction="up"),),
        )
        skip_known_non_video = PlanNode(
            kind="branch",
            node_id="skip_known_non_video",
            condition=PlanCondition(
                "page_is",
                {
                    "states": [
                        "douyin_live",
                        "douyin_live_preview",
                        "douyin_ad",
                    ]
                },
            ),
            children=(
                self._action(
                    "skip_known_non_video_page",
                    "swipe",
                    direction="up",
                ),
            ),
            otherwise=(self._action("recover_non_video", "recover_unknown"),),
        )
        process_page = PlanNode(
            kind="branch",
            node_id="process_video_page",
            condition=PlanCondition("page_is", {"states": ["douyin_video"]}),
            children=tuple(
                [
                    *video_actions,
                    self._action(
                        "record_processed_video",
                        "record_verified_result",
                        counter="processed_videos",
                        expected_state="douyin_video",
                        requirements=verified_requirements,
                    ),
                    continue_if_needed,
                ]
            ),
            otherwise=(skip_known_non_video,),
        )
        loop = PlanNode(
            kind="repeat",
            node_id="video_loop",
            condition=PlanCondition(
                "counter_less_than",
                {"counter": "processed_videos", "value": target_count},
            ),
            max_iterations=target_count + 5,
            children=(
                self._action("observe_video", "observe"),
                process_page,
            ),
        )
        root = PlanNode(
            kind="sequence",
            node_id="root",
            children=(
                self._action("open_app", "ensure_app", app_id="douyin"),
                loop,
                self._action(
                    "finish",
                    "finish",
                    counter="processed_videos",
                    expected_count=target_count,
                ),
            ),
        )
        objective = f"处理接下来的{target_count}个抖音普通视频"
        plan = TaskPlan(
            app_id="douyin",
            objective=objective,
            source_operation="douyin.batch_interact",
            root=root,
        )
        plan.validate()
        return plan

    def _compile_wechat_album(self, params: dict[str, Any]) -> TaskPlan:
        chat_name = str(params.get("chat_name") or "").strip()
        image_index = params.get("image_index")
        if (
            not chat_name
            or isinstance(image_index, bool)
            or not isinstance(image_index, int)
            or not 1 <= image_index <= 20
        ):
            raise TaskPlanError("微信图片任务需要聊天名称和1～20的图片序号。")
        root = PlanNode(
            kind="sequence",
            node_id="root",
            children=(
                self._action("open_app", "ensure_app", app_id="wechat"),
                self._action("observe_home", "observe"),
                self._action("open_search", "tap_semantic", target="open_search"),
                self._action(
                    "input_chat_name",
                    "input_verified_text",
                    field="active_input",
                    text=chat_name,
                ),
                self._action("open_chat", "tap_semantic", target="exact_chat"),
                self._action("verify_chat", "observe"),
                self._action("open_plus", "tap_semantic", target="chat_plus"),
                self._action("open_album", "tap_semantic", target="album"),
                self._action("verify_album", "observe"),
                self._action(
                    "select_image",
                    "tap_semantic",
                    target="album_image",
                    index=image_index,
                ),
                self._action("verify_selection", "observe"),
                self._action("send_image", "tap_semantic", target="album_send"),
                self._action("verify_sent", "observe"),
                self._action(
                    "finish",
                    "finish",
                    expected_state="wechat_chat",
                    requirements={
                        "chat_title": chat_name,
                        "sent_image_index": image_index,
                    },
                ),
            ),
        )
        plan = TaskPlan(
            app_id="wechat",
            objective=f"向{chat_name}发送相册第{image_index}张图片",
            source_operation="wechat.send_album_image",
            root=root,
        )
        plan.validate()
        return plan

    def _compile_wechat_text(
        self,
        operation: str,
        params: dict[str, Any],
    ) -> TaskPlan:
        chat_name = str(params.get("chat_name") or "文件传输助手").strip()
        text = str(params.get("text") or "").strip()
        if not chat_name or not text:
            raise TaskPlanError("微信聊天名称和文字不能为空。")
        root = PlanNode(
            kind="sequence",
            node_id="root",
            children=(
                self._action("open_app", "ensure_app", app_id="wechat"),
                self._action("observe_home", "observe"),
                self._action("open_search", "tap_semantic", target="open_search"),
                self._action(
                    "input_chat_name",
                    "input_verified_text",
                    field="active_input",
                    text=chat_name,
                ),
                self._action("open_chat", "tap_semantic", target="exact_chat"),
                self._action("verify_chat", "observe"),
                self._action(
                    "input_message",
                    "input_verified_text",
                    field="active_input",
                    text=text,
                ),
                self._action("send", "tap_semantic", target="send"),
                self._action("verify_sent", "observe"),
                self._action(
                    "finish",
                    "finish",
                    expected_state="wechat_chat",
                    requirements={
                        "chat_title": chat_name,
                        "message_sent_text": text,
                    },
                ),
            ),
        )
        plan = TaskPlan(
            app_id="wechat",
            objective=f"向{chat_name}发送文字",
            source_operation=operation,
            root=root,
        )
        plan.validate()
        return plan
