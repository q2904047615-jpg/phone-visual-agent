from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Protocol

from deepseek_semantic_risk_audit import (
    EXTERNAL_IMPACTS as SUBGOAL_EXTERNAL_IMPACTS,
    RISK_TYPES,
    AuditSource,
    JsonRiskAuditProvider,
    RiskAuditAssessment,
    SemanticRiskAuditReport,
    SemanticRiskAuditor,
)
from generic_intent import GenericIntentError, _parse_json_object


DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION = "2026-08-11-deepseek-task-graph-v3"
GRAPH_STATUSES = frozenset(
    {"ready", "running", "awaiting_confirmation", "completed", "blocked"}
)
SUBGOAL_STATUSES = frozenset(
    {"pending", "active", "completed", "blocked", "skipped"}
)
RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
REPLAN_TRIGGERS = frozenset(
    {
        "observation_changed",
        "action_result_matched",
        "action_mismatch",
        "action_result_mismatch",
        "subgoal_completed",
        "risk_detected",
        "constraint_discovered",
        "recovery_needed",
    }
)
VERIFIED_ACTION_TRANSITION_PROTOCOL_VERSION = (
    "2026-08-16-verified-action-transition-v1"
)
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
LOW_LEVEL_INSTRUCTION_PATTERN = re.compile(
    r"(?:"
    r"点击|轻触|点按|滑动|上划|下划|左划|右划|长按|拖动|"
    r"(?:在|向)[^，。；;]{0,12}(?:输入框|文本框|搜索框)[^，。；;]{0,8}输入|"
    r"输入(?:文字|文本|内容|字符|账号|密码|关键词|搜索词|查询词|消息|验证码|"
    r"姓名|名称|号码|地址|标题|评论|[A-Za-z0-9][^，。；;\s]{0,31})|"
    r"按下[^，。；;]{0,12}键|"
    r"按(?:返回|主页|home|音量(?:加|减)?|电源|菜单|多任务)键|"
    r"裸坐标|坐标|系统命令|shell|powershell|cmd\.exe|adb|main\.exe|"
    r"\(\s*\d{1,4}\s*[,，]\s*\d{1,4}\s*\)|\bx\s*[:=]\s*\d+|"
    r"\b(?:tap|click|swipe|long[ _-]?press|drag|type[ _-]?text|input[ _-]?text|"
    r"coordinate|keycode|press[ _-]?key|system[ _-]?command)\b"
    r")",
    re.IGNORECASE,
)
EXTERNAL_STATE_CHANGE_PATTERN = re.compile(
    r"(?:"
    r"发送|发布|点赞|"
    r"(?<!已)关注(?:该|这个|目标|账号|用户|作者)|进入已关注|"
    r"(?<!已)收藏(?:该|这个|目标|地点|内容|记录|项目)|进入已收藏|"
    r"(?<!已)(?:执行|进行|完成)?保存(?:到|该|这个|目标|地点|内容|记录|文件)|进入已保存|"
    r"发表评论|发布评论|进行评论|添加评论|"
    r"删除|移除|购买|下单|付款|支付|"
    r"转账|授权|授予|修改|创建|新增|上传|分享|加入|"
    r"退出(?:当前|该|这个|目标)?(?:账号|账户|登录|群|群组|团队|组织)|"
    r"订阅|举报|预约|提交|注册|登录|登出|"
    r"\b(?:send|publish|post|comment|like|follow|favorite|save|delete|remove|"
    r"purchase|pay|transfer|grant|modify|create|upload|share|join|leave|"
    r"subscribe|report|book|submit|register|login|logout)\b"
    r")",
    re.IGNORECASE,
)
COMMUNICATION_EFFECT_PATTERN = re.compile(
    r"(?:"
    r"(?:发送|发给|发(?:一条)?|回复|询问|通知|联系|沟通).{0,12}"
    r"(?:消息|私信|留言|需求|用户|联系人|对方)|"
    r"(?:给|向).{0,12}(?:留言|发送|发私信|发消息)|"
    r"(?:私信|留言).{0,8}(?:询问|回复|通知)|"
    r"\b(?:send|reply|message|notify|contact)\b.{0,24}"
    r"\b(?:message|user|contact|recipient)\b"
    r")",
    re.IGNORECASE,
)
ACCOUNT_RELATIONSHIP_EFFECT_PATTERN = re.compile(
    r"(?:"
    r"(?:加|添加|列为|成为|删除|移除|解除).{0,8}(?:好友|联系人)|"
    r"(?<!已)(?:取消|解除)?关注(?:该|这个|目标|用户|账号|作者|对方)|"
    r"(?:拉黑|屏蔽).{0,8}(?:用户|账号|联系人|对方)|"
    r"\b(?:add|remove|block|unblock|follow|unfollow)\b.{0,20}"
    r"\b(?:friend|contact|user|account)\b"
    r")",
    re.IGNORECASE,
)
MEMBERSHIP_EFFECT_PATTERN = re.compile(
    r"(?:"
    r"(?:拉|邀请|添加|加入|移入|移出|踢出|删除|移除|创建|建立|建|修改)"
    r".{0,12}(?:群|群组|成员|团队|组织)|"
    r"(?:群|群组|团队|组织).{0,8}(?:加人|添加成员|移除成员|修改成员)|"
    r"\b(?:invite|add|remove|join|leave|create|modify)\b.{0,20}"
    r"\b(?:group|member|team|organization)\b"
    r")",
    re.IGNORECASE,
)
PERMISSION_ROLE_EFFECT_PATTERN = re.compile(
    r"(?:"
    r"(?:设为|设置为|任命|授予|撤销|修改|提升|降为).{0,12}"
    r"(?:管理员|权限|角色|所有者|版主)|"
    r"\b(?:assign|grant|revoke|promote|demote|change)\b.{0,20}"
    r"\b(?:admin|administrator|permission|role|owner|moderator)\b"
    r")",
    re.IGNORECASE,
)
CONTENT_PUBLICATION_EFFECT_PATTERN = re.compile(
    r"(?:发布|发表评论|发布评论|上传内容|分享内容|公开内容|"
    r"\b(?:publish|post|comment|upload|share)\b.{0,16}"
    r"\b(?:content|post|comment|media)\b)",
    re.IGNORECASE,
)
DATA_DELETION_EFFECT_PATTERN = re.compile(
    r"(?:(?:删除|清除|移除).{0,10}(?:数据|文件|记录|内容|项目|照片|文档)|"
    r"\b(?:delete|erase|remove)\b.{0,16}\b(?:data|file|record|content|item)\b)",
    re.IGNORECASE,
)
DATA_MUTATION_EFFECT_PATTERN = re.compile(
    r"(?:(?:保存|创建|新增|修改|编辑|提交|上传).{0,10}"
    r"(?:数据|文件|记录|内容|项目|地点|文档|表单|草稿)|进入已保存|"
    r"(?:数据|文件|记录|内容|项目|地点|文档|表单|草稿).{0,4}"
    r"(?:已保存|已创建|已新增|已修改|已编辑|已提交|已上传)|"
    r"\b(?:save|create|modify|edit|submit|upload)\b.{0,16}"
    r"\b(?:data|file|record|content|item|form)\b)",
    re.IGNORECASE,
)
TRANSACTION_EFFECT_PATTERN = re.compile(
    r"(?:购买|下单|付款|支付|转账|退款|充值|提现|"
    r"\b(?:purchase|order|pay|transfer|refund|deposit|withdraw)\b)",
    re.IGNORECASE,
)
ACCOUNT_PERMISSION_EFFECT_PATTERN = re.compile(
    r"(?:授权|授予权限|撤销权限|注册|登录|登出|修改账号|修改账户|"
    r"\b(?:authorize|grant permission|revoke permission|register|login|logout)\b)",
    re.IGNORECASE,
)
DIRECT_EFFECT_NEGATION_PATTERN = re.compile(
    r"(?:不|未|没有|未曾|勿|不要|不得|禁止|不能|避免|无需|无须|"
    r"do\s+not|don't|never|without)\s*"
    r"(?:(?:进行|执行|发生|出现)\s*)?"
    r"(?:(?:任何|任意|一切|all|any)\s*)?$",
    re.IGNORECASE,
)
COORDINATED_EFFECT_NEGATION_PATTERN = re.compile(
    r"(?:不|未|没有|未曾|勿|不要|不得|禁止|不能|避免|无需|无须|"
    r"do\s+not|don't|never|without)\s*"
    r"(?:(?:进行|执行|发生|出现)\s*)?"
    r"(?:(?:任何|任意|一切|all|any)\s*)?"
    r"[^，。；;]{1,24}(?:或|和|及|以及|、|or|and)\s*$",
    re.IGNORECASE,
)
SAFE_NAVIGATION_SEMANTIC_PATTERN = re.compile(
    r"(?:打开|进入|启动|切换|返回|后退|关闭|取消|查看.{0,12}(?:页|界面|详情)|"
    r"\b(?:open|enter|launch|navigate|switch|back|close|view)\b)",
    re.IGNORECASE,
)
NEGATED_LOW_LEVEL_INSTRUCTION_PREFIX_PATTERN = re.compile(
    r"(?:不|未|没有|未曾|勿|不要|不得|禁止|不能|避免|无需|无须|"
    r"do\s+not|don't|never|without)\s*"
    r"(?:(?:进行|执行)\s*)?"
    r"(?:(?:任何|任意|一切|all|any)\s*)?"
    r"(?:(?!(?:但|但是|然而|不过|可以|仍可|需要|应当|然后|再|"
    r"but|however|may|can)).){0,24}$",
    re.IGNORECASE,
)
REPAIRABLE_INITIAL_GRAPH_ERRORS = (
    "任务图至少需要一个全局完成条件。",
    "可推进任务图必须且只能有一个活动子目标。",
    "可推进的任务图至少需要一个目标 App。",
)
LOCAL_TRANSIENT_NAVIGATION_PATTERN = re.compile(
    r"(?:(?:新建|打开|进入|关闭|切换|显示).{0,10}(?:空白)?(?:标签页|页签|窗口|弹层|浮层)|"
    r"(?:前台|后台|上一级|下一页|当前页面|空白页面))",
    re.IGNORECASE,
)
REVERSIBLE_NAVIGATION_EFFECT_PATTERN = re.compile(
    r"(?:刷新|重新加载|重载|重新获取|重新读取|重新导航|返回|后退|"
    r"切换.{0,10}(?:页面|页签|视图|窗口)|打开.{0,10}(?:页面|视图|详情)|"
    r"(?:页面|界面|主界面).{0,16}(?:无遮挡|不再被遮挡)|"
    r"(?:无遮挡|不再被遮挡).{0,16}(?:页面|界面|主界面)|"
    r"(?:遮挡层|弹层|浮层).{0,12}(?:不再可见|已消失|不存在)|"
    r"\b(?:refresh|reload|re\s*load|re\s*fetch|re\s*retrieve|reacquire|"
    r"navigate|return|back|switch\s+(?:page|tab|view|window)|"
    r"open\s+(?:page|view|details?))\b)",
    re.IGNORECASE,
)
LOCAL_UNSUBMITTED_INPUT_STATE_PATTERN = re.compile(
    r"(?:(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域).{0,28}"
    r"(?:文字|文本|内容|值|字符).{0,20}"
    r"(?:为|是|变为|改为|修改为|替换为|显示|保持)|"
    r"(?:填写|输入|替换|改为|修改).{0,28}"
    r"(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域)|"
    r"(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域).{0,20}"
    r"(?:填写|输入|替换|改为|修改).{0,28}"
    r"(?:未提交|草稿|文字|文本|内容|值|字符)|"
    r"(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域).{0,20}"
    r"(?:保留|留下).{0,20}(?:未发送|未提交|草稿|文字|文本|内容)|"
    r"(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域).{0,20}"
    r"(?:包含|含有|显示).{0,28}(?:未发送|未提交|草稿)|"
    r"\b(?:input|text|query)\s*(?:field|box).{0,28}(?:contains?|shows?|value|text)\b)",
    re.IGNORECASE,
)
LOCAL_EDITABLE_CARRIER_ADJECTIVE_PATTERN = re.compile(
    r"(?:可编辑(?:的)?|editable\s+)"
    r"(?=[^，。；;]{0,12}(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域|"
    r"\b(?:input|text|query)\s*(?:field|box)\b))",
    re.IGNORECASE,
)
LOCAL_INPUT_PREPARATION_STATE_PATTERN = re.compile(
    r"(?:"
    r"(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域)"
    r"[^，。；;]{0,24}(?:可见|显示|存在|可编辑|已聚焦|获得焦点|保持焦点)|"
    r"(?:可见|显示|存在|可编辑|已聚焦|获得焦点|保持焦点)"
    r"[^，。；;]{0,24}(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域)|"
    r"\b(?:input|text|query|message)\s*(?:field|box|area)\b"
    r"[^,.;\r\n]{0,24}\b(?:visible|shown|present|editable|focused)\b|"
    r"\b(?:visible|shown|present|editable|focused)\b"
    r"[^,.;\r\n]{0,24}\b(?:input|text|query|message)\s*(?:field|box|area)\b"
    r")",
    re.IGNORECASE,
)
LOCAL_TEMPORARY_DRAFT_CLEAR_STATE_PATTERN = re.compile(
    r"(?:(?:当前页面|当前前台|当前应用|本机|本地).{0,28}"
    r"(?:唯一)?(?:未发送|未提交|临时|草稿).{0,24}"
    r"(?:为空|空白|无内容|内容为空)|"
    r"(?:唯一)?(?:未发送|未提交|临时|草稿).{0,24}"
    r"(?:为空|空白|无内容|内容为空).{0,28}"
    r"(?:当前页面|当前前台|当前应用|本机|本地)|"
    r"\b(?:current|local)\b.{0,32}\b(?:temporary|unsubmitted|unsent|draft)\b"
    r".{0,24}\b(?:empty|blank|cleared)\b)",
    re.IGNORECASE,
)
PERSISTENT_DRAFT_STATE_PATTERN = re.compile(
    r"(?:已保存|云端|云同步|服务器|账号草稿|历史记录|文件|数据库|"
    r"\b(?:saved|cloud|synced|server|account|history|file|database)\b)",
    re.IGNORECASE,
)
LOCAL_UNSUBMITTED_WORKFLOW_RISK_PATTERN = re.compile(
    r"(?:(?:未提交|本机临时|本地临时|临时).{0,12}"
    r"(?:文本|文字|输入|草稿|内容)|"
    r"(?:文本|文字|输入|草稿|内容).{0,12}"
    r"(?:未提交|本机临时|本地临时|临时))",
    re.IGNORECASE,
)
RISK_EFFECT_ACTION_PATTERN = re.compile(
    r"(?:取消关注|发送|提交|删除|清除|移除|转发|发布|选择|保存|分享|回复|"
    r"联系(?!人)|关注|评论|上传|创建|修改|授权|登录|登出|购买|下单|付款|支付|转账|"
    r"\b(?:send|submit|delete|erase|remove|forward|publish|post|select|save|"
    r"share|reply|contact|follow|comment|upload|create|modify|authorize|login|"
    r"logout|purchase|order|pay|transfer)\b)",
    re.IGNORECASE,
)
READ_ONLY_RISK_CONTROL_STATE_PATTERN = re.compile(
    r"(?:(?:停在|保持在).{0,20}(?:按钮|控件|入口).{0,8}(?:之前|前)|"
    r"(?:(?:发送|提交|删除|清除|转发|发布|保存|分享|回复|关注|支付|"
    r"send|submit|delete|erase|forward|publish|save|share|reply|follow|pay)\s*)?"
    r"(?:按钮|控件|入口|button|control).{0,12}(?:可见|显示|仍能看见|可核对|"
    r"未被触发|没有触发|未触发|未被点击|没有被点击|未点击|"
    r"未被激活|未激活|没有激活|"
    r"未被启用|未启用|没有启用|"
    r"visible|shown|not\s+triggered|not\s+activated|not\s+enabled)|"
    r"\b(?:stop|stay|remain)\b.{0,28}\bbefore\b.{0,16}\b(?:button|control)\b|"
    r"\b(?:button|control)\b.{0,16}\b(?:visible|shown)\b)",
    re.IGNORECASE,
)
DIRECT_PROHIBITION_CLAUSE_PATTERN = re.compile(
    r"^\s*(?:(?:且|并且|并|and)\s*)?"
    r"(?:不要|不得|禁止|不能|避免|勿|"
    r"不(?!要|得|能|应|可|只|仅|忘记)|do\s+not|don't|never)\s*"
    r"(?!(?:忘记|漏掉|只|仅|forget\b|fail\b))",
    re.IGNORECASE,
)
CONTACT_SELECTION_ACTION_PATTERN = re.compile(
    r"(?:(?:选择|切换(?:到|至)?|改变当前).{0,8}(?:其他)?(?:联系人|聊天对象)|"
    r"\b(?:select|switch|change)\b.{0,16}\b(?:contact|recipient)\b)",
    re.IGNORECASE,
)
CONTACT_SELECTION_ANCHOR = "__contact_selection__"
LOCAL_INPUT_EFFECT_BOUNDARY_PATTERN = re.compile(
    r"(?:不|未|勿|不要|不得|禁止|不能|避免|无需|无须|"
    r"do\s+not|don't|never|without)"
    r"[^，。；;]{0,28}"
    r"(?:搜索|提交|发送|保存|发布|上传|分享|评论|回复|"
    r"search|submit|send|save|publish|post|upload|share|comment|reply)",
    re.IGNORECASE,
)
LOCAL_KEYBOARD_MODE_PATTERN = re.compile(
    r"(?:(?:输入法|软键盘|键盘).{0,24}"
    r"(?:输入模式|直输模式|英文直输|中文拼音|"
    r"direct[_ -]?latin|chinese[_ -]?pinyin)|"
    r"(?:direct[_ -]?latin|chinese[_ -]?pinyin).{0,24}"
    r"(?:input\s*method|keyboard|ime|输入法|键盘))",
    re.IGNORECASE,
)
PERSISTENT_KEYBOARD_SETTING_PATTERN = re.compile(
    r"(?:默认|全局|系统设置|账号|账户|同步|云端|词库|安装|启用|停用|"
    r"卸载|持久|default|global|system\s+settings?|account|sync|cloud|"
    r"dictionary|install|enable|disable|uninstall|persistent)",
    re.IGNORECASE,
)
REPAIRABLE_INITIAL_GRAPH_ERROR_FRAGMENTS = (
    "文本模型没有返回有效 JSON",
    "文本模型返回内容不是 JSON 对象",
    "缺少字段",
    "必须是对象",
    "必须是数组",
    "包含低层动作表达",
    "关联风险的子目标影响分类必须为",
)
REPAIRABLE_REPLAN_ERROR_FRAGMENTS = (
    "文本模型没有返回有效 JSON",
    "文本模型返回内容不是 JSON 对象",
    "协议外字段",
    "包含低层动作表达",
    "任务图至少需要一个全局完成条件",
    "可推进任务图必须且只能有一个活动子目标",
    "active_subgoal_id",
    "read_only 完成复核",
    "命名页面完成声明缺少结构化画面身份锚点",
    "子目标使用了当前观察之外的完成证据",
    "matched controller_transition 未完成其绑定的 navigation_only 子目标",
)
MISMATCH_BLOCKED_CLARIFICATION = (
    "动作后的新画面未证明预期结果，且当前没有可验证的安全替代路径；"
    "请说明希望继续原目标还是停止任务。"
)
class JsonTaskGraphProvider(Protocol):
    configured: bool

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int = 2000) -> str: ...


class TaskGraphError(ValueError):
    pass


@dataclass(frozen=True)
class TargetApp:
    app_id: str
    app_name: str

    def validate(self) -> None:
        if not ID_PATTERN.fullmatch(self.app_id):
            raise TaskGraphError(f"目标 App ID 无效：{self.app_id!r}")
        _require_text(self.app_name, "target_apps.app_name")


@dataclass(frozen=True)
class GraphGoal:
    objective: str
    target_apps: tuple[TargetApp, ...]
    entities: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text(self.objective, "goal.objective")
        _reject_low_level_instruction(self.objective, "goal.objective")
        app_ids: set[str] = set()
        for app in self.target_apps:
            app.validate()
            if app.app_id in app_ids:
                raise TaskGraphError(f"目标 App ID 重复：{app.app_id}")
            app_ids.add(app.app_id)
        _reject_control_fields(self.entities, "goal.entities")
        input_text = self.entities.get("input_text")
        if input_text is not None:
            if not isinstance(input_text, str) or not input_text or len(input_text) > 100:
                raise TaskGraphError(
                    "goal.entities.input_text 必须为1～100个逐字输入字符。"
                )
            if "\n" in input_text or "\r" in input_text:
                raise TaskGraphError("goal.entities.input_text 不得包含换行。")


@dataclass(frozen=True)
class CompletionCondition:
    condition_id: str
    description: str
    evidence_required: tuple[str, ...]
    satisfied: bool = False
    evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        _validate_id(self.condition_id, "完成条件 ID")
        _require_text(self.description, "completion_conditions.description")
        _reject_low_level_instruction(
            self.description,
            "completion_conditions.description",
        )
        _validate_text_list(self.evidence_required, "evidence_required", required=True)
        for item in self.evidence_required:
            _reject_low_level_completion_evidence(
                item,
                "completion_conditions.evidence_required",
            )
        _validate_text_list(self.evidence, "evidence", required=False)
        if self.satisfied and not self.evidence:
            raise TaskGraphError(f"已满足的完成条件缺少可见证据：{self.condition_id}")
        if not self.satisfied and self.evidence:
            raise TaskGraphError(f"未满足的完成条件不能携带完成证据：{self.condition_id}")


@dataclass(frozen=True)
class RiskAction:
    risk_id: str
    description: str
    external_effect: str
    risk_type: str
    risk_level: str
    subgoal_ids: tuple[str, ...]
    confirmation_required: bool = True

    def validate(self) -> None:
        _validate_id(self.risk_id, "风险 ID")
        _require_text(self.description, "risk_actions.description")
        _require_text(self.external_effect, "risk_actions.external_effect")
        _reject_low_level_instruction(self.description, "risk_actions.description")
        _reject_low_level_instruction(
            self.external_effect,
            "risk_actions.external_effect",
        )
        if self.risk_type not in RISK_TYPES:
            raise TaskGraphError(f"通用风险类型无效：{self.risk_type}")
        if self.risk_level not in RISK_LEVELS:
            raise TaskGraphError(f"风险等级无效：{self.risk_level}")
        if self.confirmation_required is not True:
            raise TaskGraphError(f"风险动作必须等待用户确认：{self.risk_id}")
        _validate_id_list(self.subgoal_ids, "risk_actions.subgoal_ids", required=True)


@dataclass(frozen=True)
class Subgoal:
    subgoal_id: str
    objective: str
    status: str
    depends_on: tuple[str, ...]
    constraints: tuple[str, ...]
    completion_conditions: tuple[str, ...]
    completion_evidence: tuple[str, ...]
    risk_action_ids: tuple[str, ...]
    external_impact: str

    def validate(self, *, input_text: Any = "") -> None:
        _validate_id(self.subgoal_id, "子目标 ID")
        _require_text(self.objective, "subgoals.objective")
        if self.status not in SUBGOAL_STATUSES:
            raise TaskGraphError(f"子目标状态无效：{self.status}")
        _validate_id_list(self.depends_on, "subgoals.depends_on", required=False)
        _validate_text_list(self.constraints, "subgoals.constraints", required=False)
        for item in self.constraints:
            _reject_low_level_instruction(
                item,
                "subgoals.constraints",
                allow_negated=True,
            )
        _validate_text_list(
            self.completion_conditions,
            "subgoals.completion_conditions",
            required=True,
        )
        _reject_low_level_instruction(self.objective, "subgoals.objective")
        for item in self.completion_conditions:
            _reject_low_level_completion_evidence(
                item,
                "subgoals.completion_conditions",
            )
        _validate_text_list(
            self.completion_evidence,
            "subgoals.completion_evidence",
            required=False,
        )
        _validate_id_list(
            self.risk_action_ids,
            "subgoals.risk_action_ids",
            required=False,
        )
        if self.external_impact not in SUBGOAL_EXTERNAL_IMPACTS:
            raise TaskGraphError(
                f"子目标外部影响分类无效：{self.external_impact}"
            )
        inferred_risk_types = _infer_external_risk_types(
            self.objective,
            *self.constraints,
            *self.completion_conditions,
        )
        scoped_input_texts = (
            self.objective,
            *self.constraints,
            *self.completion_conditions,
        )
        describes_local_input_state = any(
            LOCAL_UNSUBMITTED_INPUT_STATE_PATTERN.search(value)
            for value in scoped_input_texts
        )
        if (
            describes_local_input_state
            and self.external_impact in {"read_only", "navigation_only"}
            and not _state_description_binds_canonical_input_text(
                scoped_input_texts,
                input_text if isinstance(input_text, str) else "",
            )
        ):
            raise TaskGraphError(
                f"子目标输入状态未绑定 canonical input_text：{self.subgoal_id}"
            )
        proven_local_input = _is_explicitly_unsubmitted_local_input(
            *scoped_input_texts,
            input_text=input_text,
        )
        proven_local_input_preparation = _is_local_input_preparation_state(
            self.objective,
            *self.completion_conditions,
            input_text=input_text,
        )
        proven_local_keyboard_mode = _is_reversible_local_keyboard_mode(
            self.objective,
            *self.constraints,
            *self.completion_conditions,
        )
        proven_read_only_control_state = (
            self.external_impact == "read_only"
            and _is_read_only_risk_control_state(
                self.objective,
                self.constraints,
                self.completion_conditions,
            )
        )
        if (
            inferred_risk_types
            and not proven_local_input
            and not proven_local_input_preparation
            and not proven_local_keyboard_mode
            and not proven_read_only_control_state
            and self.external_impact in {
            "read_only",
            "navigation_only",
            }
        ):
            raise TaskGraphError(
                f"子目标包含外部状态变化但未声明：{self.subgoal_id}"
            )
        if self.external_impact in {"external_state", "unknown"} and not self.risk_action_ids:
            raise TaskGraphError(
                f"外部状态或未知影响子目标必须关联风险并失败关闭：{self.subgoal_id}"
            )
        if self.risk_action_ids and self.external_impact not in {
            "external_state",
            "unknown",
        }:
            raise TaskGraphError(
                f"关联风险的子目标影响分类必须为 external_state 或 unknown：{self.subgoal_id}"
            )
        if self.status == "completed" and not self.completion_evidence:
            raise TaskGraphError(f"已完成子目标缺少完成证据：{self.subgoal_id}")
        if self.status != "completed" and self.completion_evidence:
            raise TaskGraphError(f"未完成子目标不能携带完成证据：{self.subgoal_id}")


@dataclass(frozen=True)
class ReplanRecord:
    revision: int
    trigger: str
    reason: str
    scene_id: str
    evidence: tuple[str, ...]
    retained_completed_subgoal_ids: tuple[str, ...]
    added_subgoal_ids: tuple[str, ...]
    skipped_subgoal_ids: tuple[str, ...]
    consumed_action_transition_receipt_id: str = ""


@dataclass(frozen=True)
class VerifiedActionTransition:
    """Controller-owned receipt for exactly one confirmed action transition.

    This receipt is deliberately not visual evidence.  It proves which scoped
    action was consumed and which fresh observation followed it; claims about
    page state still has to be grounded in ``visible_evidence``.  Only a typed
    controller-transition reference may complete its bound navigation-only
    subgoal; it never becomes a visual or external-state fact.
    """

    receipt_id: str
    session_id: str
    task_id: str
    device_id: str
    prior_revision: int
    subgoal_id: str
    decision_node_id: str
    action_digest: str
    rebound_action_digest: str
    resolved_action_digest: str
    action_kind: str
    before_observation_id: str
    before_fingerprint: str
    after_observation_id: str
    after_fingerprint: str
    physical_actions: int
    outcome: str
    errors: tuple[str, ...] = ()
    controller_completion_evidence: tuple[str, ...] = ()
    protocol_version: str = VERIFIED_ACTION_TRANSITION_PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != VERIFIED_ACTION_TRANSITION_PROTOCOL_VERSION:
            raise TaskGraphError(
                f"动作转换回执协议版本无效：{self.protocol_version}"
            )
        _require_text(self.receipt_id, "action_transition.receipt_id")
        _require_text(self.session_id, "action_transition.session_id")
        if not ID_PATTERN.fullmatch(self.receipt_id):
            raise TaskGraphError(
                f"动作转换回执 receipt_id 无效：{self.receipt_id!r}"
            )
        if not TASK_ID_PATTERN.fullmatch(self.task_id):
            raise TaskGraphError(f"动作转换回执 task_id 无效：{self.task_id!r}")
        if not DEVICE_ID_PATTERN.fullmatch(self.device_id):
            raise TaskGraphError(f"动作转换回执 device_id 无效：{self.device_id!r}")
        if (
            isinstance(self.prior_revision, bool)
            or not isinstance(self.prior_revision, int)
            or self.prior_revision < 1
        ):
            raise TaskGraphError("动作转换回执 prior_revision 必须是正整数。")
        for field_name in (
            "subgoal_id",
            "decision_node_id",
            "action_digest",
            "rebound_action_digest",
            "resolved_action_digest",
            "action_kind",
            "before_observation_id",
            "before_fingerprint",
            "after_observation_id",
            "after_fingerprint",
        ):
            _require_text(getattr(self, field_name), f"action_transition.{field_name}")
        for field_name in (
            "action_digest",
            "rebound_action_digest",
            "resolved_action_digest",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)):
                raise TaskGraphError(
                    f"动作转换回执 {field_name} 必须是 64 位小写 SHA-256。"
                )
        if self.before_observation_id == self.after_observation_id:
            raise TaskGraphError("动作转换回执必须绑定新的动作后 observation_id。")
        if (
            isinstance(self.physical_actions, bool)
            or not isinstance(self.physical_actions, int)
            or self.physical_actions != 1
        ):
            raise TaskGraphError("动作转换回执必须且只能证明 1 次物理动作。")
        if self.outcome not in {"matched", "mismatched"}:
            raise TaskGraphError(f"动作转换回执 outcome 无效：{self.outcome}")
        _validate_text_list(self.errors, "action_transition.errors", required=False)
        _validate_text_list(
            self.controller_completion_evidence,
            "action_transition.controller_completion_evidence",
            required=False,
        )
        if self.outcome == "matched" and self.errors:
            raise TaskGraphError("matched 动作转换回执不能同时包含验证错误。")
        if self.outcome == "mismatched" and not self.errors:
            raise TaskGraphError("mismatched 动作转换回执必须包含验证错误。")
        if self.outcome == "matched" and self.before_fingerprint == self.after_fingerprint:
            raise TaskGraphError("matched 动作转换回执必须绑定变化后的 fingerprint。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "receipt_id": self.receipt_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "prior_revision": self.prior_revision,
            "subgoal_id": self.subgoal_id,
            "decision_node_id": self.decision_node_id,
            "action_digest": self.action_digest,
            "rebound_action_digest": self.rebound_action_digest,
            "resolved_action_digest": self.resolved_action_digest,
            "action_kind": self.action_kind,
            "before_observation_id": self.before_observation_id,
            "before_fingerprint": self.before_fingerprint,
            "after_observation_id": self.after_observation_id,
            "after_fingerprint": self.after_fingerprint,
            "physical_actions": self.physical_actions,
            "outcome": self.outcome,
            "errors": list(self.errors),
            "controller_completion_evidence": list(
                self.controller_completion_evidence
            ),
        }


@dataclass(frozen=True)
class ControllerTransitionEvidenceRef:
    ref_id: str
    receipt_id: str
    subgoal_id: str
    text: str
    source: str = "controller_transition"

    def validate(self) -> None:
        if self.source != "controller_transition":
            raise TaskGraphError("控制器转换证据来源无效。")
        for field_name in ("ref_id", "receipt_id", "subgoal_id", "text"):
            _require_text(
                getattr(self, field_name),
                f"controller_transition_evidence.{field_name}",
            )
        if not self.ref_id.startswith(f"controller_transition:{self.receipt_id}:"):
            raise TaskGraphError("控制器转换证据 ref_id 未绑定 receipt_id。")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "ref_id": self.ref_id,
            "source": self.source,
            "receipt_id": self.receipt_id,
            "subgoal_id": self.subgoal_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class ObservedState:
    scene_id: str
    summary: str
    visible_evidence: tuple[str, ...]
    grounded_visual_facts: tuple[str, ...] = ()
    last_action_outcome: str = "not_applicable"
    blocked_reasons: tuple[str, ...] = ()
    verified_action_transition: VerifiedActionTransition | None = None
    controller_transition_evidence_refs: tuple[
        ControllerTransitionEvidenceRef, ...
    ] = ()

    def validate(self) -> None:
        _require_text(self.scene_id, "observation.scene_id")
        _require_text(self.summary, "observation.summary")
        _validate_text_list(
            self.visible_evidence,
            "observation.visible_evidence",
            required=True,
        )
        _validate_text_list(
            self.grounded_visual_facts,
            "observation.grounded_visual_facts",
            required=False,
        )
        if self.last_action_outcome not in {
            "not_applicable",
            "matched",
            "mismatched",
            "uncertain",
        }:
            raise TaskGraphError(
                f"观察中的动作结果无效：{self.last_action_outcome}"
            )
        _validate_text_list(
            self.blocked_reasons,
            "observation.blocked_reasons",
            required=False,
        )
        if self.verified_action_transition is not None:
            self.verified_action_transition.validate()
            if self.last_action_outcome != self.verified_action_transition.outcome:
                raise TaskGraphError(
                    "观察动作结果与 verified_action_transition outcome 不一致。"
                )
        for item in self.controller_transition_evidence_refs:
            item.validate()
            if (
                self.verified_action_transition is None
                or item.receipt_id != self.verified_action_transition.receipt_id
                or item.subgoal_id != self.verified_action_transition.subgoal_id
                or item.text
                not in self.verified_action_transition.controller_completion_evidence
            ):
                raise TaskGraphError(
                    "控制器转换证据未绑定当前 verified action transition。"
                )
        if (
            self.verified_action_transition is None
            and self.controller_transition_evidence_refs
        ):
            raise TaskGraphError("无动作回执时不得携带控制器转换证据。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "scene_id": self.scene_id,
            "summary": self.summary,
            "visible_evidence": list(self.visible_evidence),
            "grounded_visual_facts": list(self.grounded_visual_facts),
            "last_action_outcome": self.last_action_outcome,
            "blocked_reasons": list(self.blocked_reasons),
            "verified_action_transition": (
                self.verified_action_transition.to_dict()
                if self.verified_action_transition is not None
                else None
            ),
            "controller_transition_evidence_refs": [
                item.to_dict()
                for item in self.controller_transition_evidence_refs
            ],
        }


@dataclass(frozen=True)
class DynamicTaskGraph:
    task_id: str
    device_id: str
    revision: int
    status: str
    goal: GraphGoal
    constraints: tuple[str, ...]
    completion_conditions: tuple[CompletionCondition, ...]
    risk_actions: tuple[RiskAction, ...]
    subgoals: tuple[Subgoal, ...]
    active_subgoal_id: str | None
    clarification_questions: tuple[str, ...] = ()
    replan_history: tuple[ReplanRecord, ...] = ()
    protocol_version: str = DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION
    raw_user_goal: str = field(default="", repr=False, compare=False)

    def validate(self) -> None:
        if self.protocol_version != DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION:
            raise TaskGraphError(f"任务图协议版本无效：{self.protocol_version}")
        if not TASK_ID_PATTERN.fullmatch(self.task_id):
            raise TaskGraphError(f"task_id 无效：{self.task_id!r}")
        if not DEVICE_ID_PATTERN.fullmatch(self.device_id):
            raise TaskGraphError(f"device_id 无效：{self.device_id!r}")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise TaskGraphError("任务图 revision 必须是正整数。")
        if self.status not in GRAPH_STATUSES:
            raise TaskGraphError(f"任务图状态无效：{self.status}")
        self.goal.validate()
        if self.status != "blocked" and not self.goal.target_apps:
            raise TaskGraphError("可推进的任务图至少需要一个目标 App。")
        _validate_text_list(self.constraints, "constraints", required=False)
        for item in self.constraints:
            _reject_low_level_instruction(item, "constraints", allow_negated=True)
        _validate_text_list(
            self.clarification_questions,
            "clarification_questions",
            required=False,
        )
        for item in self.clarification_questions:
            _reject_low_level_instruction(item, "clarification_questions")

        conditions = _unique_by_id(
            self.completion_conditions,
            lambda item: item.condition_id,
            "完成条件",
        )
        if not conditions:
            raise TaskGraphError("任务图至少需要一个全局完成条件。")
        for condition in conditions.values():
            condition.validate()

        risks = _unique_by_id(self.risk_actions, lambda item: item.risk_id, "风险")
        for risk in risks.values():
            risk.validate()
        subgoals = _unique_by_id(self.subgoals, lambda item: item.subgoal_id, "子目标")
        for subgoal in subgoals.values():
            subgoal.validate(input_text=self.goal.entities.get("input_text"))
            if subgoal.subgoal_id in subgoal.depends_on:
                raise TaskGraphError(f"子目标不能依赖自身：{subgoal.subgoal_id}")
            missing_dependencies = set(subgoal.depends_on) - set(subgoals)
            if missing_dependencies:
                raise TaskGraphError(
                    f"子目标 {subgoal.subgoal_id} 依赖不存在节点："
                    + ", ".join(sorted(missing_dependencies))
                )
            missing_risks = set(subgoal.risk_action_ids) - set(risks)
            if missing_risks:
                raise TaskGraphError(
                    f"子目标 {subgoal.subgoal_id} 引用不存在风险："
                    + ", ".join(sorted(missing_risks))
                )
        for risk in risks.values():
            missing_subgoals = set(risk.subgoal_ids) - set(subgoals)
            if missing_subgoals:
                raise TaskGraphError(
                    f"风险 {risk.risk_id} 引用不存在子目标："
                    + ", ".join(sorted(missing_subgoals))
                )
            for subgoal_id in risk.subgoal_ids:
                if risk.risk_id not in subgoals[subgoal_id].risk_action_ids:
                    raise TaskGraphError(
                        f"风险与子目标引用不对称：{risk.risk_id} / {subgoal_id}"
                    )
        for subgoal in subgoals.values():
            inferred_types = _infer_external_risk_types(
                subgoal.objective,
                *subgoal.constraints,
                *subgoal.completion_conditions,
            )
            if _is_explicitly_unsubmitted_local_input(
                subgoal.objective,
                *subgoal.constraints,
                *subgoal.completion_conditions,
                input_text=self.goal.entities.get("input_text"),
            ):
                # The helper already rechecks the whole scoped text after
                # removing only a carrier's editable-capability adjective and
                # refuses every concrete external effect.  Keep the graph
                # validator aligned with that same formal proof.
                inferred_types = frozenset()
            if _is_local_input_preparation_state(
                subgoal.objective,
                *subgoal.completion_conditions,
                input_text=self.goal.entities.get("input_text"),
            ):
                # Visibility/editability/focus are reversible carrier states.
                # Remove only the generic ambiguity; concrete effects remain.
                inferred_types = inferred_types - {"unknown_external_effect"}
            if _is_reversible_local_keyboard_mode(
                subgoal.objective,
                *subgoal.constraints,
                *subgoal.completion_conditions,
            ):
                inferred_types = inferred_types - {"unknown_external_effect"}
            if (
                subgoal.external_impact == "read_only"
                and _is_read_only_risk_control_state(
                    subgoal.objective,
                    subgoal.constraints,
                    subgoal.completion_conditions,
                )
            ):
                inferred_types = frozenset()
            linked_types = {
                risks[risk_id].risk_type for risk_id in subgoal.risk_action_ids
            }
            missing_types = inferred_types - linked_types
            if missing_types:
                raise TaskGraphError(
                    f"子目标 {subgoal.subgoal_id} 缺少匹配的通用风险类型："
                    + ", ".join(sorted(missing_types))
                )
        if (
            self.status != "blocked"
            and _describes_external_state_change(self.goal.objective)
            and not _is_explicitly_unsubmitted_local_input(
                self.raw_user_goal,
                self.goal.objective,
                *self.constraints,
                input_text=self.goal.entities.get("input_text"),
            )
            and not _is_reversible_local_keyboard_mode(
                self.raw_user_goal,
                self.goal.objective,
                *self.constraints,
            )
            and not risks
        ):
            raise TaskGraphError("外部状态目标必须声明风险动作并等待确认。")
        _reject_dependency_cycles(subgoals)

        active = [item.subgoal_id for item in self.subgoals if item.status == "active"]
        if self.status in {"ready", "running", "awaiting_confirmation"}:
            if len(active) != 1 or self.active_subgoal_id != active[0]:
                raise TaskGraphError("可推进任务图必须且只能有一个活动子目标。")
            active_node = subgoals[active[0]]
            unfinished_dependencies = [
                dependency
                for dependency in active_node.depends_on
                if subgoals[dependency].status != "completed"
            ]
            if unfinished_dependencies:
                raise TaskGraphError(
                    "活动子目标存在未完成依赖：" + ", ".join(unfinished_dependencies)
                )
            if self.status == "awaiting_confirmation" and not active_node.risk_action_ids:
                raise TaskGraphError("等待确认状态必须关联当前子目标的风险动作。")
            if (
                active_node.external_impact in {"external_state", "unknown"}
                and self.status != "awaiting_confirmation"
            ):
                raise TaskGraphError(
                    "外部状态或未知影响子目标成为 current_subgoal 时必须等待用户确认。"
                )
            if (
                self.status == "awaiting_confirmation"
                and active_node.external_impact not in {"external_state", "unknown"}
            ):
                raise TaskGraphError("等待确认状态只能用于外部状态或未知影响子目标。")
        elif active or self.active_subgoal_id is not None:
            raise TaskGraphError("完成或阻塞任务图不能保留活动子目标。")

        if self.status == "completed":
            if not all(item.satisfied for item in self.completion_conditions):
                raise TaskGraphError("任务完成必须满足全部全局完成条件。")
            if any(item.status in {"pending", "active", "blocked"} for item in self.subgoals):
                raise TaskGraphError("任务完成时不能保留未决子目标。")
        if self.status == "blocked" and not (
            self.clarification_questions
            or any(item.status == "blocked" for item in self.subgoals)
        ):
            raise TaskGraphError("阻塞任务图必须说明澄清问题或阻塞子目标。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value.pop("raw_user_goal", None)
        value["goal"]["target_apps"] = [asdict(app) for app in self.goal.target_apps]
        value["constraints"] = list(self.constraints)
        value["completion_conditions"] = [
            {
                **asdict(item),
                "evidence_required": list(item.evidence_required),
                "evidence": list(item.evidence),
            }
            for item in self.completion_conditions
        ]
        value["risk_actions"] = [
            {**asdict(item), "subgoal_ids": list(item.subgoal_ids)}
            for item in self.risk_actions
        ]
        value["subgoals"] = [
            {
                **asdict(item),
                "depends_on": list(item.depends_on),
                "constraints": list(item.constraints),
                "completion_conditions": list(item.completion_conditions),
                "completion_evidence": list(item.completion_evidence),
                "risk_action_ids": list(item.risk_action_ids),
            }
            for item in self.subgoals
        ]
        value["clarification_questions"] = list(self.clarification_questions)
        value["replan_history"] = [
            {
                **asdict(item),
                "evidence": list(item.evidence),
                "retained_completed_subgoal_ids": list(
                    item.retained_completed_subgoal_ids
                ),
                "added_subgoal_ids": list(item.added_subgoal_ids),
                "skipped_subgoal_ids": list(item.skipped_subgoal_ids),
            }
            for item in self.replan_history
        ]
        current = next(
            (
                item
                for item in value["subgoals"]
                if item["subgoal_id"] == self.active_subgoal_id
            ),
            None,
        )
        value["current_subgoal"] = current
        return value

    def active_subgoal(self) -> Subgoal | None:
        self.validate()
        if self.active_subgoal_id is None:
            return None
        return next(item for item in self.subgoals if item.subgoal_id == self.active_subgoal_id)

    def to_qwen_context(
        self,
        *,
        confirmed_risk_ids: tuple[str, ...] = (),
        confirmed_task_id: str | None = None,
        confirmed_device_id: str | None = None,
        confirmed_subgoal_id: str | None = None,
        confirmed_revision: int | None = None,
    ) -> dict[str, Any]:
        """Expose only the current high-level target and safety context to Qwen."""

        value = self.to_dict()
        current = value["current_subgoal"]
        current_risk_ids = set(current["risk_action_ids"] if current else [])
        confirmed = set(confirmed_risk_ids)
        if confirmed and confirmed_task_id != self.task_id:
            raise TaskGraphError("确认记录 task_id 不匹配，禁止跨 task 复用。")
        if confirmed and confirmed_device_id != self.device_id:
            raise TaskGraphError("确认记录 device_id 不匹配，禁止跨 device 复用。")
        if confirmed and confirmed_subgoal_id != self.active_subgoal_id:
            raise TaskGraphError("确认记录不属于 current_subgoal，禁止跨子目标复用。")
        if confirmed and confirmed_revision != self.revision:
            raise TaskGraphError("确认记录 revision 不匹配，禁止跨 revision 复用。")
        if not confirmed and (
            confirmed_task_id is not None
            or confirmed_device_id is not None
            or confirmed_subgoal_id is not None
            or confirmed_revision is not None
        ):
            raise TaskGraphError("确认作用域不能脱离 confirmed_risk_ids 单独提供。")
        unknown_confirmations = confirmed - current_risk_ids
        if unknown_confirmations:
            raise TaskGraphError(
                "确认记录不属于 current_subgoal："
                + ", ".join(sorted(unknown_confirmations))
            )
        confirmation_required = bool(
            current
            and current["external_impact"] in {"external_state", "unknown"}
        )
        confirmation_granted = bool(
            confirmation_required
            and current_risk_ids
            and current_risk_ids.issubset(confirmed)
        )
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "task_status": self.status,
            "goal": value["goal"],
            "global_constraints": value["constraints"],
            "goal_completion_conditions": value["completion_conditions"],
            "current_subgoal": current,
            "current_external_impact": (
                current["external_impact"] if current else None
            ),
            "risk_actions": [
                item
                for item in value["risk_actions"]
                if item["risk_id"] in current_risk_ids
            ],
            "confirmation_gate": {
                "required": confirmation_required,
                "state": (
                    "confirmed"
                    if confirmation_granted
                    else "awaiting_confirmation"
                    if confirmation_required
                    else "not_required"
                ),
                "risk_ids": sorted(current_risk_ids),
                "scope": {
                    "task_id": self.task_id,
                    "device_id": self.device_id,
                    "revision": self.revision,
                    "subgoal_id": self.active_subgoal_id,
                },
                "external_state_action_allowed": confirmation_granted,
            },
        }


class DeepSeekTaskGraphPlanner:
    """Create and revise high-level task graphs without any execution capability."""

    def __init__(
        self,
        provider: JsonTaskGraphProvider,
        *,
        risk_audit_provider: JsonRiskAuditProvider | None = None,
    ) -> None:
        self.provider = provider
        self.risk_auditor = SemanticRiskAuditor(risk_audit_provider or provider)
        self.last_raw_response = ""
        self.last_risk_audit: SemanticRiskAuditReport | None = None

    @property
    def risk_audit_call_count(self) -> int:
        return self.risk_auditor.call_count

    def plan(
        self,
        raw_goal: str,
        *,
        device_id: str,
        task_id: str | None = None,
    ) -> DynamicTaskGraph:
        text = " ".join(str(raw_goal or "").strip().split())
        if not text:
            raise TaskGraphError("用户目标不能为空。")
        _validate_device_id(device_id)
        resolved_task_id = task_id or uuid.uuid4().hex
        _validate_task_id(resolved_task_id)
        self._require_provider()
        prompt = _initial_prompt(text)
        try:
            graph = self._request_graph(
                prompt,
                task_id=resolved_task_id,
                device_id=device_id,
                revision=1,
                raw_user_goal=text,
                validate=False,
            )
            graph = _normalize_initial_local_navigation(graph)
            graph = _normalize_unique_active_frontier(graph)
            graph = _normalize_initial_confirmation_status(graph)
            graph.validate()
        except TaskGraphError as exc:
            if not _retryable_initial_output_error(exc):
                raise
            initial_error = exc
            graph = self._request_graph(
                _repair_initial_prompt(
                    text,
                    invalid_response=self.last_raw_response,
                    validation_error=str(exc),
                ),
                task_id=resolved_task_id,
                device_id=device_id,
                revision=1,
                raw_user_goal=text,
                validate=False,
            )
            graph = _normalize_initial_local_navigation(graph)
            graph = _normalize_unique_active_frontier(graph)
            graph = _normalize_initial_confirmation_status(graph)
            try:
                graph.validate()
            except TaskGraphError as repair_error:
                if (
                    not _retryable_initial_output_error(repair_error)
                    or _initial_repair_error_category(repair_error)
                    == _initial_repair_error_category(initial_error)
                ):
                    raise
                graph = self._request_graph(
                    _repair_initial_prompt(
                        text,
                        invalid_response=self.last_raw_response,
                        validation_error=str(repair_error),
                    ),
                    task_id=resolved_task_id,
                    device_id=device_id,
                    revision=1,
                    raw_user_goal=text,
                    validate=False,
                )
                graph = _normalize_initial_local_navigation(graph)
                graph = _normalize_unique_active_frontier(graph)
                graph = _normalize_initial_confirmation_status(graph)
                graph.validate()
        try:
            self._audit_and_validate_graph(graph)
        except TaskGraphError as exc:
            if not _retryable_safe_initial_audit_conflict(text, exc):
                raise
            graph = self._request_graph(
                _retry_safe_initial_audit_prompt(text),
                task_id=resolved_task_id,
                device_id=device_id,
                revision=1,
                raw_user_goal=text,
                validate=False,
            )
            graph = _normalize_initial_local_navigation(graph)
            graph = _normalize_unique_active_frontier(graph)
            graph = _normalize_initial_confirmation_status(graph)
            graph.validate()
            self._audit_and_validate_graph(graph)
        if (
            graph.status == "completed"
            or any(item.status == "completed" for item in graph.subgoals)
            or any(item.satisfied for item in graph.completion_conditions)
        ):
            raise TaskGraphError("初始规划没有观察证据，不能宣称目标或子目标已完成。")
        return graph

    def replan(
        self,
        graph: DynamicTaskGraph,
        observation: ObservedState,
        *,
        trigger: str,
        reason: str,
    ) -> DynamicTaskGraph:
        graph.validate()
        observation.validate()
        if trigger not in REPLAN_TRIGGERS:
            raise TaskGraphError(f"不支持的重规划触发原因：{trigger}")
        _require_text(reason, "replan.reason")
        self._require_provider()
        prompt = _replan_prompt(graph, observation, trigger=trigger, reason=reason)
        try:
            candidate = self._request_graph(
                prompt,
                task_id=graph.task_id,
                device_id=graph.device_id,
                revision=graph.revision + 1,
                raw_user_goal=graph.raw_user_goal or graph.goal.objective,
                validate=False,
            )
            candidate = _restore_completed_history_evidence(graph, candidate)
            candidate = _canonicalize_literal_visible_evidence_clauses(
                graph,
                candidate,
                observation,
            )
            candidate = _normalize_unique_active_frontier(candidate)
            self._validate_replan_candidate(
                graph,
                candidate,
                observation,
                trigger=trigger,
            )
        except TaskGraphError as exc:
            if not _retryable_replan_output_error(exc):
                raise
            invalid_response = self.last_raw_response
            candidate = self._request_graph(
                _repair_replan_prompt(
                    graph,
                    observation,
                    trigger=trigger,
                    reason=reason,
                    invalid_response=invalid_response,
                    validation_error=str(exc),
                ),
                task_id=graph.task_id,
                device_id=graph.device_id,
                revision=graph.revision + 1,
                raw_user_goal=graph.raw_user_goal or graph.goal.objective,
                validate=False,
            )
            candidate = _restore_completed_history_evidence(graph, candidate)
            candidate = _canonicalize_literal_visible_evidence_clauses(
                graph,
                candidate,
                observation,
            )
            candidate = _normalize_unique_active_frontier(candidate)
            try:
                self._validate_replan_candidate(
                    graph,
                    candidate,
                    observation,
                    trigger=trigger,
                )
            except TaskGraphError as repair_error:
                normalized = _normalize_blocked_mismatch_clarification(
                    candidate,
                    trigger=trigger,
                    error=repair_error,
                )
                if normalized is None:
                    raise
                candidate = _restore_completed_history_evidence(graph, normalized)
                self._validate_replan_candidate(
                    graph,
                    candidate,
                    observation,
                    trigger=trigger,
                )
        previous_ids = {item.subgoal_id for item in graph.subgoals}
        completed_ids = tuple(
            item.subgoal_id for item in graph.subgoals if item.status == "completed"
        )
        added_ids = tuple(
            item.subgoal_id for item in candidate.subgoals if item.subgoal_id not in previous_ids
        )
        skipped_ids = tuple(
            item.subgoal_id
            for item in candidate.subgoals
            if item.status == "skipped"
            and next(
                (old.status for old in graph.subgoals if old.subgoal_id == item.subgoal_id),
                None,
            )
            != "skipped"
        )
        record = ReplanRecord(
            revision=candidate.revision,
            trigger=trigger,
            reason=reason.strip(),
            scene_id=observation.scene_id,
            evidence=observation.visible_evidence,
            retained_completed_subgoal_ids=completed_ids,
            added_subgoal_ids=added_ids,
            skipped_subgoal_ids=skipped_ids,
            consumed_action_transition_receipt_id=(
                observation.verified_action_transition.receipt_id
                if observation.verified_action_transition is not None
                and trigger in {
                    "action_result_matched",
                    "action_result_mismatch",
                }
                else ""
            ),
        )
        revised = replace(candidate, replan_history=graph.replan_history + (record,))
        revised.validate()
        return revised

    def _validate_replan_candidate(
        self,
        graph: DynamicTaskGraph,
        candidate: DynamicTaskGraph,
        observation: ObservedState,
        *,
        trigger: str,
    ) -> None:
        """Apply every safety and evidence check to one replan candidate."""

        if candidate.revision != graph.revision + 1:
            raise TaskGraphError(
                "重规划 revision 必须严格等于上一 revision + 1。"
            )
        transition = observation.verified_action_transition
        if trigger in {"action_result_matched", "action_result_mismatch"}:
            if transition is None:
                raise TaskGraphError("动作结果重规划缺少本地 verified action transition。")
            expected_outcome = (
                "matched" if trigger == "action_result_matched" else "mismatched"
            )
            if transition.outcome != expected_outcome:
                raise TaskGraphError("重规划触发与本地动作转换回执 outcome 不一致。")
            previous_current = graph.active_subgoal()
            if (
                transition.task_id != graph.task_id
                or transition.device_id != graph.device_id
                or transition.prior_revision != graph.revision
                or transition.subgoal_id != graph.active_subgoal_id
                or transition.after_observation_id != observation.scene_id
                or previous_current is None
            ):
                raise TaskGraphError("动作转换回执未严格绑定上一任务图及当前观察。")
            consumed_receipts = {
                item.consumed_action_transition_receipt_id
                for item in graph.replan_history
                if item.consumed_action_transition_receipt_id
            }
            if transition.receipt_id in consumed_receipts:
                raise TaskGraphError("动作转换回执已经消费，禁止跨 revision 重放。")
        elif trigger == "observation_changed" and transition is not None:
            raise TaskGraphError("纯观察变化不得携带动作执行回执。")

        _validate_external_impact_revision(graph, candidate)
        _validate_preserved_risk_ids(graph, candidate)
        candidate.validate()
        previous_current = graph.active_subgoal()
        candidate_current = candidate.active_subgoal()
        if (
            trigger == "action_result_mismatch"
            and previous_current is not None
            and next(
                (
                    item.status
                    for item in candidate.subgoals
                    if item.subgoal_id == previous_current.subgoal_id
                ),
                None,
            )
            == "completed"
        ):
            raise TaskGraphError(
                "动作结果不匹配时不能完成回执绑定的上一活动子目标。"
            )
        if (
            trigger == "subgoal_completed"
            and previous_current is not None
            and previous_current.external_impact == "read_only"
            and candidate_current is not None
            and candidate_current.external_impact == "read_only"
        ):
            raise TaskGraphError(
                "read_only 完成复核不能继续保留 read_only 活动子目标；"
                "当前证据足够时应完成，证据不足时应阻塞，或推进到后续非只读子目标。"
            )
        self._audit_and_validate_graph(candidate)
        _validate_revision(graph, candidate, observation)

    def _request_graph(
        self,
        prompt: str,
        *,
        task_id: str,
        device_id: str,
        revision: int,
        raw_user_goal: str,
        validate: bool = True,
    ) -> DynamicTaskGraph:
        raw = self.provider.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=2400,
        )
        self.last_raw_response = raw
        try:
            payload = _parse_json_object(raw)
        except GenericIntentError as exc:
            raise TaskGraphError(str(exc)) from exc
        payload = _normalize_explicit_ui_label_payload(payload, raw_user_goal)
        graph = _graph_from_payload(
            payload,
            task_id=task_id,
            device_id=device_id,
            revision=revision,
            raw_user_goal=raw_user_goal,
        )
        if validate:
            graph.validate()
        return graph

    def _audit_and_validate_graph(self, graph: DynamicTaskGraph) -> None:
        sources = _risk_audit_sources(graph)
        report = self.risk_auditor.audit(sources)
        sanitized_assessments = []
        for assessment in report.assessments:
            try:
                _reject_low_level_instruction(
                    assessment.reason,
                    "risk_audit.reason",
                )
            except TaskGraphError:
                assessment = replace(
                    assessment,
                    reason=(
                        f"语义风险审计分类为 {assessment.external_impact}；"
                        "原始展示理由因包含低层操作表达已隔离"
                    ),
                )
            sanitized_assessments.append(assessment)
        report = replace(report, assessments=tuple(sanitized_assessments))
        report = _apply_local_risk_supplements(report, sources, graph=graph)
        self.last_risk_audit = report
        _validate_graph_against_risk_audit(graph, report)

    def _require_provider(self) -> None:
        if not self.provider.configured:
            raise TaskGraphError("DeepSeek 动态任务图尚未配置。")


def _initial_prompt(raw_goal: str) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图规划器。你只维护目标和高层子目标，
不观察图片、不选择控件、不输出点击/滑动/输入等动作，也不能输出坐标、Shell 或系统命令。

用户原始目标：{json.dumps(raw_goal, ensure_ascii=False)}

{_schema_prompt()}

初始规划规则：
1. 适用于任意 App 和跨 App 目标，不得生成任何 App 专用固定流程。
2. 子目标描述“应达到什么状态”，不能描述具体按钮、坐标或动作序列。
   用户原始目标可以直接包含点击、滑动、输入等自然语言动作；不要拒绝，也不要把这些动作词
   复制进任务图。应提取该动作希望达到的可见结果状态，例如把“滑动页面找到目标内容”抽象为
   “目标内容在当前页面可见”，具体下一动作仍由 Qwen 根据真实画面决定。
   但用户用“不要、不得、禁止、不能、避免”明确否定的低层动作属于安全约束，必须以同样的
   明确否定形式保留在 constraints 中；不得删除，也不得改写成含糊或双重否定的表达。
   用户对方向或次数的限制也要保留，但必须改写成动作后的状态变化，不得复述动作词。例如把
   “只能向上滑动一次”改写为“页面内容只允许向上移动一次”。
   如果用户明确指“当前页面”“当前应用”或“当前前台”但没有说 App 名称，target_apps 使用
   [{{"app_id":"current_foreground","app_name":"当前前台应用"}}]；不能只因未重复 App 名称而阻塞。
   如果目标界面的字面标签本身含有点击、滑动、输入、长按、拖动等动作词，这仍只是
   可见文字；必须将它逐字保存在goal.entities.target_ui_label，不得复制到goal.objective、
   subgoals.objective、completion_conditions或constraints。这些状态字段只能描述目标页面、区域或
   内容可见，不能把字面标签当成动作指令。
   必须按以下通用语义边界改写，而不是照抄用户动作措辞：
   - “点击或打开某入口”写成“目标页面在前台可见”；
   - “关闭遮挡层”写成“目标页面不再被遮挡，主要内容可见”；
   - “在输入框输入 X”写成“当前输入框内容为 X”，提交边界另存 constraints；
   - “计算某表达式”写成“本机临时结果区域显示该表达式的答案”。
   这些只是跨 App 的结果状态例式，不能据此生成固定步骤或控件选择。
3. 只能有一个 active 子目标；其依赖必须已经 completed（初始图通常无依赖）。
4. 初始规划没有画面证据，所有完成条件 satisfied=false，任何子目标都不能 completed。
5. 每个子目标必须用 external_impact 标为 read_only、navigation_only、external_state 或 unknown。
   会改变账号、数据、交易、发布、发送或其他外部状态的事项必须标为 external_state 并列入
   risk_actions；无法确定影响时标为 unknown。两者都必须关联风险，confirmation_required=true；
   如果成为 active，status 必须为 awaiting_confirmation。
6. read_only 只能描述查看、读取、检查等纯观察结果；navigation_only 只能描述打开或进入页面等
   导航结果。仅改变本机临时界面层级、前后台页面或临时标签页也属于 navigation_only，不得为它
   虚构 risk_actions；但登录/退出账号、修改账号数据或云端同步状态仍属于 external_state。
   当用户要查看、读取或核对某个目标页面的结果，但没有明确说明该结果已经在当前画面中时，
   必须先建立一个 navigation_only 子目标描述“目标页面或目标区域可见”，再建立 read_only 子目标
   描述要核对的结果；不得把潜在导航需求隐藏在单个 read_only 子目标中。
   如果 goal.entities.target_ui_label 是具名入口/分类，而最终完成条件要求另一个结果文字或状态，
   必须再拆分为“具名入口在列表中可见”与“入口对应的目标页面可见”两个 navigation_only 状态，
   最后才是 read_only 结果核对。入口可见绝不能证明其对应页面或最终结果已经可见。
   只改变当前可见输入框中的未提交临时文字，也可归入 navigation_only。写入文字时目标文字必须
   明确非空；将当前唯一未发送/未提交临时草稿恢复为空白时，必须把空白状态写成明确结果而不能
   虚构空字符串 input_text。两者都要求用户直接禁止该上下文中的搜索、提交、发送、保存或发布等
   效果，且句中没有任何未被否定的外部效果。输入并搜索/发送/保存、清除云端或已保存数据、未明确
   禁止提交效果、或含义不清时仍必须标为 external_state 或 unknown。
   不能证明属于这些安全类别时必须标为 unknown，不能为了免确认而猜成安全类别。
7. 风险类型只用通信、内容发布、账号关系、成员关系、权限角色、数据修改/删除、交易支付、
   账号权限或未知外部影响等跨 App 语义，不得描述 App 页面路径。
8. 信息不足时 status=blocked、active_subgoal_id=null，并填写 clarification_questions。
9. 只返回 JSON 对象，不要 Markdown。
"""


def _repair_initial_prompt(
    raw_goal: str,
    *,
    invalid_response: str,
    validation_error: str,
) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图规划器。上一次 JSON 未通过本地协议校验。
请根据校验错误重新生成完整任务图，不要解释、不要局部补丁，也不要输出点击、滑动、输入、
坐标、Shell、系统命令或任何 App 专用固定流程。

用户原始目标：{json.dumps(raw_goal, ensure_ascii=False)}
本地校验错误：{json.dumps(validation_error, ensure_ascii=False)}
上一次无效 JSON：
{invalid_response}

{_schema_prompt()}

修复规则：
1. status 为 ready、running 或 awaiting_confirmation 时，必须恰好一个子目标 status=active，
   且 active_subgoal_id 必须等于该子目标 ID。
2. blocked 或 completed 时 active_subgoal_id=null，且不能有 active 子目标。
3. 至少返回一个全局 completion_conditions；初始规划不得宣称任何条件或子目标已完成。
4. external_state 或 unknown 必须声明并关联风险；成为 active 时必须等待本地用户确认。
   read_only 或 navigation_only 不得关联 risk_actions。仅改变本机临时界面层级、前后台页面或
   临时标签页属于 navigation_only；登录/退出账号、账号数据或云端同步状态不属于此例外。
   只改变当前可见输入框中的未提交临时文字仅在目标文字非空、用户直接禁止相关提交效果、且没有
   任何未否定外部效果时属于 navigation_only；否则仍按 external_state 或 unknown 失败关闭。
5. 如果用户原始目标含有点击、滑动、输入等低层动作措辞，goal、subgoals 和
   completion_conditions 只保留动作希望达到的可见结果状态，不得复述低层动作；具体下一动作
   由 Qwen 根据真实画面决定。例如把“滑动页面找到目标内容”改写为“目标内容在当前页面可见”。
   用户以“不要、不得、禁止、不能、避免”明确否定的动作是例外：必须用同样的明确否定形式
   保留在 constraints 中，例如逐字保留“不要点击其他控件”，不得删除或改成“不点击”。
   对方向或次数的限制必须改写成动作后的状态变化，例如把“只能向上滑动一次”改写为
   “页面内容只允许向上移动一次”，不得把正向低层动作词放入 constraints。
   同样必须把“点击或打开某入口”改写为“目标页面在前台可见”，把“关闭遮挡层”改写为
   “目标页面不再被遮挡，主要内容可见”，把“在输入框输入 X”改写为“当前输入框内容为 X”，
   把“计算某表达式”改写为“本机临时结果区域显示该表达式的答案”。这些是结果状态例式，
   不是固定步骤，也不能出现在 Qwen 动作之前的本地编排中。
6. 如果用户明确指“当前页面”“当前应用”或“当前前台”但未说 App 名称，target_apps 必须使用
   [{{"app_id":"current_foreground","app_name":"当前前台应用"}}]，不得只因缺少 App 名称而阻塞。
7. 字面 UI 标签若包含点击、滑动、输入、长按、拖动等词，必须逐字放在
   goal.entities.target_ui_label，不得出现在goal.objective、subgoals.objective、
   completion_conditions或constraints；状态字段只描述目标页面、区域或内容可见。
8. 只返回符合结构的完整 JSON 对象，不要 Markdown。
"""


def _retry_safe_initial_audit_prompt(raw_goal: str) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图规划器。上一次独立语义风险审计与任务图
分类发生冲突，但本地逐词校验没有发现任何未被否定的外部状态效果。请根据用户原始目标进行
一次独立重新规划；不要沿用上一次任务图或审计结论，不要解释，也不要输出点击、滑动、输入、
坐标、Shell、系统命令或任何 App 专用固定流程。

用户原始目标：{json.dumps(raw_goal, ensure_ascii=False)}

{_schema_prompt()}

重新规划规则：
1. 明确否定或禁止的效果词是约束，不是正向目标；例如“不登录”“不要发送”本身不构成外部
   状态动作，但“不要忘记登录”“不能只查看而要发送”仍包含正向外部效果。
2. read_only、navigation_only、external_state、unknown 必须按完整句子语义重新判断。
3. external_state 或 unknown 必须声明并关联风险；成为 active 时必须等待本地用户确认。
4. 初始规划没有画面证据，不能宣称任何目标或子目标已经完成。
5. 只返回符合结构的完整 JSON 对象，不要 Markdown。
"""


def _retryable_safe_initial_audit_conflict(
    raw_goal: str,
    error: TaskGraphError,
) -> bool:
    return (
        "语义风险审计与任务图分类冲突" in str(error)
        and not _describes_external_state_change(raw_goal)
        and bool(_infer_directly_negated_risk_types(raw_goal))
    )


def _retryable_initial_output_error(error: TaskGraphError) -> bool:
    text = str(error)
    if "协议外字段" in text:
        return False
    return text in REPAIRABLE_INITIAL_GRAPH_ERRORS or any(
        fragment in text for fragment in REPAIRABLE_INITIAL_GRAPH_ERROR_FRAGMENTS
    )


def _initial_repair_error_category(error: TaskGraphError) -> str:
    text = str(error)
    if "包含低层动作表达" in text:
        return "low_level_instruction"
    if "关联风险的子目标影响分类必须为" in text:
        return "safe_impact_with_risk"
    if text in REPAIRABLE_INITIAL_GRAPH_ERRORS:
        return "graph_structure"
    return text


def _dependency_ancestor_map(
    subgoals: dict[str, Subgoal],
) -> dict[str, frozenset[str]]:
    """Return transitive dependency ancestors without assuming a valid DAG."""

    result: dict[str, frozenset[str]] = {}
    for subgoal_id, subgoal in subgoals.items():
        found: set[str] = set()
        pending = list(subgoal.depends_on)
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            parent = subgoals.get(current)
            if parent is not None:
                pending.extend(parent.depends_on)
        result[subgoal_id] = frozenset(found)
    return result


def _local_input_preparation_subgoal_ids(
    graph: DynamicTaskGraph,
    subgoals: dict[str, Subgoal],
    local_unsubmitted_input_ids: set[str],
    ancestor_map: dict[str, frozenset[str]],
) -> set[str]:
    """Prove reversible input-carrier preparation on the canonical input chain.

    This is a local structural attestation, not a model-declared permission.  A
    preparation node may only describe a formal input carrier being visible,
    editable, or focused.  It must be dependency-related to an independently
    proven canonical unsubmitted-input node, and its positive result must not
    contain any external effect.  The attestation never authorizes text entry;
    that remains the separate ``input_verified_text`` action.
    """

    input_text = graph.goal.entities.get("input_text")
    if (
        not isinstance(input_text, str)
        or not input_text.strip()
        or not local_unsubmitted_input_ids
    ):
        return set()

    def dependency_related(left_id: str, right_id: str) -> bool:
        return (
            left_id == right_id
            or left_id in ancestor_map.get(right_id, frozenset())
            or right_id in ancestor_map.get(left_id, frozenset())
        )

    result: set[str] = set()
    for subgoal in subgoals.values():
        if subgoal.subgoal_id in local_unsubmitted_input_ids:
            continue
        if not _is_local_input_preparation_state(
            subgoal.objective,
            *subgoal.completion_conditions,
            input_text=input_text,
        ):
            continue
        if not any(
            dependency_related(subgoal.subgoal_id, input_id)
            for input_id in local_unsubmitted_input_ids
        ):
            continue
        result.add(subgoal.subgoal_id)
    return result


def _risk_is_required_by_positive_result(
    graph: DynamicTaskGraph,
    risk: RiskAction,
) -> bool:
    sources = [graph.goal.objective]
    for condition in graph.completion_conditions:
        sources.extend((condition.description, *condition.evidence_required))
    for subgoal in graph.subgoals:
        sources.extend((subgoal.objective, *subgoal.completion_conditions))
    type_pattern = _external_risk_patterns().get(risk.risk_type)
    if type_pattern is None and risk.risk_type == "unknown_external_effect":
        type_pattern = EXTERNAL_STATE_CHANGE_PATTERN
    phrases = tuple(
        value.strip()
        for value in (risk.description, risk.external_effect)
        if len("".join(value.split())) >= 4
    )
    for source in sources:
        for clause in _positive_effect_clauses(source):
            if type_pattern is not None and _has_unnegated_effect_match(
                type_pattern,
                clause,
            ):
                return True
            for phrase in phrases:
                if _has_unnegated_effect_match(
                    re.compile(re.escape(phrase), re.IGNORECASE),
                    clause,
                ):
                    return True
    return False


def _subgoal_is_safe_without_forbidden_risk(
    graph: DynamicTaskGraph,
    subgoal: Subgoal,
    local_unsubmitted_input_ids: set[str],
    local_input_preparation_ids: set[str],
) -> bool:
    if subgoal.subgoal_id in (
        local_unsubmitted_input_ids | local_input_preparation_ids
    ):
        return True
    if subgoal.external_impact == "read_only":
        return _is_read_only_risk_control_state(
            subgoal.objective,
            subgoal.constraints,
            subgoal.completion_conditions,
        ) or not _infer_external_risk_types(
            subgoal.objective,
            *subgoal.completion_conditions,
        )
    positive_text = "；".join((subgoal.objective, *subgoal.completion_conditions))
    return bool(
        LOCAL_TRANSIENT_NAVIGATION_PATTERN.search(positive_text)
        or REVERSIBLE_NAVIGATION_EFFECT_PATTERN.search(positive_text)
    ) and not any(
        _has_unnegated_effect_match(pattern, positive_text)
        for pattern in _external_risk_patterns().values()
    )


def _purely_forbidden_initial_risks(
    graph: DynamicTaskGraph,
    subgoals: dict[str, Subgoal],
    local_unsubmitted_input_ids: set[str],
    local_input_preparation_ids: set[str],
) -> tuple[set[str], set[str]]:
    removable_ids: set[str] = set()
    safe_subgoal_ids: set[str] = set()
    for risk in graph.risk_actions:
        linked = tuple(subgoals.get(item) for item in risk.subgoal_ids)
        if not linked or any(item is None for item in linked):
            continue
        anchors = _risk_effect_action_anchors(
            risk.description,
            risk.external_effect,
        )
        constraints = tuple(graph.constraints) + tuple(
            constraint
            for item in linked
            if item is not None
            for constraint in item.constraints
        )
        if not anchors or not all(
            any(
                _text_directly_negates_action_anchor(constraint, anchor)
                for constraint in constraints
            )
            for anchor in anchors
        ):
            continue
        if _risk_is_required_by_positive_result(graph, risk):
            continue
        if not all(
            item is not None
            and _subgoal_is_safe_without_forbidden_risk(
                graph,
                item,
                local_unsubmitted_input_ids,
                local_input_preparation_ids,
            )
            for item in linked
        ):
            continue
        removable_ids.add(risk.risk_id)
        safe_subgoal_ids.update(
            item.subgoal_id for item in linked if item is not None
        )
    return removable_ids, safe_subgoal_ids


def _normalize_initial_local_navigation(graph: DynamicTaskGraph) -> DynamicTaskGraph:
    """Remove only self-contradictory low-risk markers from proven local navigation."""

    if graph.status not in {"ready", "running", "awaiting_confirmation"}:
        return graph
    subgoals = {item.subgoal_id: item for item in graph.subgoals}
    local_unsubmitted_input_ids = {
        item.subgoal_id
        for item in graph.subgoals
        if LOCAL_UNSUBMITTED_INPUT_STATE_PATTERN.search(
            "；".join(
                (
                    item.objective,
                    *item.constraints,
                    *item.completion_conditions,
                )
            )
        )
        and _is_explicitly_unsubmitted_local_input(
            graph.raw_user_goal or graph.goal.objective,
            graph.goal.objective,
            item.objective,
            *graph.constraints,
            *item.constraints,
            *item.completion_conditions,
            input_text=graph.goal.entities.get("input_text"),
        )
    }
    local_temporary_clear_ids = {
        item.subgoal_id
        for item in graph.subgoals
        if _is_explicit_local_temporary_draft_clear(graph, item)
    }
    ancestor_map = _dependency_ancestor_map(subgoals)
    local_input_preparation_ids = _local_input_preparation_subgoal_ids(
        graph,
        subgoals,
        local_unsubmitted_input_ids,
        ancestor_map,
    )
    removable_ids, safe_workflow_ids = _purely_forbidden_initial_risks(
        graph,
        subgoals,
        local_unsubmitted_input_ids,
        local_input_preparation_ids,
    )

    def dependency_related(left_id: str, right_id: str) -> bool:
        return (
            left_id == right_id
            or left_id in ancestor_map.get(right_id, frozenset())
            or right_id in ancestor_map.get(left_id, frozenset())
        )

    for risk in graph.risk_actions:
        linked = tuple(subgoals.get(item) for item in risk.subgoal_ids)
        if (
            risk.risk_type
            in {
                "unknown_external_effect",
                "message_or_communication",
                "content_publication",
                "data_mutation",
                "data_deletion",
            }
            and risk.subgoal_ids
            and set(risk.subgoal_ids) <= local_temporary_clear_ids
        ):
            removable_ids.add(risk.risk_id)
            safe_workflow_ids.update(risk.subgoal_ids)
            continue
        if (
            risk.risk_type not in {
                "unknown_external_effect",
                "data_mutation",
                "message_or_communication",
                "content_publication",
            }
            or not linked
            or any(item is None for item in linked)
        ):
            continue
        local_input_only = all(
            item is not None
            and item.subgoal_id
            in (local_unsubmitted_input_ids | local_input_preparation_ids)
            for item in linked
        )
        local_input_workflow = bool(
            local_unsubmitted_input_ids
            and risk.risk_level == "low"
            and _explicitly_denies_external_effect(risk.external_effect)
            and LOCAL_UNSUBMITTED_WORKFLOW_RISK_PATTERN.search(risk.description)
            and _infer_external_risk_types(risk.description)
            <= {"unknown_external_effect"}
            and all(
                item is not None
                and any(
                    dependency_related(item.subgoal_id, input_id)
                    for input_id in local_unsubmitted_input_ids
                )
                and (
                    item.subgoal_id in local_input_preparation_ids
                    or not _infer_external_risk_types(
                        item.objective,
                        *item.constraints,
                        *item.completion_conditions,
                    )
                )
                for item in linked
            )
        )
        if local_input_only:
            pass
        elif not local_input_workflow and (
            risk.risk_level != "low"
            or not _explicitly_denies_external_effect(risk.external_effect)
        ):
            continue
        if all(
            item.external_impact in {
                "read_only",
                "navigation_only",
                "external_state",
                "unknown",
            }
            and (
                LOCAL_TRANSIENT_NAVIGATION_PATTERN.search(item.objective)
                or item.subgoal_id in local_unsubmitted_input_ids
                or item.subgoal_id in local_input_preparation_ids
                or local_input_workflow
            )
            and (
                item.subgoal_id in local_input_preparation_ids
                or not _infer_external_risk_types(
                    item.objective,
                    *item.constraints,
                    *item.completion_conditions,
                )
            )
            for item in linked
            if item is not None
        ):
            removable_ids.add(risk.risk_id)
            if local_input_workflow:
                safe_workflow_ids.update(
                    item.subgoal_id for item in linked if item is not None
                )
    normalized_subgoals = tuple(
        replace(
            item,
            external_impact=(
                "navigation_only"
                if item.risk_action_ids
                and set(item.risk_action_ids) <= removable_ids
                and (
                    LOCAL_TRANSIENT_NAVIGATION_PATTERN.search(item.objective)
                    or item.subgoal_id in local_unsubmitted_input_ids
                    or item.subgoal_id in local_input_preparation_ids
                    or item.subgoal_id in local_temporary_clear_ids
                    or item.subgoal_id in safe_workflow_ids
                )
                and item.external_impact != "read_only"
                else item.external_impact
            ),
            risk_action_ids=tuple(
                risk_id
                for risk_id in item.risk_action_ids
                if risk_id not in removable_ids
            ),
        )
        for item in graph.subgoals
    )
    active_ids = [item.subgoal_id for item in normalized_subgoals if item.status == "active"]
    active_subgoal_id = graph.active_subgoal_id
    if not active_ids and active_subgoal_id is not None:
        selected = next(
            (item for item in normalized_subgoals if item.subgoal_id == active_subgoal_id),
            None,
        )
        if (
            selected is not None
            and selected.status == "pending"
            and not selected.depends_on
            and selected.external_impact in {"read_only", "navigation_only"}
            and not selected.risk_action_ids
        ):
            normalized_subgoals = tuple(
                replace(item, status="active")
                if item.subgoal_id == active_subgoal_id
                else item
                for item in normalized_subgoals
            )
            active_ids = [active_subgoal_id]
    if not active_ids and active_subgoal_id is None:
        candidates = [
            item
            for item in normalized_subgoals
            if item.status == "pending"
            and not item.depends_on
            and item.external_impact in {"read_only", "navigation_only"}
            and not item.risk_action_ids
        ]
        if len(candidates) == 1:
            selected_id = candidates[0].subgoal_id
            normalized_subgoals = tuple(
                replace(item, status="active") if item.subgoal_id == selected_id else item
                for item in normalized_subgoals
            )
            active_subgoal_id = selected_id
    normalized_status = graph.status
    if graph.status == "awaiting_confirmation" and not any(
        item.risk_action_ids for item in normalized_subgoals if item.status == "active"
    ):
        normalized_status = "ready"
    return replace(
        graph,
        status=normalized_status,
        risk_actions=tuple(
            risk for risk in graph.risk_actions if risk.risk_id not in removable_ids
        ),
        subgoals=normalized_subgoals,
        active_subgoal_id=active_subgoal_id,
    )


def _normalize_unique_active_frontier(graph: DynamicTaskGraph) -> DynamicTaskGraph:
    """Repair active markers only when the dependency graph has one safe frontier.

    DeepSeek may occasionally leave a downstream node active while its dependency
    is still unfinished, or activate both the current and its direct successor.
    The dependency DAG already determines the only runnable node in that case.
    Independent runnable roots remain ambiguous and are deliberately left for the
    strict validator to reject.
    """

    if graph.status not in {"ready", "running", "awaiting_confirmation"}:
        return graph
    completed_ids = {
        item.subgoal_id for item in graph.subgoals if item.status == "completed"
    }
    frontier = tuple(
        item
        for item in graph.subgoals
        if item.status in {"pending", "active"}
        and all(dependency in completed_ids for dependency in item.depends_on)
    )
    if len(frontier) != 1:
        return graph
    selected = frontier[0]
    if (
        selected.external_impact not in {"read_only", "navigation_only"}
        or selected.risk_action_ids
    ):
        return graph
    active_ids = tuple(
        item.subgoal_id for item in graph.subgoals if item.status == "active"
    )
    if (
        active_ids == (selected.subgoal_id,)
        and graph.active_subgoal_id == selected.subgoal_id
    ):
        return graph
    normalized_subgoals = tuple(
        replace(item, status="active")
        if item.subgoal_id == selected.subgoal_id
        else replace(item, status="pending")
        if item.status == "active"
        else item
        for item in graph.subgoals
    )
    return replace(
        graph,
        status=("ready" if graph.status == "awaiting_confirmation" else graph.status),
        subgoals=normalized_subgoals,
        active_subgoal_id=selected.subgoal_id,
    )


def _normalize_initial_confirmation_status(
    graph: DynamicTaskGraph,
) -> DynamicTaskGraph:
    """Promote a fully risk-bound active node to the mandatory confirmation gate."""

    if graph.status not in {"ready", "running"}:
        return graph
    active = tuple(item for item in graph.subgoals if item.status == "active")
    if len(active) != 1 or graph.active_subgoal_id != active[0].subgoal_id:
        return graph
    current = active[0]
    declared_risk_ids = {risk.risk_id for risk in graph.risk_actions}
    if (
        current.external_impact not in {"external_state", "unknown"}
        or not current.risk_action_ids
        or not set(current.risk_action_ids) <= declared_risk_ids
    ):
        return graph
    return replace(graph, status="awaiting_confirmation")


def _explicitly_denies_external_effect(value: str) -> bool:
    normalized = "".join(str(value or "").strip().lower().split())
    return bool(
        re.search(
            r"(?:(?:无|没有|不涉及|不会产生|不得产生|禁止产生|不改变)(?:任何)?"
            r"(?:账号或)?(?:外部)?(?:状态)?"
            r"(?:影响|变更|变化)|不影响(?:账号数据|外部系统|外部状态))",
            normalized,
        )
    )


def _risk_effect_action_anchors(*values: str) -> frozenset[str]:
    anchors = {
        match.group(0).casefold()
        for value in values
        for match in RISK_EFFECT_ACTION_PATTERN.finditer(str(value or ""))
    }
    if any(
        CONTACT_SELECTION_ACTION_PATTERN.search(str(value or ""))
        for value in values
    ):
        anchors.add(CONTACT_SELECTION_ANCHOR)
    return frozenset(anchors)


def _positive_effect_clauses(value: str) -> tuple[str, ...]:
    return tuple(
        clause.strip()
        for clause in re.split(r"[，,。；;\r\n]+", str(value or ""))
        if clause.strip()
        and not DIRECT_PROHIBITION_CLAUSE_PATTERN.search(clause)
    )


def _text_directly_negates_action_anchor(value: str, anchor: str) -> bool:
    pattern = (
        CONTACT_SELECTION_ACTION_PATTERN
        if anchor == CONTACT_SELECTION_ANCHOR
        else re.compile(re.escape(anchor), re.IGNORECASE)
    )
    for match in pattern.finditer(str(value or "")):
        prefix = str(value or "")[: match.start()].rstrip().lower()
        if (
            DIRECT_EFFECT_NEGATION_PATTERN.search(prefix)
            or COORDINATED_EFFECT_NEGATION_PATTERN.search(prefix)
            or NEGATED_LOW_LEVEL_INSTRUCTION_PREFIX_PATTERN.search(prefix)
        ):
            return True
    return False


def _is_read_only_risk_control_state(
    objective: str,
    constraints: tuple[str, ...],
    completion_conditions: tuple[str, ...],
) -> bool:
    positive_text = "；".join((objective, *completion_conditions))
    anchors = _risk_effect_action_anchors(positive_text)
    if not anchors or not READ_ONLY_RISK_CONTROL_STATE_PATTERN.search(positive_text):
        return False
    residual_positive_effects = READ_ONLY_RISK_CONTROL_STATE_PATTERN.sub(
        "",
        positive_text,
    )
    if _infer_external_risk_types(residual_positive_effects):
        return False
    return all(
        any(_text_directly_negates_action_anchor(item, anchor) for item in constraints)
        for anchor in anchors
    )


def _state_description_binds_canonical_input_text(
    values: tuple[str, ...],
    input_text: str,
) -> bool:
    """Prove an exact canonical literal belongs to a formal input-state clause.

    Descriptive words between a state relation and the literal are prose, not
    alternate candidate values.  The task graph's canonical ``input_text`` is
    therefore the only value authority.  We only check that this exact literal
    occurs in the same punctuation-delimited clause as an existing formal input
    carrier state; we never extract or infer a replacement value from prose.
    """

    literal = str(input_text or "").strip()
    if not literal:
        return False
    escaped = re.escape(literal)
    continuation = r"[A-Za-z0-9_.-]"
    prefix = rf"(?<!{continuation})" if re.match(continuation, literal[0]) else ""
    suffix = rf"(?!{continuation})" if re.match(continuation, literal[-1]) else ""
    literal_pattern = re.compile(prefix + escaped + suffix)
    for value in values:
        for clause in re.split(r"[。；;\r\n]+", str(value or "")):
            if (
                literal_pattern.search(clause)
                and LOCAL_UNSUBMITTED_INPUT_STATE_PATTERN.search(clause)
            ):
                return True
    return False


def _is_local_input_preparation_state(
    *values: str,
    input_text: Any,
) -> bool:
    """Recognize only reversible state of a formal local input carrier.

    The canonical literal proves that the graph contains a concrete input task;
    it is deliberately not treated as permission to type.  Positive clauses may
    describe only visibility, editability, or focus.  Any concrete external
    effect (send/save/delete/account/transaction and so on) rejects the proof.
    """

    if not isinstance(input_text, str) or not input_text.strip():
        return False
    texts = tuple(str(value or "") for value in values if str(value or "").strip())
    if not texts or not LOCAL_INPUT_PREPARATION_STATE_PATTERN.search("；".join(texts)):
        return False
    inferred = frozenset().union(
        *(
            _infer_external_risk_types(clause)
            for value in texts
            for clause in _positive_effect_clauses(value)
        )
    )
    return inferred <= {"unknown_external_effect"}


def _is_explicitly_unsubmitted_local_input(
    *values: str,
    input_text: Any,
) -> bool:
    target_text = input_text if isinstance(input_text, str) else ""
    if not target_text or not target_text.strip():
        return False
    texts = tuple(str(value or "") for value in values if str(value or "").strip())
    combined = "；".join(texts)
    risk_texts = tuple(
        LOCAL_EDITABLE_CARRIER_ADJECTIVE_PATTERN.sub("", value)
        for value in texts
    )
    inferred = frozenset().union(
        *(
            _infer_external_risk_types(clause)
            for value in risk_texts
            for clause in _positive_effect_clauses(value)
        )
    )
    return bool(
        _state_description_binds_canonical_input_text(texts, target_text)
        and LOCAL_INPUT_EFFECT_BOUNDARY_PATTERN.search(combined)
        # Generic wording such as "修改输入框文字" currently produces only
        # unknown_external_effect.  The explicit unsubmitted-input boundary is
        # enough to resolve that ambiguity, but never suppress a concrete
        # communication, publication, account, data, or transaction effect.
        and inferred <= {"unknown_external_effect"}
    )


def _is_reversible_local_keyboard_mode(*values: str) -> bool:
    """Recognize only an unsubmitted, device-local IME mode state.

    The exception is deliberately narrower than general settings changes. Any
    concrete account, communication, publication, data, permission, or
    transaction effect keeps the scope external and fail-closed.
    """

    texts = tuple(str(value or "") for value in values if str(value or "").strip())
    if not texts:
        return False
    combined = "；".join(texts)
    inferred = frozenset().union(
        *(_infer_external_risk_types(value) for value in texts)
    )
    return bool(
        LOCAL_KEYBOARD_MODE_PATTERN.search(combined)
        and not PERSISTENT_KEYBOARD_SETTING_PATTERN.search(combined)
        and inferred <= {"unknown_external_effect"}
    )


def _retryable_replan_output_error(error: TaskGraphError) -> bool:
    text = str(error)
    return any(fragment in text for fragment in REPAIRABLE_REPLAN_ERROR_FRAGMENTS)


def _normalize_blocked_mismatch_clarification(
    candidate: DynamicTaskGraph,
    *,
    trigger: str,
    error: TaskGraphError,
) -> DynamicTaskGraph | None:
    """Replace only a blocked model request for another low-level action.

    The result remains blocked and asks for a high-level user choice.  It never
    grants confirmation, creates a visual action, or makes an invalid active
    graph executable.
    """

    if trigger not in {"action_mismatch", "action_result_mismatch"}:
        return None
    if candidate.status != "blocked" or candidate.active_subgoal_id is not None:
        return None
    if "clarification_questions" not in str(error):
        return None
    if not candidate.clarification_questions:
        return None
    for question in candidate.clarification_questions:
        try:
            _reject_low_level_instruction(question, "clarification_questions")
        except TaskGraphError:
            continue
        return None
    return replace(
        candidate,
        clarification_questions=(MISMATCH_BLOCKED_CLARIFICATION,),
    )


def _replan_prompt(
    graph: DynamicTaskGraph,
    observation: ObservedState,
    *,
    trigger: str,
    reason: str,
) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图重规划器。根据新的只读观察，返回修订后的
完整高层任务图快照。你不能输出控件选择、点击、滑动、输入、坐标、Shell 或系统命令。

当前任务图：
{json.dumps(graph.to_dict(), ensure_ascii=False)}

重规划触发：{json.dumps(trigger, ensure_ascii=False)}
重规划原因：{json.dumps(reason, ensure_ascii=False)}
新的只读观察：
{json.dumps(observation.to_dict(), ensure_ascii=False)}

{_schema_prompt()}

重规划规则：
1. goal 必须逐字段保持不变；constraints 必须保留已有约束，可追加新发现的约束。
2. 已 completed 的子目标必须原样保留且仍为 completed；已满足的全局条件不得撤销。
3. 可修改、跳过或替换尚未完成的子目标，并新增子目标；不要坚持已失效的旧路径。
4. 新宣称 completed/satisfied 时，evidence 必须逐字复制 visible_evidence 中的证据；
   历史完成节点继续保留自己的历史证据。
5. 既有 risk_actions 必须保留，不能降低风险等级或取消 confirmation_required。
6. external_state 和 unknown 子目标都必须关联风险；成为 active 时必须返回
   awaiting_confirmation。每轮只选择一个 active 高层子目标；不要提出下一视觉动作。
7. 既有 external_state 不能降级，unknown 没有新的可靠证据时不能改成 read_only 或
   navigation_only；read_only/navigation_only 必须分别有纯观察或纯导航依据。
   trigger=observation_changed 且当前 read_only 结果无法由新画面直接证明时，如果目标页面或区域
   尚未出现，应把未完成路径改写为先达到 navigation_only 的目标页面可见状态，再保留后续
   read_only 结果核对；不得把导航动作本身写入子目标，也不得凭空宣称结果完成。
8. 只返回 JSON 对象，不要 Markdown，也不要返回 task_id、device_id、revision、协议版本、
    current_subgoal 或历史记录；这些字段由本地协议层生成。
9. 当 trigger=subgoal_completed 且当前子目标是 read_only 时，本轮必须用 visible_evidence 完成
   该只读子目标及匹配的全局条件，或明确阻塞，或推进到后续非只读子目标；不得继续保留任何
   read_only 活动子目标，避免只读复核再次请求视觉动作或形成循环。
10. completion_conditions[].evidence 只能选择 visible_evidence 中完整、逐字相同的独立短字符串。
    subgoals[].completion_evidence 通常也只能选 visible_evidence；唯一例外是当前严格绑定的
    navigation_only 旧子目标可选择 controller_transition_evidence_refs[].ref_id。每个数组最多3项，
    不得拼接多项、不得复制整个观察对象或 JSON。没有匹配证据时保持未完成或阻塞。
11. verified_action_transition 是本地控制器生成、严格绑定上一 revision/子目标/决策/动作和
    前后观察的动作回执；它与 visible_evidence 分离，不能当作页面可见事实或全局完成证据。
    outcome=matched 只证明该受控动作已执行并获得匹配验证，不代表任意子目标自动完成。
12. trigger=action_result_matched 时，可以结合回执和当前 visible_evidence 完成其严格绑定的
    navigation_only 旧子目标，或推进到不同的剩余状态目标；若证据不足，应明确重写剩余目标
    或阻塞。不得让同一活动子目标原样存活后再次请求等价动作。external_state/unknown 不能
    仅凭回执完成，仍必须由当前 visible_evidence 证明真实外部结果。
    如果旧子目标要求目标页面/结果区域可见，而新画面只出现了具名入口或分类项，绝不能完成旧
    子目标；应把未完成路径修订为先达到“具名入口可见”的 navigation_only 状态，再保留目标页面
    和结果核对状态。控制器回执只证明本轮受控动作及其可见变化，不能把入口冒充结果页面。
13. trigger=action_result_mismatch 时，不得完成回执绑定的旧子目标；必须根据当前画面重规划、
    阻塞或提出高层澄清。revision 必须严格增加 1。
14. 任何包含具名页面、卡片、区域或结果身份的 completed/satisfied 声明，其名称必须
    能从 grounded_visual_facts 的 screen_id、overlay 或可见元素 label/meaning 中找到结构化支持。
    visible_evidence 或 summary 中的自由文本描述不能单独证明具名身份。若旧路径使用了
    未被结构化画面支持的具名页面，不得硬完成；应跳过或替换该未完成节点，
    改为基于 grounded_visual_facts 中实际可见的结构化状态（例如唯一目标输入元素可见）
    继续高层规划；不得把入口名称、动作成功或场景变化冒充为目标页面身份。
"""


def _repair_replan_prompt(
    graph: DynamicTaskGraph,
    observation: ObservedState,
    *,
    trigger: str,
    reason: str,
    invalid_response: str,
    validation_error: str,
) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图重规划器。上一次修订 JSON 未通过
本地协议、安全或证据校验。请根据原任务图、新观察和校验错误重新生成一份完整修订图。
这只是唯一一次格式与高层协议修复机会；不要解释、不要局部补丁，也不要输出控件选择、
点击、滑动、输入、坐标、Shell、系统命令或任何 App 专用固定流程。

当前任务图：
{json.dumps(graph.to_dict(), ensure_ascii=False)}

重规划触发：{json.dumps(trigger, ensure_ascii=False)}
重规划原因：{json.dumps(reason, ensure_ascii=False)}
新的只读观察：
{json.dumps(observation.to_dict(), ensure_ascii=False)}

本地校验错误：{json.dumps(validation_error, ensure_ascii=False)}
上一次无效 JSON：
{invalid_response}

{_schema_prompt()}

修复规则：
1. goal 必须逐字段保持不变；constraints 必须保留已有约束，可追加新发现的约束。
2. 已 completed 的子目标和已满足的全局条件不得撤销；既有风险不得删除、降级或取消确认。
3. 只能依据 visible_evidence 新增完成证据；动作结果不匹配时不得假称预期结果已完成。
4. 可替换、跳过或新增尚未完成的高层子目标，但不能描述按钮、坐标或任何低层动作。
5. external_state 或 unknown 必须关联风险；成为 active 时必须等待本地确认。
6. 仍需通过全部本地校验；不要试图改写任务身份、设备、revision 或协议字段。
7. 只返回符合结构的完整 JSON 对象，不要 Markdown。
8. 当 trigger=subgoal_completed 且原活动子目标是 read_only 时，不得继续返回 read_only 活动
   子目标；只能依据 visible_evidence 完成、阻塞，或推进到后续非只读子目标。
9. 全局完成条件证据只能选择 visible_evidence。子目标完成证据通常也只能选择 visible_evidence；
   严格绑定的 navigation_only 旧子目标可选择 controller_transition_evidence_refs[].ref_id。
   每个数组最多3项；禁止拼接多项或复制整个观察对象/JSON。
10. verified_action_transition 是本地控制器回执而不是视觉证据；只能与当前
    visible_evidence 共同解释其严格绑定的上一 navigation_only 子目标。不能用它伪造
    external_state/unknown 完成，也不能在 matched 后原样保留旧子目标再提出等价动作。
11. action_result_mismatch 不得完成回执绑定的旧子目标；revision 必须严格增加 1。
12. 若校验错误指出“命名页面完成声明缺少结构化画面身份锚点”，不得重复该声明，
    也不得将 summary/visible_evidence 的自由文本当作身份。只能使用 grounded_visual_facts
    里的 screen_id、overlay 或元素 label/meaning；如仍无支持，应跳过或替换尚未完成的
    具名页面节点，改为基于当前结构化可见元素的高层状态。不得改写用户最终目标。
13. 若校验错误指出子目标使用了当前观察之外的完成证据，必须删除该伪证据；不得把
    subgoal_id、condition_id、目标名称或自行概括的句子当作证据。只能逐字选择
    visible_evidence，或为严格绑定的上一 navigation_only 子目标选择
    controller_transition_evidence_refs[].ref_id；没有合格证据就保持未完成、替换路径或阻塞。
14. 若校验错误指出“matched controller_transition 未完成其绑定的 navigation_only 子目标”，
    必须把该严格绑定的上一活动子目标标为 completed，并逐字使用对应
    controller_transition_evidence_refs[].ref_id；不得让该旧子目标继续 active，不得把回执用于
    其他子目标、全局条件或 external_state/unknown，也不得自行生成第二动作。
"""


def _schema_prompt() -> str:
    return """JSON 只允许以下结构：
{
  "status":"ready|running|awaiting_confirmation|completed|blocked",
  "goal":{
    "objective":"用户最终想达到的结果",
    "target_apps":[{"app_id":"稳定小写英文ID","app_name":"App名称"}],
    "entities":{"目标对象或内容":"值","input_text":"仅在确实需要输入时逐字复制用户指定文字；否则省略此键"}
  },
  "constraints":["全局约束"],
  "completion_conditions":[{
    "condition_id":"小写稳定ID",
    "description":"最终完成条件",
    "evidence_required":["需要从画面看到的事实"],
    "satisfied":false,
    "evidence":[]
  }],
  "risk_actions":[{
    "risk_id":"小写稳定ID",
    "description":"可能改变外部状态的事项",
    "external_effect":"对账号、数据、交易或他人的影响",
    "risk_type":"message_or_communication|content_publication|account_relationship_change|membership_change|permission_role_change|data_mutation|data_deletion|transaction_or_payment|account_or_permission_change|unknown_external_effect",
    "risk_level":"low|medium|high|critical",
    "subgoal_ids":["关联子目标ID"],
    "confirmation_required":true
  }],
  "subgoals":[{
    "subgoal_id":"小写稳定ID",
    "objective":"应达到的高层状态",
    "status":"pending|active|completed|blocked|skipped",
    "depends_on":["前置子目标ID"],
    "constraints":["本子目标约束"],
    "completion_conditions":["本子目标完成条件"],
    "completion_evidence":[],
    "risk_action_ids":["关联风险ID"],
    "external_impact":"read_only|navigation_only|external_state|unknown"
  }],
  "active_subgoal_id":"活动子目标ID或null",
  "clarification_questions":["阻塞时需要用户补充的信息"]
}"""


def _graph_from_payload(
    payload: dict[str, Any],
    *,
    task_id: str,
    device_id: str,
    revision: int,
    raw_user_goal: str,
) -> DynamicTaskGraph:
    _expect_keys(
        payload,
        {
            "status",
            "goal",
            "constraints",
            "completion_conditions",
            "risk_actions",
            "subgoals",
            "active_subgoal_id",
            "clarification_questions",
        },
        "任务图",
    )
    raw_goal = _expect_dict(payload.get("goal"), "goal")
    _expect_keys(raw_goal, {"objective", "target_apps", "entities"}, "goal")
    target_apps = tuple(
        _target_app_from_payload(item)
        for item in _expect_list(raw_goal.get("target_apps"), "goal.target_apps")
    )
    entities = dict(_expect_dict(raw_goal.get("entities"), "goal.entities"))
    # Some JSON providers materialize an optional example field as null or an
    # empty string.  Treat only those two representations as absence.  Any
    # non-empty value still goes through the strict exact-input validation.
    optional_input_text = entities.get("input_text")
    if optional_input_text is None or optional_input_text == "":
        entities.pop("input_text", None)
    goal = GraphGoal(
        objective=_require_text(raw_goal.get("objective"), "goal.objective"),
        target_apps=target_apps,
        entities=entities,
    )
    active_value = payload.get("active_subgoal_id")
    active_subgoal_id = None if active_value is None else str(active_value).strip()
    return DynamicTaskGraph(
        task_id=task_id,
        device_id=device_id,
        revision=revision,
        status=str(payload.get("status") or "").strip().lower(),
        goal=goal,
        constraints=_text_tuple(payload.get("constraints"), "constraints"),
        completion_conditions=tuple(
            _condition_from_payload(item)
            for item in _expect_list(
                payload.get("completion_conditions"), "completion_conditions"
            )
        ),
        risk_actions=tuple(
            _risk_from_payload(item)
            for item in _expect_list(payload.get("risk_actions"), "risk_actions")
        ),
        subgoals=tuple(
            _subgoal_from_payload(item)
            for item in _expect_list(payload.get("subgoals"), "subgoals")
        ),
        active_subgoal_id=active_subgoal_id,
        clarification_questions=_text_tuple(
            payload.get("clarification_questions"), "clarification_questions"
        ),
        raw_user_goal=raw_user_goal,
    )


def _target_app_from_payload(value: Any) -> TargetApp:
    item = _expect_dict(value, "target_apps[]")
    _expect_keys(item, {"app_id", "app_name"}, "target_apps[]")
    return TargetApp(
        app_id=str(item.get("app_id") or "").strip().lower(),
        app_name=_require_text(item.get("app_name"), "target_apps.app_name"),
    )


def _condition_from_payload(value: Any) -> CompletionCondition:
    item = _expect_dict(value, "completion_conditions[]")
    _expect_keys(
        item,
        {"condition_id", "description", "evidence_required", "satisfied", "evidence"},
        "completion_conditions[]",
    )
    if not isinstance(item.get("satisfied"), bool):
        raise TaskGraphError("completion_conditions.satisfied 必须是布尔值。")
    return CompletionCondition(
        condition_id=str(item.get("condition_id") or "").strip().lower(),
        description=_require_text(item.get("description"), "completion_conditions.description"),
        evidence_required=_text_tuple(item.get("evidence_required"), "evidence_required"),
        satisfied=item["satisfied"],
        evidence=_text_tuple(item.get("evidence"), "evidence"),
    )


def _risk_from_payload(value: Any) -> RiskAction:
    item = _expect_dict(value, "risk_actions[]")
    _expect_keys(
        item,
        {
            "risk_id",
            "description",
            "external_effect",
            "risk_type",
            "risk_level",
            "subgoal_ids",
            "confirmation_required",
        },
        "risk_actions[]",
    )
    return RiskAction(
        risk_id=str(item.get("risk_id") or "").strip().lower(),
        description=_require_text(item.get("description"), "risk_actions.description"),
        external_effect=_require_text(
            item.get("external_effect"), "risk_actions.external_effect"
        ),
        risk_type=str(item.get("risk_type") or "").strip().lower(),
        risk_level=str(item.get("risk_level") or "").strip().lower(),
        subgoal_ids=_id_tuple(item.get("subgoal_ids"), "risk_actions.subgoal_ids"),
        confirmation_required=item.get("confirmation_required") is True,
    )


def _subgoal_from_payload(value: Any) -> Subgoal:
    item = _expect_dict(value, "subgoals[]")
    _expect_keys(
        item,
        {
            "subgoal_id",
            "objective",
            "status",
            "depends_on",
            "constraints",
            "completion_conditions",
            "completion_evidence",
            "risk_action_ids",
            "external_impact",
        },
        "subgoals[]",
    )
    return Subgoal(
        subgoal_id=str(item.get("subgoal_id") or "").strip().lower(),
        objective=_require_text(item.get("objective"), "subgoals.objective"),
        status=str(item.get("status") or "").strip().lower(),
        depends_on=_id_tuple(item.get("depends_on"), "subgoals.depends_on"),
        constraints=_text_tuple(item.get("constraints"), "subgoals.constraints"),
        completion_conditions=_text_tuple(
            item.get("completion_conditions"), "subgoals.completion_conditions"
        ),
        completion_evidence=_text_tuple(
            item.get("completion_evidence"), "subgoals.completion_evidence"
        ),
        risk_action_ids=_id_tuple(
            item.get("risk_action_ids"), "subgoals.risk_action_ids"
        ),
        external_impact=str(item.get("external_impact") or "").strip().lower(),
    )


_VISUAL_IDENTITY_CONTAINER_PATTERN = re.compile(
    r"页面|界面|屏幕|视图|面板|卡片|(?:^|\b)(?:page|screen|view|panel|card)(?:\b|$)",
    re.IGNORECASE,
)
_GENERIC_VISUAL_LOCATION_PATTERN = re.compile(
    r"^(?:当前|同一|该|目标|原来|原有)(?:本地)?(?:页面|界面|屏幕|视图)"
    r"(?:中|内|上)?\s*|"
    r"^(?:the\s+)?(?:current|same|this|target|original)\s+"
    r"(?:local\s+)?(?:page|screen|view)\b\s*|"
    r"(?:在|位于)\s*(?:当前|同一|该)?(?:页面|界面|屏幕|视图)"
    r"(?:中|内|上)?\s*(?:可见|出现|显示|存在)?|"
    r"(?:visible|present|shown)\s+(?:in|on)\s+(?:the\s+)?"
    r"(?:current\s+)?(?:page|screen|view)",
    re.IGNORECASE,
)
_QUOTED_VISUAL_IDENTITY_PATTERN = re.compile(
    r"“([^”]{2,80})”|\"([^\"]{2,80})\""
)
_QUOTED_VISUAL_IDENTITY_CONTEXT_PATTERN = re.compile(
    r"标题|文字|标签|按钮|入口|链接|"
    r"\b(?:title|heading|text|label|button|entry|link)\b",
    re.IGNORECASE,
)
_LEADING_UNNAMED_VISUAL_CONTAINER_PATTERN = re.compile(
    r"^(?:页面|界面|屏幕|视图|面板|卡片)(?:中|内|上)?|"
    r"^(?:the\s+)?(?:page|screen|view|panel|card)\b",
    re.IGNORECASE,
)
_VISUAL_IDENTITY_GENERIC_TOKENS = (
    "原来的",
    "原有的",
    "当前的",
    "指定的",
    "目标的",
    "已经",
    "清晰",
    "完整",
    "当前",
    "原来",
    "原有",
    "指定",
    "目标",
    "页面",
    "界面",
    "屏幕",
    "视图",
    "面板",
    "卡片",
    "控件",
    "元素",
    "入口",
    "主标题",
    "标题",
    "文字",
    "逐字",
    "显示",
    "出现",
    "可见",
    "打开",
    "进入",
    "返回",
    "回到",
    "通过",
    "流程",
    "结果",
    "page",
    "screen",
    "view",
    "panel",
    "card",
    "visible",
    "shown",
    "displayed",
    "open",
    "opened",
    "result",
    "process",
    "flow",
    "main title",
    "title",
    "heading",
    "text",
    "verbatim",
    "current",
    "target",
    "original",
    "the",
    "is",
)


def _compact_identity_text(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.casefold()))


def _named_visual_identity_anchor(texts: tuple[str, ...]) -> str:
    anchors: list[str] = []
    for item in texts:
        value = str(item or "").strip()
        identity_value = _GENERIC_VISUAL_LOCATION_PATTERN.sub(" ", value)
        if _LEADING_UNNAMED_VISUAL_CONTAINER_PATTERN.search(
            identity_value.strip()
        ):
            continue
        if (
            not identity_value.strip()
            or not _VISUAL_IDENTITY_CONTAINER_PATTERN.search(identity_value)
        ):
            continue
        container = _VISUAL_IDENTITY_CONTAINER_PATTERN.search(identity_value)
        # A named container's identity is the modifier before "page/screen/view".
        # State predicates after the container (for example an input being visible)
        # are completion facts, not part of the page name.
        identity_name = (
            identity_value[: container.start()]
            if container is not None
            else identity_value
        )
        cleaned = identity_name.casefold()
        for token in _VISUAL_IDENTITY_GENERIC_TOKENS:
            cleaned = cleaned.replace(token, " ")
        anchor = _compact_identity_text(cleaned)
        has_stable_length = len(anchor) >= 4 or len(
            re.findall(r"[\u4e00-\u9fff]", anchor)
        ) >= 2
        if has_stable_length and anchor not in anchors:
            anchors.append(anchor)
    # Short referential phrases such as "上一页" or "详情页" are not stable
    # page identities. Longer names must be grounded in structured scene facts.
    return min(anchors, key=len) if anchors else ""


def _quoted_visual_identity_anchor(texts: tuple[str, ...]) -> str:
    """Return an explicitly quoted UI label that must match verbatim."""

    anchors: list[str] = []
    for item in texts:
        value = str(item or "").strip()
        if not _QUOTED_VISUAL_IDENTITY_CONTEXT_PATTERN.search(value):
            continue
        for match in _QUOTED_VISUAL_IDENTITY_PATTERN.finditer(value):
            literal = next(
                (group for group in match.groups() if group is not None),
                "",
            )
            anchor = _compact_identity_text(literal)
            if len(anchor) >= 4 and anchor not in anchors:
                anchors.append(anchor)
    return min(anchors, key=len) if anchors else ""


def _identity_anchor_is_grounded(anchor: str, facts: tuple[str, ...]) -> bool:
    for fact in facts:
        compact = _compact_identity_text(fact)
        if not compact:
            continue
        if anchor in compact:
            return True
        longest = SequenceMatcher(
            None,
            anchor,
            compact,
            autojunk=False,
        ).find_longest_match()
        if longest.size / len(anchor) >= 0.5:
            return True
    return False


def named_visual_identity_is_grounded(
    texts: tuple[str, ...],
    facts: tuple[str, ...],
) -> bool:
    """Return whether a named visual container is grounded by structured facts.

    Unnamed element/state descriptions are outside this page-identity gate. A
    real named page, screen, view, panel, or card must have at least one
    structured identity fact; prose summaries are intentionally not sufficient.
    """

    anchor = _named_visual_identity_anchor(texts)
    if not anchor:
        return True
    return bool(facts and _identity_anchor_is_grounded(anchor, facts))


def _require_named_visual_identity_grounding(
    texts: tuple[str, ...],
    observation: ObservedState,
    *,
    field: str,
) -> None:
    literal_anchor = _quoted_visual_identity_anchor(texts)
    if literal_anchor and observation.grounded_visual_facts:
        if not any(
            literal_anchor in _compact_identity_text(fact)
            for fact in observation.grounded_visual_facts
        ):
            raise TaskGraphError(
                f"逐字 UI 完成声明缺少完整结构化画面锚点：{field}"
            )
        return
    anchor = _named_visual_identity_anchor(texts)
    if not anchor:
        return
    if not observation.grounded_visual_facts:
        return
    if not _identity_anchor_is_grounded(anchor, observation.grounded_visual_facts):
        raise TaskGraphError(
            f"命名页面完成声明缺少结构化画面身份锚点：{field}"
        )


def _restore_completed_history_evidence(
    previous: DynamicTaskGraph,
    candidate: DynamicTaskGraph,
) -> DynamicTaskGraph:
    """Keep controller-owned completion history immutable across model replans.

    A model must still return every completed node with the same status and
    semantics; deletion, resurrection, or any other field mutation remains a
    validation error. Only the historical evidence tuple is restored from the
    trusted previous graph, so omission or paraphrase cannot erase or rewrite
    controller-accepted history.
    """

    completed = {
        item.subgoal_id: item
        for item in previous.subgoals
        if item.status == "completed"
    }
    restored = tuple(
        replace(
            item,
            completion_evidence=completed[item.subgoal_id].completion_evidence,
        )
        if item.subgoal_id in completed and item.status == "completed"
        else item
        for item in candidate.subgoals
    )
    return replace(candidate, subgoals=restored)


def _is_explicit_local_temporary_draft_clear(
    graph: DynamicTaskGraph,
    subgoal: Subgoal,
) -> bool:
    """Recognize only an explicitly local, reversible empty-draft result."""

    if (
        len(graph.goal.target_apps) != 1
        or graph.goal.target_apps[0].app_id != "current_foreground"
        or str(graph.goal.entities.get("input_text") or "").strip()
    ):
        return False
    state_texts = (subgoal.objective, *subgoal.completion_conditions)
    combined_state = "；".join(state_texts)
    if (
        not LOCAL_TEMPORARY_DRAFT_CLEAR_STATE_PATTERN.search(combined_state)
        or PERSISTENT_DRAFT_STATE_PATTERN.search(combined_state)
    ):
        return False
    boundary_texts = (
        graph.raw_user_goal or graph.goal.objective,
        graph.goal.objective,
        *graph.constraints,
        *subgoal.constraints,
    )
    if not any(_explicitly_denies_external_effect(item) for item in boundary_texts):
        return False
    positive = tuple(
        clause
        for value in (*boundary_texts, *state_texts)
        for clause in _positive_effect_clauses(value)
    )
    return not any(_infer_external_risk_types(item) for item in positive)


def _explicit_local_temporary_clear_audit_scopes(
    graph: DynamicTaskGraph,
) -> frozenset[str | None]:
    ids = {
        item.subgoal_id
        for item in graph.subgoals
        if _is_explicit_local_temporary_draft_clear(graph, item)
    }
    return frozenset({*ids, None} if ids else ())


def _explicit_local_input_audit_scopes(
    graph: DynamicTaskGraph,
    source_groups: dict[str | None, list[AuditSource]],
) -> set[str | None]:
    """Bind canonical input text and global safety constraints to each subgoal."""

    input_text = graph.goal.entities.get("input_text")
    global_context = (
        graph.raw_user_goal or graph.goal.objective,
        graph.goal.objective,
        *graph.constraints,
    )
    scopes: set[str | None] = set()
    for subgoal in graph.subgoals:
        group = source_groups.get(subgoal.subgoal_id, ())
        group_texts = tuple(item.text for item in group)
        if LOCAL_UNSUBMITTED_INPUT_STATE_PATTERN.search(
            "；".join(group_texts)
        ) and _is_explicitly_unsubmitted_local_input(
            *global_context,
            *group_texts,
            input_text=input_text,
        ):
            scopes.add(subgoal.subgoal_id)
    if scopes:
        scopes.add(None)
    return scopes


def _structured_local_input_workflow_scopes(
    graph: DynamicTaskGraph,
    source_groups: dict[str | None, list[AuditSource]],
) -> frozenset[str | None]:
    """Extend a proven local input scope only along its dependency chain."""

    if graph.risk_actions:
        return frozenset()
    direct_scopes = _explicit_local_input_audit_scopes(graph, source_groups)
    direct_ids = {item for item in direct_scopes if item is not None}
    if not direct_ids:
        return frozenset()
    subgoals = {item.subgoal_id: item for item in graph.subgoals}
    ancestor_map = _dependency_ancestor_map(subgoals)
    scopes: set[str | None] = set()
    for subgoal in graph.subgoals:
        group = source_groups.get(subgoal.subgoal_id, [])
        if (
            subgoal.external_impact not in {"read_only", "navigation_only"}
            or subgoal.risk_action_ids
            or not group
            or any(_infer_external_risk_types(item.text) for item in group)
        ):
            continue
        if any(
            subgoal.subgoal_id == input_id
            or input_id in ancestor_map[subgoal.subgoal_id]
            or subgoal.subgoal_id in ancestor_map[input_id]
            for input_id in direct_ids
        ):
            scopes.add(subgoal.subgoal_id)
    if scopes == set(subgoals) and not any(
        _infer_external_risk_types(item.text)
        for item in source_groups.get(None, [])
    ):
        scopes.add(None)
    return frozenset(scopes)


def _canonicalize_literal_visible_evidence_clauses(
    previous: DynamicTaskGraph,
    candidate: DynamicTaskGraph,
    observation: ObservedState,
) -> DynamicTaskGraph:
    """Bind a model's exact visible clause to its full controller-owned fact.

    Models sometimes copy one complete clause from a longer visible summary
    instead of copying the entire evidence item.  This normalization remains
    fail-closed: only a sufficiently specific, punctuation-delimited literal
    clause with exactly one source is replaced by that source.  Paraphrases,
    identifiers, partial phrases and ambiguous matches remain invalid.
    """

    visible = tuple(observation.visible_evidence)
    visible_set = frozenset(visible)

    def normalized(value: str) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    clause_sources: dict[str, set[str]] = {}
    for source in visible:
        for clause in re.split(r"[，,。.;；！？!?]+", source):
            key = normalized(clause)
            if key:
                clause_sources.setdefault(key, set()).add(source)

    def canonicalize(claims: tuple[str, ...]) -> tuple[str, ...]:
        result: list[str] = []
        for claim in claims:
            if claim in visible_set or claim.startswith("controller_transition:"):
                result.append(claim)
                continue
            key = normalized(claim)
            semantic_chars = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", key)
            sources = clause_sources.get(key, set())
            if len(semantic_chars) >= 6 and len(sources) == 1:
                result.append(next(iter(sources)))
            else:
                result.append(claim)
        return tuple(dict.fromkeys(result))

    old_conditions = {
        item.condition_id: item for item in previous.completion_conditions
    }

    def newly_satisfied(item: CompletionCondition) -> bool:
        old = old_conditions.get(item.condition_id)
        return item.satisfied and (old is None or not old.satisfied)

    conditions = tuple(
        replace(item, evidence=canonicalize(item.evidence))
        if newly_satisfied(item)
        else item
        for item in candidate.completion_conditions
    )
    old_subgoals = {item.subgoal_id: item for item in previous.subgoals}
    subgoals = tuple(
        replace(
            item,
            completion_evidence=canonicalize(item.completion_evidence),
        )
        if item.status == "completed"
        and (
            item.subgoal_id not in old_subgoals
            or old_subgoals[item.subgoal_id].status != "completed"
        )
        else item
        for item in candidate.subgoals
    )
    return replace(
        candidate,
        completion_conditions=conditions,
        subgoals=subgoals,
    )


def _validate_revision(
    previous: DynamicTaskGraph,
    candidate: DynamicTaskGraph,
    observation: ObservedState,
) -> None:
    if candidate.goal != previous.goal:
        raise TaskGraphError("重规划不能改写用户目标、目标 App 或目标实体。")
    if not set(previous.constraints).issubset(candidate.constraints):
        raise TaskGraphError("重规划不能删除已有全局约束。")

    old_conditions = {item.condition_id: item for item in previous.completion_conditions}
    new_conditions = {item.condition_id: item for item in candidate.completion_conditions}
    missing_conditions = set(old_conditions) - set(new_conditions)
    if missing_conditions:
        raise TaskGraphError(
            "重规划不能删除全局完成条件：" + ", ".join(sorted(missing_conditions))
        )
    evidence = set(observation.visible_evidence)
    controller_refs = {
        item.ref_id: item
        for item in observation.controller_transition_evidence_refs
    }
    for condition_id, old in old_conditions.items():
        new = new_conditions[condition_id]
        if (
            new.description != old.description
            or new.evidence_required != old.evidence_required
        ):
            raise TaskGraphError(f"重规划不能改写全局完成条件：{condition_id}")
        if old.satisfied and (not new.satisfied or not set(old.evidence).issubset(new.evidence)):
            raise TaskGraphError(f"重规划不能撤销已满足完成条件：{condition_id}")
        if not old.satisfied and new.satisfied and not set(new.evidence).issubset(evidence):
            raise TaskGraphError(f"完成条件使用了当前观察之外的证据：{condition_id}")
        if not old.satisfied and new.satisfied:
            _require_named_visual_identity_grounding(
                (new.description, *new.evidence_required),
                observation,
                field=f"completion_conditions.{condition_id}",
            )
    for condition_id in set(new_conditions) - set(old_conditions):
        condition = new_conditions[condition_id]
        if condition.satisfied and not set(condition.evidence).issubset(evidence):
            raise TaskGraphError(f"新增完成条件使用了当前观察之外的证据：{condition_id}")
        if condition.satisfied:
            _require_named_visual_identity_grounding(
                (condition.description, *condition.evidence_required),
                observation,
                field=f"completion_conditions.{condition_id}",
            )

    old_risks = {item.risk_id: item for item in previous.risk_actions}
    new_risks = {item.risk_id: item for item in candidate.risk_actions}
    missing_risks = set(old_risks) - set(new_risks)
    if missing_risks:
        raise TaskGraphError("重规划不能删除既有风险：" + ", ".join(sorted(missing_risks)))
    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    for risk_id, old in old_risks.items():
        new = new_risks[risk_id]
        if (
            new.description != old.description
            or new.external_effect != old.external_effect
            or new.risk_type != old.risk_type
            or risk_order[new.risk_level] < risk_order[old.risk_level]
            or new.confirmation_required is not True
            or not set(old.subgoal_ids).issubset(new.subgoal_ids)
        ):
            raise TaskGraphError(f"重规划不能改写或降低既有风险：{risk_id}")

    old_subgoals = {item.subgoal_id: item for item in previous.subgoals}
    new_subgoals = {item.subgoal_id: item for item in candidate.subgoals}
    completed = {
        subgoal_id: item
        for subgoal_id, item in old_subgoals.items()
        if item.status == "completed"
    }
    missing_completed = set(completed) - set(new_subgoals)
    if missing_completed:
        raise TaskGraphError(
            "重规划不能删除已完成子目标：" + ", ".join(sorted(missing_completed))
        )
    for subgoal_id, old in completed.items():
        new = new_subgoals[subgoal_id]
        if (
            new.status != "completed"
            or new.objective != old.objective
            or new.depends_on != old.depends_on
            or new.constraints != old.constraints
            or new.completion_conditions != old.completion_conditions
            or new.risk_action_ids != old.risk_action_ids
            or new.external_impact != old.external_impact
            or not set(old.completion_evidence).issubset(new.completion_evidence)
        ):
            raise TaskGraphError(f"重规划不能复活或改写已完成子目标：{subgoal_id}")
    for subgoal_id, new in new_subgoals.items():
        old = old_subgoals.get(subgoal_id)
        newly_completed = new.status == "completed" and (
            old is None or old.status != "completed"
        )
        if newly_completed:
            claimed = set(new.completion_evidence)
            controller_claims = claimed.intersection(controller_refs)
            unknown_claims = claimed - evidence - set(controller_refs)
            if unknown_claims:
                raise TaskGraphError(
                    f"子目标使用了当前观察之外的完成证据：{subgoal_id}"
                )
            if controller_claims:
                if (
                    old is None
                    or old.external_impact != "navigation_only"
                    or previous.active_subgoal_id != subgoal_id
                    or observation.verified_action_transition is None
                    or observation.verified_action_transition.outcome != "matched"
                ):
                    raise TaskGraphError(
                        "controller_transition 证据只能完成其严格绑定的上一 "
                        "navigation_only 活动子目标。"
                    )
                for ref_id in controller_claims:
                    ref = controller_refs[ref_id]
                    if ref.subgoal_id != subgoal_id:
                        raise TaskGraphError(
                            "controller_transition 证据跨子目标使用。"
                        )
        if newly_completed:
            _require_named_visual_identity_grounding(
                (new.objective, *new.completion_conditions),
                observation,
                field=f"subgoals.{subgoal_id}",
            )
    transition = observation.verified_action_transition
    if (
        transition is not None
        and transition.outcome == "matched"
        and controller_refs
        and previous.active_subgoal_id is not None
    ):
        old_active = old_subgoals.get(previous.active_subgoal_id)
        revised_old_active = new_subgoals.get(previous.active_subgoal_id)
        if (
            old_active is not None
            and old_active.external_impact == "navigation_only"
            and transition.subgoal_id == old_active.subgoal_id
            and any(
                ref.subgoal_id == old_active.subgoal_id
                for ref in controller_refs.values()
            )
            and (
                revised_old_active is None
                or revised_old_active.status != "completed"
            )
        ):
            raise TaskGraphError(
                "matched controller_transition 未完成其绑定的 navigation_only 子目标："
                f"{old_active.subgoal_id}"
            )


def _validate_external_impact_revision(
    previous: DynamicTaskGraph,
    candidate: DynamicTaskGraph,
) -> None:
    old_subgoals = {item.subgoal_id: item for item in previous.subgoals}
    new_subgoals = {item.subgoal_id: item for item in candidate.subgoals}
    for subgoal_id, old in old_subgoals.items():
        new = new_subgoals.get(subgoal_id)
        if new is None:
            continue
        if old.external_impact == "external_state" and new.external_impact != "external_state":
            raise TaskGraphError(
                f"重规划不能降低既有 external_state 影响分类：{subgoal_id}"
            )
        if old.external_impact == "unknown" and new.external_impact in {
            "read_only",
            "navigation_only",
        }:
            raise TaskGraphError(
                f"重规划不能未经证据把 unknown 降级为安全分类：{subgoal_id}"
            )


def _validate_preserved_risk_ids(
    previous: DynamicTaskGraph,
    candidate: DynamicTaskGraph,
) -> None:
    previous_risks = {item.risk_id: item for item in previous.risk_actions}
    candidate_risks = {item.risk_id: item for item in candidate.risk_actions}
    missing = set(previous_risks) - set(candidate_risks)
    if missing:
        raise TaskGraphError(
            "重规划不能删除既有风险：" + ", ".join(sorted(missing))
        )
    for risk_id, previous_risk in previous_risks.items():
        candidate_risk = candidate_risks[risk_id]
        if candidate_risk.risk_type != previous_risk.risk_type:
            raise TaskGraphError(f"重规划不能改换既有风险类别：{risk_id}")
        if candidate_risk.confirmation_required is not True:
            raise TaskGraphError(f"重规划不能取消既有风险确认：{risk_id}")


def _reject_dependency_cycles(subgoals: dict[str, Subgoal]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(subgoal_id: str) -> None:
        if subgoal_id in visiting:
            raise TaskGraphError(f"子目标依赖形成环：{subgoal_id}")
        if subgoal_id in visited:
            return
        visiting.add(subgoal_id)
        for dependency in subgoals[subgoal_id].depends_on:
            visit(dependency)
        visiting.remove(subgoal_id)
        visited.add(subgoal_id)

    for subgoal_id in subgoals:
        visit(subgoal_id)


def _unique_by_id(values: tuple[Any, ...], key: Any, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        item_id = key(value)
        if item_id in result:
            raise TaskGraphError(f"{label} ID 重复：{item_id}")
        result[item_id] = value
    return result


def _expect_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise TaskGraphError(
            f"{path} 包含协议外字段：" + ", ".join(sorted(str(item) for item in unexpected))
        )
    missing = allowed - set(value)
    if missing:
        raise TaskGraphError(
            f"{path} 缺少字段：" + ", ".join(sorted(missing))
        )


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskGraphError(f"{path} 必须是对象。")
    return value


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TaskGraphError(f"{path} 必须是数组。")
    return value


def _require_text(value: Any, path: str, *, max_length: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskGraphError(f"{path} 必须是非空字符串。")
    text = value.strip()
    if len(text) > max_length:
        raise TaskGraphError(f"{path} 超过最大长度 {max_length}。")
    return text


def _text_tuple(value: Any, path: str) -> tuple[str, ...]:
    items = _expect_list(value, path)
    return tuple(_require_text(item, f"{path}[]") for item in items)


def _id_tuple(value: Any, path: str) -> tuple[str, ...]:
    items = _expect_list(value, path)
    return tuple(str(item or "").strip().lower() for item in items)


def _validate_text_list(values: tuple[str, ...], path: str, *, required: bool) -> None:
    if required and not values:
        raise TaskGraphError(f"{path} 不能为空。")
    if len(values) != len(set(values)):
        raise TaskGraphError(f"{path} 不能包含重复项。")
    for value in values:
        _require_text(value, path)


def _validate_id_list(values: tuple[str, ...], path: str, *, required: bool) -> None:
    if required and not values:
        raise TaskGraphError(f"{path} 不能为空。")
    if len(values) != len(set(values)):
        raise TaskGraphError(f"{path} 不能包含重复 ID。")
    for value in values:
        _validate_id(value, path)


def _validate_id(value: str, label: str) -> None:
    if not ID_PATTERN.fullmatch(value):
        raise TaskGraphError(f"{label} 无效：{value!r}")


def _validate_device_id(device_id: str) -> None:
    if not DEVICE_ID_PATTERN.fullmatch(str(device_id or "")):
        raise TaskGraphError(f"device_id 无效：{device_id!r}")


def _validate_task_id(task_id: str) -> None:
    if not TASK_ID_PATTERN.fullmatch(str(task_id or "")):
        raise TaskGraphError(f"task_id 无效：{task_id!r}")


def _reject_low_level_instruction(
    value: str,
    path: str,
    *,
    allow_negated: bool = False,
) -> None:
    for match in LOW_LEVEL_INSTRUCTION_PATTERN.finditer(value):
        prefix = value[max(0, match.start() - 40) : match.start()].lower()
        if allow_negated and NEGATED_LOW_LEVEL_INSTRUCTION_PREFIX_PATTERN.search(
            prefix
        ):
            continue
        raise TaskGraphError(f"DeepSeek 高层任务图包含低层动作表达：{path}")


def _reject_low_level_completion_evidence(value: str, path: str) -> None:
    """Allow only a negated low-level token inside a visible control state.

    DeepSeek occasionally describes the safe pre-submit state as a button being
    "not activated or clicked".  That is not an instruction, but accepting all
    negated low-level prose here would let a misplaced constraint masquerade as
    completion evidence.  The exception therefore requires both the existing
    read-only risk-control state grammar and independent negation of every
    low-level token. Positive action history and direct prohibitions still fail
    and must be repaired into a high-level state or moved to constraints.
    """

    try:
        _reject_low_level_instruction(value, path)
    except TaskGraphError:
        if not READ_ONLY_RISK_CONTROL_STATE_PATTERN.search(value):
            raise
        _reject_low_level_instruction(value, path, allow_negated=True)


def _describes_external_state_change(*values: str) -> bool:
    return bool(_infer_external_risk_types(*values))


def _infer_external_risk_types(*values: str) -> frozenset[str]:
    inferred: set[str] = set()
    patterns = _external_risk_patterns()
    for value in values:
        field_inferred: set[str] = set()
        for risk_type, pattern in patterns.items():
            if _has_unnegated_effect_match(pattern, value):
                field_inferred.add(risk_type)
        if (
            _has_unnegated_effect_match(EXTERNAL_STATE_CHANGE_PATTERN, value)
            and not field_inferred
        ):
            field_inferred.add("unknown_external_effect")
        inferred.update(field_inferred)
    return frozenset(inferred)


def _external_risk_patterns() -> dict[str, re.Pattern[str]]:
    return {
        "message_or_communication": COMMUNICATION_EFFECT_PATTERN,
        "account_relationship_change": ACCOUNT_RELATIONSHIP_EFFECT_PATTERN,
        "membership_change": MEMBERSHIP_EFFECT_PATTERN,
        "permission_role_change": PERMISSION_ROLE_EFFECT_PATTERN,
        "content_publication": CONTENT_PUBLICATION_EFFECT_PATTERN,
        "data_deletion": DATA_DELETION_EFFECT_PATTERN,
        "data_mutation": DATA_MUTATION_EFFECT_PATTERN,
        "transaction_or_payment": TRANSACTION_EFFECT_PATTERN,
        "account_or_permission_change": ACCOUNT_PERMISSION_EFFECT_PATTERN,
    }


def _infer_directly_negated_risk_types(value: str) -> frozenset[str]:
    negated: set[str] = set()
    for risk_type, pattern in _external_risk_patterns().items():
        for match in pattern.finditer(value):
            prefix = value[: match.start()].rstrip().lower()
            if (
                DIRECT_EFFECT_NEGATION_PATTERN.search(prefix)
                or COORDINATED_EFFECT_NEGATION_PATTERN.search(prefix)
            ):
                negated.add(risk_type)
    return frozenset(negated)


def _has_unnegated_effect_match(pattern: re.Pattern[str], value: str) -> bool:
    for match in pattern.finditer(value):
        prefix = value[: match.start()].rstrip().lower()
        if (
            DIRECT_EFFECT_NEGATION_PATTERN.search(prefix)
            or COORDINATED_EFFECT_NEGATION_PATTERN.search(prefix)
        ):
            continue
        return True
    return False


def _risk_audit_sources(graph: DynamicTaskGraph) -> tuple[AuditSource, ...]:
    sources = [
        AuditSource(
            source_id="raw_goal",
            source_kind="raw_goal",
            text=graph.raw_user_goal or graph.goal.objective,
        ),
        AuditSource(
            source_id="goal.objective",
            source_kind="goal_objective",
            text=graph.goal.objective,
        ),
    ]
    sources.extend(
        AuditSource(
            source_id=f"constraints.{index}",
            source_kind="goal_constraint",
            text=text,
        )
        for index, text in enumerate(graph.constraints)
    )
    for condition in graph.completion_conditions:
        sources.append(
            AuditSource(
                source_id=(
                    f"completion_conditions.{condition.condition_id}.description"
                ),
                source_kind="goal_completion_condition",
                text=condition.description,
            )
        )
        sources.extend(
            AuditSource(
                source_id=(
                    f"completion_conditions.{condition.condition_id}."
                    f"evidence_required.{index}"
                ),
                source_kind="goal_completion_condition",
                text=text,
            )
            for index, text in enumerate(condition.evidence_required)
        )
    for subgoal in graph.subgoals:
        sources.append(
            AuditSource(
                source_id=f"subgoals.{subgoal.subgoal_id}.objective",
                source_kind="subgoal_objective",
                subgoal_id=subgoal.subgoal_id,
                text=subgoal.objective,
            )
        )
        sources.extend(
            AuditSource(
                source_id=f"subgoals.{subgoal.subgoal_id}.constraints.{index}",
                source_kind="subgoal_constraint",
                subgoal_id=subgoal.subgoal_id,
                text=text,
            )
            for index, text in enumerate(subgoal.constraints)
        )
        sources.extend(
            AuditSource(
                source_id=(
                    f"subgoals.{subgoal.subgoal_id}.completion_conditions.{index}"
                ),
                source_kind="subgoal_completion_condition",
                subgoal_id=subgoal.subgoal_id,
                text=text,
            )
            for index, text in enumerate(subgoal.completion_conditions)
        )
    return tuple(sources)


def _apply_local_risk_supplements(
    report: SemanticRiskAuditReport,
    sources: tuple[AuditSource, ...],
    *,
    graph: DynamicTaskGraph | None = None,
) -> SemanticRiskAuditReport:
    source_map = {item.source_id: item for item in sources}
    source_groups: dict[str | None, list[AuditSource]] = {}
    for source in sources:
        source_groups.setdefault(source.subgoal_id, []).append(source)
    transient_navigation_scopes = {
        scope_id
        for scope_id, group in source_groups.items()
        if any(LOCAL_TRANSIENT_NAVIGATION_PATTERN.search(item.text) for item in group)
        and not any(_infer_external_risk_types(item.text) for item in group)
    }
    structured_navigation_scopes = (
        _structured_reversible_navigation_scopes(graph, source_groups)
        if graph is not None
        else frozenset()
    )
    local_literal_action_scopes = (
        _structured_local_literal_action_scopes(graph, source_groups)
        if graph is not None
        else frozenset()
    )
    local_input_scopes = (
        _explicit_local_input_audit_scopes(graph, source_groups)
        if graph is not None
        else set()
    )
    local_input_workflow_scopes = (
        _structured_local_input_workflow_scopes(graph, source_groups)
        if graph is not None
        else frozenset()
    )
    local_temporary_clear_scopes = (
        _explicit_local_temporary_clear_audit_scopes(graph)
        if graph is not None
        else frozenset()
    )
    read_only_risk_control_scopes = (
        {
            subgoal.subgoal_id
            for subgoal in graph.subgoals
            if subgoal.external_impact == "read_only"
            and not subgoal.risk_action_ids
            and _is_read_only_risk_control_state(
                subgoal.objective,
                subgoal.constraints,
                subgoal.completion_conditions,
            )
        }
        if graph is not None
        else set()
    )
    local_input_graph_is_risk_free = graph is None or not graph.risk_actions
    current_foreground_keyboard_scope = bool(
        graph is not None
        and len(graph.goal.target_apps) == 1
        and graph.goal.target_apps[0].app_id == "current_foreground"
        and not graph.risk_actions
    )
    local_keyboard_mode_scopes = (
        {
            scope_id
            for scope_id, group in source_groups.items()
            if _is_reversible_local_keyboard_mode(*(item.text for item in group))
            and (
                scope_id is None
                or any(
                    subgoal.subgoal_id == scope_id
                    and subgoal.external_impact == "navigation_only"
                    and not subgoal.risk_action_ids
                    for subgoal in graph.subgoals
                )
            )
        }
        if current_foreground_keyboard_scope
        else set()
    )
    assessments = []
    for assessment in report.assessments:
        source = source_map[assessment.source_id]
        scope_is_local_input = assessment.subgoal_id in local_input_scopes
        if assessment.subgoal_id is None and None in local_input_scopes:
            scope_is_local_input = True
        scope_is_local_keyboard_mode = (
            assessment.subgoal_id in local_keyboard_mode_scopes
        )
        if assessment.subgoal_id is None and None in local_keyboard_mode_scopes:
            scope_is_local_keyboard_mode = True
        is_negated_constraint = source.source_kind in {
            "goal_constraint",
            "subgoal_constraint",
        } and source.text.strip().lower().startswith(
            ("不要", "不得", "禁止", "不能", "避免", "do not", "never")
        )
        inferred = (
            frozenset()
            if is_negated_constraint
            else _infer_external_risk_types(source.text)
        )
        negated_types = _infer_directly_negated_risk_types(source.text)
        model_types = frozenset(assessment.risk_types)
        if assessment.subgoal_id in read_only_risk_control_scopes:
            assessment = replace(
                assessment,
                external_impact="read_only",
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地校验确认这里只核对被明确禁止触发的风险控件可见状态"
                ),
            )
            model_types = frozenset()
        if (
            assessment.external_impact == "external_state"
            and model_types
            and model_types <= negated_types
            and not inferred
        ):
            corrected_impact = (
                "navigation_only"
                if SAFE_NAVIGATION_SEMANTIC_PATTERN.search(source.text)
                else "read_only"
            )
            assessment = replace(
                assessment,
                external_impact=corrected_impact,
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地逐匹配否定校验确认该风险类型仅以直接否定形式出现"
                ),
            )
        if (
            assessment.external_impact in {"external_state", "unknown"}
            and scope_is_local_input
            and local_input_graph_is_risk_free
        ):
            assessment = replace(
                assessment,
                external_impact="navigation_only",
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地校验确认只改变未提交输入框临时文字且相关提交效果被直接禁止"
                ),
            )
        if (
            assessment.external_impact in {"external_state", "unknown"}
            and assessment.subgoal_id in local_temporary_clear_scopes
            and not inferred
        ):
            assessment = replace(
                assessment,
                external_impact="navigation_only",
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地校验确认仅把当前唯一未提交临时草稿恢复为空白且禁止任何外部效果"
                ),
            )
        if (
            assessment.external_impact in {"external_state", "unknown"}
            and model_types <= {"unknown_external_effect", "data_mutation"}
            and inferred <= {"unknown_external_effect"}
            and assessment.subgoal_id in local_input_workflow_scopes
        ):
            expected_impact = next(
                item.external_impact
                for item in graph.subgoals
                if item.subgoal_id == assessment.subgoal_id
            )
            assessment = replace(
                assessment,
                external_impact=expected_impact,
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地一致性校验确认该节点只属于已证明未提交输入的同一依赖链"
                ),
            )
        if (
            assessment.external_impact in {"external_state", "unknown"}
            and model_types <= {"unknown_external_effect", "data_mutation"}
            and inferred <= {"unknown_external_effect"}
            and scope_is_local_keyboard_mode
        ):
            assessment = replace(
                assessment,
                external_impact="navigation_only",
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地校验确认只改变未提交的设备输入法临时模式"
                ),
            )
        if (
            assessment.external_impact in {"external_state", "unknown"}
            and model_types <= {"unknown_external_effect", "data_mutation"}
            and inferred <= {"unknown_external_effect"}
            and assessment.subgoal_id in local_literal_action_scopes
        ):
            expected_impact = "navigation_only"
            if assessment.subgoal_id is not None:
                expected_impact = next(
                    item.external_impact
                    for item in graph.subgoals
                    if item.subgoal_id == assessment.subgoal_id
                )
            assessment = replace(
                assessment,
                external_impact=expected_impact,
                risk_types=(),
                reason=(
                    assessment.reason
                    + "；本地一致性校验确认精确字面动作标签仅对应当前页面的本机临时状态"
                ),
            )
        if (
            assessment.external_impact == "external_state"
            and model_types
            and model_types <= {"unknown_external_effect", "data_mutation"}
            and not inferred
            and (
                assessment.subgoal_id in transient_navigation_scopes
                or assessment.subgoal_id in structured_navigation_scopes
            )
        ):
            structured_reconciliation = (
                assessment.subgoal_id in structured_navigation_scopes
            )
            assessment = replace(
                assessment,
                external_impact="navigation_only",
                risk_types=(),
                reason=(
                    assessment.reason
                    + (
                        "；本地一致性校验确认结构化影响为无风险的可逆导航，"
                        "实体和整组语义均不含外部效果"
                        if structured_reconciliation
                        else "；本地校验确认只涉及临时界面层级或标签页导航"
                    )
                ),
            )
        if (
            inferred
            and assessment.subgoal_id not in read_only_risk_control_scopes
            and not (
                scope_is_local_input and local_input_graph_is_risk_free
            )
            and not (
                scope_is_local_keyboard_mode
                and inferred <= {"unknown_external_effect"}
            )
            and assessment.external_impact != "unknown"
        ):
            assessment = replace(
                assessment,
                external_impact="external_state",
                risk_types=tuple(sorted(set(assessment.risk_types) | set(inferred))),
                reason=(
                    assessment.reason
                    + "；本地单字段防御规则提供了额外外部状态证据"
                ),
            )
        assessments.append(assessment)
    return replace(report, assessments=tuple(assessments))


def _normalize_explicit_ui_label_payload(
    payload: dict[str, Any],
    raw_user_goal: str,
) -> dict[str, Any]:
    """Keep an explicitly quoted UI label out of high-level state prose.

    The original user text is deliberately left untouched for the independent
    semantic risk audit.  Only an explicitly quoted label in either the narrow
    ``visible text/label is \"...\"`` form or immediately followed by a UI-role
    noun can mint ``target_ui_label``; this does not authorize an action.
    """

    source = str(raw_user_goal or "")
    patterns = (
        re.compile(
            r"(?:目标(?:入口|控件|元素)?的?)?"
            r"(?:可见)?(?:文字|标签|名称)\s*(?:是|为|：|:)\s*"
            r"[“‘\"]([^”’\"\r\n]{1,80})[”’\"]",
            re.IGNORECASE,
        ),
        re.compile(
            r"[“‘\"]([^”’\"\r\n]{1,80})[”’\"]\s*"
            r"(?:入口|按钮|选项|标签|控件|元素)",
            re.IGNORECASE,
        ),
    )
    action_like_labels: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(source):
            candidate = match.group(1).strip()
            try:
                _reject_low_level_instruction(
                    candidate,
                    "goal.entities.target_ui_label",
                )
            except TaskGraphError:
                if candidate not in action_like_labels:
                    action_like_labels.append(candidate)
    if not action_like_labels:
        return payload
    if len(action_like_labels) != 1:
        raise TaskGraphError("用户目标包含多个动作词字面 UI 标签，无法唯一绑定。")
    label = action_like_labels[0]

    value = json.loads(json.dumps(payload, ensure_ascii=False))
    raw_goal = value.get("goal")
    if not isinstance(raw_goal, dict):
        return value
    entities = raw_goal.get("entities")
    if not isinstance(entities, dict):
        return value
    existing = entities.get("target_ui_label")
    if existing not in (None, "", label):
        raise TaskGraphError("DeepSeek 返回的 target_ui_label 与用户字面标签冲突。")
    entities["target_ui_label"] = label

    def replace_label(item: Any) -> Any:
        if not isinstance(item, str):
            return item
        normalized = item.replace(f"“{label}”", "目标入口")
        normalized = normalized.replace(f"‘{label}’", "目标入口")
        normalized = normalized.replace(f'"{label}"', "目标入口")
        normalized = normalized.replace(label, "目标入口")
        normalized = re.sub(
            r"(?:可见)?(?:文字|标签|名称)\s*(?:是|为|：|:)\s*目标入口",
            "目标入口",
            normalized,
        )
        if "长按" in label:
            normalized = re.sub(
                r"(?:被\s*)?长按(?:完成|成功)?",
                "处于当前页面可见的本机临时结果状态",
                normalized,
            )
        if "拖动" in label:
            normalized = re.sub(
                r"(?:被\s*)?拖动\s*(?:到|至|进入)?",
                "位于当前页面的本机临时目标位置",
                normalized,
            )
        return normalized

    raw_goal["objective"] = replace_label(raw_goal.get("objective"))
    conditions = value.get("completion_conditions")
    if isinstance(conditions, list):
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            condition["description"] = replace_label(condition.get("description"))
            evidence = condition.get("evidence_required")
            if isinstance(evidence, list):
                condition["evidence_required"] = [
                    replace_label(item) for item in evidence
                ]
    subgoals = value.get("subgoals")
    if isinstance(subgoals, list):
        for subgoal in subgoals:
            if not isinstance(subgoal, dict):
                continue
            subgoal["objective"] = replace_label(subgoal.get("objective"))
            completion = subgoal.get("completion_conditions")
            if isinstance(completion, list):
                subgoal["completion_conditions"] = [
                    replace_label(item) for item in completion
                ]
    return value


def _structured_reversible_navigation_scopes(
    graph: DynamicTaskGraph,
    source_groups: dict[str | None, list[AuditSource]],
) -> frozenset[str | None]:
    """Reconcile only navigation scopes whose structure and semantics agree."""

    entity_text = json.dumps(
        graph.goal.entities,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if _infer_external_risk_types(entity_text):
        return frozenset()
    subgoals = {item.subgoal_id: item for item in graph.subgoals}
    safe_scopes: set[str | None] = set()
    for scope_id, group in source_groups.items():
        scoped_subgoals = (
            tuple(graph.subgoals)
            if scope_id is None
            else ((subgoals[scope_id],) if scope_id in subgoals else ())
        )
        if not scoped_subgoals:
            continue
        if any(
            item.external_impact != "navigation_only" or item.risk_action_ids
            for item in scoped_subgoals
        ):
            continue
        if scope_id is None and graph.risk_actions:
            continue
        texts = tuple(item.text for item in group)
        if any(_infer_external_risk_types(text) for text in texts):
            continue
        if not any(
            _has_reversible_navigation_semantics(text)
            for text in (*texts, entity_text)
        ):
            continue
        safe_scopes.add(scope_id)
    return frozenset(safe_scopes)


def _structured_local_literal_action_scopes(
    graph: DynamicTaskGraph,
    source_groups: dict[str | None, list[AuditSource]],
) -> frozenset[str | None]:
    """Reconcile only a fully structured, explicitly local gesture-label task."""

    if (
        len(graph.goal.target_apps) != 1
        or graph.goal.target_apps[0].app_id != "current_foreground"
        or graph.risk_actions
        or _infer_external_risk_types(graph.raw_user_goal)
        or not _infer_directly_negated_risk_types(graph.raw_user_goal)
    ):
        return frozenset()
    label = str(graph.goal.entities.get("target_ui_label") or "").strip()
    marker_groups = (
        ("长按", "long_press", "longpress"),
        ("拖动", "drag"),
    )
    matched = tuple(
        group
        for group in marker_groups
        if any(marker in label.casefold() for marker in group)
    )
    if len(matched) != 1 or _infer_external_risk_types(label):
        return frozenset()

    subgoals = {item.subgoal_id: item for item in graph.subgoals}
    safe_navigation_ids: set[str] = set()
    safe_read_only_ids: set[str] = set()
    for subgoal in graph.subgoals:
        group = source_groups.get(subgoal.subgoal_id, [])
        texts = tuple(item.text for item in group)
        if (
            subgoal.external_impact not in {"navigation_only", "read_only"}
            or subgoal.risk_action_ids
            or not texts
            or any(_infer_external_risk_types(text) for text in texts)
        ):
            continue
        combined = " ".join(texts)
        if subgoal.external_impact == "navigation_only" and (
            "本机临时" in combined and "当前页面" in combined
        ):
            safe_navigation_ids.add(subgoal.subgoal_id)
        elif subgoal.external_impact == "read_only" and re.search(
            r"(?:显示|可见|观察|状态区域)", combined
        ):
            safe_read_only_ids.add(subgoal.subgoal_id)
    if not safe_navigation_ids:
        return frozenset()

    scopes: set[str | None] = set(safe_navigation_ids | safe_read_only_ids)
    global_group = source_groups.get(None, [])
    if (
        global_group
        and not any(_infer_external_risk_types(item.text) for item in global_group)
        and all(
            item.subgoal_id in scopes
            for item in graph.subgoals
        )
    ):
        scopes.add(None)
    return frozenset(scopes)


def _has_reversible_navigation_semantics(value: str) -> bool:
    normalized = re.sub(r"[_-]+", " ", value)
    return bool(
        LOCAL_TRANSIENT_NAVIGATION_PATTERN.search(normalized)
        or REVERSIBLE_NAVIGATION_EFFECT_PATTERN.search(normalized)
    )


def _validate_graph_against_risk_audit(
    graph: DynamicTaskGraph,
    report: SemanticRiskAuditReport,
) -> None:
    if graph.status == "blocked":
        return
    risks = {item.risk_id: item for item in graph.risk_actions}
    subgoals = {item.subgoal_id: item for item in graph.subgoals}
    for subgoal_id, subgoal in subgoals.items():
        assessments = tuple(
            item for item in report.assessments if item.subgoal_id == subgoal_id
        )
        if not assessments:
            raise TaskGraphError(
                f"语义风险审计遗漏子目标，失败关闭为 unknown：{subgoal_id}"
            )
        impact, risk_types = _aggregate_audit_assessments(assessments)
        safe_classification_disagreement = {
            impact,
            subgoal.external_impact,
        } <= {"read_only", "navigation_only"}
        if impact != subgoal.external_impact and not safe_classification_disagreement:
            if impact == "unknown":
                raise TaskGraphError(
                    f"语义风险审计为 unknown 且任务图未声明匹配风险：{subgoal_id}"
                )
            raise TaskGraphError(
                f"语义风险审计与任务图分类冲突：{subgoal_id} "
                f"audit={impact} graph={subgoal.external_impact}"
            )
        linked_types = {
            risks[risk_id].risk_type
            for risk_id in subgoal.risk_action_ids
            if risk_id in risks
        }
        missing = set(risk_types) - linked_types
        if missing:
            raise TaskGraphError(
                f"语义风险审计要求子目标关联匹配风险：{subgoal_id} / "
                + ", ".join(sorted(missing))
            )

    global_assessments = tuple(
        item for item in report.assessments if item.subgoal_id is None
    )
    if not global_assessments:
        raise TaskGraphError("语义风险审计遗漏用户目标，失败关闭为 unknown。")
    global_impact, global_risk_types = _aggregate_audit_assessments(
        global_assessments
    )
    graph_impacts = {item.external_impact for item in graph.subgoals}
    graph_risk_types = {item.risk_type for item in graph.risk_actions}
    if global_impact == "external_state" and "external_state" not in graph_impacts:
        raise TaskGraphError("语义风险审计与任务图分类冲突：全局目标包含外部状态。")
    if global_impact == "unknown" and "unknown" not in graph_impacts:
        raise TaskGraphError("语义风险审计为 unknown 且任务图未声明匹配风险：全局目标。")
    missing_global = set(global_risk_types) - graph_risk_types
    if missing_global:
        raise TaskGraphError(
            "语义风险审计要求任务图关联匹配风险："
            + ", ".join(sorted(missing_global))
        )


def _aggregate_audit_assessments(
    assessments: tuple[RiskAuditAssessment, ...],
) -> tuple[str, tuple[str, ...]]:
    impacts = {item.external_impact for item in assessments}
    if "external_state" in impacts:
        impact = "external_state"
        risk_types = {
            risk_type
            for item in assessments
            if item.external_impact == "external_state"
            for risk_type in item.risk_types
        }
    elif "unknown" in impacts:
        impact = "unknown"
        risk_types = {"unknown_external_effect"}
    elif "navigation_only" in impacts:
        impact = "navigation_only"
        risk_types = set()
    else:
        impact = "read_only"
        risk_types = set()
    return impact, tuple(sorted(risk_types))


def _reject_control_fields(value: Any, path: str) -> None:
    forbidden = {
        "action",
        "actions",
        "step",
        "steps",
        "tap",
        "click",
        "swipe",
        "coordinate",
        "coordinates",
        "x",
        "y",
        "shell",
        "command",
        "powershell",
        "python",
        "main_exe",
        "execution_plan",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in forbidden:
                raise TaskGraphError(f"任务图包含低层控制字段：{path}.{key}")
            _reject_control_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_control_fields(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise TaskGraphError(f"任务图字段类型不受支持：{path}")
