"""Domain contracts shared by the modular monolith."""

from .canonical_selection import CANONICAL_SELECTION_RECEIPT_VERSION, CanonicalSelectionReceipt
from .confirmation_authority import ConfirmationAuthority, EffectConfirmationAuthority
from .device_execution import (
    DEVICE_EXECUTOR_PROTOCOL,
    DeviceActionRequest,
    DeviceExecutionError,
    DeviceExecutionResult,
    DeviceExecutor,
    DeviceTaskRegistryError,
    DeviceTaskRegistryPort,
)
from .session import (
    ACTIVE_SESSION_STATUSES,
    AgentSession,
    AgentSessionConflictError,
    AgentSessionDeviceMismatchError,
    AgentSessionNotFoundError,
    AgentSessionRecoveryRequiredError,
    AgentSessionRepository,
    RECOVERY_REQUIRED_STATUS,
    require_session_device,
)
from .session_evidence import AgentEvidenceStoreFactory, AgentEvidenceStorePort, EvidenceStoreError
from .tool_intent import ToolIntent, ToolIntentError
from .model_gateway import ModelGateway, ModelGatewayRegistration, ModelGatewayRegistry
from .tool_registry import ToolRegistration, ToolRegistry, build_canonical_tool_registry
from .runtime_event import (
    FAILURE_CATEGORIES,
    MAX_REPLAY_PAGE_SIZE,
    RUNTIME_EVENT_REPLAY_PROTOCOL,
    RuntimeEvent,
    RuntimeEventPage,
    RuntimeFailure,
    classify_runtime_failure,
    validate_replay_cursor,
    validate_replay_page_size,
)

__all__ = ['ACTIVE_SESSION_STATUSES', 'CANONICAL_SELECTION_RECEIPT_VERSION', 'DEVICE_EXECUTOR_PROTOCOL',
    'AgentSession', 'AgentSessionConflictError', 'AgentSessionDeviceMismatchError',
    'AgentSessionNotFoundError', 'AgentSessionRecoveryRequiredError', 'AgentSessionRepository',
    'RECOVERY_REQUIRED_STATUS',
    'AgentEvidenceStoreFactory', 'AgentEvidenceStorePort', 'CanonicalSelectionReceipt', 'ConfirmationAuthority',
    'DeviceActionRequest', 'DeviceExecutionError', 'DeviceExecutionResult', 'DeviceExecutor', 'DeviceTaskRegistryError',
    'DeviceTaskRegistryPort', 'EvidenceStoreError', 'EffectConfirmationAuthority', 'ToolIntent', 'ToolIntentError',
    'ModelGateway', 'ModelGatewayRegistration', 'ModelGatewayRegistry', 'ToolRegistration', 'ToolRegistry',
    'build_canonical_tool_registry',
    'FAILURE_CATEGORIES', 'MAX_REPLAY_PAGE_SIZE', 'RUNTIME_EVENT_REPLAY_PROTOCOL',
    'RuntimeEvent', 'RuntimeEventPage', 'RuntimeFailure', 'classify_runtime_failure',
    'validate_replay_cursor', 'validate_replay_page_size',
    'require_session_device']
