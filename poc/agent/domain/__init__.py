"""Domain contracts shared by the modular monolith."""

from .device_execution import (
    DEVICE_EXECUTOR_PROTOCOL,
    EXECUTABLE_ACTION_KINDS,
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
    AgentSessionRepository,
    require_session_device,
)
from .session_evidence import (
    AgentEvidenceStoreFactory,
    AgentEvidenceStorePort,
    EvidenceStoreError,
)

__all__ = [
    "ACTIVE_SESSION_STATUSES",
    "DEVICE_EXECUTOR_PROTOCOL",
    "EXECUTABLE_ACTION_KINDS",
    "AgentSession",
    "AgentSessionConflictError",
    "AgentSessionDeviceMismatchError",
    "AgentSessionNotFoundError",
    "AgentSessionRepository",
    "AgentEvidenceStoreFactory",
    "AgentEvidenceStorePort",
    "DeviceActionRequest",
    "DeviceExecutionError",
    "DeviceExecutionResult",
    "DeviceExecutor",
    "DeviceTaskRegistryError",
    "DeviceTaskRegistryPort",
    "EvidenceStoreError",
    "require_session_device",
]
