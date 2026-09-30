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
    AgentSessionRepository,
    require_session_device,
)
from .session_evidence import AgentEvidenceStoreFactory, AgentEvidenceStorePort, EvidenceStoreError

__all__ = ['ACTIVE_SESSION_STATUSES', 'CANONICAL_SELECTION_RECEIPT_VERSION', 'DEVICE_EXECUTOR_PROTOCOL',
    'AgentSession', 'AgentSessionConflictError', 'AgentSessionDeviceMismatchError',
    'AgentSessionNotFoundError', 'AgentSessionRepository',
    'AgentEvidenceStoreFactory', 'AgentEvidenceStorePort', 'CanonicalSelectionReceipt', 'ConfirmationAuthority',
    'DeviceActionRequest', 'DeviceExecutionError', 'DeviceExecutionResult', 'DeviceExecutor', 'DeviceTaskRegistryError',
    'DeviceTaskRegistryPort', 'EvidenceStoreError', 'EffectConfirmationAuthority',
    'require_session_device']
