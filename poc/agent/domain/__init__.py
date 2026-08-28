"""Domain contracts shared by the modular monolith."""

from .app_surface_lineage import (
    AppSurfaceLineageAuthority,
    AppSurfaceLineageError,
    VerifiedAppSurfaceLineage,
)
from .canonical_selection import (
    CANONICAL_SELECTION_RECEIPT_VERSION,
    CanonicalSelectionReceipt,
)
from .confirmation_authority import (
    ConfirmationAuthority,
    EffectConfirmationAuthority,
)
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
    "CANONICAL_SELECTION_RECEIPT_VERSION",
    "DEVICE_EXECUTOR_PROTOCOL",
    "EXECUTABLE_ACTION_KINDS",
    "AgentSession",
    "AgentSessionConflictError",
    "AgentSessionDeviceMismatchError",
    "AgentSessionNotFoundError",
    "AgentSessionRepository",
    "AppSurfaceLineageAuthority",
    "AppSurfaceLineageError",
    "AgentEvidenceStoreFactory",
    "AgentEvidenceStorePort",
    "CanonicalSelectionReceipt",
    "ConfirmationAuthority",
    "DeviceActionRequest",
    "DeviceExecutionError",
    "DeviceExecutionResult",
    "DeviceExecutor",
    "DeviceTaskRegistryError",
    "DeviceTaskRegistryPort",
    "EvidenceStoreError",
    "EffectConfirmationAuthority",
    "VerifiedAppSurfaceLineage",
    "require_session_device",
]
