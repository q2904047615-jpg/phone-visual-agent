"""Infrastructure adapters for the agent modular monolith."""

from .camera_coordinator import CameraPreviewUnavailable, DeviceCameraCoordinator
from .device_controller_registry import (
    DeviceControllerRegistry,
    DeviceControllerRegistryError,
    ProvisionalDeviceControllerError,
)
from .device_exclusivity import InterProcessLease, SHARED_DEVICE_LEASE_DIR
from .device_executor import ReplayDeviceExecutor, RobotDeviceExecutor
from .device_runtime_resources import DeviceRuntimeResourceError, DeviceRuntimeResourceRegistry
from .device_task_registry import DeviceTaskRegistry
from .file_system_evidence_store import FileSystemAgentEvidenceStore
from .in_memory_session_repository import InMemoryAgentSessionRepository

__all__ = [
    "DeviceTaskRegistry",
    "CameraPreviewUnavailable",
    "DeviceCameraCoordinator",
    "DeviceControllerRegistry",
    "DeviceControllerRegistryError",
    "DeviceRuntimeResourceError",
    "DeviceRuntimeResourceRegistry",
    "FileSystemAgentEvidenceStore",
    "InMemoryAgentSessionRepository",
    "InterProcessLease",
    "ReplayDeviceExecutor",
    "RobotDeviceExecutor",
    "ProvisionalDeviceControllerError",
    "SHARED_DEVICE_LEASE_DIR",
]
