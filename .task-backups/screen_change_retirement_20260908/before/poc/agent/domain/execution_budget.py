"""One task-wide budget; no per-field or per-gesture retry authority."""
from dataclasses import dataclass

DEFAULT_DEVICE_ACTION_BUDGET = 100
DEFAULT_OBSERVATION_BUDGET = 200


class ExecutionBudgetExhausted(RuntimeError):
    pass


def positive_budget(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("整任务预算必须是正整数。")
    return value


@dataclass
class TaskExecutionBudget:
    max_physical_actions: int = DEFAULT_DEVICE_ACTION_BUDGET
    max_observations: int = DEFAULT_OBSERVATION_BUDGET
    observation_attempts: int = 0

    def __post_init__(self) -> None:
        positive_budget(self.max_physical_actions)
        positive_budget(self.max_observations)

    def configure(self, *, max_physical_actions: int | None = None,
        max_observations: int | None = None) -> None:
        # Validate both before changing either; updating limits never resets use.
        actions = positive_budget(self.max_physical_actions if max_physical_actions is None else max_physical_actions)
        observations = positive_budget(self.max_observations if max_observations is None else max_observations)
        self.max_physical_actions, self.max_observations = actions, observations

    def request_observation(self, *, physical_actions: int, will_execute: bool = False) -> None:
        if will_execute and physical_actions >= self.max_physical_actions:
            raise ExecutionBudgetExhausted("达到整任务设备动作预算，已暂停并保留进度。")
        if self.observation_attempts >= self.max_observations:
            raise ExecutionBudgetExhausted("达到整任务观察预算，已暂停并保留进度。")
        self.observation_attempts += 1

    def snapshot(self, physical_actions: int) -> dict:
        return {"max_physical_actions": self.max_physical_actions,
            "max_observations": self.max_observations,
            "physical_actions": physical_actions, "observation_attempts": self.observation_attempts,
            "remaining_actions": max(0, self.max_physical_actions - physical_actions),
            "remaining_observations": max(0, self.max_observations - self.observation_attempts)}
