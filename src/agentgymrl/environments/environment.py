from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from agentgymrl.inference.model_output import ModelOutput

STATE = TypeVar("STATE")


@dataclass
class ActionExecutionResult:
    """Result of executing an agent action in the environment."""

    raw_result: str
    exception: Optional[Exception] = None

    @property
    def has_error(self) -> bool:
        return self.exception is not None


@dataclass
class EnvironmentResult(Generic[STATE]):
    """Result of handling an action in the environment."""

    action_exec_results: Optional[List[ActionExecutionResult]] = None
    updated_state: Optional[STATE] = None
    should_end: bool = False
    has_error: bool = False


class Environment(Generic[STATE], ABC):
    """Abstract environment interface that can handle agent outputs and maintain state."""

    def __init__(self, env_idx: int = 0):
        self.env_idx = env_idx

    @abstractmethod
    def handle_output(
        self, current_state: STATE, model_output: ModelOutput
    ) -> EnvironmentResult[STATE]:
        """
        Process agent output and update the state. This should handle:
        1. Action execution if an action is present
        2. State updates for any agent output (e.g., updating conversation history)
        3. Deciding whether the environment (& therefore the sequence) should end

        Args:
            current_state: Current environment state
            model_output: Output from the agent (may or may not contain actions)

        Returns:
            EnvironmentResult containing updated state and result information
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources when done with this environment."""
