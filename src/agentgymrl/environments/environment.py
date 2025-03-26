from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from agentgymrl.inference.model_output import ModelOutput


@dataclass
class EnvironmentResult:
    """Result of handling an action in the environment."""

    should_end_sequence: bool = False
    tool_call_output: Optional[str] = None
    exception: Optional[Exception] = None

    @property
    def has_error(self) -> bool:
        return self.exception is not None


class Environment(ABC):
    """Abstract environment interface that can handle agent outputs and maintain its own state if required.

    It will receive agent outputs and is responsible for updating the environment state and returning a response for the agent if required.
    """

    def __init__(self, env_idx: int = 0):
        self.env_idx = env_idx

    @abstractmethod
    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        """
        This should handle:
        1. Tool call execution
        2. Internal state updates (if any)
        3. Deciding when the environment (& therefore the sequence generation) should end

        Args:
            model_output: Output from the agent

        Returns:
            EnvironmentResult
        """

    @abstractmethod
    def get_state(self) -> Optional[dict]:
        """Get the environment state as a dictionary."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources when done with this environment."""
