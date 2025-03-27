from dataclasses import dataclass, field
from typing import Generic, List
from agentgymrl.environments.state import STATE
import torch


@dataclass
class SampleResult(Generic[STATE]):
    """Result of a sample generation."""
    env_state: STATE
    input_ids: torch.Tensor
    source_mask: torch.Tensor
    env_call_count: int = 0
    env_exceptions: List[Exception] = field(default_factory=list)

@dataclass
class ToolSampleResult(Generic[STATE]):
    """Stores a tool-based conversation along with its expected answer and any exceptions"""
    state: STATE
    answer: str
    env_exceptions: List[Exception] = field(default_factory=list)
