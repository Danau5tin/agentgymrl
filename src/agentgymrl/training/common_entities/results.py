from dataclasses import dataclass, field
from typing import Any, List
import torch


@dataclass
class SampleResult:
    """Result of a sample generation."""
    env_state: dict[str, any]
    input_ids: torch.Tensor
    source_mask: torch.Tensor
    env_call_count: int = 0
    env_exceptions: List[Exception] = field(default_factory=list)

@dataclass
class ToolSampleResult:
    """Stores a tool-based conversation along with its expected answer and any exceptions"""
    state: dict[str, Any]
    answer: str
    env_exceptions: List[Exception] = field(default_factory=list)
