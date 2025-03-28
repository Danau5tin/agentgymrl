from dataclasses import dataclass, field
from typing import Any, Callable, Generic, List, Optional

from agentgymrl.environments.state import STATE
from agentgymrl.tool_calling.tool_func_schema_generator import generate_function_schema
from agentgymrl.training.common_entities.results import ToolSampleResult
from agentgymrl.training.environment_pool import EnvironmentConfig


@dataclass
class AgentConfig:
    sys_msg: str
    tools: List[Callable[..., Any]]
    temperature: float = 0.9
    max_env_calls: int = 20
    max_new_tokens: int = 1000
    tool_schemas: list[dict[str, any]] = field(init=False)

    def __post_init__(self):
        """Generate tool schemas after initialization."""
        self.tool_schemas = [generate_function_schema(tool) for tool in self.tools] if self.tools else []


@dataclass
class ReportingConfig:
    report_to: List[str]
    report_every_n_steps: int

    @staticmethod
    def create_wandb(report_every_n_steps: int = 10) -> 'ReportingConfig':
        """
        Creates a ReportingConfig for Weights & Biases reporting.
        
        Args:
            report_every_n_steps: Number of steps between reports
            
        Returns:
            ReportingConfig configured for wandb
        """
        return ReportingConfig(
            report_to=["wandb"],
            report_every_n_steps=report_every_n_steps
        )
    
@dataclass
class TrainingConfig(Generic[STATE]):
    hf_model_name: str
    train_csv_dataset_path: str
    output_dir: str
    agent_config: AgentConfig
    environment_config: EnvironmentConfig
    reward_func: Callable[[List[ToolSampleResult[STATE]]], List[float]]
    reporting_config: Optional[ReportingConfig] = None
    