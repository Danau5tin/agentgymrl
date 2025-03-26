from dataclasses import dataclass
from typing import List


@dataclass
class AgentInstructions:
    sys_msg: str
    tool_schemas: list[dict[str, any]]

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