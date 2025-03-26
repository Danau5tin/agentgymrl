from dataclasses import dataclass


@dataclass
class AgentInstructions:
    sys_msg: str
    tool_schemas: list[dict[str, any]]
