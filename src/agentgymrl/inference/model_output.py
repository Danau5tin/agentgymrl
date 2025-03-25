from typing import List, Optional


class AgentAction:
    """
    An AgentAction is anything an agent outputs that can be handled by the environment.
    """

    def __init__(
        self,
        identifier: str,
        is_final: bool = False,
    ):
        self.identifier = identifier
        self.is_final = is_final


class ModelOutput:
    """
    A ModelOutput contains the raw content of the output including any actions the agent decided to take.
    """

    def __init__(self, raw_content: str, agent_actions: Optional[List[AgentAction]]):
        self.raw_content = raw_content
        self.agent_actions = agent_actions
