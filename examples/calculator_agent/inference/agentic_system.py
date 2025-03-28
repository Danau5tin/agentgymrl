
from typing import Callable, Generic

from agentgymrl.environments.environment import Environment
from agentgymrl.environments.state import STATE
from agentgymrl.inference.model_output import ModelOutput



class AgenticSystem(Generic[STATE]):
    """
    Generic agent execution loop that manages the interaction between
    an agent and its environment.
    """
    
    def __init__(
        self,
        environment: Environment[STATE],
        agent_callback: Callable[[STATE], ModelOutput],
        max_iterations: int = 10
    ):
        """
        Initialize the agent loop.
        
        Args:
            environment: Environment to execute actions in
            agent_callback: Function that produces agent outputs from state
            max_iterations: Maximum number of loop iterations
        """
        self.environment = environment
        self.agent_callback = agent_callback
        self.max_iterations = max_iterations
        self.current_iteration = 0
    
    def run(self, state: STATE) -> STATE:
        """
        Run the agent loop until completion or max iterations.
        
        Args:
            initial_state: Starting state
            
        Returns:
            Final state after execution
        """
        self.current_iteration += 1
        
        # Get agent output
        model_output = self.agent_callback(state)
        
        # Always let the environment update the state
        result = self.environment.handle_output(model_output)

        new_state = self.environment.get_state()
        
        if result.should_end_sequence or result.has_error or self.current_iteration >= self.max_iterations:
            self.environment.cleanup()
            return new_state
            
        # Continue the loop with updated state
        return self.run(new_state)
