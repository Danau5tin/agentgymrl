from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from agentgymrl.environments.environment import Environment, EnvironmentResult
from agentgymrl.inference.model_output import ModelOutput, ToolCall
from environment.entities.expression import Expression
from environment.tools.calculator import calculate


class CalculatorError(Exception):
    """Base exception for calculator environment errors."""
    pass


class UnsupportedToolCallError(CalculatorError):
    """Exception raised when an unsupported tool call is made."""


class MultipleToolCallsError(CalculatorError):
    """Exception raised when multiple tool calls are made simultaneously."""


@dataclass
class CalculatorState:
    """State for calculator environment."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def copy(self) -> "CalculatorState":
        """Create a deep copy of the current state."""
        return CalculatorState(messages=self.messages.copy())
    
    def add_message(self, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add a message to the state."""
        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)


class CalculatorEnvironment(Environment):
    """Environment for calculator agent."""
    
    def __init__(self, initial_state: Optional[CalculatorState] = None, env_idx: int = 0):
        """Initialize the calculator environment with an empty state."""
        super().__init__(env_idx=env_idx)
        self.state = initial_state or CalculatorState()

    def initialise_state_with_user_prompt(self, user_prompt: str) -> None:
        """Initialise the calculator environment with a user prompt."""
        new_state = CalculatorState()
        new_state.add_message("user", user_prompt)
        self.state = new_state
    
    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        """Process model output and update environment state accordingly."""
        new_state = self.state.copy()
        
        tool_calls_dict = [tc.to_dict() for tc in model_output.tool_calls] if model_output.tool_calls else None
        new_state.add_message("assistant", model_output.raw_content, tool_calls_dict)
        
        if len(model_output.tool_calls) == 0:
            self.state = new_state
            return EnvironmentResult(should_end_sequence=True)
        
        result = self._handle_tool_calls(model_output.tool_calls, new_state)
        self.state = new_state
        return result
    
    def _handle_tool_calls(self, tool_calls: List[ToolCall], state: CalculatorState) -> EnvironmentResult:
        """Process and validate tool calls from the model output."""
        try:
            if len(tool_calls) > 1:
                raise MultipleToolCallsError("Multiple tool calls not supported. Try one at a time.")
            
            tool_call = tool_calls[0]
            if tool_call.tool_name != "calculate":
                raise UnsupportedToolCallError(f"Unsupported tool call: {tool_call.tool_name}")
            
            return self._execute_calculator_call(tool_call, state)
            
        except CalculatorError as e:
            error_msg = f"Error: {str(e)}"
            state.add_message("tool", error_msg)
            return EnvironmentResult(
                should_end_sequence=False,  # Model should continue and try to recover
                output_to_show_model=error_msg,
                exception=e,
            )
    
    def _execute_calculator_call(self, tool_call: ToolCall, state: CalculatorState) -> EnvironmentResult:
        """Execute a calculator tool call with the given parameters."""
        try:
            expression = Expression(**tool_call.tool_parameters["expression"])
            result = calculate(expression)
            
            tool_call_output = str(result)
            state.add_message("tool", tool_call_output)
            
            return EnvironmentResult(
                should_end_sequence=False,
                output_to_show_model=tool_call_output,
            )
        except Exception as e:
            tool_call_output = f"Error: {str(e)}"
            state.add_message("tool", tool_call_output)
            
            return EnvironmentResult(
                should_end_sequence=False,
                output_to_show_model=tool_call_output,
                exception=e,
            )
        
    def get_state(self) -> CalculatorState:
        """Get the current state of the calculator environment."""
        return self.state
    
    def cleanup(self):
        """Clean up any resources. No cleanup needed for this environment."""
        pass
