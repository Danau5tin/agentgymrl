import re
import json
from typing import List
from agentgymrl.inference.model_output import ToolCall
from agentgymrl.training.tool_call_parsers.tool_call_parser import ToolCallParser


class Phi4MiniInstructToolCallParser(ToolCallParser):
    """
    Parser for extracting tool calls from Phi-4-mini-instruct model outputs.
    
    The expected format is:
    <|tool_call|>[{"type": "function", "function": {"name": "tool_name", "arguments": {...}}}]<|/tool_call|>
    """
    
    def parse_tool_calls(self, response_text: str) -> List[ToolCall]:
        """
        Parse tool calls from the response text.
        
        Args:
            response_text (str): The raw text output from the model
            
        Returns:
            List[ToolCall]: List of parsed tool calls, empty if none found
        """
        tool_calls = []
        
        pattern = r'<\|tool_call\|>(.*?)<\|/tool_call\|>'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call_data = json.loads(match)
                
                if isinstance(tool_call_data, list):
                    tool_calls.extend(self._process_tool_call_list(tool_call_data))
                else:
                    tool_call = self._process_single_tool_call(tool_call_data)
                    if tool_call:
                        tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
            except Exception:
                # Skip any other parsing errors
                continue
                
        return tool_calls
    
    def _process_tool_call_list(self, tool_call_list: List[dict]) -> List[ToolCall]:
        result = []
        for tool_call_data in tool_call_list:
            tool_call = self._process_single_tool_call(tool_call_data)
            if tool_call:
                result.append(tool_call)
        return result
    
    def _process_single_tool_call(self, tool_call_data: dict) -> ToolCall:
        if not isinstance(tool_call_data, dict):
            return None
            
        if 'type' in tool_call_data and tool_call_data.get('type') == 'function' and 'function' in tool_call_data:
            function_data = tool_call_data['function']
            if 'name' in function_data and 'arguments' in function_data:
                arguments = function_data['arguments']
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                
                return ToolCall(
                    tool_name=function_data['name'],
                    tool_parameters=arguments
                )
        
        return None
    