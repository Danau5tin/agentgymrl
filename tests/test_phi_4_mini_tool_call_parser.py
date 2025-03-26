import unittest

from agentgymrl.training.tool_call_parsers.phi_4_mini_instruct import Phi4MiniInstructToolCallParser


class TestPhi4MiniInstructToolCallParser(unittest.TestCase):
    def setUp(self):
        self.parser = Phi4MiniInstructToolCallParser()

    def test_empty_response(self):
        """Test parsing an empty response."""
        response = ""
        tool_calls = self.parser.parse_tool_calls(response)
        self.assertEqual(len(tool_calls), 0)

    def test_response_without_tool_calls(self):
        """Test parsing a response without any tool calls."""
        response = "This is a regular response without any tool calls."
        tool_calls = self.parser.parse_tool_calls(response)
        self.assertEqual(len(tool_calls), 0)

    def test_single_tool_call(self):
        """Test parsing a response with a single tool call."""
        response = "Let me search that for you. <|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": {\"query\": \"python programming\"}}}]<|/tool_call|><|end|>"
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].tool_name, "search")
        self.assertEqual(tool_calls[0].tool_parameters, {"query": "python programming"})

    def test_multiple_tool_calls(self):
        """Test parsing a response with multiple tool calls."""
        response = """Let me help you with that.
        <|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": {\"query\": \"weather forecast\"}}}]<|/tool_call|>
        Now let me check the time.
        <|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}}]<|/tool_call|>
        <|end|>"""
        
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].tool_name, "search")
        self.assertEqual(tool_calls[0].tool_parameters, {"query": "weather forecast"})
        self.assertEqual(tool_calls[1].tool_name, "get_time")
        self.assertEqual(tool_calls[1].tool_parameters, {"timezone": "UTC"})

    def test_tool_call_with_empty_arguments(self):
        """Test parsing a tool call with empty arguments."""
        response = "<|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"list_files\", \"arguments\": {}}}]<|/tool_call|><|end|>"
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].tool_name, "list_files")
        self.assertEqual(tool_calls[0].tool_parameters, {})

    def test_tool_call_with_nested_arguments(self):
        """Test parsing a tool call with nested arguments."""
        response = """<|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"create_user\", \"arguments\": {\"user\": {\"name\": \"John Doe\", \"age\": 30, \"address\": {\"street\": \"123 Main St\", \"city\": \"Anytown\"}}}}}]<|/tool_call|><|end|>"""
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].tool_name, "create_user")
        self.assertEqual(tool_calls[0].tool_parameters, {
            "user": {
                "name": "John Doe", 
                "age": 30, 
                "address": {
                    "street": "123 Main St", 
                    "city": "Anytown"
                }
            }
        })

    def test_tool_call_array_in_single_block(self):
        """Test parsing a response with an array of tool calls in a single block."""
        response = """<|tool_call|>[
            {\"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": {\"query\": \"python\"}}},
            {\"type\": \"function\", \"function\": {\"name\": \"translate\", \"arguments\": {\"text\": \"hello\", \"to_language\": \"es\"}}}
        ]<|/tool_call|><|end|>"""
        
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].tool_name, "search")
        self.assertEqual(tool_calls[0].tool_parameters, {"query": "python"})
        self.assertEqual(tool_calls[1].tool_name, "translate")
        self.assertEqual(tool_calls[1].tool_parameters, {"text": "hello", "to_language": "es"})

    def test_malformed_tool_call(self):
        """Test parsing a response with malformed tool call syntax."""
        response = """<|tool_call|>This is not valid JSON<|/tool_call|><|end|>"""
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 0)

    def test_string_arguments(self):
        """Test parsing a tool call with string arguments that need to be parsed as JSON."""
        response = """<|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": \"{\\\"query\\\": \\\"python\\\"}\"}}]<|/tool_call|><|end|>"""
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].tool_name, "search")
        self.assertEqual(tool_calls[0].tool_parameters, {"query": "python"})

    def test_mixed_valid_and_invalid_tool_calls(self):
        """Test parsing a response with both valid and invalid tool calls."""
        response = """
        <|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": {\"query\": \"python\"}}}]<|/tool_call|>
        <|tool_call|>Invalid JSON<|/tool_call|>
        <|tool_call|>[{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}}]<|/tool_call|>
        """
        
        tool_calls = self.parser.parse_tool_calls(response)
        
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].tool_name, "search")
        self.assertEqual(tool_calls[1].tool_name, "get_time")


if __name__ == "__main__":
    unittest.main()
