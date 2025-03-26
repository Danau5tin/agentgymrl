import json
from typing import Any, List

from transformers import PreTrainedTokenizer

from agentgymrl.training.token_handlers.token_handler import TokenHandler


class Phi4MiniInstructTokenHandler(TokenHandler):

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)
        self.eos_token_id = tokenizer.eos_token_id  # <|endoftext|>
        self.user_token_id = 200021  # <|user|>
        self.assistant_token_id = 200019  # <|assistant|>
        self.eot_token_id = 200020  # <|end|>
        self.tool_token_id = 200023  # <|tool|> 
        

    def start_sequence(
        self, 
        sys_msg: str, 
        tools: List[dict[str, Any]],
        should_train_on: bool = False
    ) -> None:
        """Start a new conversation sequence"""
        messages = [
            {'role': 'system', 'content': sys_msg, 'tools': json.dumps(tools)},
        ]
        new_tokens = self.tokenizer.apply_chat_template(messages, tokenize=True)
        if new_tokens[-1] != self.eos_token_id:
            raise ValueError("Expected EOS token at the end of the sequence")
        
        new_tokens_without_eos = new_tokens[:-1]
        self.add_tokens(new_tokens_without_eos, should_train_on=should_train_on)

    def add_user_message(self, content: str, should_train_on: bool=False) -> None:
        """Add a user message"""
        content_ids = self.tokenizer.encode(content)
        new_tokens = [
            self.user_token_id,
            *content_ids,
            self.eot_token_id,
        ]
        self.add_tokens(new_tokens, should_train_on=should_train_on)

    def add_assistant_generation_prompt(self, should_train_on: bool=True):
        self.add_tokens([self.assistant_token_id], should_train_on=should_train_on)

    def add_tool_output_message(self, content: str, should_train_on = False):
        content_tokens = self.tokenizer.tokenize(content, add_special_tokens=False)
        new_tokens = [self.tool_token_id, *content_tokens, self.eot_token_id]
        self.add_tokens(new_tokens, should_train_on=should_train_on)

    def add_end_of_sequence_if_not_present(self):
        if self.all_token_ids[-1] != self.eos_token_id:
            self.add_tokens([self.eos_token_id], should_train_on=True)
    