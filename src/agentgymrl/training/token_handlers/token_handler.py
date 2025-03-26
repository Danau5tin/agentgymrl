from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List
from transformers import PreTrainedTokenizer

@dataclass
class TokenHandler(ABC):
    """Abstract base class for handling different LLM chat templates and managing token state"""
    tokenizer: PreTrainedTokenizer
    
    # State storage
    all_token_ids: List[int] = field(default_factory=list)
    source_mask: List[int] = field(default_factory=list)
    
    def add_tokens(self, tokens: List[int], should_train_on: bool) -> None:
        """Add tokens to the state with appropriate masking"""
        if not tokens:
            return
            
        self.all_token_ids.extend(tokens)
        self.source_mask.extend([1 if should_train_on else 0] * len(tokens))
    
    @abstractmethod
    def start_sequence(
        self, 
        sys_msg: str, 
        tools: List[dict[str, Any]],
        should_train_on: bool = False
    ) -> None:
        """Start a new conversation sequence"""

    
    @abstractmethod
    def add_user_message(self, content: str, should_train_on: bool=False) -> None:
        """Add a user message"""
    
    @abstractmethod
    def add_assistant_generation_prompt(self, should_train_on: bool=True) -> None:
        """Prompt the assistant to generate"""
    
    @abstractmethod
    def add_tool_output_message(self, content: str, should_train_on: bool = False) -> None:
        """Add a tool message"""
    
    @abstractmethod
    def add_end_of_sequence_if_not_present(self) -> None:
        """End the conversation sequence"""
    
    def get_all_tokens(self) -> List[int]:
        """Get all tokens"""
        return self.all_token_ids
    
    def get_source_mask(self) -> List[int]:
        """Get the source mask"""
        return self.source_mask