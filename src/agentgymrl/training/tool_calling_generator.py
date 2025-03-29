import logging
from typing import Generic

from agentgymrl.environments.state import STATE
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from agentgymrl.inference.model_output import ModelOutput
from agentgymrl.training.environment_pool import EnvironmentPool
from agentgymrl.training.common_entities.config import AgentConfig
from agentgymrl.training.common_entities.results import SampleResult
from agentgymrl.training.token_handlers.phi_4_mini_instruct import (
    Phi4MiniInstructTokenHandler,
)
from agentgymrl.training.tool_call_parsers.phi_4_mini_instruct import (
    Phi4MiniInstructToolCallParser,
)


class ToolCallingGenerator(Generic[STATE]):
    """
    Handles the sequential generation process for tool-calling models.
    This class manages:
        - The generation of model output
        - Model output parsing
        - Providing the output to the relevant environment
        - Continuation of generation including tool call outputs
        - Looping through this until the environment decides to end the sequence or model provides eos token
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        environment_pool: EnvironmentPool[STATE],
        agent_config: AgentConfig,
        device: torch.device,
        log_level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.environment_pool = environment_pool
        self.agent_config = agent_config

        self.token_handler = Phi4MiniInstructTokenHandler(tokenizer=tokenizer)  # TODO (AIT-733): Create a factory and handle different models
        self.tool_call_parser = Phi4MiniInstructToolCallParser()  # TODO (AIT-733): Create a factory and handle different models

    def _initialise_conversation(self, env_idx: int, prompt: str) -> None:
        self.environment_pool.initialise_state_with_user_prompt(
            env_idx=env_idx, 
            user_prompt=prompt
        )

        self.token_handler.start_sequence(
            sys_msg=self.agent_config.sys_msg, 
            tools=self.agent_config.tool_schemas
        )
        self.token_handler.add_user_message(prompt)
        self.token_handler.add_assistant_generation_prompt()
        self.logger.debug("Initialized conversation and added assistant generation prompt")

    def _run_inference_on_model(self) -> ModelOutput:
        """
        Generate the next tokens from the model.
        """
        all_token_ids = self.token_handler.get_all_tokens()
        inputs = {'input_ids': torch.tensor([all_token_ids], device=self.device)}
        self.logger.debug(f"Generating next tokens from model with input length: {len(all_token_ids)}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.agent_config.temperature,
                do_sample=True,
                max_new_tokens=self.agent_config.max_new_tokens,
            )

        new_token_ids = outputs[0, len(all_token_ids):].tolist()
        self.token_handler.add_tokens(new_token_ids, should_train_on=True)

        new_text_with_spec_tokens = self.tokenizer.decode(new_token_ids, skip_special_tokens=False)
        new_text_without_spec_tokens = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        tool_calls = self.tool_call_parser.parse_tool_calls(response_text=new_text_with_spec_tokens)

        self.logger.debug(f"Generated {len(new_token_ids)} new tokens, with number of tool calls: {len(tool_calls)}")
        return ModelOutput(raw_content=new_text_without_spec_tokens, tool_calls=tool_calls)
        

    def generate_with_tool_calls(
            self,
            prompt: str,
            env_idx: int = 0,
    ) -> SampleResult[STATE]:
        
        self._initialise_conversation(env_idx=env_idx, prompt=prompt)
        env_call_count = 0
        env_exceptions = []

        while env_call_count < self.agent_config.max_env_calls:
            model_output = self._run_inference_on_model()
            env_result = self.environment_pool.handle_output(
                env_idx=env_idx,
                model_output=model_output,
            )
            env_call_count += 1
            if env_result.has_error:
                env_exceptions.append(env_result.exception)
                self.logger.debug(f"Error in environment call {env_call_count}: {env_result.exception}")

            if env_result.should_end_sequence:
                self.logger.debug("Environment decided to end the sequence")
                break

            self.token_handler.add_tool_output_message(content=env_result.output_to_show_model)
            self.token_handler.add_assistant_generation_prompt()

        # Sanity call to cover scenarios such as env ending sequence before the model does
        self.token_handler.add_end_of_sequence_if_not_present()

        env_state = self.environment_pool.get_state(env_idx=env_idx)
        return SampleResult[STATE](
            env_state=env_state,
            input_ids=torch.tensor(self.token_handler.get_all_tokens(), device=self.device),
            source_mask=torch.tensor(self.token_handler.get_source_mask(), device=self.device),
            env_call_count=env_call_count,
            env_exceptions=env_exceptions,
        )
