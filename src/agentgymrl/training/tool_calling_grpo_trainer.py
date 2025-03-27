import logging
from typing import Any, Dict, Generic, List, Tuple

from agentgymrl.environments.state import STATE
import torch
from accelerate.utils import gather, is_peft_model
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import selective_log_softmax

from agentgymrl.training.common_entities.config import TrainingConfig
from agentgymrl.training.common_entities.results import ToolSampleResult
from agentgymrl.training.environment_pool import EnvironmentPool
from agentgymrl.training.tool_calling_generator import ToolCallingGenerator


class ToolCallingGRPOTrainer(GRPOTrainer, Generic[STATE]):
    """
    GRPO Trainer for tool-calling language models.
    This class extends the standard GRPOTrainer to handle the unique aspects
    of training tool-calling models.
    """

    def __init__(
            self,
            config: TrainingConfig[STATE],
            peft_config=None,
            log_level: int = logging.INFO,
            **kwargs
    ):
        # TODO: Add doc string
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.num_envs = config.environment_config.num_envs
        
        train_dataset = load_dataset("csv", data_files=config.train_csv_dataset_path)['train']
        self.logger.info(f"Dataset loaded with {len(train_dataset['prompt'])} prompts")
            
        agent_config = config.agent_config
        reporting_config = config.reporting_config
        super().__init__(
            model=config.hf_model_name,
            reward_funcs=[],  # Initialize empty, we'll use our custom single reward function anyway
            args=GRPOConfig(
                num_generations=self.num_envs,
                per_device_train_batch_size=self.num_envs, # TODO: Understand this better and create config
                gradient_checkpointing=True,
                temperature=agent_config.temperature,
                max_completion_length=agent_config.max_new_tokens,
                report_to=reporting_config.report_to if reporting_config else [], # TODO: Check defaults
                logging_steps=reporting_config.report_every_n_steps if reporting_config else 100, # TODO: Check defaults
                num_train_epochs=3,
                save_steps=100,
                save_total_limit=2,
                output_dir=config.output_dir,
            ),
            train_dataset=train_dataset,
            peft_config=peft_config,
            **kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.logger.info("Setting pad_token to eos_token as it was not defined")
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.reward_func = config.reward_func
        self.reward_weights = torch.tensor([1.0], device=self.accelerator.device)  # Single reward function

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            self.generator = ToolCallingGenerator[STATE](
                model=unwrapped_model,
                tokenizer=self.tokenizer,
                agent_config=agent_config,
                environment_pool=EnvironmentPool[STATE](config=config.environment_config),
                device=self.accelerator.device,
                log_level=log_level,
            )

        if self.ref_model is not None:
            self.logger.info("Adding SyncRefModelCallback")
            self.add_callback(SyncRefModelCallback(
                ref_model=self.ref_model,
                accelerator=self.accelerator
            ))

        self._init_metrics()
        self._log_details()

    def _init_metrics(self) -> None:
        self._metrics["tool_calls_per_sample"] = []
        self._metrics["model_token_ratio"] = []
        self._metrics["model_tokens_percent"] = []
        self._metrics["attn_tokens_percent"] = []
        self._metrics["completion_length"] = []
        self._metrics["mean_reward"] = []
        self._metrics["reward_std"] = []
        self._metrics["kl"] = []
        self._metrics["policy_loss"] = []

    def _log_details(self):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        is_peft = is_peft_model(unwrapped_model)
        
        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)

        self.logger.info("\n\n✅ GRPO Trainer is ready. Details:")
        self.logger.info(f"PEFT model: {is_peft}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Percentage of parameters being trained: {trainable_params/total_params:.4%}")
        self.logger.info(f"Number of environments: {self.num_envs}")
        self.logger.info(f"Device: {self.accelerator.device}")
        self.logger.info("\n\n")
        self.logger.info("Ready to train tool-calling model")
        self.logger.info("\n\n")

    def _pad_sequences_to_equal_length(
            self, 
            sequences: List[torch.Tensor], 
            source_masks: List[torch.Tensor],
            pad_token_id: int,
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Pad all sequences to the same length by adding padding tokens to the left side.
        Also handle the corresponding source_masks correctly.
        
        Args:
            sequences: List of token id tensors
            source_masks: List of source mask tensors (1 for model tokens, 0 for others)
            pad_token_id: Token ID to use for padding
            
        Returns:
            Tuple of (padded_sequences, padded_source_masks, padded_attention_masks)
        """
        device = self.accelerator.device
        max_sequence_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        padded_source_masks = []
        padded_attention_masks = []
        
        for ids, source_mask in zip(sequences, source_masks):
            padding_length = max_sequence_length - len(ids)
            
            if padding_length > 0:
                # Create padding tensors
                padding = torch.full((padding_length,), pad_token_id, dtype=ids.dtype, device=device)
                source_padding = torch.zeros(padding_length, dtype=torch.long, device=device)
                attention_padding = torch.zeros(padding_length, dtype=torch.long, device=device)
                
                # Add padding to left side
                padded_seq = torch.cat([padding, ids.to(device)])
                padded_src_mask = torch.cat([source_padding, source_mask.to(device)])
                padded_attn_mask = torch.cat([attention_padding, torch.ones_like(ids, device=device)])
            else:
                # Already at max length
                padded_seq = ids.to(device)
                padded_src_mask = source_mask.to(device)
                padded_attn_mask = torch.ones_like(ids, device=device)
            
            padded_sequences.append(padded_seq)
            padded_source_masks.append(padded_src_mask)
            padded_attention_masks.append(padded_attn_mask)
        
        # Validate that all sequences and masks have the same length
        lengths = [(len(seq), len(mask)) for seq, mask in zip(padded_sequences, padded_source_masks)]
        self.logger.debug(f"Padded sequence lengths: {lengths}")
            
        return padded_sequences, padded_source_masks, padded_attention_masks
    

    def _create_eos_mask(self, input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        """
        Create a mask that keeps tokens up to and including the first EOS token,
        masking out everything that comes after.
        
        Args:
            input_ids: Tensor of token IDs of shape (batch_size, sequence_length)
            eos_token_id: The ID of the EOS token to detect
            
        Returns:
            torch.Tensor: Binary mask of shape (batch_size, sequence_length)
                        where 1 = keep token, 0 = mask token out
        """
        # TODO: Is this needed?
        self.logger.debug("Creating EOS mask")
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Step 1: Identify positions of all EOS tokens
        is_eos = (input_ids == eos_token_id)
        
        # Step 2: Initialize each sequence's EOS position to the end of sequence
        # If a sequence has no EOS token, we'll keep the whole sequence
        eos_positions = torch.full((batch_size,), seq_length, dtype=torch.long, device=device)
        
        # Step 3: For sequences that have at least one EOS token, find the position of the first one
        sequences_with_eos = is_eos.any(dim=1)
        num_with_eos = sequences_with_eos.sum().item()
        self.logger.info(f"Found {num_with_eos}/{batch_size} sequences with EOS token")
        
        if sequences_with_eos.any():
            first_eos_positions = is_eos.int().argmax(dim=1)
            eos_positions[sequences_with_eos] = first_eos_positions[sequences_with_eos]
        
        # Step 4: Create a mask by comparing each position to the first EOS position
        # For each sequence, positions <= the first EOS position get a 1, others get a 0
        position_indices = torch.arange(seq_length, device=device).expand(batch_size, -1)
        eos_mask = (position_indices <= eos_positions.unsqueeze(1)).int()
        
        return eos_mask
    
    def _calculate_reference_log_pbs(self, input_ids, attention_mask, source_mask):
        """
        Calculate reference log probabilities and adjust source mask accordingly.
        
        Returns:
            Tensor of shape (batch_size, seq_len-1) containing reference log probabilities
        """
        self.logger.debug("Calculating reference log probabilities")
        with torch.inference_mode():
            if self.ref_model is not None:
                self.logger.debug("Using reference model for log probabilities")
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    input_ids.size(1)  # Use full sequence length
                )
            else:
                model_to_unwrap = self.accelerator.unwrap_model(self.model)
                if is_peft_model(model_to_unwrap):
                    self.logger.debug("Using main model with disabled PEFT adapter for log probabilities")
                    with model_to_unwrap.disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            input_ids,
                            attention_mask,
                            input_ids.size(1)
                        )
                else:
                    self.logger.debug("Using main model for log probabilities (no PEFT adapter)")
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        input_ids,
                        attention_mask,
                        input_ids.size(1)  # Use full sequence length
                    )
        
        # Adjust source_mask size to match ref_per_token_logps shape
        # ref_per_token_logps has shape (batch_size, seq_len-1) due to next-token prediction
        if source_mask.size(1) != ref_per_token_logps.size(1):
            self.logger.debug(f"Adjusting source_mask from {source_mask.size(1)} to {ref_per_token_logps.size(1)}")
            # Slice source_mask to match the size of ref_per_token_logps
            adjusted_source_mask = source_mask[:, 1:source_mask.size(1)]
        else:
            adjusted_source_mask = source_mask
            
        # Zero out log probabilities for non-model-generated tokens
        ref_per_token_logps = ref_per_token_logps * adjusted_source_mask.float()
        
        return ref_per_token_logps

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for the model, handling GRPO's pre-duplicated prompts.
        """
        self.logger.info("Preparing inputs for model")
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        answers = [x.get("answer", "") for x in inputs]
        self.logger.info(f"Processing {len(prompts)} prompt instances (already duplicated by GRPO)")
        
        input_ids_list = []
        source_masks = []
        env_call_counts = []
        tool_convo_results = [] 

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            is_peft = is_peft_model(unwrapped_model)
            if is_peft:
                self.logger.debug("Using PEFT model with merged adapter for generation")
                unwrapped_model.merge_adapter()
                
            self.generator.model = unwrapped_model
            self.generator.device = unwrapped_model.device
            
            for i, prompt in enumerate(prompts):
                # Map to environment based on position within batch
                env_idx = i % self.num_envs
                
                self.logger.debug(f"Processing prompt instance {i+1}/{len(prompts)} using environment {env_idx}")
                sample = self.generator.generate_with_tool_calls(
                    prompt=prompt,
                    env_idx=env_idx
                )
                
                tool_convo_results.append(
                    ToolSampleResult[STATE](
                        state=sample.env_state,
                        answer=answers[i],
                        env_exceptions=sample.env_exceptions
                    )
                )
                
                input_ids_list.append(sample.input_ids)
                source_masks.append(sample.source_mask)
                env_call_counts.append(sample.env_call_count)
            
            if is_peft:
                self.logger.debug("Unmerging PEFT adapter after generation")
                unwrapped_model.unmerge_adapter()

        avg_tool_calls = sum(env_call_counts) / max(len(env_call_counts), 1)
        self._metrics["tool_calls_per_sample"].append(avg_tool_calls)

        padded_input_ids, padded_source_masks, padded_attention_masks = self._pad_sequences_to_equal_length(
            input_ids_list, 
            source_masks,
            self.tokenizer.pad_token_id
        )

        assert all(len(ids) == len(mask) for ids, mask in zip(padded_input_ids, padded_source_masks)), \
        "Source masks and input IDs must have the same length after padding"

        # Stack tensors
        input_ids = torch.stack(padded_input_ids).to(device)
        attention_mask = torch.stack(padded_attention_masks).to(device)
        source_mask = torch.stack(padded_source_masks).to(device)
        
        eos_mask = self._create_eos_mask(input_ids, self.tokenizer.eos_token_id)

        # Apply EOS mask to attention mask
        attention_mask = attention_mask * eos_mask
        self.logger.debug(f"Applied EOS mask, remaining tokens: {attention_mask.sum().item()}")

        ref_per_token_logps = self._calculate_reference_log_pbs(input_ids, attention_mask, source_mask)

        self.logger.info("Computing rewards and advantages")
        rewards = self._compute_rewards(tool_convo_results)
        advantages = self._compute_advantages(rewards)
        
        self.logger.info("Input preparation complete")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "source_mask": source_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages
        }

    def _compute_rewards(self, tool_convo_results: List[ToolSampleResult[STATE]]):
        """
        Compute rewards for each conversation using our single reward function.

        Args:
            convo_answer_pairs: List of ConversationWithAnswer objects

        Returns:
            Tensor of rewards for each completion
        """
        self.logger.info(f"Computing rewards for {len(tool_convo_results)} conversations")
        device = self.accelerator.device
        
        output_rewards = self.reward_func(tool_convo_results)
        
        rewards = torch.tensor(
            output_rewards,
            dtype=torch.float32,
            device=device
        )

        rewards = gather(rewards)
        
        mean_reward = rewards.mean().item()
        min_reward = rewards.min().item()
        max_reward = rewards.max().item()
        self.logger.info(f"Rewards - Mean: {mean_reward:.4f}, Min: {min_reward:.4f}, Max: {max_reward:.4f}")

        self._metrics["mean_reward"].append(mean_reward)

        process_slice = slice(
            self.accelerator.process_index * len(tool_convo_results),
            (self.accelerator.process_index + 1) * len(tool_convo_results),
        )

        return rewards[process_slice]

    def _compute_advantages(self, rewards):
        """
        Compute advantages by normalizing rewards within each prompt group.

        Args:
            rewards: Tensor of rewards for all completions

        Returns:
            Tensor of advantages
        """
        self.logger.debug("Computing advantages")
        # Reshape to (num_prompts, num_generations)
        grouped_rewards = rewards.view(-1, self.num_generations)

        # Compute mean and std for each group
        mean_rewards = grouped_rewards.mean(dim=1, keepdim=True)
        std_rewards = grouped_rewards.std(dim=1, keepdim=True)

        # Track standard deviation in metrics
        reward_std = std_rewards.mean().item()
        self._metrics["reward_std"].append(reward_std)
        self.logger.debug(f"Average reward std: {reward_std:.4f}")

        # Normalize to get advantages (avoid division by zero)
        advantages = (grouped_rewards - mean_rewards) / (std_rewards + 1e-4)

        # Reshape back to original shape
        return advantages.reshape(-1)
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """
        Modified to handle full sequence lengths and clearly document the output shape.
        
        Returns:
            per_token_logps: Tensor of shape (batch_size, seq_len-1) containing log probabilities
                            for predicting tokens at positions 1 to seq_len
        """
        self.logger.debug(f"Getting per-token log probabilities for sequence length {logits_to_keep}")
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit
        
        # For full sequence, we need all corresponding input tokens for the logits
        next_tokens = input_ids[:, 1:logits_to_keep+1]
        
        # This produces a tensor of shape (batch_size, seq_len-1)
        return selective_log_softmax(logits, next_tokens)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the GRPO loss using source mask to identify model-generated tokens.
        The source mask distinguishes between model-generated tokens (1) and 
        prompt/tool output tokens (0).
        """
        self.logger.debug("Computing loss")
        if return_outputs:
            raise ValueError("ToolCallingGRPOTrainer does not support returning outputs")
        
        # Get inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        source_mask = inputs["source_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]
        
        # Ensure model is in training mode
        model.train()
        
        # Debug model state
        unwrapped_model = self.accelerator.unwrap_model(model)
        is_peft = is_peft_model(unwrapped_model)
        self.logger.debug(f"Model uses PEFT: {is_peft}")
        self.logger.debug(f"Model in training mode: {model.training}")
        
        # CRITICAL FIX: Instead of trying to get gradients directly from the model output,
        # we'll create a differentiable computation using the parameters
        
        # Step 1: Get the input embeddings - these are differentiable w.r.t model parameters
        inputs_embeds = unwrapped_model.get_input_embeddings()(input_ids)
        
        # Step 2: Force gradient tracking on the embeddings
        if not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)
        self.logger.debug(f"Input embeddings require_grad: {inputs_embeds.requires_grad}")
        
        # Step 3: Forward pass with embeddings instead of input_ids
        outputs = unwrapped_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        self.logger.debug(f"Logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")
        
        # Continue with loss computation
        # Note: We need to shift logits and next_tokens by 1 to match the expected pattern
        logits = logits[:, :-1, :]  # (B, L-1, V)
        next_tokens = input_ids[:, 1:]
        
        # Force gradient retention by creating a new computation path
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)
        self.logger.debug(f"token_log_probs requires_grad: {token_log_probs.requires_grad}")
        
        # Adjust masks to match dimensions
        adjusted_attention_mask = attention_mask[:, 1:attention_mask.size(1)]
        adjusted_source_mask = source_mask[:, 1:source_mask.size(1)]
        
        # Make sure shapes match
        if token_log_probs.shape != adjusted_source_mask.shape:
            min_len = min(token_log_probs.shape[1], adjusted_source_mask.shape[1])
            token_log_probs = token_log_probs[:, :min_len]
            ref_per_token_logps = ref_per_token_logps[:, :min_len]
            adjusted_attention_mask = adjusted_attention_mask[:, :min_len]
            adjusted_source_mask = adjusted_source_mask[:, :min_len]
        
        # Compute KL divergence and policy gradient loss
        per_token_kl = torch.exp(ref_per_token_logps - token_log_probs) - (ref_per_token_logps - token_log_probs) - 1
        per_token_policy_loss = torch.exp(token_log_probs - token_log_probs.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_policy_loss - self.beta * per_token_kl)
        
        # Create mask for model-generated tokens
        model_token_mask = adjusted_attention_mask * adjusted_source_mask
        
        # Apply mask and compute average loss
        if model_token_mask.sum() > 0:
            loss = (per_token_loss * model_token_mask).sum() / model_token_mask.sum()
        else:
            self.logger.warning("No model-generated tokens found, using attention mask for loss")
            loss = (per_token_loss * adjusted_attention_mask).sum() / (adjusted_attention_mask.sum() + 1e-8)
        
        self.logger.debug(f"Final loss value: {loss.item()}, requires_grad: {loss.requires_grad}")
        
        kl_loss = ((self.beta * per_token_kl) * model_token_mask).sum() / (model_token_mask.sum() + 1e-8)
        policy_loss = (per_token_policy_loss * model_token_mask).sum() / (model_token_mask.sum() + 1e-8)
        completion_length = attention_mask.sum(1).float().mean().item()
        model_tokens_in_attn = model_token_mask.sum().item()
        non_pad_tokens = adjusted_attention_mask.sum().item()
        model_token_ratio = model_tokens_in_attn / non_pad_tokens if non_pad_tokens > 0 else 0.0
        total_adj_tokens = adjusted_source_mask.numel()
        model_adj_tokens = (adjusted_source_mask > 0).sum().item()

        self._metrics["completion_length"].append(completion_length)
        self._metrics["kl"].append(kl_loss.item())
        self._metrics["policy_loss"].append(-policy_loss.item())
        self._metrics["model_token_ratio"].append(model_token_ratio)
        self._metrics["model_tokens_percent"].append(model_adj_tokens / total_adj_tokens)
        self._metrics["attn_tokens_percent"].append(non_pad_tokens / total_adj_tokens)
        
        return loss
