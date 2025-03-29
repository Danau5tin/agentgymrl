import datetime
import os

from accelerate import Accelerator
from huggingface_hub import HfApi
import wandb

from agentgymrl.training.tool_calling_grpo_trainer import ToolCallingGRPOTrainer
from agentgymrl.training.common_entities.config import TrainingConfig, AgentConfig, ReportingConfig
from agentgymrl.training.environment_pool import EnvironmentConfig

from environment.calculation_environment import CalculatorEnvironment
from inference.prompting.sys_msg import get_sys_msg
from training.rewards.calculator_reward_func import calculate_reward

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    training_data_path = os.getenv("TRAINING_DATA_PATH")
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    use_peft = os.getenv("USE_PEFT", "false").lower() == "true"


    env_config = EnvironmentConfig(
        env_class=CalculatorEnvironment,
        num_envs=16,
    )
    agent_config = AgentConfig(
        sys_msg=get_sys_msg("phi_4_minimal"),
        tools=[], # Already injected in the system message
        temperature=0.9,
        max_env_calls=20,
        max_new_tokens=1000,
    )
    reporting_config = ReportingConfig.create_wandb(report_every_n_steps=10)


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_short_name = model_name.split('/')[-1]
    prefix = "peft" if use_peft else "full"
    run_name = f"{prefix}_{model_short_name}_calculator_envs{env_config.num_envs}_{timestamp}"
    repo_name = f"{hf_username}/{run_name}"
    output_dir = f"./results/{run_name}"

    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process

    if is_main_process:
        wandb.init(
            project="calculator-agent-training",
            name=run_name,
            config={
                "model": model_name,
                "environment": env_config.env_class.__name__,
                "num_envs": env_config.num_envs,
                "training_mode": "peft" if use_peft else "full_model",
            }
        )

    peft_config = None
    if use_peft:
        from peft import LoraConfig
        peft_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear",
            "modules_to_save": None,
        }
        peft_conf = LoraConfig(**peft_config)

    training_config = TrainingConfig(
        hf_model_name=model_name,
        train_csv_dataset_path=training_data_path,
        output_dir=output_dir,
        agent_config=agent_config,
        environment_config=env_config,
        reward_func=calculate_reward,
        reporting_config=reporting_config,
    )

    trainer = ToolCallingGRPOTrainer(
        config=training_config,
        peft_config=peft_config,
    )

    trainer.train()
    if use_peft:
        model_description = f"""
# {run_name}

This is a PEFT adapter trained with GRPO for calculator tool use.

## Training Details
- Base model: {model_name}
- Environment: {env_config.env_class.__name__}
- Number of environments: {env_config.num_envs}
- Training timestamp: {timestamp}

## Usage
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Load and apply the adapter
peft_config = PeftConfig.from_pretrained("{repo_name}")
model = PeftModel.from_pretrained(model, "{repo_name}")
```
""".strip()
    else:
        model_description = f"""
# {run_name}

This is a full model trained with GRPO for calculator tool use.

## Training Details
- Base model: {model_name}
- Environment: {env_config.env_class.__name__}
- Number of environments: {env_config.num_envs}
- Training timestamp: {timestamp}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the full model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")  # Tokenizer can be loaded from original or fine-tuned model
```
    """.strip()
    
    # Push model to HF Hub
    if is_main_process:
        try:
            print(f"Pushing {'adapter' if use_peft else 'full model'} to Hugging Face Hub: {repo_name}")
            
            # Initialize the Hub API
            api = HfApi(token=hf_token)
            
            # Create the repository
            api.create_repo(repo_id=repo_name, exist_ok=True, private=False)
            print(f"Created repository: {repo_name}")
            
            # Create README with model description
            with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(model_description)
            
            # For full models, provide a warning about potential long upload times
            if not use_peft:
                print("WARNING: Uploading a full model. This may take a significant amount of time depending on model size.")
                print("For very large models, consider using git-lfs or other chunked upload methods if this fails.")
            
            # Upload the model files to the Hub with increased timeout for full models
            timeout = 3600 if not use_peft else 900  # 1 hour for full models, 15 mins for PEFT
            
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                commit_message=f"Upload {'full model' if not use_peft else 'PEFT adapter'} for {run_name}",
                create_pr=False,  # We want to upload directly to main
                ignore_patterns=["*.pyc", ".git*", "__pycache__", "*.log", "wandb/*"],  # Ignore unnecessary files
            )
            
            print(f"Successfully pushed to {repo_name}")
            
            # Add additional metadata if needed
            api.update_repo_visibility(repo_id=repo_name, private=True)
            
            # Log the link in wandb
            hf_model_url = f"https://huggingface.co/{repo_name}"
            wandb.log({"hf_model_url": hf_model_url})
            print(f"Model available at: {hf_model_url}")
            
        except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {e}")
            print("If uploading a full model, consider using git-lfs or splitting the upload into chunks.")

    if wandb.run is not None:
        wandb.finish()