import os # <-- Import os module
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

import pandas as pd
import torch
from agentgymrl.environments.environment import Environment
from agentgymrl.environments.state import STATE
from agentgymrl.inference.model_output import ModelOutput
from agentgymrl.training.tool_call_parsers.tool_call_parser import ToolCallParser
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from inference.agentic_system import AgenticSystem


@dataclass
class EvaluationTask:
    """Configuration for a specific evaluation task."""
    task_name: str
    eval_csv_path: str
    model_name: str
    environment_class: Type[Environment]
    create_initial_state: Callable[[str], STATE]
    verify_answer: Callable[[str, str], bool] # Takes model_response and expected_answer, returns bool
    prompt_column: str = "prompt"
    answer_column: str = "answer"
    temperature: float = 0.9
    max_env_calls: int = 20
    max_new_tokens: int = 1000


class EvalRunner:
    """Runs evaluations based on a provided task configuration."""

    def __init__(
        self,
        tool_call_parser: ToolCallParser,
    ):
        """
        Initializes the EvalRunner.

        Args:
            tool_call_parser: An instance of a class implementing ToolCallParser.
        """
        print("Initializing EvalRunner...")
        self.tool_call_parser = tool_call_parser

        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None
        self.device: torch.device = None


    def _load_model(self, model_path: str) -> None:
        """Loads the model and tokenizer."""
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            print("Warning: Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _agent_callback(
        self,
        state: STATE,
        task: EvaluationTask
    ) -> ModelOutput:
        """
        Generic agent callback function that interacts with the model.
        """
        if not hasattr(state, 'messages') or not isinstance(state.messages, list):
             raise AttributeError(f"State object of type {type(state)} must have a 'messages' attribute (list).")

        inputs = self.tokenizer.apply_chat_template(
            state.messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                temperature=task.temperature,
                do_sample=task.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=task.max_new_tokens,
            )

        input_length = inputs.shape[1]
        generated_tokens = outputs[0, input_length:]

        new_text_with_spec_tokens = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        new_text_without_spec_tokens = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        tool_calls = self.tool_call_parser.parse_tool_calls(response_text=new_text_with_spec_tokens)

        return ModelOutput(raw_content=new_text_without_spec_tokens, tool_calls=tool_calls)

    def _run_single_eval(
        self,
        prompt: str,
        expected_answer: str,
        task: EvaluationTask
    ) -> Dict[str, Any]:
        """Runs a single evaluation item."""
        initial_state = task.create_initial_state(prompt)
        environment = task.environment_class(initial_state=initial_state)

        # Create a wrapper for the agent callback that includes task
        def agent_callback_wrapper(state: STATE):
            return self._agent_callback(state, task)

        system = AgenticSystem[type(initial_state)](
            environment=environment,
            agent_callback=agent_callback_wrapper,
            max_iterations=task.max_env_calls,
        )

        final_state = system.run(state=initial_state)

        # Extract final response - assumes last message is assistant's
        final_response = ""
        if final_state.messages and final_state.messages[-1].get("role") == "assistant":
             content = final_state.messages[-1].get("content", "")
             tool_calls_exist = bool(final_state.messages[-1].get("tool_calls"))

             if content:
                 final_response = content
             elif tool_calls_exist:
                 final_response = "[Assistant used tools but provided no final text]"
             else:
                 final_response = "[Assistant provided no final text or tool calls]"
        elif final_state.messages:
             # Handle cases where the last message isn't from the assistant (e.g., error, max iterations)
             final_response = f"[Evaluation ended. Last message role: {final_state.messages[-1].get('role', 'Unknown')}]"
        else:
             final_response = "[No messages generated in final state]"


        is_correct = task.verify_answer(final_response, expected_answer)

        return {
            "final_state_messages": final_state.messages,
            "final_response": final_response,
            "is_correct": is_correct
        }

    def run_evaluation(
        self,
        task: EvaluationTask,
        output_dir: str = './evaluation_results',
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all prompts in a CSV file based on the provided task.

        Args:
            task: The EvaluationTask configuration.
            output_dir: The base directory where results will be saved.
                        A subdirectory named after the task will be created here.
        """
        csv_path = task.eval_csv_path
        print(f"Starting evaluation for task: {task.task_name}")
        print(f"Reading evaluation data from: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return []
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []

        if task.prompt_column not in df.columns:
            raise ValueError(f"Prompt column '{task.prompt_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        if task.answer_column not in df.columns:
             raise ValueError(f"Answer column '{task.answer_column}' not found in CSV. Available columns: {df.columns.tolist()}")

        task_output_dir = os.path.join(output_dir, task.task_name)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        detailed_results_path = os.path.join(task_output_dir, f'detailed_results_{timestamp}.csv')
        summary_results_path = os.path.join(task_output_dir, f'summary_results_{timestamp}.csv')

        try:
            os.makedirs(task_output_dir, exist_ok=True)
            print(f"Results will be saved in: {task_output_dir}")
        except OSError as e:
            print(f"Error creating output directory {task_output_dir}: {e}")
            return []


        print(f"Found {len(df)} problems to evaluate.")

        self._load_model(task.model_name)

        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {task.task_name}"):
            prompt = row[task.prompt_column]
            expected_answer = str(row[task.answer_column])

            try:
                result = self._run_single_eval(
                    prompt=prompt,
                    expected_answer=expected_answer,
                    task=task
                )

                result_entry = {
                    "index": idx,
                    "prompt": prompt,
                    "expected_answer": expected_answer,
                    "model_response": result["final_response"],
                    "is_correct": result["is_correct"],
                    "conversation_history": result["final_state_messages"]
                }
                results.append(result_entry)

            except Exception as e:
                print(f"Error evaluating problem {idx}: {e}")
                # Log error details, including traceback if needed
                import traceback
                tb_str = traceback.format_exc()
                print(tb_str)
                results.append({
                    "index": idx,
                    "prompt": prompt,
                    "expected_answer": expected_answer,
                    "model_response": f"ERROR: {e}",
                    "is_correct": False,
                    "error": str(e),
                    "traceback": tb_str,
                    "conversation_history": []
                })

        if not results:
            print("\nNo results generated.")
            return []

        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_evaluated = len(results)
        accuracy = (correct_count / total_evaluated * 100) if total_evaluated > 0 else 0

        print("\n=== Evaluation Summary ===")
        print(f"Task: {task.task_name}")
        print(f"Total problems evaluated: {total_evaluated}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")

        try:
            results_df = pd.DataFrame(results)
            if "conversation_history" in results_df.columns:
                 results_df["conversation_history"] = results_df["conversation_history"].astype(str)
            if "traceback" in results_df.columns:
                 results_df["traceback"] = results_df["traceback"].astype(str)

            results_df.to_csv(detailed_results_path, index=False, encoding='utf-8')
            print(f"Detailed results saved to {detailed_results_path}")
        except Exception as e:
            print(f"Error saving detailed results to CSV: {e}")

        summary_data = [{
            "task_name": task.task_name,
            "model_name": task.model_name,
            "total_evaluated": total_evaluated,
            "correct_count": correct_count,
            "accuracy_percent": accuracy
        }]
        summary_df = pd.DataFrame(summary_data)
        try:
            summary_df.to_csv(summary_results_path, index=False, encoding='utf-8') # Added encoding
            print(f"Summary results saved to {summary_results_path}")
        except Exception as e:
            print(f"Error saving summary results to CSV: {e}")


        return results