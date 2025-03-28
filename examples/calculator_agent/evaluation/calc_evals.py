from agentgymrl.training.tool_call_parsers.phi_4_mini_instruct import (
    Phi4MiniInstructToolCallParser,
)

from environment.calculation_environment import CalculatorEnvironment, CalculatorState
from evaluation.runner import EvaluationTask
from evaluation.verifiers.answer_verifier import is_correct_answer
from inference.prompting.sys_msg import get_sys_msg


def create_calculator_initial_state(prompt: str) -> CalculatorState:
    """Creates the initial state for the calculator task."""
    state = CalculatorState(
        messages=[
            {"role": "system", "content": get_sys_msg("phi_4_tools")},
            {"role": "user", "content": prompt},
        ]
    )
    return state


calculator_task = EvaluationTask(
    task_name="Phi-4-mini-instruct-standard-calculator",
    model_name="microsoft/Phi-4-mini-instruct",
    eval_csv_path="evaluation/data/calculator_evals.csv",
    environment_class=CalculatorEnvironment,
    create_initial_state=create_calculator_initial_state,
    verify_answer=is_correct_answer,
)


calculator_tool_parser = Phi4MiniInstructToolCallParser()
