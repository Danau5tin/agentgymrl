from dotenv import load_dotenv

from evaluation.runner import EvalRunner, EvaluationTask
from evaluation.calc_evals import calculator_task, calculator_tool_parser


if __name__ == "__main__":
    load_dotenv()

    current_task: EvaluationTask = calculator_task
    current_parser = calculator_tool_parser

    runner = EvalRunner(tool_call_parser=current_parser)

    try:
        results = runner.run_evaluation(
            task=current_task,
            output_path="./results.csv",
        )
        print("\nEvaluation script finished successfully.")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nEvaluation script finished with errors.")